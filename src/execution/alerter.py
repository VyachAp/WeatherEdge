"""Telegram alerting system for trade signals and portfolio updates."""

from __future__ import annotations

import asyncio
import enum
import logging
import re
import traceback
from datetime import datetime, time, timezone
from typing import TYPE_CHECKING

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TelegramError
from telegram.ext import Application, CallbackQueryHandler, ContextTypes

from sqlalchemy import select

from src.config import settings
from src.db.engine import async_session
from src.db.models import BankrollLog, Market, Signal, Trade, TradeDirection, TradeStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.risk.drawdown import DrawdownState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MarkdownV2 escaping
# ---------------------------------------------------------------------------

_MD2_SPECIAL = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")


def _escape_md2(text: object) -> str:
    """Escape Telegram MarkdownV2 special characters in dynamic text."""
    return _MD2_SPECIAL.sub(r"\\\1", str(text))


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.75:
        return "HIGH"
    if confidence >= 0.55:
        return "MEDIUM"
    return "LOW"




# ---------------------------------------------------------------------------
# AlertType enum
# ---------------------------------------------------------------------------


class AlertType(str, enum.Enum):
    NEW_SIGNAL = "new_signal"
    POSITION_UPDATE = "position_update"
    RESOLUTION = "resolution"
    DAILY_SUMMARY = "daily_summary"
    DRAWDOWN_WARNING = "drawdown_warning"
    SYSTEM_ERROR = "system_error"


# ---------------------------------------------------------------------------
# Alerter
# ---------------------------------------------------------------------------


class Alerter:
    """Async Telegram alerter with rate-limited message queue."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self._token = bot_token or settings.TELEGRAM_BOT_TOKEN
        self._chat_id = chat_id or settings.TELEGRAM_CHAT_ID
        self._enabled = bool(self._token and self._chat_id)

        self._app: Application | None = None  # type: ignore[type-arg]
        self._queue: asyncio.Queue[tuple[str, InlineKeyboardMarkup | None]] = (
            asyncio.Queue()
        )
        self._drain_task: asyncio.Task[None] | None = None

    # -- Lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        """Initialise the bot and start the message-drain loop."""
        if not self._enabled:
            logger.info("Alerter running in dry-run mode (no Telegram credentials)")
            return

        self._app = (
            Application.builder().token(self._token).build()
        )
        self._app.add_handler(CallbackQueryHandler(self._handle_callback))

        await self._app.initialize()
        # Terminate any lingering getUpdates session from a previous instance.
        # The delete_webhook call with a unique offset forces Telegram to
        # invalidate the old long-poll connection.  We then wait for the old
        # instance's poll to time out before we start our own.
        await self._app.bot.delete_webhook(drop_pending_updates=True)
        logger.info("Waiting for previous Telegram polling session to expire…")
        await asyncio.sleep(5)
        await self._app.start()
        await self._app.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES,
        )  # type: ignore[union-attr]

        self._drain_task = asyncio.create_task(self._drain_queue())
        logger.info("Alerter started — Telegram delivery enabled")

    async def shutdown(self) -> None:
        """Gracefully stop the alerter."""
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass

        if self._app is not None:
            await self._app.updater.stop()  # type: ignore[union-attr]
            await self._app.stop()
            await self._app.shutdown()

        logger.info("Alerter shut down")

    # -- Rate-limited queue ---------------------------------------------------

    async def _enqueue(
        self,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        if not self._enabled:
            logger.info("[DRY-RUN] %s", text)
            return
        await self._queue.put((text, reply_markup))

    async def _drain_queue(self) -> None:
        """Send queued messages respecting Telegram rate limits."""
        assert self._app is not None  # noqa: S101
        while True:
            text, markup = await self._queue.get()
            try:
                msg = await self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_markup=markup,
                    disable_web_page_preview=True,
                )
                logger.info("Telegram message delivered (id=%s, queue=%d)",
                            msg.message_id, self._queue.qsize())
            except RetryAfter as exc:
                logger.warning("Rate limited — retrying after %ss", exc.retry_after)
                await asyncio.sleep(exc.retry_after)
                await self._queue.put((text, markup))
            except TelegramError as exc:
                if "can't parse entities" in str(exc).lower():
                    logger.warning("MarkdownV2 parse failed, falling back to plain text")
                    plain = re.sub(r"\\(.)", r"\1", text)  # strip escapes
                    plain = re.sub(r"[*_~`]", "", plain)
                    try:
                        await self._app.bot.send_message(
                            chat_id=self._chat_id,
                            text=plain,
                            reply_markup=markup,
                            disable_web_page_preview=True,
                        )
                    except TelegramError as inner:
                        logger.error("Plain-text fallback also failed: %s", inner)
                else:
                    logger.error("Failed to send Telegram message: %s", exc)
            finally:
                self._queue.task_done()
            await asyncio.sleep(0.05)  # ~20 msg/sec headroom

    # -- Alert: NEW_SIGNAL ----------------------------------------------------

    # -- Alert: POSITION_UPDATE -----------------------------------------------

    async def send_position_update(
        self,
        trade: Trade,
        market: Market,
        old_price: float,
        new_price: float,
    ) -> None:
        """Alert when a market moves >5% on an open position."""
        e = _escape_md2
        change_pct = (new_price - old_price) / old_price * 100 if old_price else 0
        direction_emoji = "\U0001f7e2" if change_pct > 0 else "\U0001f534"
        unrealised = (new_price - (trade.entry_price or 0)) * trade.stake_usd

        text = (
            f"{direction_emoji} *Position Update*\n"
            f"\n"
            f"\"{e(market.question)}\"\n"
            f"\n"
            f"Price: {e(f'{old_price * 100:.0f}%')} → {e(f'{new_price * 100:.0f}%')} "
            f"\\({e(f'{change_pct:+.1f}%')}\\)\n"
            f"Direction: {e(trade.direction.value)}\n"
            f"Stake: {e(f'${trade.stake_usd:,.0f}')}\n"
            f"Unrealised P&L: {e(f'${unrealised:+,.2f}')}"
        )
        await self._enqueue(text)

    # -- Alert: RESOLUTION ----------------------------------------------------

    async def send_resolution(self, trade: Trade, market: Market) -> None:
        """Alert when a market resolves — show P&L."""
        e = _escape_md2
        won = trade.status == TradeStatus.WON
        emoji = "\u2705" if won else "\u274c"
        result = "WON" if won else "LOST"
        pnl = trade.pnl or 0.0

        text = (
            f"{emoji} *Resolution: {e(result)}*\n"
            f"\n"
            f"\"{e(market.question)}\"\n"
            f"\n"
            f"Direction: {e(trade.direction.value)}\n"
            f"Stake: {e(f'${trade.stake_usd:,.0f}')}\n"
            f"P&L: {e(f'${pnl:+,.2f}')}"
        )
        await self._enqueue(text)

    # -- Alert: DAILY_SUMMARY -------------------------------------------------

    async def send_daily_summary(self, session: AsyncSession) -> None:
        """End-of-day recap: open positions, P&L, bankroll, signals."""
        from sqlalchemy import func, select

        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )

        # Signals detected today
        signal_count = (
            await session.execute(
                select(func.count(Signal.id)).where(Signal.created_at >= today_start)
            )
        ).scalar_one()

        # Trades opened today
        opened_count = (
            await session.execute(
                select(func.count(Trade.id)).where(
                    Trade.opened_at >= today_start,
                    Trade.status.in_([TradeStatus.OPEN, TradeStatus.PENDING]),
                )
            )
        ).scalar_one()

        # Trades resolved today
        won_count = (
            await session.execute(
                select(func.count(Trade.id)).where(
                    Trade.closed_at >= today_start,
                    Trade.status == TradeStatus.WON,
                )
            )
        ).scalar_one()
        lost_count = (
            await session.execute(
                select(func.count(Trade.id)).where(
                    Trade.closed_at >= today_start,
                    Trade.status == TradeStatus.LOST,
                )
            )
        ).scalar_one()

        # Daily P&L from resolved trades
        daily_pnl = (
            await session.execute(
                select(func.coalesce(func.sum(Trade.pnl), 0.0)).where(
                    Trade.closed_at >= today_start,
                    Trade.status.in_([TradeStatus.WON, TradeStatus.LOST]),
                )
            )
        ).scalar_one()

        # Latest bankroll state
        latest_log = (
            await session.execute(
                select(BankrollLog)
                .order_by(BankrollLog.timestamp.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        bankroll = latest_log.balance if latest_log else settings.INITIAL_BANKROLL
        drawdown = latest_log.drawdown_pct if latest_log else 0.0

        from src.risk.drawdown import DrawdownLevel

        if drawdown >= 0.20:
            dd_label = DrawdownLevel.PAUSED.value.upper()
        elif drawdown >= 0.10:
            dd_label = DrawdownLevel.CAUTION.value.upper()
        else:
            dd_label = DrawdownLevel.NORMAL.value.upper()

        e = _escape_md2
        date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")

        text = (
            f"\U0001f4cb *Daily Summary* — {e(date_str)}\n"
            f"\n"
            f"\U0001f50e Signals detected: {e(signal_count)}\n"
            f"\U0001f4c8 Trades opened: {e(opened_count)}\n"
            f"\u2705 Resolved: {e(won_count)} won, {e(lost_count)} lost\n"
            f"\U0001f4b0 P&L today: {e(f'${daily_pnl:+,.2f}')}\n"
            f"\n"
            f"\U0001f4bc Bankroll: {e(f'${bankroll:,.2f}')}\n"
            f"\U0001f4c9 Drawdown: {e(f'{drawdown * 100:.1f}%')} \\({e(dd_label)}\\)"
        )
        await self._enqueue(text)

    # -- Alert: DRAWDOWN_WARNING ----------------------------------------------

    async def send_drawdown_warning(self, state: DrawdownState) -> None:
        """Alert when entering CAUTION or PAUSED state."""
        e = _escape_md2
        emoji = "\u26a0\ufe0f" if state.level.value == "caution" else "\U0001f6d1"

        text = (
            f"{emoji} *Drawdown Warning: {e(state.level.value.upper())}*\n"
            f"\n"
            f"Drawdown: {e(f'{state.drawdown_pct * 100:.1f}%')}\n"
            f"Peak: {e(f'${state.peak:,.0f}')} → Current: {e(f'${state.current:,.0f}')}\n"
            f"Action: {e(state.action)}\n"
            f"Size multiplier: {e(f'{state.size_multiplier:.1f}x')}"
        )
        await self._enqueue(text)

    # -- Alert: SYSTEM_ERROR --------------------------------------------------

    async def send_system_error(
        self,
        error: Exception,
        context: str = "",
    ) -> None:
        """Alert when ingestion or signal pipeline fails."""
        e = _escape_md2
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        tb_short = "".join(tb[-3:])[:500]

        ctx_line = f"\nContext: {e(context)}" if context else ""

        text = (
            f"\U0001f6a8 *System Error*{ctx_line}\n"
            f"\n"
            f"`{e(type(error).__name__)}`: {e(str(error))}\n"
            f"\n"
            f"```\n{e(tb_short)}\n```"
        )
        await self._enqueue(text)

    # -- Alert: MARKET_DISCOVERY ----------------------------------------------

    async def send_market_discovery(
        self,
        station_icao: str,
        current_temp_f: float,
        markets: list,
    ) -> None:
        """Alert when backward resolution discovers markets near current obs."""
        if not markets:
            return

        e = _escape_md2
        lines = [
            f"\U0001f50d *Market Discovery*  `{e(station_icao)}`  {current_temp_f:.1f}F",
            "",
        ]

        for m in markets[:5]:
            q = (m.question or "?")[:60]
            yes = f"{m.current_yes_price:.0%}" if m.current_yes_price else "?"
            thresh = f"{m.parsed_threshold:.0f}F" if m.parsed_threshold is not None else "?"
            dist = ""
            if m.parsed_threshold is not None:
                d = current_temp_f - m.parsed_threshold
                dist = f", {d:+.1f}F"
            lines.append(f"  {e(q)}")
            lines.append(f"    YES: {yes}  thresh: {thresh}{dist}")

        if len(markets) > 5:
            lines.append(f"\n_\\+{len(markets) - 5} more_")

        await self._enqueue("\n".join(lines))

    # -- Callback handler -----------------------------------------------------

    async def _handle_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle inline keyboard button presses."""
        query = update.callback_query
        if query is None or query.data is None:
            return
        await query.answer()

        parts = query.data.split(":", 1)
        if len(parts) != 2:
            return
        action, market_id = parts

        if action == "exec":
            await query.edit_message_reply_markup(reply_markup=None)
            # Attempt live execution for the most recent PENDING trade
            from src.execution.polymarket_client import place_order as _place_order
            try:
                async with async_session() as sess:
                    stmt = (
                        select(Trade)
                        .where(Trade.market_id == market_id)
                        .where(Trade.status == TradeStatus.PENDING)
                        .order_by(Trade.opened_at.desc())
                        .limit(1)
                    )
                    result = await sess.execute(stmt)
                    trade = result.scalar_one_or_none()
                    if trade is not None:
                        ok = await _place_order(trade, sess)
                        if ok:
                            trade.status = TradeStatus.OPEN
                            await sess.commit()
                            await query.message.reply_text("\u2705 Order placed")  # type: ignore[union-attr]
                        else:
                            await sess.commit()
                            await query.message.reply_text(  # type: ignore[union-attr]
                                f"\u274c Order failed: {trade.exchange_status}",
                            )
                    else:
                        await query.message.reply_text("\u26a0\ufe0f No pending trade found")  # type: ignore[union-attr]
            except Exception:
                logger.exception("Callback execution failed for %s", market_id)
                await query.message.reply_text("\u274c Execution error")  # type: ignore[union-attr]

        elif action == "skip":
            logger.info("Signal skipped for market %s", market_id)
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("\u23ed Skipped")  # type: ignore[union-attr]

        elif action == "detail":
            detail_text = await self._build_detail_message(market_id)
            await self._enqueue(detail_text)

    async def _build_detail_message(self, market_id: str) -> str:
        """Build a detailed ensemble breakdown for a market."""
        from sqlalchemy import select

        e = _escape_md2

        async with async_session() as session:
            signal_row = (
                await session.execute(
                    select(Signal)
                    .where(Signal.market_id == market_id)
                    .order_by(Signal.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

            market_row = await session.get(Market, market_id)

        if signal_row is None or market_row is None:
            return f"\U0001f4ca *Detail*\n\nNo signal data found for {e(market_id)}"

        aviation = f"{signal_row.aviation_prob * 100:.1f}%" if signal_row.aviation_prob is not None else "n/a"
        consensus = f"{signal_row.model_prob * 100:.1f}%"
        market_p = f"{signal_row.market_prob * 100:.1f}%"

        return (
            f"\U0001f4ca *Signal Detail*\n"
            f"\n"
            f"\"{e(market_row.question)}\"\n"
            f"\n"
            f"Aviation: {e(aviation)}\n"
            f"Consensus: {e(consensus)}\n"
            f"Market: {e(market_p)}\n"
            f"Edge: {e(f'{signal_row.edge * 100:+.1f}%')}\n"
            f"Direction: {e(signal_row.direction.value)}\n"
            f"Confidence: {e(f'{signal_row.confidence:.2f}')}"
        )


# ---------------------------------------------------------------------------
# Singleton & convenience
# ---------------------------------------------------------------------------

_alerter: Alerter | None = None


def get_alerter() -> Alerter:
    """Return the module-level Alerter singleton."""
    global _alerter  # noqa: PLW0603
    if _alerter is None:
        _alerter = Alerter()
    return _alerter


async def send_daily_summary() -> None:
    """Entry point for APScheduler daily job."""
    alerter = get_alerter()
    async with async_session() as session:
        await alerter.send_daily_summary(session)
