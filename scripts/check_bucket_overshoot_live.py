"""Dry-run the `bucket_overshoot` lock branch against LIVE Polymarket + METAR data.

Run this before/after flipping `BUCKET_OVERSHOOT_LOCK_ENABLED` to see exactly which
buckets the rule considers dead right now and what the book would charge for NO.
Fires no orders and touches no DB.

    python -m scripts.check_bucket_overshoot_live

A healthy run mid-day shows many dead buckets, nearly all "skip (repriced)" — real
fires happen only in the ~seconds after a fresh killing METAR. See
docs/mastering_playbook.md Â§5 (2026-07-10).

Read-only: fetches today's markets and today's METARs, runs the real decision
function, and prints what the bot *would* do. Places no orders.
"""
import asyncio, json, sys
from datetime import datetime, timezone, timedelta
import httpx, pandas as pd
sys.path.insert(0, '.')
from src.config import settings
from src.signals.lock_rules import evaluate_lock
from src.signals.state_aggregator import WeatherState
from src.signals.mapper import CITY_ICAO, icao_timezone
from src.ingestion.polymarket import _city_to_polymarket_slug, parse_question

settings.BUCKET_OVERSHOOT_LOCK_ENABLED = True

class MK:
    def __init__(self, q, thr, op, end):
        self.question, self.parsed_threshold, self.parsed_operator, self.end_date = q, thr, op, end
        self.parsed_target_date = None

async def main():
    now = datetime.now(timezone.utc)
    today = pd.Timestamp(now).normalize()
    cities = [c for c in CITY_ICAO if CITY_ICAO[c] not in settings.bucket_overshoot_excluded]
    async with httpx.AsyncClient(timeout=30) as c:
        # live METARs (2h window so we get the day's history)
        icaos = sorted({CITY_ICAO[x] for x in cities})
        rows = []
        for i in range(0, len(icaos), 15):
            batch = icaos[i:i+15]
            rr = await c.get('https://aviationweather.gov/api/data/metar',
                             params={'ids': ','.join(batch), 'format':'json', 'hours': 30})
            try: rows.extend(rr.json())
            except Exception: pass
        met = pd.DataFrame(rows)
        met['obs'] = pd.to_datetime(met.reportTime, utc=True, format='mixed')
        met['is_speci'] = met.rawOb.str.startswith('SPECI')
        met = met[~met.is_speci]
        met['temp_f'] = met.temp*9/5+32

        n_fire = 0; n_mkt = 0
        for city in cities:
            icao = CITY_ICAO[city]
            slug = f"highest-temperature-in-{_city_to_polymarket_slug(city)}-on-{today.strftime('%B').lower()}-{today.day}-{today.year}"
            try:
                ev = (await c.get('https://gamma-api.polymarket.com/events', params={'slug': slug})).json()
            except Exception: continue
            if not ev: continue
            st_met = met[met.icaoId == icao]
            if st_met.empty: continue
            tz = icao_timezone(icao)
            hist = tuple((o.obs.to_pydatetime(), float(o.temp_f)) for o in st_met.sort_values('obs').itertuples())
            state = WeatherState(station_icao=icao, current_max_f=max(t for _,t in hist),
                metar_trend_rate=0.0, dewpoint_trend_rate=0.0, forecast_peak_f=0.0,
                hours_until_peak=0.0, solar_declining=False, solar_decline_magnitude=0.0,
                cloud_rising=False, cloud_rise_magnitude=0.0, routine_count_today=len(hist),
                has_forecast=False, routine_history=hist)
            for m in ev[0].get('markets', []):
                q = m.get('question') or ''
                p = parse_question(q)
                if not p or p.operator != 'exactly' or p.threshold is None: continue
                end = pd.Timestamp(m.get('endDate')).to_pydatetime()
                n_mkt += 1
                mk = MK(q, float(p.threshold), 'exactly', end)
                d = evaluate_lock(state, mk, now_utc=now)
                if d.branch == 'bucket_overshoot':
                    n_fire += 1
                    ct = m.get('clobTokenIds'); ct = json.loads(ct) if isinstance(ct,str) else ct
                    oc = m.get('outcomes'); oc = json.loads(oc) if isinstance(oc,str) else oc
                    low=[o.lower() for o in oc]; yi = low.index('yes') if 'yes' in low else 0
                    book = (await c.get('https://clob.polymarket.com/book', params={'token_id': ct[yi]})).json()
                    bids = sorted([(float(x['price']), float(x['size'])) for x in book.get('bids',[])], reverse=True)
                    best_bid = bids[0][0] if bids else 0.0
                    no_cost = 1 - best_bid
                    cap = sum(sz*(1-pr) for pr,sz in bids if pr >= best_bid-0.03)
                    gate = 'WOULD BUY NO' if no_cost <= settings.BUCKET_OVERSHOOT_MAX_COST else 'skip (repriced)'
                    print(f'{icao} | {q[:58]:58s} | max={state.current_max_f:.1f}F rc={len(hist)} '
                          f'| NO cost={no_cost:.3f} depth=${cap:6.0f} | {gate}')
        print(f'\nexactly-markets evaluated: {n_mkt}   bucket_overshoot fired: {n_fire}')
        print(f'excluded stations: {sorted(settings.bucket_overshoot_excluded)}')

asyncio.run(main())
