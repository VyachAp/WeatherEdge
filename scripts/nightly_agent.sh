#!/usr/bin/env bash
#
# Nightly orchestrator launcher (propose-only). Runs the WeatherEdge nightly
# mastering agent headless via Claude Code, then prints the .env/src guard diff.
#
# Runtime: LOCAL path (this Mac reaches the prod DB via .env DATABASE_URL). A
# cloud routine is the alternative but needs the DO managed-Postgres exposed to
# Anthropic's cloud + secrets injected — see docs/improvements.md "[in-flight]
# Nightly orchestrator agent".
#
# Wire it up:  see the launchd plist at deploy/com.weatheredge.nightly.plist
# Dry-run now:  bash scripts/nightly_agent.sh
#
set -euo pipefail

REPO="/Users/viacheslav.apasov/Business/WeatherEdge"
cd "$REPO"

# launchd runs with a minimal PATH; make venv python + claude resolvable.
export PATH="$REPO/.venv/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

mkdir -p runtime
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="runtime/nightly_agent_$(date -u +%Y%m%d).log"
RAW="runtime/nightly_agent_$(date -u +%Y%m%d).jsonl"   # raw stream-json event log (debug)
PROMPT_FILE="docs/agents/nightly_orchestrator_prompt.md"

# Pretty one-line-per-event view of the stream-json feed (readable live progress).
prettify() {
  jq -rc --unbuffered '
    if .type=="system" and .subtype=="init" then "⚙️  session start (model \(.model // "?"))"
    elif .type=="assistant" then
      ( .message.content[]? |
          if .type=="text" and ((.text // "")|length>0) then "💬 " + ((.text)|gsub("\n";" ")|.[0:200])
          elif .type=="tool_use" then "🔧 " + .name + "  " + (((.input.command // .input.file_path // .input.description // "")|tostring)|gsub("\n";" ")|.[0:160])
          else empty end )
    elif .type=="result" then "✅ done: " + (((.result // "")|tostring)|gsub("\n";" ")|.[0:200]) + "   [cost $\(.total_cost_usd // 0), \(.num_turns // "?") turns]"
    else empty end' 2>/dev/null
}

# Stream live to the terminal AND append to the log via tee, so an interactive
# run isn't silent. Under launchd there's no terminal — tee still writes the log.
# -p / --output-format stream-json / --verbose : headless agentic mode that
#               STREAMS newline-delimited JSON events as they happen (text mode
#               buffers to the end — the reason early runs looked dead). We tee the
#               raw JSONL to $RAW for debugging and pipe it through prettify() for a
#               readable live line-per-step view on the terminal + $LOG.
# --settings  : scoped permissions + deny-hooks for THIS run only (see
#               nightly_agent.settings.json). DEFAULT permission mode (NO
#               --dangerously-skip-permissions): verified 2026-07-09 that headless
#               default mode DENIES any tool not in permissions.allow and CONTINUES
#               (no hang), so the allow/deny surface is the real guard.
# --max-turns : runaway backstop (generous; the run is meant to be token-heavy).
echo "================ nightly orchestrator ${STAMP} ================" | tee -a "$LOG"
echo "(streaming; raw events → $RAW)" | tee -a "$LOG"

claude -p "$(cat "$PROMPT_FILE")" \
  --output-format stream-json --verbose \
  --settings scripts/nightly_agent.settings.json \
  --max-turns 80 \
  2>>"$LOG" \
  | tee -a "$RAW" \
  | prettify \
  | tee -a "$LOG" \
  || echo "!! claude pipeline exited non-zero (see $LOG / $RAW)" | tee -a "$LOG"

{
  echo "---------------- .env/src guard diff (MUST be empty) ----------------"
  git diff --stat -- .env src/ alembic/ || true
  echo "================ done ${STAMP} ================"
} | tee -a "$LOG"

echo "nightly orchestrator finished; log: $LOG"
