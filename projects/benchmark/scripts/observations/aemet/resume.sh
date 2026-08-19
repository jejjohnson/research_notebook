#!/usr/bin/env bash
# Resume an AEMET scrape inside tmux with a CPU keepalive sidecar.
#
# Why a keepalive: Azure ML's dsiidlestopagent monitors CPU activity to
# decide when to shut down idle compute. Long AEMET scrapes are mostly
# blocked in HTTP waits / rate-limit gates, so CPU stays near zero and
# Azure has previously killed running scrapes mid-period — that is what
# ended the 2026-05-05 run at period 1955-1959, with no error in the log.
# The sidecar below burns CPU on a deterministic ~0.5% duty cycle (50 ms
# busy every 10 s) so a sample-based idle detector consistently sees
# activity.
#
# Usage:
#   scripts/observations/aemet/resume.sh monthly --start 1955
#   scripts/observations/aemet/resume.sh daily --start 1950 --end 1999
#   scripts/observations/aemet/resume.sh smoke
#
# Inside the resulting tmux session, detach with Ctrl-b d. Re-attach with
# `tmux attach -t aemet-<flavor>`. Logs at projects/benchmark/logs/.

set -euo pipefail

flavor="${1:?usage: $0 <smoke|monthly|daily> [extra args...]}"
shift || true

case "$flavor" in
  smoke|monthly|daily) ;;
  *) echo "unknown flavor: $flavor (expected smoke|monthly|daily)" >&2; exit 2 ;;
esac

session="aemet-${flavor}"
script="scripts/observations/aemet/${flavor}.py"
# scripts/observations/aemet/ -> project root is three levels up.
project_root="$(cd "$(dirname "$0")/../../.." && pwd)"

# Accept either env var (matches scratch_root() resolution in common.py).
scratch_root="${AEMET_SCRATCH_ROOT:-${BENCHMARK_AEMET_ROOT:-}}"
if [[ -z "$scratch_root" ]]; then
    echo "AEMET_SCRATCH_ROOT (or BENCHMARK_AEMET_ROOT) is not set." >&2
    echo "Set it (recommended in ~/.bashrc) to keep data out of the repo:" >&2
    echo "  export AEMET_SCRATCH_ROOT=\$HOME/scratch/aemet" >&2
    exit 1
fi

if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session '$session' already exists. Attach with: tmux attach -t $session" >&2
    exit 1
fi

# Forward extra args safely — preserves quoting / spaces / metacharacters.
escaped_args=""
for arg in "$@"; do
    escaped_args+=" $(printf '%q' "$arg")"
done

# Launcher script with the keepalive sidecar + the scrape. Self-deletes on
# exit. Written to a temp file so the tmux command stays a clean exec —
# no nested shell-string escaping landmines.
launcher="$(mktemp -t aemet-resume-XXXXXX.sh)"
cat > "$launcher" <<LAUNCHER
#!/usr/bin/env bash
set -e
self="\$0"
trap 'rm -f "\$self"' EXIT

# CPU keepalive: deterministic ~0.5% duty cycle so dsiidlestopagent
# samples consistent activity even when the scrape is blocked on I/O.
python3 - <<'PY' &
import time
PERIOD = 10.0
BUSY = 0.05
while True:
    started = time.perf_counter()
    while time.perf_counter() - started < BUSY:
        pass
    elapsed = time.perf_counter() - started
    time.sleep(max(0.0, PERIOD - elapsed))
PY
keepalive_pid=\$!
trap 'kill \$keepalive_pid 2>/dev/null || true; rm -f "\$self"' EXIT

uv run python ${script}${escaped_args}

echo
echo '--- scrape exited; press any key to close ---'
read -n 1
LAUNCHER
chmod +x "$launcher"

# Propagate the scratch root via tmux -e (handles escaping internally).
tmux new-session -d -s "$session" \
    -e "AEMET_SCRATCH_ROOT=$scratch_root" \
    -c "$project_root" \
    "$launcher"

echo "started tmux session: $session"
echo "  attach:  tmux attach -t $session"
echo "  log:     tail -f ${project_root}/logs/aemet_${flavor}.log"
