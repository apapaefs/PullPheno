#!/usr/bin/env bash
# Run explicitly selected phases on Odysseus. --check never prepares a batch.
set -eo pipefail
package_dir=$(cd -- "$(dirname -- "$0")" && pwd)
repository=${PULLPHENO_REPOSITORY:-/home/apapaefs/Projects/PullPheno}
output_dir=${PULLPHENO_ASYMMETRY_OUTPUT:-/micron/Projects/PullPheno/analysis-results/campaign22-beam-asymmetries-20260929}
batch_dir="$output_dir/batches/campaign22-beam-asymmetries-20260929"
source /home/apapaefs/Projects/root/root_v6.40.02/bin/thisroot.sh
set -u
analysis_python=/home/apapaefs/Projects/PullPheno/.venv-odysseus/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
if [[ $# != 1 ]]; then
  printf 'Usage: %s --check|prepare|smoke|analysis|plots|report\n' "$0" >&2
  exit 2
fi
phase=$1
case "$phase" in
  --check) exec "$analysis_python" "$package_dir/prepare.py" --repository "$repository" --output-root "$output_dir" ;;
  prepare)
    mkdir -p "$output_dir"
    exec 9>"$output_dir/launch.lock"
    flock -n 9 || { printf 'Another launcher is active.\n' >&2; exit 1; }
    exec "$analysis_python" "$package_dir/prepare.py" --repository "$repository" --output-root "$output_dir" --prepare ;;
  smoke|analysis|plots|report) ;;
  *) printf 'Unknown phase: %s\n' "$phase" >&2; exit 2 ;;
esac
if [[ ! -f "$batch_dir/inputs.json" || ! -f "$batch_dir/source-provenance.json" ]]; then
  printf 'Batch is not prepared; run bash run.sh prepare first.\n' >&2
  exit 1
fi
exec 9>"$output_dir/launch.lock"
flock -n 9 || { printf 'Another study phase is live; leave it running.\n' >&2; exit 1; }
exec > >(tee -a "$output_dir/launch.log") 2>&1
export MPLCONFIGDIR="$output_dir/matplotlib" GIT_CEILING_DIRECTORIES="$batch_dir"
mkdir -p "$MPLCONFIGDIR"
trap 'printf "Phase %s stopped at line %s; see %s/%s.log\n" "$phase" "$LINENO" "$batch_dir" "$phase" >&2' ERR
if [[ "$phase" == smoke || "$phase" == analysis ]]; then
  export THERMAL_GUARD_CONFIG="$batch_dir/thermal-guard.conf"
  thermal_start="$batch_dir/thermal-start-$phase-$(date -u +%Y%m%dT%H%M%SZ)-$$.json"
  "$HOME/.local/bin/pullpheno-thermal-guard" status > "$thermal_start"
  "$analysis_python" - "$thermal_start" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert not s['sensor_error'] and not s['state'].get('latched'), 'Thermal guard requires inspection'
assert set(s['temperatures_c']) == set(s['configured_sensors'])
assert all(t <= s['manual_resume_thresholds_c'][n] for n,t in s['temperatures_c'].items()), 'Machine is too hot to start'
PY
  guard_unit="pullpheno-c22-asymmetry-guard-$$"
  systemd-run --user --unit="$guard_unit" --collect --on-active=1s --on-unit-active=15s \
    --timer-property=AccuracySec=1s --property=Nice=19 \
    /usr/bin/env "THERMAL_GUARD_CONFIG=$THERMAL_GUARD_CONFIG" "$HOME/.local/bin/pullpheno-thermal-guard" check
  trap 'systemctl --user stop "$guard_unit.timer" || true' EXIT
  systemctl --user is-active --quiet "$guard_unit.timer"
fi
printf 'Starting %s; progress: %s/%s.log\n' "$phase" "$batch_dir" "$phase"
if [[ "$phase" == report ]]; then
  nice -n 19 "$analysis_python" "$batch_dir/code/scripts/build_beam_asymmetry_report.py" \
    --status "$batch_dir/status.json" --destination "$output_dir/report" >> "$batch_dir/report.log" 2>&1
  printf 'Report: %s/report/index.html\n' "$output_dir"
else
  nice -n 19 "$analysis_python" "$batch_dir/code/scripts/run_beam_asymmetry_study.py" \
    --phase "$phase" --batch "$batch_dir" --inputs "$batch_dir/inputs.json" \
    --output-root "$output_dir" >> "$batch_dir/$phase.log" 2>&1
fi
printf '%s completed successfully.\n' "$phase"
