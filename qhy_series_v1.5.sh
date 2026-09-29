#!/usr/bin/env bash
# Capture a series of QHY exposures using qhy_single_frame.py.
#
# Runs 10, 20, 50, 100, 200, 500, 1000 ms in ONE camera session (one Python
# process), so the camera is opened once and never closed/reopened between
# exposure steps. Closing and reopening the camera between steps was
# triggering USB re-enumeration failures ("No QHY camera found" / "SDK not
# support this camera now"), so this avoids that entirely rather than
# retrying around it.
#
# Usage:
#   ./qhy_series.sh                 # 10 frames per exposure, SDK default gain
#   ./qhy_series.sh -g 80           # set gain to 80
#   ./qhy_series.sh -n 5 -o mydir   # 5 frames per exposure, output to ./mydir
#   ./qhy_series.sh -u 0            # low USB traffic value, for USB timeouts
#
# Output files: <outdir>/exp_<ms>ms.fits (or exp_<ms>ms_<nn>.fits if -n > 1)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAPTURE_PY="${SCRIPT_DIR}/qhy_single_frame_v1.5.py"

EXPOSURES_MS="10,20,50,100,200,500,1000"
COUNT=10
GAIN=""
USB_TRAFFIC=""
OUTDIR="qhy_series_$(date +%Y%m%d_%H%M%S)"

usage() {
    echo "Usage: $0 [-n frames_per_exposure] [-g gain] [-u usb_traffic] [-o output_dir]"
    exit 1
}

while getopts "n:g:u:o:h" opt; do
    case "$opt" in
        n) COUNT="$OPTARG" ;;
        g) GAIN="$OPTARG" ;;
        u) USB_TRAFFIC="$OPTARG" ;;
        o) OUTDIR="$OPTARG" ;;
        *) usage ;;
    esac
done

if [[ ! -f "$CAPTURE_PY" ]]; then
    echo "Error: cannot find $CAPTURE_PY" >&2
    exit 1
fi

if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || (( COUNT < 1 )); then
    echo "Error: -n must be a positive integer" >&2
    exit 1
fi

args=(--exposures "$EXPOSURES_MS" --count "$COUNT" --output-dir "$OUTDIR")
[[ -n "$GAIN" ]] && args+=(--gain "$GAIN")
[[ -n "$USB_TRAFFIC" ]] && args+=(--usb-traffic "$USB_TRAFFIC")

echo "Running exposure series (single camera session): $EXPOSURES_MS ms, ${COUNT} frames each"
echo "Output dir: $OUTDIR"
[[ -n "$GAIN" ]] && echo "Gain: $GAIN"
[[ -n "$USB_TRAFFIC" ]] && echo "USB traffic: $USB_TRAFFIC"

start=$(date +%s)

if ! python3 "$CAPTURE_PY" "${args[@]}"; then
    echo "Error: capture series failed." >&2
    exit 1
fi

elapsed=$(( $(date +%s) - start ))
echo
echo "Done: series completed in ${elapsed} s. Files in ${OUTDIR}."
