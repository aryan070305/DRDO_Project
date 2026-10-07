#!/usr/bin/env bash
# Low-latency audio + CPU configuration for the D-ANC prototype on Jetson AGX Orin (Ubuntu / JetPack).
# NOT VERIFIED ON HARDWARE IN THIS PROJECT.  Run with sudo; review each step first.
set -euo pipefail

echo "== ALSA capture devices";  arecord -l || true
echo "== ALSA playback devices"; aplay -l   || true

# 1) Maximum clocks for deterministic latency (benchmark ALSO at the mode you will field: 15W/30W).
#    Mode IDs on AGX Orin 64GB (Jetson Linux r36 docs): 0=MAXN, 1=15W, 2=30W, 3=50W.
if command -v nvpmodel >/dev/null; then nvpmodel -q || true; fi
# nvpmodel -m 0 && jetson_clocks          # uncomment to select MAXN + lock clocks

# 2) CPU governor -> performance (avoid DVFS-induced latency spikes)
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [ -w "$g" ] && echo performance > "$g" || true
done

# 3) Example ALSA config: card "USB" = 2-in/2-out USB audio interface
#    ch0 = primary (boom) mic, ch1 = reference (outward-facing) mic; stereo headphone out.
cat > /tmp/asound.conf.danc <<'EOF'
pcm.danc_in {
    type hw
    card USB
    device 0
    format S16_LE
    rate 16000          # if the interface cannot run at 16 kHz, capture at 48 kHz and resample in software
    channels 2
}
pcm.danc_out {
    type hw
    card USB
    device 0
    format S16_LE
    rate 16000
    channels 2
}
EOF
echo "Example ALSA config written to /tmp/asound.conf.danc (copy into /etc/asound.conf after review)."

# 4) Desktop/session audio servers add buffering: either stop them for the device
#    (systemctl --user stop pipewire pipewire-pulse wireplumber) or run through pw-jack with a
#    160-frame quantum: PIPEWIRE_QUANTUM=160/16000 pw-jack python3 -m danc.inference.realtime ...
# 5) The NVIDIA forum reports GPU launch stalls caused by the desktop session on AGX Orin; for a
#    headless prototype:  systemctl set-default multi-user.target
echo "Done. Suggested run: chrt -f 80 taskset -c 10,11 python3 -m danc.inference.realtime --onnx exports/dancnet_v3hq_cont.onnx"
