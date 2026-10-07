#!/usr/bin/env bash
# Log Jetson power rails with tegrastats while the engine runs.  NOT VERIFIED ON HARDWARE IN THIS PROJECT.
# Usage: ./log_power.sh <seconds> [out.csv]
# Rail names printed by tegrastats differ between JetPack releases and modules (e.g. VDD_GPU_SOC,
# VDD_CPU_CV, VIN_SYS_5V0 on AGX Orin dev kits) - inspect the raw log and adapt the parser.
set -euo pipefail
DUR="${1:-60}"
OUT="${2:-power_$(date +%Y%m%d_%H%M%S).csv}"
RAW="${OUT%.csv}.raw.txt"
sudo tegrastats --interval 100 --logfile "$RAW" &
PID=$!
sleep "$DUR"
sudo kill "$PID" || true
python3 - "$RAW" "$OUT" <<'EOF'
import re, sys, csv
raw, out = sys.argv[1], sys.argv[2]
rows = []
for line in open(raw):
    rails = dict((m.group(1), (int(m.group(2)), int(m.group(3))))
                 for m in re.finditer(r"([A-Z0-9_]+) (\d+)mW/(\d+)mW", line))
    if rails:
        rows.append({k: v[0] for k, v in rails.items()})
keys = sorted({k for r in rows for k in r})
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
for k in keys:
    vals = [r[k] for r in rows if k in r]
    print(f"{k:16s} mean {sum(vals)/len(vals):8.0f} mW  max {max(vals):6d} mW  (n={len(vals)})")
EOF
echo "raw: $RAW  csv: $OUT"
