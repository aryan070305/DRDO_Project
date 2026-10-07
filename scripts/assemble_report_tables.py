"""Report assembly helper: copy the auto-generated table blocks of reports/results/tables.md into reports/REPORT.md.

    python scripts/make_tables.py > reports/results/tables.md
    python scripts/assemble_report_tables.py --check     # report whether REPORT.md tables equal tables.md (no write)
    python scripts/assemble_report_tables.py --report    # replace every "#### Table Rx" block in REPORT.md by tables.md

Used on 2026-10-05 to put the round-2 tables into REPORT.md (it was run from the session scratchpad then; this is that
script with the project root made relative). --draft filled a scratch draft of sections 8.6-8.10 (historical; the draft
is not part of the project). Tables not produced by make_tables.py (R15, from paired_stats.md) are left untouched.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = Path(__file__).parent   # --draft: folder holding the (historical) sec8_round2.md draft
tables = (ROOT / "reports/results/tables.md").read_text().splitlines()


def blocks(lines):
    out, cur = [], None
    for ln in lines:
        if ln.startswith("#### "):
            if cur:
                out.append(cur)
            cur = [ln]
        elif cur is not None:
            cur.append(ln)
    if cur:
        out.append(cur)
    res = []
    for b in out:
        while b and not b[-1].strip():
            b.pop()
        key = re.match(r"#### (Table R\d+[a-d]?):", b[0]).group(1)
        res.append((key, "\n".join(b)))
    return res


TB = blocks(tables)


def get(key, nth=0):
    hits = [t for k, t in TB if k == key]
    return hits[nth]


def is_table_line(ln):
    s = ln.strip()
    return ((not s) or s.startswith("|") or bool(re.fullmatch(r"\*[^*].*[^*]\*", s))
            or s.startswith("Cells: PESQ-WB") or s.startswith("Linear interpolation of the per-SNR means"))


def replace_blocks(text):
    lines = text.splitlines()
    out, i, seen = [], 0, {}
    while i < len(lines):
        m = re.match(r"#### (Table R\d+[a-d]?):", lines[i])
        if m and any(k == m.group(1) for k, _ in TB):
            key = m.group(1)
            n = seen.get(key, 0)
            seen[key] = n + 1
            j = i + 1
            while j < len(lines) and is_table_line(lines[j]):
                j += 1
            out.extend(get(key, n).splitlines())
            out.append("")
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    rep = ROOT / "reports/REPORT.md"
    if "--check" in sys.argv:
        cur = rep.read_text()
        new = replace_blocks(cur)
        print("REPORT.md tables are identical to tables.md" if new == cur else "REPORT.md tables DIFFER from tables.md (run --report)")
    if "--report" in sys.argv:
        rep.write_text(replace_blocks(rep.read_text()))
        print("REPORT tables replaced")
    if "--draft" in sys.argv:
        d = S / "sec8_round2.md"
        t = d.read_text()
        for key, ph in [("Table R11", "⟨R11⟩"), ("Table R12a", "⟨R12a⟩"), ("Table R12b", "⟨R12b⟩"),
                        ("Table R12c", "⟨R12c⟩"), ("Table R12d", "⟨R12d⟩"), ("Table R13", "⟨R13⟩")]:
            t = t.replace(ph, get(key))
        ps = (ROOT / "reports/results/paired_stats.md").read_text().strip()
        t = t.replace("⟨R15⟩", "#### Table R15: paired differences with 95 % bootstrap confidence intervals "
                      "(`scripts/paired_stats.py` → `reports/results/paired_stats.json`)\n\n" + ps)
        d.with_name("sec8_round2_filled.md").write_text(t)
        print("draft filled; remaining placeholders:", re.findall(r"⟨[^⟩]+⟩", t))
