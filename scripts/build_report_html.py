"""Build reports/site/index.html (shareable web version of reports/REPORT.md).

The report body is converted from REPORT.md; the target-compliance matrix at the top is computed directly
from reports/results/*/per_file.csv so no number is transcribed by hand.

    python scripts/build_report_html.py
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import markdown
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "site"
TARGETS = (("snr", 15.0, "SNR", "dB"), ("stoi", 0.85, "STOI", ""), ("pesq_wb", 2.5, "PESQ", ""))
ROWS = [("defence_v1", "dancnet:v3hq_cont", "HQ · 40 ms", "1 mic, deployed (round 2)"),
        ("defence_v1", "dancnet:ll", "LL · 20 ms", "1 mic, deployed"),
        ("defence_v1", "dancnet:hq_metric", "HQ · 40 ms", "1 mic, round 1"),
        ("dualmic_v1", "hybrid2:v3_2mic", "Two-mic cascade · 40 ms", "2 mics, deployed (round 2)"),
        ("dualmic_v1", "hybrid:v3hq_cont", "HQ · 40 ms", "2 mic hybrid"),
        ("dualmic_v1", "hybrid:ll", "LL · 20 ms", "2 mic hybrid"),
        ("defence_v1", "file:dfn3", "DeepFilterNet3", "1 mic, reference")]
SNRS = [-10, -5, 0, 5, 10, 15]


def compliance_matrix() -> str:
    cache = {}
    head = "".join(f'<th scope="col">{s:+d} dB</th>' for s in SNRS)
    rows = []
    for st, sysname, label, kind in ROWS:
        if st not in cache:
            cache[st] = pd.read_csv(ROOT / "reports" / "results" / st / "per_file.csv")
        df = cache[st]
        g = df[df.system == sysname].groupby("snr_db")[["snr", "stoi", "pesq_wb"]].mean()
        cells = []
        for s in SNRS:
            if s not in g.index:
                cells.append('<td class="na">not tested</td>')
                continue
            r = g.loc[s]
            chips = []
            for key, thr, name, unit in TARGETS:
                v = r[key]
                ok = v > thr
                val = f"{v:.1f}" if key == "snr" else (f"{v:.3f}" if key == "stoi" else f"{v:.2f}")
                chips.append(f'<span class="chip {"pass" if ok else "fail"}" title="{name} {val}{(" " + unit) if unit else ""} '
                             f'(target &gt; {thr:g})"><b>{name}</b> {val}</span>')
            n_ok = sum(r[k] > t for k, t, *_ in TARGETS)
            cells.append(f'<td class="m{n_ok}">{"".join(chips)}</td>')
        rows.append(f'<tr><th scope="row"><span class="rl">{label}</span><span class="rk">{kind}</span></th>{"".join(cells)}</tr>')
    return (f'<div class="tablewrap matrix"><table><thead><tr><th scope="col">Model · input SNR</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


CSS = r"""
/* Layout: technical report. Sticky section rail + one reading column; data blocks may run wider. */
:root{
  --paper:#f6f7f5; --panel:#ffffff; --ink:#1b2228; --muted:#59656c; --rule:#d3d9d6;
  --accent:#23697a; --accent-soft:#e3eff1; --pass:#2c7a4c; --pass-bg:#e4f2e9; --fail:#a8382c; --fail-bg:#f8e7e4;
  --code:#edf0ee; --plate:#ffffff;
  --f-display:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;
  --f-body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --paper:#11161a; --panel:#171e23; --ink:#e2e7e9; --muted:#9aa6ac; --rule:#2b343a;
  --accent:#71b7c7; --accent-soft:#1c2c31; --pass:#63b985; --pass-bg:#16281e; --fail:#e47468; --fail-bg:#2e1a18;
  --code:#1b2328; --plate:#ffffff; color-scheme:dark}}
:root[data-theme="dark"]{
  --paper:#11161a; --panel:#171e23; --ink:#e2e7e9; --muted:#9aa6ac; --rule:#2b343a;
  --accent:#71b7c7; --accent-soft:#1c2c31; --pass:#63b985; --pass-bg:#16281e; --fail:#e47468; --fail-bg:#2e1a18;
  --code:#1b2328; --plate:#ffffff; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font:16px/1.62 var(--f-body);padding-inline:16px;padding-block:0 64px}
.shell{max-width:1240px;margin:0 auto;display:grid;grid-template-columns:230px minmax(0,1fr);gap:48px}
nav.rail{position:sticky;top:env(safe-area-inset-top,0px);align-self:start;max-height:100vh;overflow:auto;padding-block:32px;font-size:13.5px}
nav.rail .t{font:600 12px/1.2 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}
nav.rail ol{list-style:none;margin:0;padding:0;display:grid;gap:2px}
nav.rail a{display:block;padding:4px 8px;border-radius:4px;color:var(--ink);text-decoration:none}
nav.rail a:hover,nav.rail a:focus-visible{background:var(--accent-soft);color:var(--accent);outline:none}
main{min-width:0;padding-block:32px}
header.hero{border-bottom:2px solid var(--ink);padding-bottom:22px;margin-bottom:28px;display:grid;gap:10px}
.eyebrow{font:600 12px/1.3 var(--f-mono);letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}
h1{font:600 clamp(30px,4.6vw,46px)/1.08 var(--f-display);letter-spacing:-.01em;margin:0;text-wrap:balance}
.sub{color:var(--muted);max-width:68ch;margin:0}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;font:13px/1.4 var(--f-mono);color:var(--muted)}
.meta b{color:var(--ink);font-weight:600}
section.summary{display:grid;gap:14px;margin-bottom:40px}
section.summary h2{margin-top:0;border-top:0;padding-top:0}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px;color:var(--muted)}
.chip{display:inline-flex;align-items:baseline;gap:4px;font:12.5px/1.25 var(--f-mono);padding:2px 6px;border-radius:3px;white-space:nowrap;font-variant-numeric:tabular-nums}
.chip b{font-weight:600;font-size:11px;letter-spacing:.04em}
.chip.pass{background:var(--pass-bg);color:var(--pass)}
.chip.fail{background:var(--fail-bg);color:var(--fail)}
.matrix td{vertical-align:top}
.matrix td .chip{display:flex;margin-bottom:3px}
.matrix td.m3{box-shadow:inset 0 3px 0 var(--pass)}
.matrix td.na{color:var(--muted);font-size:12px}
.matrix th .rl{display:block;font:600 14px/1.2 var(--f-display)}
.matrix th .rk{display:block;font:12px/1.3 var(--f-mono);color:var(--muted)}
h2{font:600 26px/1.2 var(--f-display);margin:56px 0 12px;padding-top:12px;border-top:1px solid var(--rule);text-wrap:balance}
h3{font:600 20px/1.25 var(--f-display);margin:34px 0 8px;text-wrap:balance}
h4{font:600 15px/1.3 var(--f-body);margin:26px 0 8px;color:var(--ink)}
.prose p,.prose li{max-width:76ch}
.prose a{color:var(--accent)}
.prose code{font:0.88em var(--f-mono);background:var(--code);padding:1px 4px;border-radius:3px}
.prose pre{background:var(--code);padding:14px 16px;border-radius:4px;overflow-x:auto;font:13px/1.5 var(--f-mono)}
.prose pre code{background:none;padding:0}
.prose blockquote{margin:16px 0;padding:10px 16px;border-left:3px solid var(--accent);background:var(--accent-soft)}
.tablewrap{overflow-x:auto;margin:14px 0 18px;border:1px solid var(--rule);border-radius:4px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
thead th{font:600 12px/1.3 var(--f-mono);letter-spacing:.03em;color:var(--muted);background:var(--code);position:sticky;top:0}
tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}
td strong,th strong{color:var(--accent)}
figure,.prose p>img{display:block}
.prose img{background:var(--plate);padding:8px;border:1px solid var(--rule);border-radius:4px;margin:12px 0}
.fig-caption{font-size:13.5px;color:var(--muted)}
.foot{margin-top:56px;padding-top:16px;border-top:1px solid var(--rule);font:13px/1.5 var(--f-mono);color:var(--muted)}
details.mnav{display:none}
@media (max-width:900px){
  .shell{grid-template-columns:minmax(0,1fr);gap:0}
  nav.rail{display:none}
  details.mnav{display:block;position:sticky;top:env(safe-area-inset-top,0px);z-index:2;background:var(--paper);border-bottom:1px solid var(--rule);padding-block:10px;margin-inline:-16px;padding-inline:16px}
  details.mnav summary{cursor:pointer;font:600 13px var(--f-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--accent)}
  details.mnav ol{list-style:none;padding:8px 0 0;margin:0;display:grid;gap:4px;font-size:14px}
  details.mnav a{color:var(--ink);text-decoration:none}
  h2{font-size:22px}
}
@media (prefers-reduced-motion:no-preference){nav.rail a{transition:background .15s,color .15s}}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""


def main():
    md_text = (ROOT / "reports" / "REPORT.md").read_text()
    # drop the markdown H1 + byline (rendered by the designed header instead)
    body_md = md_text.split("\n---\n", 1)[1] if "\n---\n" in md_text else md_text
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"],
                           extension_configs={"toc": {"permalink": False}})
    body = md.convert(body_md)
    body = re.sub(r"<table>", '<div class="tablewrap"><table>', body)
    body = re.sub(r"</table>", "</table></div>", body)
    body = body.replace('src="figures/', 'loading="lazy" src="figures/')
    # section navigation from H2s
    secs = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body)
    nav = "".join(f'<li><a href="#{i}">{re.sub("<[^>]+>", "", t)}</a></li>' for i, t in secs)
    page = f"""<title>D-ANC Hybrid Noise Cancellation</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&display=swap">
<style>{CSS}</style>
<div class="shell">
<nav class="rail" aria-label="Sections"><div class="t">Sections</div><ol><li><a href="#compliance">Target compliance</a></li>{nav}</ol></nav>
<main>
<details class="mnav"><summary>Sections</summary><ol><li><a href="#compliance">Target compliance</a></li>{nav}</ol></details>
<header class="hero">
  <div class="eyebrow">Technical report · defence voice communications</div>
  <h1>D-ANC: hybrid AI/ML adaptive noise cancellation</h1>
  <p class="sub">A reference-microphone sub-band NLMS filter coupled to a causal complex-domain deep-filtering network, built, trained and
  validated for gunfire, artillery, rotor, drone, vehicle, siren and wind noise, with an edge deployment path to the NVIDIA Jetson AGX Orin.</p>
  <div class="meta"><span><b>Modes</b> HQ 40 ms · LL 20 ms · two-mic cascade 40 ms</span><span><b>HQ model</b> 609 k params · 1.37 GMAC/s</span>
  <span><b>Compute</b> 0.67 ms per 10 ms hop, two-mic 1.27 ms (Apple M4, 1 thread)</span><span><b>Date</b> 2026-10-05 (round 2)</span></div>
</header>
<section class="summary" id="compliance">
  <h2>Target compliance by input SNR</h2>
  <p class="sub">Each cell shows the mean over the test files at that input SNR, with target SNR &gt; 15 dB, STOI &gt; 0.85 and PESQ-WB &gt; 2.5.
  Green meets the target, red misses it; a bar on top marks cells where all three are met. Single-mic rows use the 900-file defence test set,
  two-microphone rows the 240-scene dual-microphone set, which covers −5…10 dB. Computed directly from the per-file results.</p>
  {compliance_matrix()}
  <div class="legend"><span><span class="chip pass"><b>STOI</b> 0.951</span> meets target</span><span><span class="chip fail"><b>PESQ</b> 2.01</span> below target</span></div>
</section>
<article class="prose">{body}</article>
<div class="foot">Source: reports/REPORT.md in the project folder · tables generated by scripts/make_tables.py · per-file results in reports/results/ ·
claim verification in docs/research/claims_verification.md · file checksums in reports/MANIFEST.sha256</div>
</main></div>
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page)
    imgs = sorted(set(re.findall(r'src="(figures/[^"]+)"', page)))
    print("wrote", OUT / "index.html", "figures:", imgs)


if __name__ == "__main__":
    main()
