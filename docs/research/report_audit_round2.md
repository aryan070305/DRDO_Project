# Round-2 adversarial audit of reports/REPORT.md (read-only)

*Requested 2026-10-05 ~07:32 +04 on the assembled round-2 report; final report delivered 2026-10-05T03:48:22.191Z (UTC). All confirmed errors and most possible issues were corrected in `reports/REPORT.md` and the decision records afterwards; the round-1 audit is `docs/research/report_audit.json`.* Exported verbatim from the agent's final report in the Claude Code session transcript.

---

Audit of reports/REPORT.md (round 2): findings

Nothing in the project folder was changed. Scratch files are in the scratchpad only. I recomputed every number from per_file.csv with the venv: defence_v1 −5…15 dB (800 files), dualmic_v1 (240), babble (60), edge (304), high-SNR (160), vbdemand (824).

## A. Confirmed errors (the data contradict the text)

1. **L448, §6.3 kit table: "Builds dist/jetson_bundle/ (≈ 8 MB) with only … the deployed ONNX models".**
   - The bundle is 13.4 MB and holds 116 files (`du` = 13M; `logs/package_bundle_round2.out` ends "( 13M)").
   - It also still contains the round-1 `exports/dancnet_hqft.onnx` and `.json`, which are not deployed. They are listed in `SHA256SUMS` (lines 63–64), because the bundle was "updated in place".
   - `deploy/jetson/QUICKSTART.md` L90 has the same stale figure ("about 8 MB and 110 files").
   - Fix: say ≈ 13 MB, and either remove the stale model or mention it.

2. **L1136: "The first chain log wrongly recorded 'v3_2mic finished' … the log is annotated."**
   - `logs/chain2b.out` line 1 reads "[chain2b] v3_2mic finished Sun Oct 4 17:06:19" with no annotation.
   - A grep of `logs/` for annotat/wrongly/crashed finds nothing.
   - Fix: add the annotation, or drop the claim.

3. **L1056, R12: "Measured single-mic: STOI 0.725 → 0.750, PESQ-WB 1.10 → 1.24 (§8.6)".**
   - These are the v3hq values. The deployed v3hq_cont gives 0.753 / 1.25, as §0 L79 and §8.6 L827 say.
   - "The NLMS hybrid raises STOI … to 0.823" is `hybrid:v3hq`. The deployed-HQ hybrid (`hybrid:v3hq_cont`) gives 0.825, as in the §0 table.
   - Fix: use 0.753 / 1.25 and 0.825.

4. **L64: "STOI > 0.85 is met on average in every evaluation".**
   - It fails on crew babble: edge crew_babble, deployed HQ 0.753. On dualmic_babble_v1, single-mic HQ gives 0.773 and the NLMS hybrid 0.825.
   - Fix: "on the main defence and dual-mic test sets".

5. **"v3-LL was better on every single-mic metric" (L95, L984, L994; the same wording is in `v3_decision_ll.md`).**
   - True on defence_v1 (every metric, every SNR, every category), on VoiceBank and on transparency.
   - False on the edge set and the high-SNR set:

     | Where | v3-LL | v2-LL | Δ |
     |---|---|---|---|
     | Edge noise_only attenuation (Table R11) | 35.3 dB | 36.9 dB | −1.7 dB |
     | Edge level_high SNR / SI-SDR | 14.0 dB | 14.1 dB | −0.09 / −0.23 dB |
     | Edge level_low PESQ-WB / SI-SDR | | | −0.005 / −0.11 dB |
     | Edge noise_switch PESQ-WB | | | −0.006 |
     | High-SNR 25 dB PESQ-WB | 3.647 | 3.653 | −0.006 |
     | High-SNR 25 dB PESQ-NB | 4.026 | 4.034 | −0.008 |

   - Fix: "better on every metric of defence_v1 and VoiceBank+DEMAND".

6. **L68: "+0.033 STOI" for round-2 HQ vs DeepFilterNet3.** 0.91783 − 0.88537 = +0.0325, which rounds to **+0.032**. The PESQ-WB (+0.468) and SI-SDR (+2.65 dB) leads are correct.

7. **L41: noisy PESQ-NB "1.95".** The mean is 1.9446, so **1.94**. This is a double-rounding of R1's 1.945.

8. **L53: hybrid with round-2 HQ, output SNR "13.6 dB".** The mean is 13.549, so **13.5**. Same double-rounding (R6 shows 13.55).

9. **L416: "The exported graphs match PyTorch to ≤ 1.3e-5".** This is the round-1 figure.
   - Round-2 exports: v3hq_cont 1.72e-5 (`exports/dancnet_v3hq_cont.json`; the decision record says 1.7e-5), v3ll 2.0e-5, v3hq 1.19e-5, v3_2mic 1.14e-5.
   - Fix: "≤ 2.0e-5".

10. **L285–290: the provenance of the v3 rationale does not hold up chronologically.**
    - The text presents the per-band diagnosis and the failed post-filter as the reasons for adding capacity ("therefore … the remaining lever was model capacity").
    - `scripts/band_error_analysis.py` and its CSV were first written on 2026-10-04 at 22:57 (birth time = modification time). That is about 14 h after `v3_decision_rule.md` and `configs/train_v3_hq.yaml` (08:40:45) and after the v3hq run started (about 08:41). The CSV already contains v3hq.
    - The post-filter result (`val_hq_metric.json`) dates from 08:49, also after the decision.
    - Fix: describe both as retrospective checks, or say the pre-decision diagnosis was not recorded.

11. **L350: "One job at a time … the runs are queued strictly one after another".** Training runs were sequential, but evaluations repeatedly ran alongside GPU training:

    | Evaluation | Time (4–5 Oct) | Training running at the time |
    |---|---|---|
    | v3hq test evals | 15:55–16:24 | v3_2mic |
    | v3ll evals | 19:03–19:27 | abl_base |
    | babble single-mic eval, DFN3 and high-SNR evals | 22:53–23:12 | abl_la4 / abl_nopmsqe |
    | cascade test evals | 02:00–02:17 | v3hq_cont |
    | post-filter tuning, edge v2 eval | 08:44–09:03 | v3hq |

    - Contention is visible: the cascade eval took 363 s of processing against 88 s for dnn2 earlier.
    - Relatedly, L1096 calls the list an "exact sequence", but build_dual_babble (22:44) and build_highsnr (23:00) ran after v3ll and during the ablations.
    - Fix: "training runs were queued one at a time; some evaluations overlapped, so wall times include contention"; call the list "grouped, not strictly chronological".

12. **L359: v3hq_cont wall time "5.1 h".** elapsed_h is 5.047, so 5.0 h (the decision record says 5.05 h). Trivial.

## B. Medium or possible issues

13. **Jetson timing for the two-mic configuration (L413).**
    - "About 3–4 ms, still inside the 10 ms deadline" scales only the unpaced 1.27 ms.
    - §6.3 (L434) scales the paced figures for a single round-1 network and concludes the worst case "could exceed the 10 ms deadline" (8.5–13 ms).
    - Paced runs were never repeated for v3 or the two-mic configuration: `simulated_live_runs.json` holds only four hqft runs from 4 Oct 08:08.
    - By the report's own method, the two-network configuration is more at risk, so drop "still inside".
    - Stale round-1 compute figures are also left in round-2 context: L157 (0.37–0.76 GMAC/s, 0.5 ms), L422 budget (0.5 ms; 2.1 / 4.3 ms paced), L433 (0.5 ms → 1–1.5 ms). The deployed v3 is 1.37 GMAC/s and 0.67 ms; the two-mic configuration runs two networks at 1.27 ms.

14. **The cascade's pre-registration caveat is missing.**
    - L35 and §8.10 say every change was pre-registered.
    - `cascade_decision_rule.md` itself discloses that the cascade idea came from the two-mic network's test-set per-category results. The report should say so.
    - The timing is consistent but tight: the rule was written at 02:00:33, and the test eval started about 02:00:51 (end time minus logged durations).

15. **L1001: "keeps both strengths".** Against the deployed round-2 hybrid (`hybrid:v3hq_cont`), the cascade is worse in:
    - stationary noise: PESQ-WB −0.013, STOI −0.0075, SI-SDR −0.30 dB;
    - −5 dB input: PESQ-WB −0.026, STOI −0.006.

    Against `hybrid:v3hq`, stationary STOI is still −0.0058. Overall STOI vs the deployed hybrid is −0.0009 [−0.0023, +0.0005]. State the trade-off.

16. **Different SNR axes and test sets are juxtaposed in §0 L71–73 and R14.**
    - defence_v1 uses a nominal, active-speech SNR; its measured whole-signal SNR is about 1.2 dB lower (nominal 5 dB = 3.87 dB measured).
    - dualmic_v1 nominal equals measured exactly.
    - So "single-mic 5.3 dB" against "cascade 2.6 dB" mixes both the set and the SNR definition.
    - The valid within-set comparison (7.1 vs 2.6 dB) is only in §8.7.

17. **"PESQ-WB > 2.5 is now met on average: 2.51" (L65, L995) is a point estimate.**
    - The bootstrap 95 % CI of the mean is [2.47, 2.55] (SNR-stratified), which straddles 2.5.
    - The result depends on the SNR mix: all 900 files including −10 dB give 2.40.
    - Only 49.0 % of files exceed 2.5.

18. **L473: "Both launchers were run end-to-end with the cascade on a recorded scene".** No log or result file records such a run. Add the artefact or soften the claim.

19. **L68: "DeepFilterNet3, the strongest public causal model".** §2.1 lists FRCRN (Apache-2.0, VB-D 3.21 > 3.17) and DPDFNet-8. DFN3 also beats D-ANC on VoiceBank (2.705 vs 2.485). Suggest "a strong public causal model, not trained on defence noise".

20. **Stale round-1 labels.**
    - L285: "deployed HQ model" (this was hq_metric).
    - L327: "`hq_metric` (deployed HQ)".
    - L387: "HQ deployed (`dancnet_hqft.onnx`)".
    - L728: "13.4 dB SNR for both" (the deployed-HQ hybrid is now 13.55).

    All should read "round-1".

21. **L813: "All round-1 and round-2 models were evaluated on the 11 edge cases".** Round-1 `hq` (base) and `ll_metric` were not.

22. **L827: "the reference microphone solves it: STOI 0.904".** That figure comes from a different set (dualmic_babble_v1), and PESQ-WB is still 2.06. Use "largely addresses".

23. **Small wording points.**
    - L985 "larger batches gave no throughput": v2 gave +9 % and +14 %, which is below the 25 % bar, not zero.
    - L31 "six controlled ablations": there are six arms, i.e. a reference plus five changes.
    - L358 "(2 500 steps, crash at ≈ 2 500…)": 5 500 steps were actually executed.
    - L84: the EMA-on-resume bug is a training bug, not a deployment bug.

24. **Possible reproducibility gap.** `src/danc/data/build_dual_babble.py` was modified at 23:35, after dualmic_babble_v1 was built (`meta.csv` 22:44) and evaluated (22:53).

25. **Evidence gaps outside the report text.**
    - `cascade_decision.md` cites `--fallback-onnx exports/dancnet_v3hq.onnx` in one paragraph and `v3hq_cont` elsewhere.
    - `logs/eval_highsnr_aborted_missing_dfn3.out` is 0 bytes.
    - No artefact backs "swap 7.3 GB" or "RNN input dtype".

## C. Sections that check out

- **§0 tables:** every other cell matches the recomputed means.
- **R1–R13:** identical to `tables.md`; spot-recomputed values match. R12a–d percentages and interpolated crossings match (5.3 / 9.1 / 14.7 / 18.6; 2.6 / 9.6; 7.1).
- **R14:** every cell matches `latency_mac_m4_round2_final.json`. Round-1 models reproduce within 0.03 ms.
- **R15:** identical to `paired_stats.md`. All 27 mean differences and "B better" shares reproduce from per_file.csv.
- **§5.1:** batch table matches `batch_benchmark.json`; validation scores match the logs; 10 runs, 21.77 h total; ablations 0.76–0.96 h each.
- **§3.5:** 304 files, 64 MB; babble TIR and measured SNR −0.4 / 3.8 / 7.0 dB are correct.
- **§8.6–8.9 numbers:** correct (post-blast 0.796 / 0.796, 1.74 / 1.80; post-filter +0.004 / −0.0014 / −0.03).
- **§8.10 table:** correct.
- **Decision rules:** thresholds are applied correctly in all five decisions. Rule file times precede each test result:
  - v3: 08:40:45, before training started about 08:41 and the first eval at 15:55.
  - two-mic: 22:48, before the first dnn2 test eval at about 01:52.
  - v3hq_cont: 22:48, before training started about 01:58 and the eval at 07:04.
  - cascade: 02:00:33, before the eval started about 02:00:51.
- **Deployment files and §4:** `deployment_selection.json` and `DEPLOYMENT.json` match the §4 deployed-model statement. Parameter counts are right (609 156 / 610 236), and state sizes ≈ 68 KB.
- **Self-check:** `verify_install_mac_round2.json` gives 115 PASS / 0 FAIL. It covers HQ, LL and TWO_MIC, including a reference that dies at 1 s, a run past 30 s, and timing with both networks (1.285 ms).
- **Code claims:** the 30-s trim bug description and the health-monitor parameters match the code.
- **Paths:** every file path named in the round-2 text exists.
