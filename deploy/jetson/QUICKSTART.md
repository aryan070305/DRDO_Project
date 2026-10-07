# D-ANC on NVIDIA Jetson AGX Orin 64GB: Quick Start

> **Status: NOT VERIFIED ON HARDWARE IN THIS PROJECT.** D-ANC was developed and verified on an Apple M4
> (macOS). Every Jetson-specific step below (JetPack, TensorRT, ALSA, CPU pinning, power, latency) is a
> procedure prepared from NVIDIA documentation and the project code. It has **not** been run on a Jetson by this
> project. Steps 5, 7 and 9 to 11 are how you verify it on your board. All reference numbers in this guide are
> **Mac numbers** unless stated otherwise.

This guide takes you from a fresh Jetson AGX Orin 64GB developer kit running JetPack 6.2.x to live operation.
Run the steps in order. Each step ends with a **Check** you can confirm. Background and design rationale are in
[`README.md`](README.md) (Jetson bring-up guide) and [`../../reports/REPORT.md`](../../reports/REPORT.md) §6 to §7.

| Step | What | Time |
|---|---|---|
| 1 | Check JetPack | 2 min |
| 2 | Set the power mode | 2 min |
| 3 | Copy the project or bundle to `/opt/danc` | 5 min |
| 4 | Install the runtime (`install.sh`) | 10 to 20 min |
| 5 | Self-check (`verify_install.py`) | about 2 to 5 min |
| 6 | (optional) Build and check TensorRT engines | 15 min |
| 7 | Connect the audio interface and check the channels | 10 min |
| 8 | Run the engine: HQ (40 ms), LL (20 ms) or the two-mic cascade (40 ms) | 5 min |
| 9 | Measure loop-back latency | 15 min |
| 10 | Log power | 10 min per mode |
| 11 | Real-time acceptance run (10 min) | 15 min |
| 12 | Start automatically at boot (systemd) | 5 min |

Conventions: commands run on the **Jetson** unless marked *(dev machine)*. The install directory is
`/opt/danc`. `$` lines are copy-paste ready.

---

## What you need

- Jetson AGX Orin 64GB developer kit, flashed with **JetPack 6.2.x** (Jetson Linux R36.4.3 / R36.4.4 / R36.5)
  using NVIDIA SDK Manager. Flashing is not covered here; follow NVIDIA's JetPack documentation.
- A class-compliant **2-in / 2-out USB audio interface** with two mic pre-amps (16 kHz or 48 kHz).
- **Primary mic:** a noise-cancelling boom mic, 2 to 3 cm from the lips. **Reference mic:** omni electret or MEMS,
  on the outside of the earcup or helmet, facing away from the mouth, 8 to 15 cm from the primary.
- Headset (earcups) on the interface's headphone output. For radio integration: line out → attenuator pad
  → audio isolation transformer → radio MIC input (`README.md` §4).
- Network access for `apt` / `pip`, or the offline wheel set described in step 4.
- For step 9: a loop-back cable. A second USB interface is optional, for measuring with the engine in the path.

---

## 1. Check the JetPack version

```bash
$ cat /etc/nv_tegra_release          # first line, e.g. "# R36 (release), REVISION: 4.3, ..."
$ dpkg -l nvidia-jetpack             # meta-package version (absent if only Jetson Linux was flashed)
$ apt show nvidia-jetpack 2>/dev/null | grep -E '^(Package|Version)'
$ python3 --version                  # JetPack 6.x: Python 3.10
$ uname -m                           # aarch64
```

| `/etc/nv_tegra_release` | JetPack | TensorRT / CUDA |
|---|---|---|
| R36, REVISION 4.3 | 6.2 | 10.3.0 / 12.6 |
| R36, REVISION 4.4 | 6.2.1 | 10.3.0 / 12.6 (NVIDIA: "the same libraries ... as JetPack 6.2") |
| R36, REVISION 5.x | 6.2.2 | 10.3.0 / 12.6 (NVIDIA: the same libraries as 6.2.1) |

**Check:** R36 and Python 3.10. The CPU path (the default) needs only Jetson Linux. The TensorRT path in step 6
also needs the JetPack components: `sudo apt-get update && sudo apt-get install -y nvidia-jetpack`.
Other branches such as JetPack 7.x (Ubuntu 24.04, Python 3.12) need a different pip index (`jp7/...`). They are
not covered here.

## 2. Set the power mode

```bash
$ sudo nvpmodel -q                    # current mode
$ sudo nvpmodel -m 0                  # 0 = MAXN, 1 = 15W, 2 = 30W, 3 = 50W (AGX Orin 64GB)
$ sudo jetson_clocks                  # optional: lock clocks at the mode's maximum (benchmarking)
$ cat /sys/devices/system/cpu/online  # CPUs online in this mode
```

| Mode (ID) | Online CPUs | CPU max | Note for D-ANC (one real-time thread) |
|---|---|---|---|
| MAXN (0) | 12 | 2.2 GHz | fastest per core, for the first bring-up |
| 15W (1) | 4 (CPU0 to 3) | 1.11 GHz | battery; slowest per core, so check the deadline here |
| 30W (2) | 8 (CPU0 to 7) | 1.73 GHz | default; faster per core than 50W |
| 50W (3) | 12 | 1.50 GHz | more cores but a *lower* per-core clock than 30W |

**Check:** do the final acceptance (steps 5, 10 and 11) in the mode you will field. `taskset -c 10,11` and the old
`danc.service` pin CPUs 10 and 11, which are **offline in the 15W and 30W modes**. The new launcher and unit
(`danc_live.py --cpus auto`, `danc-live.service`) pick the last two *online* CPUs instead.

## 3. Copy the project to the Jetson

**Option A: deployment bundle (recommended, about 13 MB and 116 files in round 2).** Build it on the development machine with
`deploy/jetson/package_for_jetson.sh`. The bundle has the code, the three deployed models (HQ, LL, TWO_MIC; an older bundle updated in place also keeps
the round-1 `dancnet_hqft.onnx`), the deploy kit, the 16
verification scenes and their reference values, and SHA-256 checksums.

If you received the shared project archive, the bundle is already built in `dist/jetson_bundle/` (check it with
`cd dist/jetson_bundle && sha256sum -c SHA256SUMS`, or `shasum -a 256 -c SHA256SUMS` on macOS). Pack it with
`tar -czf dist/jetson_bundle.tar.gz -C dist jetson_bundle` instead of rebuilding, then continue with `scp` below.
If the archive was unpacked on Windows, the executable bits are lost: run the scripts as `bash deploy/jetson/install.sh`.

```bash
# (dev machine) from the project root
$ deploy/jetson/package_for_jetson.sh --tar        # -> dist/jetson_bundle/ and dist/jetson_bundle.tar.gz
$ scp dist/jetson_bundle.tar.gz <user>@<jetson>:/tmp/
# (Jetson)
$ sudo mkdir -p /opt/danc && sudo chown "$USER": /opt/danc
$ tar -xzf /tmp/jetson_bundle.tar.gz -C /opt/danc --strip-components=1
$ cd /opt/danc && sha256sum -c SHA256SUMS | grep -v ': OK$' ; echo "checksum check done"   # no lines = all OK
$ cat BUNDLE_INFO.txt
```

**Option B: the full project tree.** Copy the project, for example with
`rsync -a --exclude .venv --exclude '__pycache__' --exclude '*.egg-info' --exclude dist --exclude data/raw --exclude checkpoints --exclude third_party <project>/ <user>@<jetson>:/opt/danc/`.
**Never copy the development `.venv`**: it holds macOS binaries, and `install.sh` would reuse a directory of that name.
The Jetson needs `src/`, `exports/`, `deploy/`, `pyproject.toml`,
`data/testsets/{dualmic_v1,defence_v1}` (or just the 16 files) and `reports/results/{dualmic_v1,defence_v1}/per_file.csv`.

**Check:** `ls /opt/danc` shows `src exports deploy pyproject.toml`. `/opt/danc` must be writable by you,
because `pip install -e` writes package metadata there.

## 4. Install the runtime

```bash
$ cd /opt/danc
$ ./deploy/jetson/install.sh                 # CPU runtime, venv at /opt/danc/.venv (no PyTorch)
$ source /opt/danc/.venv/bin/activate
```

What it does, idempotently:
1. apt: `python3-venv python3-pip python3-dev build-essential libsndfile1 libportaudio2 alsa-utils`.
2. Creates the venv.
3. Installs `deploy/jetson/requirements-jetson.txt`: numpy, scipy, soundfile, sounddevice, soxr, the
   **onnxruntime CPU wheel from PyPI**, pyyaml, pandas, pesq and pystoi. `pesq` is compiled from source with
   `--no-build-isolation`.
4. Runs `pip install -e .` for the project.
5. Runs a smoke test.

Options (`./deploy/jetson/install.sh --help`):

| Option | Use |
|---|---|
| `--gpu` | Replaces the CPU wheel with **onnxruntime-gpu** (CUDA and TensorRT execution providers) from the Jetson AI Lab index `https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/`, used as `--index-url` (PyPI's own aarch64 onnxruntime-gpu wheels are generic server-ARM builds, not Jetson builds). The CPU and GPU packages cannot coexist. |
| `--with-trt` | Sets up the TensorRT path for `trt_runner.py`. The venv is created with `--system-site-packages`, because TensorRT's Python bindings come from apt (`python3-libnvinfer`) and are not on pip for Jetson. It also installs `cuda-python==12.6.*` from the index. Use a separate venv, for example `./deploy/jetson/install.sh /opt/danc/.venv-trt --with-trt`. |
| `--offline-wheels DIR` | Air-gapped install from a wheel directory (see below). |
| `--no-apt`, `--no-pesq`, `--python PY`, `--index URL` | As named. |

**Air-gapped sites.** On a *connected* Jetson with the same JetPack, build all wheels once and carry the
directory across:

```bash
$ python3 -m venv /tmp/wb && . /tmp/wb/bin/activate && pip install -U pip "setuptools>=64" wheel numpy cython pytest-runner
$ pip download -d wheels pip "setuptools>=64" wheel cython pytest-runner
$ pip wheel --no-build-isolation -r deploy/jetson/requirements-jetson.txt -w wheels
# target: sudo apt-get install the apt packages above from your local mirror, then
$ ./deploy/jetson/install.sh --offline-wheels /path/to/wheels
```
For `--gpu` offline, also put the Jetson wheel in the directory:
`pip download --no-deps -d wheels --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/ onnxruntime-gpu`.

**Check:** the installer ends with `torch: not installed (expected)` and a "Next steps" block.
`python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"` prints a version.
With Python 3.10 this is at most 1.23.x for the CPU wheel, because later releases on PyPI need Python 3.11 or newer.

> Why no PyTorch: `python -m danc.inference.realtime` imports `danc.inference.runners`, which imports torch at
> module level. On the Jetson, use **`deploy/jetson/danc_live.py`**. It registers a torch-free ONNX Runtime
> runner with the same math and then runs `danc.inference.realtime` unchanged. On the Mac its output was
> bit-identical to the original path. Like the project's own runner, it also accepts two-microphone models
> (ONNX input `spec_ref`), in case one is deployed later.

## 5. Self-check

```bash
$ cd /opt/danc && source .venv/bin/activate
$ python deploy/jetson/verify_install.py --json verify_report.json
```

It blocks `import torch`, as on a clean Jetson, and then checks the following.
- **env:** Python and packages, ORT providers, Jetson release, CPU governor.
- **torch-free:** the engine and the live app import without torch.
- **contract:** every deployed model in `exports/DEPLOYMENT.json` (round 2: `HQ`, `LL`, `TWO_MIC`): SHA-256, input and output names and shapes (the two-mic model has a second input, `spec_ref`), recurrent states, look-ahead and STFT metadata, and one zero-input step.
- **quality:** the streaming engine on 8 dual-mic scenes (reference-mic NLMS + DNN, or the NLMS → two-mic cascade for `TWO_MIC`) and 8 single-mic defence scenes for each single-mic model. PESQ-WB, PESQ-NB, STOI and SNR are compared per file against `reports/results/*/per_file.csv`, with tolerances |ΔPESQ| < 0.05, |ΔSTOI| < 0.01 and |ΔSNR| < 0.25 dB.
- **robust:** silence; full-scale clipping (the limiter must hold −1 dBFS); reference mic absent or dead; NaN/Inf bursts; `reset()` determinism; bounded statistics memory for long runs; a run past 30 s. For `TWO_MIC`, also a reference that dies after 1 s, which must switch to the single-mic fallback.
- **timing:** 3000 hops of the full live engine, with p50, p99, p99.9 and max against the 10 ms deadline. For `TWO_MIC`, both networks run, as live.

**Check:** the last line is `SUMMARY: ... 0 FAIL ... -> PASS` (exit code 0). **Round 2, Mac, idle (2026-10-05):** `115 PASS, 0 FAIL, 0 WARN,
10 INFO -> PASS` in 30 s for HQ, LL and TWO_MIC. All 40 file comparisons matched their references exactly, and p99.9 was 0.78 / 0.60 / 1.46 ms (`reports/results/verify_install_mac_round2.json`; the 3000-hop self-check timing. The 6000-hop benchmark in `REPORT.md` §6 gives 0.75 / 0.60 / 1.41 ms).
Round 1 (2026-10-04, with a training job loading the CPU; `logs/verify_install_round1.out`, `reports/results/verify_install_mac_round1.json`): the result was `83 PASS, 0 FAIL, 1 WARN, 7 INFO -> PASS` in 73 s. All 32 file comparisons (16 files × HQ/LL)
matched their references exactly (max |Δ| = 0.0000). The WARN was LL p99.9 = 5.15 ms against the 5 ms target, caused
by the background load. The same check run inside `dist/jetson_bundle/` gave `83 PASS, 0 FAIL`. An independent
re-run by the review (2026-10-04; `scripts/verify_install_jetsonlike.py`, `logs/verify_install_round1_review_jetsonlike.out`), with torch **and every other package that is not in `requirements-jetson.txt`**
blocked, gave the same `83 PASS, 0 FAIL, 1 WARN, 7 INFO -> PASS` (77 s; this time the WARN was HQ p99.9 = 5.01 ms)
and a largest per-file difference of 2.4e-5 across all metrics.
On the Jetson, expect tiny differences from the different ONNX Runtime version and Arm floating point, still well
inside the tolerances. WARN lines about the 5 ms p99.9 target or individual hops over 10 ms are possible on an
untuned system. Fix them in step 11.

## 6. (optional) TensorRT engines

The **recommended default is the CPU path**. The network is batch-1, single-frame and tiny (7.6 M MAC per hop), so
GPU kernel-launch overhead dominates (`README.md` §2). Use TensorRT only if the CPU misses the deadline in your
power mode, or if the GPU is used anyway.

```bash
$ ./deploy/jetson/install.sh /opt/danc/.venv-trt --with-trt && source /opt/danc/.venv-trt/bin/activate
$ cd /opt/danc/deploy/jetson
$ ./build_trt_engine.sh ../../exports/dancnet_v3hq_cont.onnx ./engines     # HQ -> engines/dancnet_v3hq_cont_fp16.engine
$ ./build_trt_engine.sh ../../exports/dancnet_ll.onnx   ./engines     # LL -> engines/dancnet_ll_fp16.engine
$ python3 trt_runner.py --engine engines/dancnet_v3hq_cont_fp16.engine --onnx ../../exports/dancnet_v3hq_cont.onnx \
          --compare --cuda-graph --json trt_fp16_hq.json
$ cd /opt/danc && python deploy/jetson/verify_install.py --trt-engines deploy/jetson/engines --skip-robustness
```

**Check:** `trtexec` finishes without errors. `--compare` gives max |TRT − ORT| < 1e-2 on `spec_out`.
`verify_install.py` with `--trt-engines` also runs the 16-file quality check through the FP16 engine.
To use an engine live, add `--backend trt --trt-engine deploy/jetson/engines/dancnet_v3hq_cont_fp16.engine
--cuda-graph` to the step 8 command. This path is **not verified on hardware**.

## 7. Connect the audio interface and check the channels

| Interface input | Engine channel | Connect |
|---|---|---|
| Input 1 (left, "ch0") | **primary** (`--primary-ch 0`) | boom mic at the mouth |
| Input 2 (right, "ch1") | **reference** (`--reference-ch 1`) | outward-facing reference mic |
| Outputs L+R | processed speech (same on both) | headset earcups and/or radio line input |

```bash
$ arecord -l                                          # ALSA card list: note the USB card NUMBER ("card 2: ...")
$ python deploy/jetson/danc_live.py --list-devices    # PortAudio names: pick a UNIQUE substring
$ arecord --dump-hw-params -D hw:2 -c 2 -r 16000 -f S16_LE -d 1 /dev/null 2>&1 | grep -E 'RATE|CHANNELS'   # 2 = your card number
$ python deploy/jetson/check_channels.py --device USB  # quiet room: speak into the boom mic for 6 s
```
`USB` in these commands is a PortAudio name substring; replace it with the unique part of your interface's name
from `--list-devices` (for example `"USB Audio"` or `"hw:2,0"`). Set both microphone pre-amp gains to the **same**
position before running `check_channels.py`, because it compares the two channel levels.

**Check:** `check_channels.py` prints `PASS`. That means:
- the primary is at least 6 dB louder than the reference while you speak;
- there is no clipping;
- speech is not too quiet (aim for speech peaks around −12 dBFS on the primary).

If it reports SWAPPED, swap the cables, or pass `--primary-ch 1 --reference-ch 0` to every command. With swapped
channels the canceller uses your voice as the noise reference and removes speech.
Phantom power: enable it on the interface only if your microphones need it.

## 8. Run the engine (HQ, LL or two-mic)

Start with a hardware-free simulation, then go live:

```bash
$ cd /opt/danc && source .venv/bin/activate
# simulation: a recorded dual-mic scene through the same per-block callback, paced at real time
$ python deploy/jetson/danc_live.py --onnx exports/dancnet_v3hq_cont.onnx --paced \
    --simulate data/testsets/dualmic_v1/primary/0120_impulsive_m05.flac \
    --simulate-ref data/testsets/dualmic_v1/reference/0120_impulsive_m05.flac --out /tmp/sim_hq.wav
# LIVE, HQ model: 40 ms algorithmic latency (2-frame look-ahead), best quality
$ sudo -E env PATH="$PATH" python deploy/jetson/danc_live.py --onnx exports/dancnet_v3hq_cont.onnx \
    --in-device USB --out-device USB --cpus auto --rt-priority 80 --log /tmp/engine_hq.jsonl
# LIVE, LL model: 20 ms algorithmic latency (no look-ahead)
$ sudo -E env PATH="$PATH" python deploy/jetson/danc_live.py --onnx exports/dancnet_ll.onnx \
    --in-device USB --out-device USB --cpus auto --rt-priority 80 --log /tmp/engine_ll.jsonl
```

**Two-microphone headsets (recommended for crewed platforms: it handles other crew members talking).** This is the
deployed `TWO_MIC` configuration. The gated NLMS runs first and feeds the two-microphone network
(`--two-mic-cascade`). The HQ model runs in hot standby (`--fallback-onnx`) and takes over within 1 s if the reference
mic dies:
```bash
# simulation first
$ python deploy/jetson/danc_live.py --onnx exports/dancnet_v3_2mic.onnx --two-mic-cascade \
    --fallback-onnx exports/dancnet_v3hq_cont.onnx --paced \
    --simulate data/testsets/dualmic_v1/primary/0120_impulsive_m05.flac \
    --simulate-ref data/testsets/dualmic_v1/reference/0120_impulsive_m05.flac --out /tmp/sim_2mic.wav
# LIVE
$ sudo -E env PATH="$PATH" python deploy/jetson/danc_live.py --onnx exports/dancnet_v3_2mic.onnx --two-mic-cascade \
    --fallback-onnx exports/dancnet_v3hq_cont.onnx \
    --in-device USB --out-device USB --cpus auto --rt-priority 80 --log /tmp/engine_2mic.jsonl
```
Always use `--fallback-onnx` with the two-mic model. With a dead reference and no fallback, the two-mic network is
barely better than no processing (`reports/results/cascade_decision.md`). The TensorRT path (step 6) supports the
single-mic models only. The two-mic configuration runs on ONNX Runtime and costs about 2× the single-mic compute
(1.3 ms per hop on the Mac).

`sudo` is needed only for `--rt-priority` (SCHED_FIFO). The alternative is to allow real-time priority for the
`audio` group and then log out and back in:
`echo -e '@audio - rtprio 95\n@audio - memlock unlimited' | sudo tee /etc/security/limits.d/95-danc-audio.conf`
and `sudo usermod -aG audio $USER`. Useful options, all passed through to `danc.inference.realtime`:

| Option | Effect |
|---|---|
| `--no-lms` | DNN only, ignoring the reference mic |
| `--bypass` | STFT pass-through with no processing (for latency tests) |
| `--limiter-dbfs -1` | output peak limiter, the default (hearing protection) |
| `--gain-floor-db -25` | comfort-noise floor |
| `--threads 1` | ORT threads (1 is the real-time setting) |
| `--providers CUDAExecutionProvider,CPUExecutionProvider` | GPU EP, with the `--gpu` install |
| `--two-mic-cascade` | two-mic model only: NLMS first, then the two-mic network (the deployed `TWO_MIC` configuration) |
| `--fallback-onnx exports/dancnet_v3hq_cont.onnx` | two-mic model only: single-mic network in hot standby, used when the reference mic is dead (digital silence, or ≥ 40 dB below the active primary for 0.5 s); the live log shows `ref_dead` / `ref_switches` |
| `--ref-floor-dbfs -100`, `--ref-rel-dead-db 40` | thresholds of that reference-health monitor |
| `--watchdog-s 3` | launcher option (default 3 s; 0 = off): if no audio block arrives for that long, for example because the USB interface was unplugged, the launcher exits with code 4 so that systemd restarts it instead of hanging silently |

**Check:** the launcher prints `[preflight] OK` and `[live] running: input latency ... output latency ...`.
After that it prints a JSON line every 5 s with `mean_ms`, `p99_ms`, `max_ms`, `xruns` and `late`.
`mean_ms`, `p99_ms` and `max_ms` are over the last 15 to 30 s; `xruns` and `late` count from the start.
`late` and `xruns` must stay at 0. Talk with noise playing: the processed voice is heard in the earcups.
**The sidetone (own voice) must stay on the analog path**; it is not routed through the Jetson.

## 9. Measure loop-back latency

**(a) Audio I/O only.** Cable OUT-L → IN-1 at line level, using a pad if IN-1 is a mic input. Stop the engine first.
```bash
$ python deploy/jetson/measure_latency.py --device USB --rate 16000 --block 160 --channel 0 --repeats 5
```
**(b) With the engine in the path.** This needs a second interface B, because the engine holds interface A.
Connect B-OUT → A-IN1 and A-OUT-L → B-IN1. First measure B's own loop-back as in (a). Then run the engine on A,
once with `--bypass` and once with the model, and each time measure through B. With two USB interfaces
connected, `USB` is no longer a unique name: use a unique substring for each (from `--list-devices`).
```bash
$ python deploy/jetson/danc_live.py --onnx exports/dancnet_ll.onnx --in-device "<A name>" --out-device "<A name>" --bypass &
$ python deploy/jetson/measure_latency.py --device "<B name>" --rate 16000 --block 160
$ kill %1        # stop the engine on A before the next run
```
- Engine path = measured − B's loop-back.
- `--bypass` contains A's I/O plus the 20 ms STFT framing.
- By construction, the model adds exactly look-ahead × 10 ms: **0 ms for LL and 20 ms for HQ** (unit-tested
  streaming delay).
- The DNN treats a sine sweep as noise. If the model run gives an unstable peak, use the bypass value plus the
  look-ahead.

**Expected (estimate [E], not measured):** I/O ≈ 25 to 40 ms; mouth-to-line-out ≈ 45 to 60 ms (LL) or 65 to 80 ms (HQ),
inside ITU-T G.114's 150 ms. Record your measured values. The `measure_latency.py` docstring mentions
`--passthrough-test`; the actual option is `--bypass`.

## 10. Log power

```bash
$ ./deploy/jetson/log_power.sh 120 power_30W_hq.csv     # tegrastats every 100 ms for 120 s, while the engine runs
$ sudo tegrastats --interval 1000                       # interactive view; stop with Ctrl-C (or tegrastats --stop)
$ sudo pip3 install -U jetson-stats && sudo reboot      # optional: 'jtop' dashboard (system pip, not the venv)
$ jtop
```
**Check:** the CSV has one column per power rail (for example VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0). Rail names
vary between releases, so inspect the `.raw.txt` file. Repeat for each nvpmodel mode you may field. Record idle
(engine stopped), engine on the CPU, and engine on TensorRT if used.

## 11. Real-time acceptance run (do this in the fielded power mode)

```bash
$ for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee "$g" >/dev/null; done
$ sudo -E env PATH="$PATH" python deploy/jetson/verify_install.py --skip-quality --strict-timing --cpus auto --rt-priority 80
$ sudo -E env PATH="$PATH" timeout -s INT 660 python deploy/jetson/danc_live.py --onnx exports/dancnet_v3hq_cont.onnx \
    --in-device USB --out-device USB --cpus auto --rt-priority 80 --log /tmp/accept_hq.jsonl
$ tail -n 1 /tmp/accept_hq.jsonl
```
**Pass criteria:**
- `--strict-timing` PASS: no hop over 10 ms and p99.9 < 5 ms.
- 10-minute live run: `late` = 0 and `xruns` = 0.
- `check_channels.py` PASS.
- Loop-back latency recorded.
- No audible clicks or dropouts.
- Final acceptance is a listening test through the radio chain: Modified Rhyme Test (MIL-STD-1474E: at least 80 %
  in worst-case noise). PESQ and STOI are engineering proxies only.

If timing fails, try these in order: performance governor and MAXN or 30W instead of 15W; disable the desktop
(`sudo systemctl set-default multi-user.target`); the LL model; the LL-small model
(`exports/dancnet_pilot.onnx`, 0.37 ms per block on the Mac, copied only with `package_for_jetson.sh --all-models`;
it is not part of the verified deployment); the TensorRT path (step 6).

## 12. Start at boot (systemd)

```bash
$ sudo useradd --system --home /opt/danc --shell /usr/sbin/nologin --groups audio danc    # once
$ sudo cp /opt/danc/deploy/jetson/danc-live.service /etc/systemd/system/
$ printf 'DANC_ONNX=/opt/danc/exports/dancnet_v3hq_cont.onnx\nDANC_IN_DEVICE=USB\nDANC_OUT_DEVICE=USB\nDANC_CPUS=auto\nDANC_EXTRA_ARGS=\n' | sudo tee /etc/default/danc
$ sudo systemctl daemon-reload && sudo systemctl enable --now danc-live
$ systemctl status danc-live --no-pager ; journalctl -u danc-live -f
```
- For the LL model, set `DANC_ONNX=/opt/danc/exports/dancnet_ll.onnx`.
- For two-microphone headsets, set `DANC_ONNX=/opt/danc/exports/dancnet_v3_2mic.onnx` and
  `DANC_EXTRA_ARGS=--two-mic-cascade --fallback-onnx /opt/danc/exports/dancnet_v3hq_cont.onnx`.
- For swapped channels, set `DANC_EXTRA_ARGS=--primary-ch 1 --reference-ch 0`.
- After any change, run `sudo systemctl restart danc-live`.
- The unit runs as user `danc` (group `audio`) with SCHED_FIFO priority 80 and restarts on failure, retrying
  indefinitely (for example while the USB interface is unplugged). If the interface disappears while the engine
  runs, the launcher's stall watchdog ends the process after 3 s (exit code 4) and systemd restarts it.
- Logs go to `/var/log/danc/engine.jsonl`.
- `danc-live.service` replaces `danc.service`. The old unit imports torch and pins offline CPUs in 15W and 30W.

**Check:** `systemctl is-active danc-live` prints `active`. After a reboot the engine runs without a login, and
`journalctl -u danc-live` shows `[preflight] OK`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | `python -m danc.inference.realtime` or the old `danc.service` | use `deploy/jetson/danc_live.py` / `danc-live.service` |
| `pip` finds no `onnxruntime` for Python 3.10 above 1.23 | PyPI's CPU wheels from 1.24 on need Python 3.11+ | expected; `install.sh` resolves to 1.23.x automatically |
| `pesq` fails to build | missing compiler or headers, or build isolation | `sudo apt-get install build-essential python3-dev`, then re-run `install.sh` |
| `[preflight] FAIL ... Invalid sample rate` | interface does not support 16 kHz | check rates with `arecord --dump-hw-params`; add an ALSA plug PCM that resamples, e.g. in `~/.asoundrc`: `pcm.danc16k { type plug; slave { pcm "hw:USB,0"; rate 48000 } }`, then use `--in-device danc16k --out-device danc16k` if it appears in `--list-devices` (not verified; for the systemd service put it in `/etc/asound.conf`, because the `danc` user does not read your `~/.asoundrc`) |
| `Multiple input devices found for 'USB'` | substring not unique | use a longer substring from `--list-devices`, e.g. `"hw:2,0"` |
| `Device unavailable` / `busy` | PulseAudio/PipeWire or another app holds the device | `fuser -v /dev/snd/*`; stop the desktop audio server or run headless |
| `late` > 0 or `xruns` > 0 | CPU frequency scaling, desktop session, IRQs on the RT core | performance governor; `--cpus auto --rt-priority 80`; `multi-user.target`; LL or LL-small model; ALSA periods 2 → 3 |
| `SCHED_FIFO ... refused` | no RT privilege | run via `sudo`, the systemd unit, or the `limits.d` rule in step 8 |
| `requested CPUs ... are not online` | 15W/30W mode with CPUs 10,11 | use `--cpus auto` |
| speech sounds thin or is removed when the reference mic is near the mouth | talker leaks into the reference | move the reference outward, away from the mouth; re-run `check_channels.py` (target ≥ 6 dB) |
| output silent | channels swapped, or wrong output device | `check_channels.py`; `--list-devices`; try `--bypass` |
| clicks during gunfire or blasts | microphone clipping | lower the pre-amp gain, use high-SPL capsules; the limiter keeps the output at −1 dBFS or below |
| `onnxruntime-gpu` or `import tensorrt` error mentioning NumPy 1.x/2.x | wheel or apt bindings built against another NumPy | `pip install "numpy<2"` in that venv (GPU / TensorRT venv only), then re-run `verify_install.py` |
| TensorRT `could not deserialize` | engine built with another TensorRT or GPU | rebuild on this device with `build_trt_engine.sh` |
| `--backend trt` refused: "two-microphone model" | a model with a `spec_ref` input is deployed; `trt_runner.py` feeds only `spec` | use the ONNX Runtime backend (default) for that model |
| service restarts every few seconds, journal shows `[watchdog] no audio block ...` | USB interface unplugged, or the stream stalled | re-seat the USB cable; `--list-devices`; check `dmesg` for USB resets |
| `verify_install.py` quality FAIL with large Δ | wrong model or reference files (models re-exported or re-trained without updating `per_file.csv`) | check the SHA-256 lines (contract) and the bundle's `SHA256SUMS`; for a re-trained model, re-run `danc.eval.evaluate` for it, add `"reference_systems": {"dualmic_v1": "<system>", "defence_v1": "<system>"}` to its `exports/DEPLOYMENT.json` entry, and rebuild the bundle |

---

## Expected results (reference: Apple M4, macOS; NOT Jetson)

Latency, round 2 (`reports/results/latency_mac_m4_round2_final.json`; ONNX Runtime 1.30 CPU EP, 1 thread, 6000 frames, idle machine):

| Deployed configuration | Algorithmic latency | Network step mean / p99.9 | Full engine block mean / p99 / p99.9 / max | Misses > 10 ms |
|---|---|---|---|---|
| HQ `exports/dancnet_v3hq_cont.onnx` (+ NLMS) | 40 ms | 0.57 / 0.63 ms | 0.67 / 0.71 / 0.75 / 1.02 ms | 0 |
| LL `exports/dancnet_ll.onnx` (+ NLMS) | 20 ms | 0.43 / 0.48 ms | 0.51 / 0.55 / 0.60 / 0.73 ms | 0 |
| TWO_MIC: NLMS → `dancnet_v3_2mic.onnx`, HQ in hot standby | 40 ms | — | 1.27 / 1.35 / 1.41 / 1.52 ms | 0 |

- **Paced real-time simulation** (round 1, round-1 HQ model; `reports/results/simulated_live_runs.json`): 2.0 to 2.2 ms mean, max ≤ 4.3 ms, 0 misses. Blocks are slower when paced because the CPU idles between them.
- **Jetson estimate [E]** (`REPORT.md` §6.3): an A78AE core is assumed 2–3× slower than an M4 core, so about 1–2 ms per hop for HQ and LL and 2.5–4 ms for TWO_MIC. The paced worst case is higher, so it **must be measured** (steps 5 and 11).
- **`verify_install.py` timing on the Mac, round 2** (idle, unpaced, no RT settings): p99.9 0.78 ms (HQ), 0.60 ms (LL), 1.46 ms (TWO_MIC), 0 of 3000 hops over 10 ms each. Under background load (round 1, a training job running) the p99.9 rose to 4–5 ms, with a few hops over 10 ms. That is why the field acceptance uses `--strict-timing` with `--cpus auto --rt-priority 80` on an otherwise idle Jetson.

Quality: Mac reference values, full test sets (`reports/results/*/summary.json`, round 2, 2026-10-05; `verify_install.py` always reads the current per-file values):

| Set | System | PESQ-WB | PESQ-NB | STOI | SNR (dB) |
|---|---|---|---|---|---|
| dualmic_v1 (240 scenes, −5…10 dB) | input (primary mic) | 1.23 | 1.78 | 0.798 | 2.50 |
| | **HQ hybrid** (`hybrid:v3hq_cont`) | 2.50 | 3.12 | 0.932 | 13.55 |
| | **LL hybrid** (`hybrid:ll`) | 2.37 | 2.99 | 0.923 | 13.43 |
| | **TWO_MIC cascade** (`hybrid2:v3_2mic`) | 2.62 | 3.20 | 0.931 | 13.65 |
| defence_v1 (800 files, −5…15 dB) | noisy input | 1.34 | 1.95 | 0.815 | 3.80 |
| | **HQ** (`dancnet:v3hq_cont`) | 2.51 | 3.17 | 0.918 | 14.31 |
| | **LL** (`dancnet:ll`) | 2.33 | 2.98 | 0.904 | 13.04 |
| dualmic_babble_v1 (60 crew-babble scenes) | input (primary mic) | 1.12 | 1.60 | 0.740 | — |
| | HQ hybrid | 1.37 | 2.01 | 0.825 | — |
| | **TWO_MIC cascade** | 2.06 | 2.80 | 0.904 | — |

The 8-file verification subsets give these means on the Mac. They mix −5 dB and 10 or 15 dB scenes, so they
differ from the full-set means:

| Subset | HQ PESQ-WB / NB / STOI / SNR | LL PESQ-WB / NB / STOI / SNR | TWO_MIC PESQ-WB / NB / STOI / SNR |
|---|---|---|---|
| dualmic_v1, 8 scenes | 2.36 / 2.91 / 0.925 / 12.37 dB | 2.19 / 2.74 / 0.916 / 12.05 dB | 2.44 / 3.04 / 0.928 / 12.86 dB |
| defence_v1, 8 files | 2.36 / 2.95 / 0.908 / 13.83 dB | 2.26 / 2.84 / 0.888 / 12.66 dB | (single-mic set: not applicable) |

---

## Sources (accessed 2026-10-04) and what could not be verified

| Item | Source | Finding |
|---|---|---|
| JetPack 6.2 contents | https://developer.nvidia.com/embedded/jetpack-sdk-62 | Jetson Linux 36.4.3, TensorRT 10.3.0, CUDA 12.6.10, cuDNN 9.3.0 |
| JetPack 6.2.1 / 6.2.2 | NVIDIA forum announcements https://forums.developer.nvidia.com/t/jetpack-6-2-1-jetson-linux-36-4-4-is-now-live/337334 and https://forums.developer.nvidia.com/t/jetpack-6-2-2-jetson-linux-36-5-is-now-live/359622 (opened by the review on 2026-10-04) | 6.2.1 = Jetson Linux 36.4.4, "the same libraries, components and ... SDKs as JetPack 6.2"; 6.2.2 = Jetson Linux 36.5, the same libraries as 6.2.1; both Ubuntu 22.04, kernel 5.15 |
| onnxruntime-gpu for JetPack 6 | https://pypi.jetson-ai-lab.io/jp6/cu126 and `.../+simple/onnxruntime-gpu/` | index lists `onnxruntime_gpu-1.24.0-cp310-cp310-linux_aarch64.whl` and 1.23.0; also `cuda-python 12.6.2.post1`; pip index URL `https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/`. Packages it does not host (e.g. `.../+simple/numpy/`) are HTTP-302 redirected to pypi.org, so it works as the only `--index-url` |
| onnxruntime-gpu on PyPI | https://pypi.org/simple/onnxruntime-gpu/ | aarch64 wheels only from 1.29.0 on, for cp311 and later (`manylinux_2_34_aarch64`), no cp310; these are generic builds, which is why `install.sh --gpu` uses the Jetson index as `--index-url` |
| onnxruntime CPU on PyPI | https://pypi.org/pypi/onnxruntime/json | last cp310 aarch64 wheel: 1.23.2; 1.24 and later need Python ≥ 3.11 |
| TensorRT Python on JetPack 6.2 | NVIDIA moderator, https://forums.developer.nvidia.com/t/how-to-install-tensorrt-in-jetpack-6-2/335222 | apt packages `tensorrt-libs tensorrt python3-libnvinfer python3-libnvinfer-dev`; TensorRT does not build pip wheels for Tegra |
| JetPack check / meta-package | https://docs.pytorch.org/TensorRT/getting_started/jetpack.html | `apt show nvidia-jetpack`, `sudo apt-get install nvidia-jetpack`; Jetson AI Lab index pattern `https://pypi.jetson-ai-lab.io/jp6/cu126` |
| Power modes | https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html | AGX Orin 64GB: 0 MAXN 12 CPU @ 2.2 GHz, 1 15W 4 CPU @ 1.11 GHz, 2 30W 8 CPU @ 1.73 GHz, 3 50W 12 CPU @ 1.50 GHz; "The default mode is 30W (mode ID 2)" (re-checked 2026-10-04); `sudo nvpmodel -q / -m <x>`; mode persists across reboots |
| tegrastats | https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html | options `--interval <ms>`, `--logfile <file>`, `--verbose`, `--stop` |
| jetson-stats / jtop | https://rnext.it/jetson_stats/ | `sudo pip3 install -U jetson-stats`, then log out and in or reboot, then `jtop` |

**Not verified (treat as assumptions until checked on the board):**
- every Jetson runtime result: latency, deadline misses, power, ALSA behaviour, TensorRT engines and the TensorRT live path;
- that `onnxruntime-gpu` 1.24.0 from the Jetson AI Lab index loads these models with the CUDA and TensorRT EPs, and with which NumPy;
- that `cuda-python==12.6.*` from the index works with `trt_runner.py`;
- the exact `dpkg -l nvidia-jetpack` version string;
- the AGX Orin power-rail names printed by tegrastats on JetPack 6.2 (not on the fetched page);
- whether jetson-stats officially supports JetPack 6.2.x (not stated on the fetched page);
- whether PortAudio lists ALSA PCMs defined in `asound.conf` / `~/.asoundrc` (the 16 kHz plug workaround);
- whether the TensorRT 10.3 Python bindings from apt work with NumPy 2 in the `--with-trt` venv;
- the stall watchdog's behaviour on a real USB unplug (it was tested only with a simulated stall).
