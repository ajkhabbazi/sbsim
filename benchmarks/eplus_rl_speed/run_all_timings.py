"""Clean session: rerun every benchmark timing from scratch, under the same conditions.

Runs steps 2, 3, 4, 5, 6b, 7a, 7b and 7c of the README in order, each as a fresh
process in the sbsim-bench env. Before each step it waits until the PC is idle.
During each step it samples how much CPU the other programs use, so any
interference shows in the results. While the other programs are above 2 cores
it also records which programs they are, so a busy step names its cause. It
stops at the first failed step, which
includes an annual energy total that differs from the known value (225.20 GJ
uncontrolled, 271.51 GJ at 21/24 C).

Everything goes to out-all/<YYMMDD-HHMMSS>/: one folder per step (log.txt and
the step's output files), plus results.json and results.md.

A full run that passes every step with no interference creates
sbsim-bench-done.flag in the system's temporary folder, which another job can
wait on before it starts something CPU-heavy. A quick run never does.

The EnergyPlus and SBSim folders are set in bench_paths.py.

Usage (in the sbsim-bench env, from this folder):
  python run_all_timings.py --quick   # short trainings in 7b and 7c, to test this script
  python run_all_timings.py           # full run, about 1-1.5 h
"""
import argparse, ctypes, datetime, json, math, os, platform, re, shutil, subprocess, sys, tempfile, threading, time
from ctypes import wintypes
from importlib import metadata
from pathlib import Path
import psutil
from bench_paths import ENERGYPLUS_DIR, SBSIM_DIR, WEATHER

HERE = Path(__file__).resolve().parent
PY = sys.executable
EPLUS_DIR = ENERGYPLUS_DIR
EPLUS_EXE = EPLUS_DIR / "energyplus.exe"
EPW = WEATHER
IDF = HERE / "5Zone_IdealLoadsAirSystems_ReturnPlenum.idf"
SBSIM = SBSIM_DIR
FLAG = Path(tempfile.gettempdir()) / "sbsim-bench-done.flag"
OUTPUT_FOLDERS = ("out-api", "out-control", "out-env", "out-tf-env", "out-train")

UNCONTROLLED_GJ = 225.20   # annual site energy with the IDF's own setpoint schedules
CONTROLLED_GJ = 271.51     # annual site energy at 21/24 C every step
STEPS_PER_YEAR = 105120    # 5-minute steps in the run period

IDLE_MAX_PCT = 20.0        # before a step, whole-PC CPU use must be below this [%]
IDLE_WINDOW_S = 10         # ... measured over this window
IDLE_WAIT_MAX_S = 600      # give up if the PC is not idle within 10 minutes
OTHERS_MEAN_MAX = 2.0      # during a step, other programs' mean CPU use [cores]
OTHERS_10S_MAX = 3.0       # during a step, other programs' highest 10-second mean [cores]
NAME_OTHERS_ABOVE = 2.0    # while other programs are above this [cores], record which programs they are
BUSY_LOG_MAX = 600         # keep at most this many of those one-second records per step
BLOCKERS = ("train_bench", "check_sbsim_env", "run_all_timings")  # in a Python command line

QUICK_TRAIN = {   # quick mode only: just enough to test the script end to end
    "eplus": ["--iterations", "1", "--collect-steps", "12", "--learner-iterations", "10"],
    "sbsim": ["--iterations", "1", "--collect-steps", "3", "--learner-iterations", "10"],
}


# ---- small helpers ---------------------------------------------------------
def say(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def fmt(value, spec=".2f"):
    return "n/a" if value is None else format(value, spec)


def grab(pattern, text, cast=float):
    m = re.search(pattern, text, re.MULTILINE)
    return cast(m.group(1)) if m else None


def site_gj(htm):
    if not htm.exists():
        return None
    return grab(r"Total Site Energy</td>\s*<td[^>]*>\s*([\d.]+)", htm.read_text(errors="ignore"))


def gj_check(value, expected):
    ok = value is not None and abs(value - expected) < 0.005
    return ("annual site energy", ok, f"{value} GJ (expected {expected:.2f})")


def kill_tree(pid):
    try:
        root = psutil.Process(pid)
        procs = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


# ---- CPU: wait for an idle PC, and sample during a step ---------------------
def blockers():
    """Other benchmark processes that must not run next to a step."""
    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.info["pid"] == os.getpid():
            continue
        name = (p.info["name"] or "").lower()
        cmd = " ".join(p.info["cmdline"] or []).lower()
        if name == "energyplus.exe" or (name.startswith("python") and any(b in cmd for b in BLOCKERS)):
            found.append(f"{name} (PID {p.info['pid']})")
    return found


def wait_until_idle():
    """Return (CPU %, seconds waited) once the PC is idle; raise after IDLE_WAIT_MAX_S."""
    t0 = time.monotonic()
    while True:
        procs = []
        for p in psutil.process_iter(["pid", "name"]):
            if p.info["pid"] in (0, os.getpid()):
                continue
            try:
                p.cpu_percent(None)
                procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        pct = psutil.cpu_percent(interval=IDLE_WINDOW_S)
        found = blockers()
        if pct < IDLE_MAX_PCT and not found:
            return pct, time.monotonic() - t0
        top = []
        for p in procs:
            try:
                top.append((p.cpu_percent(None) / 100.0, p.info["name"]))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        top.sort(reverse=True)
        busiest = ", ".join(f"{n} {c:.1f} cores" for c, n in top[:4])
        say(f"  waiting for an idle PC: {pct:.0f}% CPU ({busiest})"
            + (f"; running: {', '.join(found)}" if found else ""))
        if time.monotonic() - t0 > IDLE_WAIT_MAX_S:
            raise RuntimeError(f"the PC was not idle within {IDLE_WAIT_MAX_S // 60} minutes")


class _UnicodeString(ctypes.Structure):
    _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT), ("Buffer", ctypes.c_void_p)]


class _ProcessEntry(ctypes.Structure):   # the start of Windows' SYSTEM_PROCESS_INFORMATION
    _fields_ = [("NextEntryOffset", wintypes.ULONG), ("NumberOfThreads", wintypes.ULONG),
                ("WorkingSetPrivateSize", ctypes.c_longlong), ("HardFaultCount", wintypes.ULONG),
                ("NumberOfThreadsHighWatermark", wintypes.ULONG), ("CycleTime", ctypes.c_ulonglong),
                ("CreateTime", ctypes.c_longlong), ("UserTime", ctypes.c_longlong),
                ("KernelTime", ctypes.c_longlong), ("ImageName", _UnicodeString),
                ("BasePriority", wintypes.LONG), ("UniqueProcessId", ctypes.c_void_p)]


def process_times():
    """CPU seconds used so far by every process: {pid: (name, seconds)}.

    One Windows call for all processes, protected ones (Defender) included. psutil
    needs a call per process, which took over a second and half a core per sweep.
    """
    size = wintypes.ULONG(1 << 20)
    while True:
        buf = ctypes.create_string_buffer(size.value)
        status = ctypes.windll.ntdll.NtQuerySystemInformation(5, buf, size, ctypes.byref(size)) & 0xFFFFFFFF
        if status == 0:
            break
        if status != 0xC0000004:   # anything but "buffer too small"
            raise OSError(f"NtQuerySystemInformation failed: {status:#x}")
        size = wintypes.ULONG(size.value + (1 << 16))
    times, offset = {}, 0
    while True:
        e = _ProcessEntry.from_buffer(buf, offset)
        name = ctypes.wstring_at(e.ImageName.Buffer, e.ImageName.Length // 2) if e.ImageName.Buffer else "?"
        times[e.UniqueProcessId or 0] = (name, (e.UserTime + e.KernelTime) / 1e7)
        if not e.NextEntryOffset:
            return times
        offset += e.NextEntryOffset


class CpuSampler(threading.Thread):
    """Once a second: whole-PC CPU use, the step's own processes, and the rest [cores]."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        try:
            self.root = psutil.Process(pid)
        except psutil.NoSuchProcess:
            self.root = None
        self.ncpu = psutil.cpu_count()
        self.stop_event = threading.Event()
        self.samples = []   # (pc_cores, step_cores, others_cores)
        self._seen = {}     # pid -> latest CPU seconds, kept after a process exits
        self.busy_log = []  # (second of the step, others_cores, [(program, cores), ...]) while others are busy

    def _step_cpu(self):
        procs = []
        if self.root is not None:
            try:
                procs = [self.root] + self.root.children(recursive=True)
            except psutil.NoSuchProcess:
                pass
        for p in procs:
            try:
                t = p.cpu_times()
                self._seen[p.pid] = t.user + t.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return sum(self._seen.values())

    @staticmethod
    def _pc_cpu():
        t = psutil.cpu_times()
        total = sum(t)
        return total, total - t.idle

    def _others_snapshot(self):
        """CPU seconds used so far by every process outside the step: {pid: (name, seconds)}."""
        snap = process_times()
        for pid in [0, *self._seen]:   # drop the idle process and the step's own processes
            snap.pop(pid, None)
        if os.getpid() in snap:
            snap[os.getpid()] = ("run_all_timings (this script)", snap[os.getpid()][1])
        return snap

    def _name_others(self, others, prev):
        """While other programs are busy, record the top ones. Does nothing while they are quiet.

        Needs two snapshots, so the first second of each busy stretch goes unnamed.
        """
        if others <= NAME_OTHERS_ABOVE:
            return None
        now, snap = time.monotonic(), self._others_snapshot()
        if prev is not None and now > prev[0] and len(self.busy_log) < BUSY_LOG_MAX:
            use = {}
            for pid, (name, seconds) in snap.items():
                if pid in prev[1]:
                    use[name] = use.get(name, 0.0) + max(0.0, seconds - prev[1][pid][1]) / (now - prev[0])
            top = sorted(use.items(), key=lambda kv: kv[1], reverse=True)[:5]
            self.busy_log.append((len(self.samples), round(others, 2), [(n, round(c, 2)) for n, c in top]))
        return now, snap

    def run(self):
        w0, (tot0, busy0), s0 = time.monotonic(), self._pc_cpu(), self._step_cpu()
        prev = None   # the last snapshot of the other programs, kept only while they are busy
        while not self.stop_event.wait(1.0):
            w1, (tot1, busy1), s1 = time.monotonic(), self._pc_cpu(), self._step_cpu()
            if tot1 > tot0 and w1 > w0:
                pc = (busy1 - busy0) / (tot1 - tot0) * self.ncpu
                step = (s1 - s0) / (w1 - w0)
                others = max(0.0, pc - step)
                self.samples.append((pc, step, others))
                try:
                    prev = self._name_others(others, prev)
                except Exception:
                    prev = None   # naming the programs must never stop the sampling
            w0, tot0, busy0, s0 = w1, tot1, busy1, s1

    def busy_top(self):
        """The programs behind the busy seconds: [(name, mean cores, highest cores)], largest first."""
        if not self.busy_log:
            return []
        total, peak = {}, {}
        for _, _, top in self.busy_log:
            for name, cores in top:
                total[name] = total.get(name, 0.0) + cores
                peak[name] = max(peak.get(name, 0.0), cores)
        names = sorted(total, key=total.get, reverse=True)[:5]
        return [(n, round(total[n] / len(self.busy_log), 2), round(peak[n], 2)) for n in names]

    def summary(self):
        if not self.samples:
            return dict(seconds=0, pc_mean=None, step_mean=None,
                        others_mean=None, others_max_10s=None, clean=True,
                        busy_seconds=0, busy_top=[], busy_log=[])
        pcs, steps, others = zip(*self.samples)
        win = min(10, len(others))
        max10 = max(sum(others[i:i + win]) / win for i in range(len(others) - win + 1))
        mean = sum(others) / len(others)
        return dict(seconds=len(self.samples),
                    pc_mean=round(sum(pcs) / len(pcs), 2),
                    step_mean=round(sum(steps) / len(steps), 2),
                    others_mean=round(mean, 2), others_max_10s=round(max10, 2),
                    clean=mean <= OTHERS_MEAN_MAX and max10 <= OTHERS_10S_MAX,
                    busy_seconds=sum(1 for o in others if o > NAME_OTHERS_ABOVE),
                    busy_top=self.busy_top(), busy_log=self.busy_log)


# ---- what each step prints, and the checks on it ----------------------------
def parse_cli(text, step_dir):
    out = step_dir / "eplus-out"
    err_file = out / "eplusout.err"
    err = err_file.read_text(errors="ignore") if err_file.exists() else ""
    m = re.search(r"Elapsed Time=(\d+)hr\s+(\d+)min\s+([\d.]+)sec", err)
    elapsed = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None
    gj = site_gj(out / "eplustbl.htm")
    return (dict(eplus_elapsed_s=elapsed, site_gj=gj),
            [("EnergyPlus completed successfully", "EnergyPlus Completed Successfully" in err, "eplusout.err"),
             gj_check(gj, UNCONTROLLED_GJ)])


def parse_api(text, step_dir):
    gj = site_gj(step_dir / "out-api" / "eplustbl.htm")
    code = grab(r"^exit code: (-?\d+)", text, int)
    return (dict(script_s=grab(r"^time: ([\d.]+) s", text), site_gj=gj),
            [("EnergyPlus exit code 0", code == 0, f"{code}"), gj_check(gj, UNCONTROLLED_GJ)])


def parse_control(text, step_dir):
    gj = site_gj(step_dir / "out-control" / "eplustbl.htm")
    code = grab(r"^exit code: (-?\d+)", text, int)
    calls = grab(r"^control steps by kind of sim: \{3: (\d+)\}$", text, int)
    return (dict(script_s=grab(r"^time: ([\d.]+) s", text), control_steps=calls, site_gj=gj),
            [("EnergyPlus exit code 0", code == 0, f"{code}"),
             (f"{STEPS_PER_YEAR} control steps", calls == STEPS_PER_YEAR, f"{calls}"),
             gj_check(gj, CONTROLLED_GJ)])


def parse_env(text, step_dir):
    steps = grab(r"^steps: (\d+)", text, int)
    gj = grab(r"^total site energy: ([\d.]+) GJ", text)
    return (dict(script_s=grab(r"^time: ([\d.]+) s", text), steps=steps, site_gj=gj),
            [(f"{STEPS_PER_YEAR} steps", steps == STEPS_PER_YEAR, f"{steps}"),
             gj_check(gj, CONTROLLED_GJ),
             ("early stop and new reset", "early stop after 10 steps: OK" in text, "")])


def parse_tf_env(text, step_dir):
    steps = grab(r"^steps: (\d+)", text, int)
    gj = grab(r"^total site energy: ([\d.]+) GJ", text)
    return (dict(script_s=grab(r"^time: ([\d.]+) s", text),
                 spec_check_s=grab(r"^spec check, one year of random actions: passed, ([\d.]+) s", text),
                 steps=steps, site_gj=gj),
            [("action mapping", "action mapping: OK" in text, ""),
             ("TF-Agents spec check", "spec check, one year of random actions: passed" in text, ""),
             (f"{STEPS_PER_YEAR} steps", steps == STEPS_PER_YEAR, f"{steps}"),
             gj_check(gj, CONTROLLED_GJ),
             ("new episode after the last step", "starts a new episode: OK" in text, "")])


def parse_sbsim_check(text, step_dir):
    m = re.search(r"^(\d+) steps: mean ([\d.]+) s per step \(first ([\d.]+) s, last ([\d.]+) s\)",
                  text, re.MULTILINE)
    n = int(m.group(1)) if m else None
    obs = grab(r"^observation size: \((\d+),\)", text, int)
    per_episode = grab(r"^steps per episode: (\d+)", text, int)
    return (dict(load_s=grab(r"^load: ([\d.]+) s", text), reset_s=grab(r"^reset: ([\d.]+) s", text),
                 steps=n, mean_step_s=float(m.group(2)) if m else None,
                 first_step_s=float(m.group(3)) if m else None, last_step_s=float(m.group(4)) if m else None),
            [("12 steps timed", n == 12, f"{n}"),
             ("53 observations", obs == 53, f"{obs}"),
             ("288 steps per episode", per_episode == 288, f"{per_episode}")])


def parse_train(text, step_dir):
    files = list((step_dir / "out-train").glob("*/summary.json"))
    if len(files) != 1:
        return {}, [("one summary.json", False, f"found {len(files)}")]
    s = json.loads(files[0].read_text())
    a = s["args"]
    keys = ("sim_steps", "resets", "setup_s", "first_reset_s", "learner_setup_s", "simulator_s",
            "simulator_ms_per_step", "collect_s", "agent_and_buffer_s", "train_s",
            "train_s_after_first_loop", "gradient_updates", "final_loss")
    metrics = {k: round(s[k], 4) if isinstance(s[k], float) else s[k] for k in keys}
    want_steps = a["iterations"] * a["collect_steps"]
    want_updates = a["iterations"] * a["learner_iterations"]
    # an episode boundary adds one environment call (the reset) beyond the counted steps
    return metrics, [("all steps collected", s["sim_steps"] >= want_steps, f"{s['sim_steps']} of {want_steps}"),
                     ("all updates done", s["gradient_updates"] == want_updates,
                      f"{s['gradient_updates']} of {want_updates}"),
                     ("finite loss", math.isfinite(s["final_loss"]), f"{s['final_loss']:.4g}")]


def train_cmd(sim):
    def cmd(step_dir, quick):
        return [PY, "train_bench.py", "--sim", sim] + (QUICK_TRAIN[sim] if quick else [])
    return cmd


def train_key(m):
    return (f"simulator {fmt(m.get('simulator_s'))} s ({fmt(m.get('simulator_ms_per_step'), '.3f')} ms per step), "
            f"agent + buffer {fmt(m.get('agent_and_buffer_s'))} s, training {fmt(m.get('train_s'))} s")


STEPS = [
    dict(id="2", name="energyplus-cli", what="EnergyPlus command line, full year, no control",
         cmd=lambda d, q: [str(EPLUS_EXE), "-w", str(EPW), "-d", str(d / "eplus-out"), str(IDF)],
         moves=[], timeout=600, parse=parse_cli,
         key=lambda m: f"{fmt(m.get('eplus_elapsed_s'))} s (EnergyPlus's own count)"),
    dict(id="3", name="run-api", what="Python API, full year, no control",
         cmd=lambda d, q: [PY, "run_api.py"], moves=["out-api"], timeout=600, parse=parse_api,
         key=lambda m: f"{fmt(m.get('script_s'))} s"),
    dict(id="4", name="run-control", what="Python API, setpoints 21/24 every step",
         cmd=lambda d, q: [PY, "run_control.py"], moves=["out-control"], timeout=600, parse=parse_control,
         key=lambda m: f"{fmt(m.get('script_s'))} s"),
    dict(id="5", name="test-env", what="RL environment, full year at 21/24",
         cmd=lambda d, q: [PY, "test_env.py"], moves=["out-env"], timeout=900, parse=parse_env,
         key=lambda m: f"{fmt(m.get('script_s'))} s"),
    dict(id="6b", name="test-tf-env", what="TF-Agents environment, full year at 21/24",
         cmd=lambda d, q: [PY, "test_tf_env.py"], moves=["out-tf-env"], timeout=1200, parse=parse_tf_env,
         key=lambda m: f"{fmt(m.get('script_s'))} s (spec check {fmt(m.get('spec_check_s'))} s)"),
    dict(id="7a", name="check-sbsim-env", what="SBSim environment, 12 steps at a fixed action",
         cmd=lambda d, q: [PY, "check_sbsim_env.py"], moves=[], timeout=1200, parse=parse_sbsim_check,
         key=lambda m: (f"{fmt(m.get('mean_step_s'))} s per step "
                        f"(load {fmt(m.get('load_s'))} s, reset {fmt(m.get('reset_s'))} s)")),
    dict(id="7b", name="train-eplus", what="SAC training on EnergyPlus",
         cmd=train_cmd("eplus"), moves=["out-train"], timeout=1800, parse=parse_train, key=train_key),
    dict(id="7c", name="train-sbsim", what="SAC training on SBSim",
         cmd=train_cmd("sbsim"), moves=["out-train"], timeout=4 * 3600, parse=parse_train, key=train_key),
]


# ---- running one step -------------------------------------------------------
def run_step(step, out_root, quick):
    step_dir = out_root / f"{step['id']}-{step['name']}"
    step_dir.mkdir()
    cmd = step["cmd"](step_dir, quick)
    pct, waited = wait_until_idle()
    say(f"step {step['id']}: {step['what']}; PC at {pct:.0f}% CPU"
        + (f" after waiting {waited:.0f} s" if waited > IDLE_WINDOW_S + 5 else ""))

    log_path = step_dir / "log.txt"
    started = datetime.datetime.now()
    code, sampler = None, None
    t0 = time.perf_counter()
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            # default buffering, so a warning on stderr cannot split a printed result line
            proc = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
            sampler = CpuSampler(proc.pid)
            sampler.start()
            try:
                code = proc.wait(timeout=step["timeout"])
            except subprocess.TimeoutExpired:
                kill_tree(proc.pid)
                proc.wait()
            except KeyboardInterrupt:
                kill_tree(proc.pid)
                proc.wait()
                raise
            finally:
                sampler.stop_event.set()
                sampler.join()
    finally:
        wall = time.perf_counter() - t0
        for name in step["moves"]:   # keep the outputs with this step, and the folder clean for the next
            if (HERE / name).exists():
                shutil.move(str(HERE / name), str(step_dir / name))

    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics, checks = {}, []
    if code == 0:
        try:
            metrics, checks = step["parse"](text, step_dir)
        except Exception as e:
            checks = [("read the step's results", False, repr(e))]
    checks.insert(0, ("process exit code 0", code == 0, "timed out" if code is None else f"{code}"))
    cpu = sampler.summary()
    result = dict(id=step["id"], name=step["name"], what=step["what"], cmd=" ".join(map(str, cmd)),
                  started=started.isoformat(timespec="seconds"), wall_s=round(wall, 2),
                  key=step["key"](metrics), metrics=metrics,
                  checks=[dict(name=n, ok=bool(ok), detail=d) for n, ok, d in checks],
                  cpu=cpu, passed=all(ok for _, ok, _ in checks))

    say(f"step {step['id']}: {'passed' if result['passed'] else 'FAILED'} in {wall:.1f} s; {result['key']}; "
        f"others' CPU {fmt(cpu['others_mean'])} cores (max 10 s {fmt(cpu['others_max_10s'])})"
        + ("" if cpu["clean"] else " -- BUSY, timing not clean"))
    if cpu["busy_top"]:
        say(f"  others above {NAME_OTHERS_ABOVE} cores for {cpu['busy_seconds']} s; mean cores then: "
            + ", ".join(f"{n} {mean:.2f}" for n, mean, _ in cpu["busy_top"]))
    for c in result["checks"]:
        if not c["ok"]:
            say(f"  failed check: {c['name']}: {c['detail']} (see {log_path})")
    return result


# ---- run record -------------------------------------------------------------
def machine_info(battery):
    def version(*names):
        for n in names:
            try:
                return metadata.version(n)
            except metadata.PackageNotFoundError:
                pass
        return None

    def output(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
        except Exception as e:
            return f"n/a ({e})"

    return dict(
        cpu=platform.processor(), physical_cores=psutil.cpu_count(logical=False),
        logical_cpus=psutil.cpu_count(), memory_gb=round(psutil.virtual_memory().total / 2**30, 1),
        on_charger=None if battery is None else battery.power_plugged,
        battery_pct=None if battery is None else battery.percent,
        power_plan=output(["powercfg", "/getactivescheme"]),
        python=platform.python_version(),
        tensorflow=version("tensorflow", "tensorflow-intel", "tensorflow-cpu"),
        tf_agents=version("tf-agents"),
        energyplus=output([str(EPLUS_EXE), "--version"]),
        sbsim_commit=output(["git", "-C", str(SBSIM), "rev-parse", "--short", "HEAD"]),
        sbsim_modified_files=output(["git", "-C", str(SBSIM), "status", "--porcelain"]) or "none",
    )


def save(run, out_root):
    (out_root / "results.json").write_text(json.dumps(run, indent=2), encoding="utf-8")


def write_md(run, out_root):
    m = run["machine"]
    lines = [
        f"# Benchmark timings, {run['mode']} run",
        "",
        f"- Started {run['started']}, finished {run['finished']}. Status: **{run['status']}**",
        f"- {m['cpu']}, {m['physical_cores']} cores, {m['logical_cpus']} threads, {m['memory_gb']} GB. "
        f"On charger: {m['on_charger']} ({m['battery_pct']}%). {m['power_plan']}",
        f"- Python {m['python']}, TensorFlow {m['tensorflow']}, tf-agents {m['tf_agents']}. {m['energyplus']}. "
        f"SBSim commit {m['sbsim_commit']}, modified files: {m['sbsim_modified_files']}",
        f"- Each step started only when the PC was below {IDLE_MAX_PCT:.0f}% CPU. \"Others\" is the CPU used by "
        f"everything except the step itself (about 1 core is the normal background). A step is marked busy "
        f"if others average over {OTHERS_MEAN_MAX} cores or any 10 s stretch is over {OTHERS_10S_MAX} cores.",
        "",
        "| Step | What | Time | Process wall | Others' CPU, mean / max 10 s | Result |",
        "|---|---|---|---|---|---|",
    ]
    for s in run["steps"]:
        c = s["cpu"]
        cpu = f"{fmt(c['others_mean'])} / {fmt(c['others_max_10s'])} cores" + ("" if c["clean"] else " **busy**")
        bad = [f"{k['name']}: {k['detail']}" for k in s["checks"] if not k["ok"]]
        result = "passed" if s["passed"] else "FAILED: " + "; ".join(bad)
        lines.append(f"| {s['id']} | {s['what']} | {s['key']} | "
                     f"{s['wall_s']:.1f} s | {cpu} | {result} |")
    lines += [
        "",
        "- \"Time\" is what each script measures itself; \"Process wall\" also includes starting Python "
        "and loading libraries.",
    ]
    for s in run["steps"]:
        c = s["cpu"]
        if c.get("busy_top"):
            lines.append(f"- Step {s['id']}: others were above {NAME_OTHERS_ABOVE} cores for {c['busy_seconds']} s. "
                         "Mean cores in those seconds (highest in brackets): "
                         + ", ".join(f"{n} {mean:.2f} ({peak:.2f})" for n, mean, peak in c["busy_top"]))
    if run["mode"] == "quick":
        lines.append("- Quick mode: 7b and 7c ran short trainings to test this script. "
                     "Their numbers are not benchmark results.")
    (out_root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Rerun every benchmark timing under the same conditions.")
    ap.add_argument("--quick", action="store_true",
                    help="short trainings in 7b and 7c, to test this script; never creates the flag")
    args = ap.parse_args()

    if "sbsim-bench" not in PY:
        sys.exit(f"run this with the sbsim-bench Python, not {PY}")
    stale = [n for n in OUTPUT_FOLDERS if (HERE / n).exists()]
    if stale:
        sys.exit(f"old output folders exist; move or delete them first: {', '.join(stale)}")
    battery = psutil.sensors_battery()
    if battery is not None and battery.power_plugged is False and not args.quick:
        sys.exit("the laptop is on battery; plug in the charger before the full run")

    mode = "quick" if args.quick else "full"
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    out_root = HERE / "out-all" / (stamp + ("-quick" if args.quick else ""))
    out_root.mkdir(parents=True)
    run = dict(mode=mode, started=datetime.datetime.now().isoformat(timespec="seconds"), finished=None,
               status="running", machine=machine_info(battery), steps=[])
    save(run, out_root)
    say(f"{mode} run, {len(STEPS)} steps; results in {out_root}")

    try:
        for step in STEPS:
            result = run_step(step, out_root, args.quick)
            run["steps"].append(result)
            save(run, out_root)
            if not result["passed"]:
                run["status"] = f"failed at step {step['id']}"
                break
        else:
            run["status"] = "passed"
    except KeyboardInterrupt:
        run["status"] = "stopped by hand"
    except Exception as e:
        run["status"] = f"error: {e}"

    run["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    save(run, out_root)
    write_md(run, out_root)

    busy = [s["id"] for s in run["steps"] if not s["cpu"]["clean"]]
    if busy:
        say(f"other programs were busy during step(s) {', '.join(busy)}; those timings are not clean")
    if args.quick:
        say("quick run: no flag")
    elif run["status"] == "passed" and not busy:
        FLAG.touch()
        say(f"created {FLAG}")
    else:
        say("flag NOT created (the run failed or was not clean)")
    say(f"status: {run['status']}; see {out_root / 'results.md'}")
    sys.exit(0 if run["status"] == "passed" and not busy else 1)


if __name__ == "__main__":
    main()
