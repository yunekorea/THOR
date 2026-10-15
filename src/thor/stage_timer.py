"""
thor/stage_timer.py -- per-layer / per-stage timing that is safe with several streams.

Drop-in for thor.timer.Timer: forward_layer() and the weight-loading blocks in
he.py only use print_legend(), stage(), layer(), setup() and paused(), and this
class provides all of them. The difference is that every piece of state is kept
per thread, so N inference streams sharing one HE object each get their own,
correct numbers instead of corrupting a shared pause/resume counter.

For every stage of every layer it records:

    seconds    wall time of the stage
    weights    time inside `timer.paused()` blocks -- in he.py those blocks do
               nothing but read this stage's light plaintexts from disk
    bs_n/bs    number of offloaded bootstraps and their total round-trip time
    wait       time spent waiting for ANOTHER stream: engine-lock contention in
               the default mode, GPU-token waits with --gpu-token. Waiting is not
               a separate slice of time -- it happens inside the other columns --
               so it is reported alongside them, not added to them.
    other      seconds - weights - bs: host-side GPU work (plus any waiting that
               happened outside weight loading and bootstraps)

Live, one line is printed per stage and per layer. At the end, report() prints a
per-stage table (averaged over layers) and a per-layer table, and write_csv()
dumps every row for later analysis.
"""

import csv
import sys
import threading
import time
from contextlib import contextmanager

LAYER_STAGES = 16
HEAD = "head"          # layer label for pooler (17) and classifier (18)


def _fmt(seconds):
    if seconds >= 60:
        m = int(seconds // 60)
        return f"{m:2d}m{seconds - 60 * m:05.2f}s"
    return f"{seconds:7.2f}s"


class StageTimer:
    def __init__(self, live=True, out=None):
        self.live = live
        self.out = out or sys.stdout
        self._t0 = time.perf_counter()
        self._tls = threading.local()
        self._rows_lock = threading.Lock()
        self._print_lock = threading.Lock()
        self.rows = []            # one dict per (stream, layer, stage)
        self.layer_rows = []      # one dict per (stream, layer)
        self.setup_seconds = None
        self._legend_printed = False

    # ------------------------------------------------------------------ helpers
    def _state(self):
        s = self._tls
        if not hasattr(s, "frames"):
            s.frames = []          # stack of open stages
            s.layer = None
            s.layer_t0 = None
            s.layer_acc = None     # totals for the open layer
            s.paused_total = 0.0
        return s

    @staticmethod
    def _stream():
        name = threading.current_thread().name
        return "main" if name == "MainThread" else name

    def _emit(self, text):
        if not self.live:
            return
        with self._print_lock:
            self.out.write(text + "\n")
            self.out.flush()

    def _frame(self):
        frames = self._state().frames
        return frames[-1] if frames else None

    # ------------------------------------------- Timer-compatible surface
    @property
    def elapsed(self):
        return _fmt(time.perf_counter() - self._t0)

    @property
    def true_elapsed(self):
        return _fmt(time.perf_counter() - self._t0 - self._state().paused_total)

    def reset(self):
        self._t0 = time.perf_counter()
        self._state().paused_total = 0.0

    def pause(self):
        self._state().pause_t0 = time.perf_counter()

    def resume(self):
        s = self._state()
        self._add_paused(time.perf_counter() - s.pause_t0)

    def print_legend(self):
        # forward_layer calls this at the top of every layer; one header per run
        # is enough, and with several streams one per layer would be noise.
        if self._legend_printed or not self.live:
            return
        self._legend_printed = True
        self._emit("  stream layer stage name                      stage time |"
                   "    other  weights  bootstraps       wait")

    def _add_paused(self, dt):
        s = self._state()
        s.paused_total += dt
        for acc in (self._frame(), s.layer_acc):
            if acc is not None:
                acc["weights"] += dt

    @contextmanager
    def paused(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._add_paused(time.perf_counter() - t0)

    @contextmanager
    def setup(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.setup_seconds = time.perf_counter() - t0
            self._emit(f"Setup: {_fmt(self.setup_seconds).strip()}")
            self.reset()

    @contextmanager
    def layer(self, layer_index):
        s = self._state()
        prev = (s.layer, s.layer_t0, s.layer_acc)
        s.layer, s.layer_t0 = layer_index, time.perf_counter()
        s.layer_acc = dict(weights=0.0, bs_n=0, bs=0.0, wait=0.0)
        try:
            yield
        finally:
            secs = time.perf_counter() - s.layer_t0
            acc = s.layer_acc
            row = dict(stream=self._stream(), layer=layer_index,
                       seconds=round(secs, 4), weights=round(acc["weights"], 4),
                       bs_n=acc["bs_n"], bs=round(acc["bs"], 4),
                       wait=round(acc["wait"], 4),
                       other=round(secs - acc["weights"] - acc["bs"], 4))
            with self._rows_lock:
                self.layer_rows.append(row)
            self._emit(f"  [{row['stream']}] LAYER {layer_index:>2} done in "
                       f"{_fmt(secs).strip()}  (other {row['other']:.1f}s, "
                       f"weights {row['weights']:.1f}s, {row['bs_n']} bootstraps "
                       f"{row['bs']:.1f}s, waited {row['wait']:.1f}s)")
            s.layer, s.layer_t0, s.layer_acc = prev

    @contextmanager
    def stage(self, stage_index, stage_name):
        s = self._state()
        frame = dict(index=stage_index, name=stage_name, t0=time.perf_counter(),
                     weights=0.0, bs_n=0, bs=0.0, wait=0.0)
        s.frames.append(frame)
        try:
            yield
        finally:
            s.frames.pop()
            secs = time.perf_counter() - frame["t0"]
            # stages 1-16 belong to the open layer; 17 (pooler) and 18
            # (classifier) run once per inference, after the last layer
            layer = s.layer if stage_index <= LAYER_STAGES else HEAD
            row = dict(stream=self._stream(), layer=layer,
                       stage=stage_index, name=stage_name,
                       seconds=round(secs, 4),
                       weights=round(frame["weights"], 4),
                       bs_n=frame["bs_n"], bs=round(frame["bs"], 4),
                       wait=round(frame["wait"], 4),
                       other=round(secs - frame["weights"] - frame["bs"], 4))
            with self._rows_lock:
                self.rows.append(row)
            # A stage nested inside another stage hands its totals upward so the
            # outer stage's breakdown still adds up.
            parent = self._frame()
            if parent is not None:
                for k in ("weights", "bs_n", "bs", "wait"):
                    parent[k] += frame[k]
            lay = "  -" if row["layer"] in (None, HEAD) else f"{row['layer']:>3}"
            bs = (f"{row['bs_n']:>3} x {row['bs'] / row['bs_n']:5.2f}s"
                  if row["bs_n"] else "          -")
            self._emit(f"  [{row['stream']:>4}] {lay}   {stage_index:>2}  "
                       f"{stage_name:<24} {_fmt(secs)} | {row['other']:7.2f}s "
                       f"{row['weights']:7.2f}s  {bs}  {row['wait']:7.2f}s")

    # ------------------------------------------------ hooks for HENDP / locks
    def note_bootstrap(self, seconds, wait=0.0):
        """Called by HENDP / HENDPParallel after each offloaded bootstrap."""
        s = self._state()
        for acc in (self._frame(), s.layer_acc):
            if acc is not None:
                acc["bs_n"] += 1
                acc["bs"] += seconds
                acc["wait"] += wait

    def note_wait(self, seconds):
        """Time this stream spent blocked by another stream (lock or token)."""
        if seconds <= 0:
            return
        s = self._state()
        for acc in (self._frame(), s.layer_acc):
            if acc is not None:
                acc["wait"] += seconds

    # ------------------------------------------------------------- reporting
    def _streams(self):
        return sorted({r["stream"] for r in self.rows + self.layer_rows})

    def stage_summary(self):
        """Per stage, averaged over every (stream, layer) it ran in."""
        groups = {}
        for r in self.rows:
            key = (r["stage"], r["name"])
            g = groups.setdefault(key, dict(n=0, seconds=0.0, other=0.0,
                                            weights=0.0, bs_n=0, bs=0.0, wait=0.0))
            g["n"] += 1
            for k in ("seconds", "other", "weights", "bs", "wait"):
                g[k] += r[k]
            g["bs_n"] += r["bs_n"]
        grand = sum(g["seconds"] for g in groups.values()) or 1.0
        out = []
        for (idx, name), g in sorted(groups.items()):
            n = g["n"]
            out.append(dict(stage=idx, name=name, runs=n,
                            mean_seconds=round(g["seconds"] / n, 3),
                            total_seconds=round(g["seconds"], 3),
                            share=round(100 * g["seconds"] / grand, 2),
                            mean_other=round(g["other"] / n, 3),
                            mean_weights=round(g["weights"] / n, 3),
                            mean_bs_n=round(g["bs_n"] / n, 2),
                            mean_bs=round(g["bs"] / n, 3),
                            mean_wait=round(g["wait"] / n, 3)))
        return out

    def layer_summary(self):
        """{stream: {layer: seconds}}, including 'head' for pooler+classifier."""
        table = {}
        for r in self.layer_rows:
            table.setdefault(r["stream"], {})[r["layer"]] = r["seconds"]
        for r in self.rows:
            if r["layer"] == HEAD:
                t = table.setdefault(r["stream"], {})
                t[HEAD] = round(t.get(HEAD, 0.0) + r["seconds"], 4)
        return table

    def report(self, out=None):
        out = out or self.out
        w = out.write
        rows = self.stage_summary()
        if not rows:
            return
        streams = self._streams()
        w("\n================ per-stage time (mean per layer, all streams) "
          "================\n")
        w(f"{'stage':>5}  {'name':<24} {'runs':>4} {'mean':>9} {'total':>10} "
          f"{'share':>6} | {'other':>8} {'weights':>8} {'bs/run':>6} "
          f"{'bs time':>8} {'wait':>8}\n")
        for r in rows:
            w(f"{r['stage']:>5}  {r['name']:<24} {r['runs']:>4} "
              f"{r['mean_seconds']:8.2f}s {r['total_seconds']:9.1f}s "
              f"{r['share']:5.1f}% | {r['mean_other']:7.2f}s "
              f"{r['mean_weights']:7.2f}s {r['mean_bs_n']:6.1f} "
              f"{r['mean_bs']:7.2f}s {r['mean_wait']:7.2f}s\n")
        tot = {k: sum(r[k] * r["runs"] for r in rows)
               for k in ("mean_other", "mean_weights", "mean_bs", "mean_wait")}
        all_s = sum(r["total_seconds"] for r in rows) or 1.0
        w(f"{'':>5}  {'all stages':<24} {'':>4} {'':>9} {all_s:9.1f}s "
          f"{100.0:5.1f}% | {tot['mean_other']:7.1f}s {tot['mean_weights']:7.1f}s "
          f"{'':>6} {tot['mean_bs']:7.1f}s {tot['mean_wait']:7.1f}s\n")
        w(f"  breakdown of all stage time: other {100*tot['mean_other']/all_s:.1f}%, "
          f"weights {100*tot['mean_weights']/all_s:.1f}%, "
          f"bootstraps {100*tot['mean_bs']/all_s:.1f}%  "
          f"(waiting for other streams, included in those: "
          f"{100*tot['mean_wait']/all_s:.1f}%)\n")

        table = self.layer_summary()
        if table:
            w("\n================ per-layer time "
              "================\n")
            w(f"{'layer':>6}" + "".join(f"{s:>12}" for s in streams) + "\n")
            layers = sorted({k for t in table.values() for k in t
                             if k != HEAD}) + [HEAD]
            for lay in layers:
                if all(lay not in table.get(s, {}) for s in streams):
                    continue
                w(f"{lay!s:>6}" + "".join(
                    f"{table.get(s, {}).get(lay, float('nan')):11.1f}s"
                    for s in streams) + "\n")
        out.flush()

    def write_csv(self, path):
        fields = ["stream", "layer", "stage", "name", "seconds", "other",
                  "weights", "bs_n", "bs", "wait"]
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            with self._rows_lock:
                for r in self.rows:
                    wr.writerow({k: r.get(k) for k in fields})
        return path

    def as_dict(self):
        return dict(setup_seconds=(round(self.setup_seconds, 3)
                                   if self.setup_seconds is not None else None),
                    stages=self.stage_summary(),
                    layers=self.layer_summary())
