"""
Disk Scheduling Simulator
-------------------------
Simulates FCFS, SSTF, SCAN, C-SCAN, LOOK and C-LOOK on the same request
queue and compares them on:

  * total head movement (cylinders)
  * average seek distance per request
  * average / maximum waiting distance (head movement before a request is
    served, used as a proxy for response time; the maximum shows starvation)
  * standard deviation of waiting distance (lower = fairer)

It has two modes:

  1. Static: the whole queue is known up front (the textbook model).
  2. Dynamic: requests keep arriving (Poisson process) while the head moves
     one cylinder per time unit. This is where starvation becomes visible,
     because a greedy algorithm like SSTF can keep serving new nearby
     requests while a far-away request waits.

Usage:
  python disk_sched.py                       # textbook example + all workloads
  python disk_sched.py --head 53 --requests 98 183 37 122 14 124 65 67
  python disk_sched.py --direction down --disk-size 500
"""

import argparse
import csv
import os
import random
import statistics

import matplotlib
matplotlib.use("Agg")  # write PNG files, no window needed
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Algorithms
#
# Each algorithm returns a "path": a list of (cylinder, served_request_index)
# tuples. served_request_index is None for points the head passes through
# without serving anything (e.g. the disk edge in SCAN / C-SCAN).
# ---------------------------------------------------------------------------

def fcfs(requests, head, disk_size, direction):
    return [(head, None)] + [(r, i) for i, r in enumerate(requests)]


def sstf(requests, head, disk_size, direction):
    pending = list(enumerate(requests))
    path = [(head, None)]
    pos = head
    while pending:
        # closest request; ties go to the one that arrived first
        i, r = min(pending, key=lambda p: (abs(p[1] - pos), p[0]))
        pending.remove((i, r))
        path.append((r, i))
        pos = r
    return path


def _split(requests, head):
    """Requests at/above the head (ascending) and below it (ascending)."""
    indexed = list(enumerate(requests))
    upper = sorted([p for p in indexed if p[1] >= head], key=lambda p: (p[1], p[0]))
    lower = sorted([p for p in indexed if p[1] < head], key=lambda p: (p[1], p[0]))
    return upper, lower


def _swap(p):
    return (p[1], p[0])


def scan_up(requests, head, disk_size):
    upper, lower = _split(requests, head)
    path = [(head, None)] + [_swap(p) for p in upper]
    if lower:
        path.append((disk_size - 1, None))            # go all the way to the end
        path += [_swap(p) for p in reversed(lower)]   # then sweep back down
    return path


def cscan_up(requests, head, disk_size):
    upper, lower = _split(requests, head)
    path = [(head, None)] + [_swap(p) for p in upper]
    if lower:
        path.append((disk_size - 1, None))            # go to the end
        path.append((0, None))                        # jump back to cylinder 0
        path += [_swap(p) for p in lower]             # sweep up again
    return path


def look_up(requests, head, disk_size):
    upper, lower = _split(requests, head)
    path = [(head, None)] + [_swap(p) for p in upper]
    path += [_swap(p) for p in reversed(lower)]       # turn at last request
    return path


def clook_up(requests, head, disk_size):
    upper, lower = _split(requests, head)
    path = [(head, None)] + [_swap(p) for p in upper]
    path += [_swap(p) for p in lower]                 # jump to lowest request
    return path


def _directional(up_fn):
    """Wrap an 'upward' algorithm so it can also start moving downward.

    Moving down on a disk is the same as moving up on a mirrored disk, so we
    mirror the cylinders, run the upward version, and mirror the result back.
    """
    def run(requests, head, disk_size, direction):
        if direction == "up":
            return up_fn(requests, head, disk_size)
        mirror = lambda c: disk_size - 1 - c
        path = up_fn([mirror(r) for r in requests], mirror(head), disk_size)
        return [(mirror(c), i) for c, i in path]
    return run


ALGORITHMS = {
    "FCFS": fcfs,
    "SSTF": sstf,
    "SCAN": _directional(scan_up),
    "C-SCAN": _directional(cscan_up),
    "LOOK": _directional(look_up),
    "C-LOOK": _directional(clook_up),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(path, n_requests):
    total = 0
    wait = [0] * n_requests
    for (prev, _), (cur, idx) in zip(path, path[1:]):
        total += abs(cur - prev)
        if idx is not None:
            wait[idx] = total
    return {
        "total": total,
        "avg_seek": total / n_requests,
        "avg_wait": statistics.mean(wait),
        "max_wait": max(wait),
        "std_wait": statistics.pstdev(wait),
        "order": [c for c, i in path if i is not None],
    }


def run_all(requests, head, disk_size, direction):
    results = {}
    for name, fn in ALGORITHMS.items():
        path = fn(requests, head, disk_size, direction)
        results[name] = {"path": path, **evaluate(path, len(requests))}
    return results


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

def make_workloads(disk_size, seed=42):
    rng = random.Random(seed)

    def clamp(x):
        return max(0, min(disk_size - 1, int(x)))

    random_wl = [rng.randrange(disk_size) for _ in range(50)]

    # a few "hot" regions, e.g. a database file and its index
    hotspots = [rng.randrange(disk_size) for _ in range(3)]
    clustered = [clamp(rng.gauss(rng.choice(hotspots), disk_size * 0.03)) for _ in range(50)]

    # mostly sequential reads with occasional random jumps
    sequential, pos = [], rng.randrange(disk_size // 2)
    for _ in range(50):
        pos = rng.randrange(disk_size) if rng.random() < 0.1 else clamp(pos + rng.randint(1, 5))
        sequential.append(pos)

    heavy = [rng.randrange(disk_size) for _ in range(500)]

    head = rng.randrange(disk_size)
    return head, {
        "Random (50)": random_wl,
        "Clustered (50)": clustered,
        "Sequential (50)": sequential,
        "Heavy load (500)": heavy,
    }


# ---------------------------------------------------------------------------
# Dynamic simulation (requests arrive over time)
#
# Time model: the head moves one cylinder per time unit, and serving a
# request at the current cylinder takes no extra time. At every time step:
#   1. requests whose arrival time has passed join the pending queue
#   2. requests at the head's cylinder are served
#   3. the algorithm picks a target cylinder and the head moves one step
# Response time of a request = time it was served - time it arrived.
# ---------------------------------------------------------------------------

def simulate_dynamic(algo, arrivals, head, disk_size, direction):
    """arrivals: list of (arrival_time, cylinder), sorted by time."""
    d = 1 if direction == "up" else -1
    edge = lambda d: disk_size - 1 if d == 1 else 0

    pending = []           # (arrival_time, cylinder, request_id), arrival order
    response = [0.0] * len(arrivals)
    t, nxt, moved = 0.0, 0, 0
    returning = None       # cylinder C-SCAN / C-LOOK is jumping back to
    trace = [(t, head)]

    def serve(p):
        pending.remove(p)
        response[p[2]] = t - p[0]

    while nxt < len(arrivals) or pending or returning is not None:
        while nxt < len(arrivals) and arrivals[nxt][0] <= t:
            pending.append((arrivals[nxt][0], arrivals[nxt][1], nxt))
            nxt += 1

        # serve requests at the head (not while C-SCAN/C-LOOK is returning)
        if returning is None:
            if algo == "FCFS":
                while pending and pending[0][1] == head:
                    serve(pending[0])
            else:
                for p in [p for p in pending if p[1] == head]:
                    serve(p)

        if not pending and returning is None:
            if nxt < len(arrivals):
                t = max(t, arrivals[nxt][0])   # idle until next arrival
            continue

        # choose where to move next
        def ahead(dir_):
            return [p for p in pending if (p[1] - head) * dir_ > 0]

        def nearest(ps):
            return min(ps, key=lambda p: (abs(p[1] - head), p[0]))[1]

        if returning is not None:
            target = returning
        elif algo == "FCFS":
            target = pending[0][1]
        elif algo == "SSTF":
            target = nearest(pending)
        elif algo == "LOOK":
            if not ahead(d):
                d = -d                          # reverse at the last request
            target = nearest(ahead(d))
        elif algo == "SCAN":
            if ahead(d):
                target = nearest(ahead(d))
            elif head != edge(d):
                target = edge(d)                # keep going to the disk edge
            else:
                d = -d
                target = nearest(ahead(d))
        elif algo == "C-SCAN":
            if ahead(d):
                target = nearest(ahead(d))
            elif head != edge(d):
                target = edge(d)
            else:
                returning = target = edge(-d)   # jump to the other edge
        elif algo == "C-LOOK":
            if ahead(d):
                target = nearest(ahead(d))
            else:                               # jump to the farthest request
                far = min if d == 1 else max
                returning = target = far(p[1] for p in pending)

        head += 1 if target > head else -1
        moved += 1
        t += 1
        trace.append((t, head))
        if head == returning:
            returning = None

    return {
        "total": moved,
        "avg_resp": statistics.mean(response),
        "p95_resp": sorted(response)[int(0.95 * (len(response) - 1))],
        "max_resp": max(response),
        "std_resp": statistics.pstdev(response),
        "throughput": len(arrivals) / t,
        "trace": trace,
        "response": response,
    }


def make_arrivals(n, rate, cylinder_fn, rng):
    """n requests with exponential gaps (Poisson arrivals at `rate` per time unit)."""
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append((t, cylinder_fn()))
    return out


def dynamic_workloads(disk_size, rate, seed=42, n=300):
    rng = random.Random(seed)
    hot = int(disk_size * 0.3)

    def uniform():
        return rng.randrange(disk_size)

    def hotspot():
        # 85% of requests hit one busy region (e.g. a database table),
        # the other 15% are scattered across the disk
        if rng.random() < 0.85:
            return max(0, min(disk_size - 1, int(rng.gauss(hot, disk_size * 0.04))))
        return rng.randrange(disk_size)

    return {
        "Uniform": make_arrivals(n, rate, uniform, rng),
        "Hotspot": make_arrivals(n, rate, hotspot, rng),
    }, hotspot


def run_all_dynamic(arrivals, head, disk_size, direction):
    return {name: simulate_dynamic(name, arrivals, head, disk_size, direction)
            for name in ALGORITHMS}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(title, results):
    print(f"\n=== {title} ===")
    print(f"{'Algorithm':<8} {'Total':>8} {'AvgSeek':>9} {'AvgWait':>9} {'MaxWait':>9} {'StdWait':>9}")
    best = min(r["total"] for r in results.values())
    for name, r in results.items():
        mark = "  <- lowest" if r["total"] == best else ""
        print(f"{name:<8} {r['total']:>8} {r['avg_seek']:>9.1f} {r['avg_wait']:>9.1f} "
              f"{r['max_wait']:>9} {r['std_wait']:>9.1f}{mark}")


def plot_paths(results, head, disk_size, filename, title):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, (name, r) in zip(axes.flat, results.items()):
        xs = [c for c, _ in r["path"]]
        ys = list(range(len(xs)))
        ax.plot(xs, ys, "-", color="tab:blue")
        served = [(c, y) for (c, i), y in zip(r["path"], ys) if i is not None]
        ax.plot([c for c, _ in served], [y for _, y in served], "o", color="tab:blue", ms=4)
        ax.plot(head, 0, "s", color="tab:red", ms=6)
        ax.set_title(f"{name}  (total = {r['total']})")
        ax.set_xlim(-5, disk_size + 4)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.supxlabel("Cylinder")
    fig.supylabel("Step")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_comparison(all_results, metric, ylabel, filename):
    workloads = list(all_results)
    algos = list(ALGORITHMS)
    width = 0.8 / len(algos)
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, algo in enumerate(algos):
        values = [all_results[w][algo][metric] for w in workloads]
        xs = [i + k * width for i in range(len(workloads))]
        ax.bar(xs, values, width, label=algo)
    ax.set_xticks([i + width * (len(algos) - 1) / 2 for i in range(len(workloads))])
    ax.set_xticklabels(workloads)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=6, loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def print_dynamic_table(title, results):
    print(f"\n=== {title} ===")
    print(f"{'Algorithm':<8} {'Movement':>9} {'AvgResp':>9} {'P95Resp':>9} {'MaxResp':>9} "
          f"{'StdResp':>9} {'Thruput':>8}")
    for name, r in results.items():
        print(f"{name:<8} {r['total']:>9} {r['avg_resp']:>9.1f} {r['p95_resp']:>9.1f} "
              f"{r['max_resp']:>9.1f} {r['std_resp']:>9.1f} {r['throughput']:>8.3f}")


def plot_dynamic_bars(dyn_results, filename):
    """Average vs maximum response time for each dynamic workload."""
    algos = list(ALGORITHMS)
    fig, axes = plt.subplots(1, len(dyn_results), figsize=(13, 4.5))
    for ax, (wl, results) in zip(axes, dyn_results.items()):
        xs = range(len(algos))
        ax.bar([x - 0.2 for x in xs], [results[a]["avg_resp"] for a in algos], 0.4, label="Average")
        ax.bar([x + 0.2 for x in xs], [results[a]["max_resp"] for a in algos], 0.4, label="Maximum")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(algos)
        ax.set_title(f"{wl} workload")
        ax.set_ylabel("Response time (time units)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_load_sweep(sweep, metric, ylabel, filename):
    """sweep: {rate: {algo: result}}"""
    rates = sorted(sweep)
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ALGORITHMS:
        ax.plot(rates, [sweep[r][algo][metric] for r in rates], "o-", label=algo)
    ax.set_xlabel("Arrival rate (requests per time unit)")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_starvation(results, arrivals, disk_size, filename, algos=("SSTF", "LOOK")):
    """Head position over time, with the longest-waiting request marked."""
    fig, axes = plt.subplots(1, len(algos), figsize=(13, 5), sharey=True)
    for ax, algo in zip(axes, algos):
        r = results[algo]
        ax.plot([t for t, _ in r["trace"]], [c for _, c in r["trace"]], lw=0.8)
        worst = max(range(len(arrivals)), key=lambda i: r["response"][i])
        arr_t, cyl = arrivals[worst]
        ax.plot([arr_t, arr_t + r["response"][worst]], [cyl, cyl], "r-", lw=3,
                label=f"longest wait: {r['response'][worst]:.0f} units")
        ax.set_title(algo)
        ax.set_xlabel("Time")
        ax.set_ylim(0, disk_size - 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")
    axes[0].set_ylabel("Head position (cylinder)")
    fig.suptitle("Hotspot workload: head position over time")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def save_dynamic_csv(dyn_results, sweep, filename):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "arrival_rate", "algorithm", "head_movement", "avg_resp",
                    "p95_resp", "max_resp", "std_resp", "throughput"])
        rows = [(wl, None, res) for wl, res in dyn_results.items()]
        rows += [("Hotspot sweep", rate, res) for rate, res in sorted(sweep.items())]
        for wl, rate, results in rows:
            for algo, r in results.items():
                w.writerow([wl, rate, algo, r["total"], round(r["avg_resp"], 2),
                            round(r["p95_resp"], 2), round(r["max_resp"], 2),
                            round(r["std_resp"], 2), round(r["throughput"], 4)])


def save_csv(all_results, filename):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "algorithm", "total_head_movement", "avg_seek",
                    "avg_wait", "max_wait", "std_wait"])
        for wl, results in all_results.items():
            for algo, r in results.items():
                w.writerow([wl, algo, r["total"], round(r["avg_seek"], 2),
                            round(r["avg_wait"], 2), r["max_wait"], round(r["std_wait"], 2)])


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Disk scheduling simulator")
    ap.add_argument("--head", type=int, default=53, help="initial head position")
    ap.add_argument("--requests", type=int, nargs="+",
                    default=[98, 183, 37, 122, 14, 124, 65, 67],
                    help="request queue (cylinder numbers)")
    ap.add_argument("--disk-size", type=int, default=200, help="number of cylinders")
    ap.add_argument("--direction", choices=["up", "down"], default="up",
                    help="initial head direction for SCAN/C-SCAN/LOOK/C-LOOK")
    ap.add_argument("--out", default="output", help="folder for charts and CSV")
    ap.add_argument("--seed", type=int, default=42, help="random seed for workloads")
    ap.add_argument("--rate", type=float, default=0.2,
                    help="arrival rate for the dynamic simulation (requests per time unit)")
    args = ap.parse_args()

    for r in args.requests + [args.head]:
        if not 0 <= r < args.disk_size:
            ap.error(f"cylinder {r} is outside 0..{args.disk_size - 1}")

    os.makedirs(args.out, exist_ok=True)

    # 1. Single queue (defaults to the classic textbook example)
    print(f"Head = {args.head}, direction = {args.direction}, cylinders = 0..{args.disk_size - 1}")
    print(f"Queue = {args.requests}")
    example = run_all(args.requests, args.head, args.disk_size, args.direction)
    print_table("Given request queue", example)
    for name, r in example.items():
        print(f"  {name:<7} order: {r['order']}")
    plot_paths(example, args.head, args.disk_size,
               os.path.join(args.out, "head_movement_example.png"),
               f"Head movement - head at {args.head}, moving {args.direction}")

    # 2. Synthetic workloads
    wl_head, workloads = make_workloads(args.disk_size, args.seed)
    all_results = {}
    for wl_name, reqs in workloads.items():
        all_results[wl_name] = run_all(reqs, wl_head, args.disk_size, args.direction)
        print_table(f"{wl_name} workload, head = {wl_head}", all_results[wl_name])

    plot_comparison(all_results, "total", "Total head movement (cylinders)",
                    os.path.join(args.out, "compare_total_movement.png"))
    plot_comparison(all_results, "max_wait", "Max waiting distance (starvation)",
                    os.path.join(args.out, "compare_max_wait.png"))
    plot_comparison(all_results, "std_wait", "Std. dev. of waiting distance (fairness)",
                    os.path.join(args.out, "compare_fairness.png"))
    save_csv(all_results, os.path.join(args.out, "results.csv"))

    # 3. Dynamic simulation: requests arrive while the head is moving
    dyn_head = args.disk_size // 2
    dyn_workloads, _ = dynamic_workloads(args.disk_size, args.rate, args.seed)
    dyn_results = {}
    for wl_name, arrivals in dyn_workloads.items():
        dyn_results[wl_name] = run_all_dynamic(arrivals, dyn_head, args.disk_size, args.direction)
        print_dynamic_table(f"Dynamic {wl_name} workload, 300 requests, "
                            f"rate = {args.rate}/time unit", dyn_results[wl_name])

    # same hotspot pattern at increasing load
    sweep = {}
    for rate in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        wls, _ = dynamic_workloads(args.disk_size, rate, args.seed)
        sweep[rate] = run_all_dynamic(wls["Hotspot"], dyn_head, args.disk_size, args.direction)

    plot_dynamic_bars(dyn_results, os.path.join(args.out, "dynamic_response_times.png"))
    plot_load_sweep(sweep, "max_resp", "Maximum response time (log scale)",
                    os.path.join(args.out, "load_sweep_max_response.png"))
    plot_load_sweep(sweep, "avg_resp", "Average response time (log scale)",
                    os.path.join(args.out, "load_sweep_avg_response.png"))
    plot_starvation(dyn_results["Hotspot"], dyn_workloads["Hotspot"], args.disk_size,
                    os.path.join(args.out, "starvation_sstf_vs_look.png"))
    save_dynamic_csv(dyn_results, sweep, os.path.join(args.out, "dynamic_results.csv"))
    print(f"\nCharts and CSV files written to '{args.out}/'")


if __name__ == "__main__":
    main()
