"""Training-progress plots from a run's episodes.csv and eval.csv (used by scripts/plot_training.py)."""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # no GUI backend, for headless servers
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

# Style rules shared by every training plot: colors in this fixed order,
# assigned per task (or per run) and never by rank, so a task keeps its color in every plot.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TASKS = ["squat", "high_knees", "quick_steps", "backward_walk", "walk", "jog", "high_knees_travel"]
DPI = 130
LINE_W = 2 * 72 / DPI  # matplotlib widths are in points (1/72 inch): this is 2 px at DPI


def read_csv(path: str, cols: list[str]) -> dict:
    """Read the given numeric columns of a CSV into {col: [float, ...]}.

    A missing file gives empty lists. A row with a missing or unreadable value is skipped:
    that is a half-written last line you get while training is still writing the file.
    """
    out = {c: [] for c in cols}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                values = [float(row[c]) for c in cols]
            except (TypeError, ValueError):
                continue
            for c, v in zip(cols, values):
                out[c].append(v)
    return out


def load_episodes(run_dir: str) -> dict:
    """step, return, mean_effort from <run_dir>/episodes.csv, as numpy arrays."""
    ep = read_csv(os.path.join(run_dir, "episodes.csv"), ["step", "return", "mean_effort"])
    return {k: np.array(v) for k, v in ep.items()}


def load_eval(run_dir: str) -> dict:
    """{task: {"step", "reduction", "fail"}} from <run_dir>/eval.csv, as numpy arrays.

    eval.csv has one row per task per evaluation; each task's rows stay in step order.
    Returns {} when there is no eval.csv yet (before the first evaluation).
    """
    path = os.path.join(run_dir, "eval.csv")
    if not os.path.exists(path):
        return {}
    by_task = defaultdict(lambda: {"step": [], "reduction": [], "fail": []})
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = float(row["step"])
                reduction = float(row["effort_reduction_pct"])
                fail = float(row["fail_rate"])
            except (TypeError, ValueError):
                continue
            t = by_task[row["task"]]
            t["step"].append(step)
            t["reduction"].append(reduction)
            t["fail"].append(fail)
    return {task: {k: np.array(v) for k, v in g.items()} for task, g in by_task.items()}


def load_run(run_dir: str) -> tuple[dict, dict]:
    """(episodes, eval) for one run folder; see load_episodes and load_eval for the dict contents."""
    return load_episodes(run_dir), load_eval(run_dir)


def smooth(step: np.ndarray, x: np.ndarray, n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Moving average of x over the last n episodes, paired with the step each window ends at.

    n shrinks to the run length for short runs; n <= 1 returns the data unchanged.
    """
    n = min(n, len(x))
    if n <= 1:
        return step, x
    # mode="valid" keeps only full windows: len(x) - n + 1 points, the first ending at step[n - 1]
    return step[n - 1:], np.convolve(x, np.ones(n) / n, mode="valid")


def best_metric(run_dir: str) -> str:
    """The run's train.best_metric from its config.yaml ("mean" if unset or unreadable)."""
    try:
        with open(os.path.join(run_dir, "config.yaml")) as f:
            return (yaml.safe_load(f) or {}).get("train", {}).get("best_metric", "mean")
    except OSError:
        return "mean"


def eval_scores(ev: dict, metric: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """(steps, checkpoint score) for every complete evaluation; empty arrays with no eval data.

    Same rule training uses to save best.pt (eval_score in exo_rl/evaluation.py):
    mean: mean effort reduction over tasks minus 100 x mean fail rate;
    worst: the same per task, worst task counts.
    """
    if not ev:
        return np.array([]), np.array([])
    n = min(len(g["step"]) for g in ev.values())  # an eval still being written has fewer tasks
    steps = next(iter(ev.values()))["step"][:n]
    if metric == "worst":
        return steps, np.min([g["reduction"][:n] - 100.0 * g["fail"][:n] for g in ev.values()], axis=0)
    reduction = np.mean([g["reduction"][:n] for g in ev.values()], axis=0)
    fail = np.mean([g["fail"][:n] for g in ev.values()], axis=0)
    return steps, reduction - 100.0 * fail


def best_step(ev: dict, metric: str = "mean") -> tuple[float, float] | None:
    """Eval step with the highest checkpoint score, and that score; None with no eval data."""
    steps, score = eval_scores(ev, metric)
    if len(score) == 0:
        return None
    i = int(np.argmax(score))
    return float(steps[i]), float(score[i])


def task_color(task: str) -> str:
    """The task's fixed color (configs/default.yaml order, then the extra motions)."""
    return COLORS[TASKS.index(task)] if task in TASKS else COLORS[-1]


def style_axis(ax, title: str, ylabel: str) -> None:
    """Shared look: title, y label with units, light gray grid behind the lines."""
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e3e3e3", linewidth=0.8)
    ax.set_axisbelow(True)  # grid behind lines, not over them


def run_name(run_dir: str) -> str:
    """Folder name of a run, e.g. "demo" for runs/demo/ (used in titles and legends)."""
    return os.path.basename(os.path.normpath(run_dir))


def draw_episode_curves(axes, ep: dict, n: int, color: str, label: str | None = None) -> None:
    """Smoothed total reward on axes[0] and smoothed human effort on axes[1], for one run."""
    for ax, key in zip(axes[:2], ("return", "mean_effort")):
        s, y = smooth(ep["step"], ep[key], n)
        ax.plot(s, y, color=color, linewidth=LINE_W, label=label)


def style_episode_axes(axes, n: int) -> None:
    """Titles and labels for the two training-episode graphs."""
    style_axis(axes[0], f"Total reward per training episode (moving average of {n} episodes)", "reward")
    style_axis(axes[1], f"Human effort in training (moving average of {n} episodes, lower is better)",
               "human effort")


def outside_legend(ax) -> None:
    """Legend just right of the graph, so it never covers a line."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)


def no_evals_note(ax) -> None:
    """Placeholder text for a run with no evaluations yet (before the first train.eval_every)."""
    ax.text(0.5, 0.5, "no evaluations yet", transform=ax.transAxes, ha="center", color="#555555")


def save_figure(fig, title: str, out: str) -> str:
    """Title, fit, save as a PNG at out (creating its folder), free the memory; returns out."""
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def plot_run(run_dir: str, out: str, n: int = 20) -> str:
    """Return, human effort and eval effort reduction per task for one run, saved as a PNG at out."""
    ep, ev = load_run(run_dir)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 9))

    draw_episode_curves(axes, ep, n, COLORS[0])
    style_episode_axes(axes, n)

    axes[2].axhline(0, color="#999999", linewidth=0.8)
    for task, g in ev.items():
        axes[2].plot(g["step"], g["reduction"], color=task_color(task), linewidth=LINE_W, marker="o",
                     markersize=4, label=task)
    metric = best_metric(run_dir)
    best = best_step(ev, metric)
    if best is not None:
        label = f"best.pt: step {best[0]:,.0f} ({metric} score {best[1]:.1f})"
        for ax in axes:
            ax.axvline(best[0], color="#555555", linestyle="--", linewidth=1, label=label)
    if ev:
        outside_legend(axes[2])
    else:
        no_evals_note(axes[2])
    style_axis(axes[2], "Evaluation: effort saved by the exo, per task", "effort reduction vs exo off (%)")
    axes[2].set_xlabel("training step")
    return save_figure(fig, f"Training progress: {run_name(run_dir)}", out)


def plot_compare(run_dirs: list[str], out: str, n: int = 20) -> str:
    """Several runs on the same three graphs, one fixed color per run (its position in run_dirs).

    The eval graph shows each run's checkpoint score instead of per-task lines (five tasks times
    several runs would be unreadable), with a star at the step its best.pt was saved.
    """
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 9))
    any_evals = False
    for i, run_dir in enumerate(run_dirs):
        ep, ev = load_run(run_dir)
        color, name = COLORS[i % len(COLORS)], run_name(run_dir)
        draw_episode_curves(axes, ep, n, color, label=name if len(ep["step"]) else f"{name} (no episodes yet)")
        metric = best_metric(run_dir)
        steps, score = eval_scores(ev, metric)
        if len(score) == 0:
            continue
        any_evals = True
        best = best_step(ev, metric)
        axes[2].plot(steps, score, color=color, linewidth=LINE_W, marker="o", markersize=4,
                     label=f"{name} (best.pt: step {best[0]:,.0f}" + (f", {metric})" if metric != "mean" else ")"))
        axes[2].plot(best[0], best[1], color=color, marker="*", markersize=14, linestyle="none")

    style_episode_axes(axes, n)
    outside_legend(axes[0])
    outside_legend(axes[1])
    axes[2].axhline(0, color="#999999", linewidth=0.8)
    if any_evals:
        outside_legend(axes[2])
    else:
        no_evals_note(axes[2])
    style_axis(axes[2], "Evaluation: checkpoint score (★ = best.pt)",
               "effort reduction − 100 × fail rate (%)")
    axes[2].set_xlabel("training step")
    return save_figure(fig, f"Training comparison: {', '.join(run_name(r) for r in run_dirs)}", out)


def plot_training(run_dirs: list[str], out: str | None = None, n: int = 20) -> str:
    """One run: plot_run, saved to <run>/training.png by default.
    Several runs: plot_compare, saved to runs/compare_training.png by default."""
    if len(run_dirs) == 1:
        return plot_run(run_dirs[0], out or os.path.join(run_dirs[0], "training.png"), n)
    return plot_compare(run_dirs, out or os.path.join("runs", "compare_training.png"), n)
