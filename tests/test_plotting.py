"""Training plots from fake episodes.csv / eval.csv files: loading, best step, and the saved PNG."""
from exo_rl.plotting import best_metric, best_step, load_run, plot_training


def make_run(path, evals=True):
    """A small fake run folder: 30 episodes, and (optionally) 2 evaluations of 2 tasks."""
    path.mkdir()
    rows = [f"{600 * (i + 1)},{500 + i},{0.4 - 0.001 * i}" for i in range(30)]
    (path / "episodes.csv").write_text("step,return,mean_effort\n" + "\n".join(rows) + "\n18600,51")  # half-written
    if evals:
        (path / "eval.csv").write_text("step,task,effort_reduction_pct,fail_rate\n"
                                       "10000,squat,20.0,0.0\n10000,walk,0.0,0.0\n"      # score 10
                                       "20000,squat,30.0,0.0\n20000,walk,10.0,0.1\n")    # score 20 - 5 = 15
    return str(path)


def test_load_run_skips_half_written_row_and_finds_best_step(tmp_path):
    ep, ev = load_run(make_run(tmp_path / "a"))
    assert len(ep["step"]) == 30
    assert sorted(ev) == ["squat", "walk"]
    assert best_step(ev) == (20000.0, 15.0)
    assert load_run(make_run(tmp_path / "b", evals=False))[1] == {}  # no eval.csv


def test_best_step_follows_the_runs_best_metric(tmp_path):
    run = make_run(tmp_path / "a")
    assert best_metric(run) == "mean"                                # no config.yaml
    (tmp_path / "a" / "config.yaml").write_text("train:\n  best_metric: worst\n")
    _, ev = load_run(run)
    assert best_metric(run) == "worst"
    assert best_step(ev, "worst") == (10000.0, 0.0)  # worst task 0 at both evals; a tie keeps the first


def test_plot_training_saves_png_for_one_and_several_runs(tmp_path):
    a, b = make_run(tmp_path / "a"), make_run(tmp_path / "b", evals=False)
    out = plot_training([a], str(tmp_path / "one.png"))
    assert out.endswith("one.png") and (tmp_path / "one.png").stat().st_size > 0
    plot_training([a, b], str(tmp_path / "cmp.png"), n=5)
    assert (tmp_path / "cmp.png").stat().st_size > 0
