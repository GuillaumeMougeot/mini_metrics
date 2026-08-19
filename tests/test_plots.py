import pytest

from mini_metrics.plots import cli, get_df_names, normalize_path


def test_normalize_path():
    path_win = r"C:\Users\test\metrics.csv"
    norm_win = normalize_path(path_win)
    assert "\\" not in norm_win
    assert norm_win.endswith("/metrics.csv")

    path_posix = "/tmp/test/metrics.csv"
    norm_posix = normalize_path(path_posix)
    assert norm_posix.startswith("/")
    assert "\\" not in norm_posix


def test_get_df_names_single_file():
    path = "/tmp/dir/my_metrics.csv"
    names = get_df_names([path])
    assert len(names) == 1
    expected_norm = normalize_path(path)
    if len(expected_norm) > 20:
        assert names[0] == expected_norm[:17] + "..."
    else:
        assert names[0] == expected_norm


def test_get_df_names_multiple_common_prefix_suffix():
    paths = [
        "/tmp/experiments/run_2026_model_1_metrics.csv",
        "/tmp/experiments/run_2026_model_2_metrics.csv",
    ]
    names = get_df_names(paths)
    # Common prefix: "/tmp/experiments/run_2026_model_"
    # Common suffix: "_metrics.csv"
    # Stripped names: "1" and "2"
    assert names == ["1", "2"]


def test_get_df_names_truncation():
    paths = [
        "/tmp/exp/model_very_long_descriptive_name_1_result.csv",
        "/tmp/exp/model_very_long_descriptive_name_2_result.csv",
    ]
    names = get_df_names(paths)
    assert names == ["1", "2"]

    # Test names where stripped portion > 20 chars
    paths_long = [
        "/tmp/exp/groupA_different_suffix_here_1.csv",
        "/tmp/exp/groupB_different_suffix_here_2.csv",
    ]
    names_long = get_df_names(paths_long)
    # Common prefix: /tmp/exp/group
    # Common suffix: .csv
    # Stripped 1: "A_different_suffix_here_1" (26 chars > 20)
    # Expected 1: "A_different_suffi..." (17 chars + "...")
    assert len(names_long[0]) == 20
    assert names_long[0].endswith("...")
    assert names_long[0] == "A_different_suffi..."


def test_get_df_names_regex():
    paths = [
        "/project/runs/v1_alpha/eval_metrics.csv",
        "/project/runs/v2_beta/eval_metrics.csv",
    ]
    pattern = r"/runs/([^/]+)/"
    names = get_df_names(paths, pattern=pattern)
    assert names == ["v1_alpha", "v2_beta"]


def test_get_df_names_regex_invalid():
    paths = ["/project/runs/v1/metrics.csv"]
    # No capture group
    with pytest.raises(ValueError, match="no capture group"):
        get_df_names(paths, pattern=r"/runs/v1/")

    # No match
    with pytest.raises(ValueError, match="did not match"):
        get_df_names(paths, pattern=r"/nonexistent/([^/]+)")


def test_cli_regex_arg(monkeypatch):
    test_args = ["mini_metric_plot", "-i", "file1.csv", "file2.csv", "-r", r"/([^/]+)\.csv$"]
    monkeypatch.setattr("sys.argv", test_args)
    parsed = cli()
    assert parsed["input"] == ["file1.csv", "file2.csv"]
    assert parsed["pattern"] == r"/([^/]+)\.csv$"
