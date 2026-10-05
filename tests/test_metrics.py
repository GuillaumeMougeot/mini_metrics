import csv

import pytest

from mini_metrics.metrics import main

test_files = ("demo_trunc", "demo", "flemming_fastai_v1", "small")

POST_FIX = ".csv.zip"


@pytest.mark.parametrize("filename_base", test_files)
def test_main_with_example_files(tmp_path, examples_dir, filename_base):
    input_file = examples_dir / f"{filename_base}{POST_FIX}"
    assert input_file.exists()

    output_file = tmp_path / f"{filename_base}_metrics.csv"
    main(files=input_file, output_dir=tmp_path)
    assert output_file.exists()

    expected_file = examples_dir / "expected" / f"{filename_base}_metrics.csv"

    # Read both files
    with open(output_file) as f_out, open(expected_file) as f_exp:
        reader_actual = list(csv.DictReader(f_out))
        reader_expected = list(csv.DictReader(f_exp))

        # Ensure row counts match
        assert len(reader_actual) == len(reader_expected)

        for actual_row, expected_row in zip(reader_actual, reader_expected):
            for key in expected_row:
                # If the value is a number, cast to float and use pytest.approx
                try:
                    exp_val = float(expected_row[key])
                    act_val = float(actual_row[key])
                    diff = act_val - exp_val
                    assert act_val == pytest.approx(exp_val, rel=1e-5), (
                        f"Metric '{key}': actual {act_val} != expected {exp_val} (diff: {diff})"
                    )
                except ValueError:
                    # If it's just text (like a name or ID), compare normally
                    assert actual_row[key] == expected_row[key], (
                        f"Column '{key}': actual {actual_row[key]} != expected {expected_row[key]}"
                    )


def test_precision_options(tmp_path, examples_dir):
    input_file = examples_dir / "flemming_fastai_v1.csv.zip"

    # Default precision (6)
    out_default = tmp_path / "default_metrics.csv"
    main(files=input_file, output_dir=tmp_path, output_name="default_metrics")
    assert out_default.exists()
    with open(out_default) as f:
        reader = list(csv.DictReader(f))
        for row in reader:
            for k, v in row.items():
                if k != "level" and "." in v:
                    decimals = len(v.split(".")[1])
                    assert decimals <= 6

    # Custom precision (2)
    out_custom = tmp_path / "custom_metrics.csv"
    main(files=input_file, output_dir=tmp_path, output_name="custom_metrics", precision=2)
    assert out_custom.exists()
    with open(out_custom) as f:
        reader = list(csv.DictReader(f))
        for row in reader:
            for k, v in row.items():
                if k != "level" and "." in v:
                    decimals = len(v.split(".")[1])
                    assert decimals <= 2

    # Disabled precision (full float precision)
    out_full = tmp_path / "full_metrics.csv"
    main(files=input_file, output_dir=tmp_path, output_name="full_metrics", precision=-1)
    assert out_full.exists()
    with open(out_full) as f:
        reader = list(csv.DictReader(f))
        has_long_decimal = any(
            len(v.split(".")[1]) > 6 for row in reader for k, v in row.items() if k != "level" and "." in v
        )
        assert has_long_decimal


def test_per_class_values_without_support_are_undefined_not_one():
    import math

    import pandas as pd

    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import evaluate_file

    # A: one correct and one rejected "D" prediction; B: never accepted; C: correct; D: predicted only.
    df = MetricDF(
        pd.DataFrame(
            {
                "instance_id": range(4),
                "filename": list("wxyz"),
                "level": 0,
                "label": ["A", "A", "B", "C"],
                "prediction": ["A", "D", "B", "C"],
                "confidence": [0.9, 0.1, 0.1, 0.9],
                "threshold": 0.5,
            }
        )
    )
    kwargs = dict(
        threshold=[0.5], simple=True, hierarchical=False, verbose=0, pattern="^(precision|recall|f1)$"
    )
    per_class = evaluate_file(df, per_class=True, **kwargs)
    precision, f1 = per_class["precision"][0], per_class["f1"][0]
    assert math.isnan(precision["B"][0]) and precision["B"][1] == 0
    assert math.isnan(precision["D"][0]) and math.isnan(f1["D"][0]) and f1["D"][1] == 0
    assert f1["B"] == (0.0, 1.0)  # Supported but never accepted: no true positives.
    aggregate = evaluate_file(df, **kwargs)
    assert aggregate["precision"][0] == 1.0  # A and C only.
    assert aggregate["recall"][0] == (0.5 + 0.0 + 1.0) / 3
    assert aggregate["f1"][0] == (2 / 3 + 0.0 + 1.0) / 3
