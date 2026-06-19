import csv

import pytest

from mini_metrics.metrics import main

test_files = ("demo_trunc", "demo", "flemming_fastai_v1", "small")

POST_FIX = ".csv.zip"


@pytest.mark.parametrize("filename_base", test_files)
def test_main_with_example_files(tmp_path, examples_dir, filename_base):
    input_file = examples_dir / f"{filename_base}{POST_FIX}"
    assert input_file.exists()

    output = tmp_path / filename_base
    output_file = tmp_path / (filename_base + ".csv")
    main(file=input_file, output=output)
    assert output_file.exists()

    expected_file = examples_dir / "expected" / f"{filename_base}.csv"

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
                    assert act_val == pytest.approx(exp_val, rel=1e-5)
                except ValueError:
                    # If it's just text (like a name or ID), compare normally
                    assert actual_row[key] == expected_row[key]
