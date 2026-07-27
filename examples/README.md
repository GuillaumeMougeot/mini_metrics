# Examples

All files in this folder should be zip archives with a single contained comma-delimited CSV.

On Linux these can be created with:

```bash
zip [name].csv.zip [name].csv
```

To run an example simply run:

```bash
uv run mm_metrics -f examples/[file].csv.zip
```

## Generate the test case expected results

The expected results for each example are stored in [./expected](expected) and are used for unit-tests.

These files should only be regenerated with great care when metrics are changed, added, or removed.

```sh
uv run mm_metrics -f examples/*.csv.zip -d examples/expected
```
