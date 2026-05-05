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