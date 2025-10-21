All files in this folder should be zip archives with a single contained comma-delimited CSV.

On Linux these can be created with:
```sh
zip [name].csv.zip [name].csv
```

To run an example simply run:
```sh
python -m mini_metrics.metrics -f examples/[file].csv.zip
```