# Task Recipes

These task-oriented recipes show how to combine registry metadata, workflow
presets, and reusable search workspaces.

## Inspect supported sensors

```bash
hypercoast registry --json
hypercoast inspect scene.nc --json
hypercoast validate pace scene.nc
hypercoast summarize scene.nc --json
```

## Run a coastal workflow

```bash
hypercoast workflows
hypercoast workflow ndwi scene.nc ndwi.nc --variable reflectance
hypercoast workflow chlorophyll scene.nc chlorophyll.csv --format csv
```

In Python:

```python
import xarray as xr
import hypercoast

dataset = xr.open_dataset("scene.nc")
ndwi = hypercoast.apply_workflow(dataset, "ndwi", variable="reflectance")
```

## Subset and extract spectra from the CLI

```bash
hypercoast subset scene.nc subset.nc --bbox -91.5 28.5 -90.5 29.5
hypercoast spectra pace scene.nc --points stations.csv --output spectra.csv
```

The point CSV should include `x` and `y` columns by default. Use
`--x-column`, `--y-column`, and `--crs` when your coordinate columns use
different names or a projected CRS.

## Search, download, and visualize by sensor

PACE:

```bash
hypercoast search pace --bbox -91 28 -90 29 --temporal 2024-06-01/2024-06-30 --workspace-output pace.json
hypercoast download pace pace.json --out-dir data/pace
hypercoast summarize data/pace/scene.nc --sensor pace --json
```

EMIT:

```bash
hypercoast search emit --bbox -91 28 -90 29 --temporal 2024-06-01/2024-06-30 --workspace-output emit.json
hypercoast download emit emit.json --out-dir data/emit
hypercoast workflow ndwi data/emit/scene.nc emit-ndwi.nc --sensor emit
```

Tanager:

```bash
hypercoast search tanager --bbox -91 28 -90 29 --count 10 --workspace-output tanager.json
hypercoast download tanager tanager.json --out-dir data/tanager
hypercoast summarize data/tanager/scene.h5 --sensor tanager --json
```

AVIRIS:

```bash
hypercoast search aviris --bbox -122 36 -121 37 --count 5 --workspace-output aviris.json
hypercoast download aviris aviris.json --out-dir data/aviris
hypercoast summarize data/aviris/scene.nc --sensor aviris --json
```

Generic NetCDF or GeoTIFF:

```bash
hypercoast summarize scene.nc --json
hypercoast workflow anomaly scene.nc anomaly.nc
```

## Save and reload search results

```python
import hypercoast

items = hypercoast.search_sensor("pace", count=5)
workspace = hypercoast.SearchResult.from_result(items, sensor="pace")
workspace.to_json("pace-search.json")
workspace.to_csv("pace-search.csv")
```

## Match spectra to a small library

```python
import hypercoast

library = hypercoast.read_spectral_library("library.csv")
matches = hypercoast.match_spectra([0.02, 0.03, 0.01], library)
```
