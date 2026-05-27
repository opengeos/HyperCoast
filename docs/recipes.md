# Task Recipes

These task-oriented recipes show how to combine registry metadata, workflow
presets, and reusable search workspaces.

## Inspect supported sensors

```bash
hypercoast registry --json
hypercoast inspect scene.nc --json
hypercoast validate pace scene.nc
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
