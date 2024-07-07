# Usage

## Import library

To use HyperCoast in a project:

```python
import hypercoast
```

## Search for datasets

To download and access NASA hyperspectral data, you will need to create an
Earthdata login. You can register for an account at
[urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov). Once you have an account,
run the following code to log in:

```python
hypercoast.nasa_earth_login()
```

Collections on NASA Earthdata are discovered with the search_datasets function,
which accepts an instrument filter as an easy way to get started. Each of the
items in the list of collections returned has a "short-name". For example, to
search for all datasets with the instrument "oci":

```python
results = hypercoast.search_datasets(instrument="oci")
datasets = set()
for item in results:
    summary = item.summary()
    short_name = summary["short-name"]
    if short_name not in datasets:
        print(short_name)
    datasets.add(short_name)
print(f"\nFound {len(datasets)} unique datasets")
```

## Search for data by short name

Next, we use the `search_nasa_data` function to find granules within a collection.
Let's use the short_name for the PACE/OCI Level-2 data product for bio-optical
and biogeochemical properties.

```python
results = hypercoast.search_nasa_data(
    short_name="PACE_OCI_L2_BGC_NRT",
    count=1,
)
```

We can refine our search by passing more parameters that describe the
spatiotemporal domain of our use case. Here, we use the temporal parameter to
request a date range and the bounding_box parameter to request granules that
intersect with a bounding box. We can even provide a cloud_cover threshold to
limit files that have a lower percentage of cloud cover. We do not provide a
count, so we'll get all granules that satisfy the constraints.

```python
tspan = ("2024-04-01", "2024-04-16")
bbox = (-76.75, 36.97, -75.74, 39.01)
clouds = (0, 50)

results, gdf = hypercoast.search_nasa_data(
    short_name="PACE_OCI_L2_BGC_NRT",
    temporal=tspan,
    bounding_box=bbox,
    cloud_cover=clouds,
    return_gdf=True,
)
```

Display the footprints of the granules that match the search criteria.

```python
gdf.explore()
```

We can also download all the results with one command.

```python
hypercoast.download_nasa_data(results, out_dir="data")
```

## Search for PACE data

To search for PACE data, we can use the `search_pace` function:

```python
results, gdf = hypercoast.search_pace(
    bounding_box=(-83, 25, -81, 28),
    temporal=("2024-05-10", "2024-05-16"),
    count=10,  # use -1 to return all datasets
    return_gdf=True,
)
```

To download the PACE data, we can use the `download_pace` function:

```python
hypercoast.download_pace(results, out_dir="data")
```

## Search for EMIT data

To search for EMIT data, we can use the `search_emit` function:

```python
results, gdf = hypercoast.search_emit(
    bounding_box=(-83, 25, -81, 28),
    temporal=("2024-04-01", "2024-05-16"),
    count=10,  # use -1 to return all datasets
    return_gdf=True,
)
```

To download the EMIT data, we can use the `download_emit` function:

```python
hypercoast.download_emit(results, out_dir="data")
```

## Visualize PACE data

Load the dataset as a `xarray.Dataset` object:

```python
dataset = hypercoast.read_pace(filepath)
```

Visualize selected bands of the dataset:

```python
hypercoast.viz_pace(dataset, wavelengths=[500, 510, 520, 530], ncols=2, crs="default")
```

Visualize the dataset on an interactive map:

```python
m = hypercoast.Map()
m.add_basemap("Hybrid")
wavelengths = [450]
m.add_pace(dataset, wavelengths, colormap="jet", vmin=0, vmax=0.02, layer_name="PACE")
m.add_colormap(cmap="jet", vmin=0, vmax=0.02, label="Reflectance")
m.add("spectral")
m
```

## Visualize EMIT data

To visualize EMIT data, we can use the `read_emit` function:

```python
dataset = hypercoast.read_emit(filepath)
```

Visualize the dataset on an interactive map:

```python
m = hypercoast.Map()
m.add_basemap("SATELLITE")
m.add_emit(dataset, wavelengths=[1000, 600, 500], vmin=0, vmax=0.3, layer_name="EMIT")
m.add("spectral")
m
```

## Create an image cube

First , load the dataset as a `xarray.Dataset` object. Select a region of interest (ROI) using the `sel` method:

```python
dataset = hypercoast.read_emit(filepath)
ds = dataset.sel(longitude=slice(-90.1482, -89.7321), latitude=slice(30.0225, 29.7451))
```

Create an image cube using the `image_cube` function:

```python
cube = hypercoast.image_cube(
    ds,
    variable="reflectance",
    cmap="jet",
    clim=(0, 0.4),
    rgb_wavelengths=[1000, 700, 500],
    rgb_gamma=2,
    title="EMIT Reflectance",
)
cube.show()
```

## Interactive slicing and thresholding

First , load the dataset as a `xarray.Dataset` object. Select a region of interest (ROI) using the `sel` method:

```python
dataset = hypercoast.read_emit(filepath)
ds = dataset.sel(longitude=slice(-90.05, -89.99), latitude=slice(30.00, 29.93))
```

Interactive slicing along the z-axis (band):

```python
p = hypercoast.image_cube(
    ds,
    variable="reflectance",
    cmap="jet",
    clim=(0, 0.5),
    rgb_wavelengths=[1000, 700, 500],
    rgb_gamma=2,
    title="EMIT Reflectance",
    widget="plane",
)
p.add_text("Band slicing", position="upper_right", font_size=14)
p.show()
```

Interactive thresholding:

```python
p = hypercoast.image_cube(
    ds,
    variable="reflectance",
    cmap="jet",
    clim=(0, 0.5),
    rgb_wavelengths=[1000, 700, 500],
    rgb_gamma=2,
    title="EMIT Reflectance",
    widget="threshold",
)
p.add_text("Thresholding", position="upper_right", font_size=14)
p.show()
```

## Visualizing PACE chlorophyll-a concentration data

Load all the data files in a directory as an `xarray.DataArray`:

```python
files = "data/*nc"
array = hypercoast.read_pace_chla(files)
```

Select a date and visualize the chlorophyll-a concentration data with Matplotlib.

```python
hypercoast.viz_pace_chla(array, date="2024-06-01", cmap="jet", size=6)
```

If the date is not specified, the data are averaged over the entire time range.

```python
hypercoast.viz_pace_chla(array, cmap="jet", size=6)
```

Convert the data array to an image that can be displayed on an interactive map.

```python
single_image = hypercoast.pace_chla_to_image(single_array)
```

Create an interactive map and display the image on the map.

```python
m = hypercoast.Map(center=[40, -100], zoom=4)
m.add_basemap("Hybrid")
m.add_raster(
    single_image,
    cmap="jet",
    vmin=-1,
    vmax=2,
    layer_name="Chlorophyll a",
    zoom_to_layer=False,
)
label = "Chlorophyll Concentration [lg(lg(mg m^-3))]"
m.add_colormap(cmap="jet", vmin=-1, vmax=2, label=label)
m
```
