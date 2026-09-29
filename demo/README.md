# demo/: every file used in the demo video

One folder per part of the video script (`PPT Details/video_script.md`).
Start the app first: double-click `start_demo.bat` (or `python serve.py --live`).
Rebuild this folder with `python docs/make_demo_folder.py --gamus <GAMUS test folder>`.

| folder | file | type | upload it in | then |
|---|---|---|---|---|
| `1_geotiff_with_lidar/` | `park_city_naip2021.tif` | **GeoTIFF** (US aerial photo, 0.6 m) | **Run a model** → Run | full DSM in metres above sea level |
| | `park_city_lidar_reference.tif` | **reference**: USGS airborne LiDAR surface, GeoTIFF | **Check against a reference** → Score this run | accuracy table, then **Compare side by side** |
| `2_png_jpg_tiff/` | `chungthang_plain.png` | **PNG**, no map info | Run a model: tick "I don't know the pixel size" | relative shape (rDSM) |
| | `chungthang_plain.jpg` | **JPG**, same pixels | Run a model: pixel size **0.3** | heights in metres |
| | `chungthang_plain.tif` | **TIFF without map info**, same pixels | Run a model: pixel size **0.3** | same as PNG/JPG (the PS lists TIFF too) |
| `3_gamus_png_with_lidar/` | `<landscape>_<tile>_image.png` | **PNG**, GAMUS test tile (0.3 m) | Run a model: pixel size **0.3** | heights in metres |
| | `<landscape>_<tile>_lidar_heights.tif` | **reference**: LiDAR height above ground for the same picture | Check against a reference | a PNG checked against LiDAR, then Compare |
| `4_geotiff_india/` | `chungthang_wv2.tif` | **GeoTIFF**, WorldView-2, Sikkim | Run a model → Run | Himalayan DSM: fly, slope, contours, buildings, flood |
| `5_results/` | charts | images | show at the end of the video | numbers from RESULTS.md and the LiDAR benchmark |

The GAMUS tiles are the same three as `python serve.py --demo-lidar` (the
median-error tile of the urban, sparse and forest test tiles).

Note: GeoTIFF runs fetch the Copernicus 30 m terrain map from the internet
(small, a few seconds). Keep the laptop online while recording parts 1 and 4.

Sources and licences: NAIP and USGS 3DEP (US public domain); GAMUS (CC BY 4.0);
Maxar Open Data WorldView-2 (CC BY-NC 4.0). See LICENSE.
