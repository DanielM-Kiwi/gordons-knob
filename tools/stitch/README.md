# Panorama stitching pipeline

These scripts turn the phone photos into `assets/gordons-knob-360.jpg`. Run them from the repo root with `WORK` files in `./work/`. They need `pip install opencv-python-headless scipy numpy pillow`.

| Step | Script | What it does |
|---|---|---|
| 1 | `sky.py` | Traces the skyline in `pano2.jpg` (red-channel drop under the brightening sky). |
| 2 | `demsky.py` | Builds a ±160 km elevation mosaic (AWS Terrain Tiles z10) and computes the true skyline from the summit. |
| 3 | `fit3.py` | Solves bearing, horizon, tilt, sweep warp and camera position by fitting the photo skyline to the DEM skyline. The loop drift of 368 px is fixed from the self-overlap. |
| 4 | `sim.py`, `sim2.py` + `rotfit.py` | SIFT-match the extra photos against the pano, then solve each camera's 3D rotation and focal length (Kabsch). |
| 5 | `stitch.py` | Resamples pano2 into an equirectangular band (+36° to −54°), feathers the 360° loop seam, extends the sky to the zenith and quilts tussock ground below the photo edge. |
| 6 | `insert.py` | Projects the summit-crew photo into the band with low-frequency colour matching and feathered edges. |

Key numbers from the fit:
- pano2 circumference: 18,032 px
- bearing at its left edge: 347.3°
- camera position: 41.6075°S 172.9429°E, about 640 m ESE of the gazetteer point
- skyline match: median 0.2°
- summit-crew photo: 0.035° median error, focal length 7,002 px

`pano1.jpg` is a second partial sweep of the same view, so it isn't needed. `looking at mot.jpg` is pinned as a photo marker instead of being blended. Its near ground doesn't match the pano, which means it was shot a few metres away, and blending it would ghost the foreground.
