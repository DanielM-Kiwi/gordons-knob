# Panorama stitching pipeline

These scripts turn the phone photos into `assets/gordons-knob-360.jpg`. Run them from the repo root with `WORK` files in `./work/`. They need `pip install opencv-python-headless scipy numpy pillow`.

| Step | Script | What it does |
|---|---|---|
| 1 | `sky.py` | Traces the skyline in `pano2.jpg` (red-channel drop under the brightening sky). |
| 2 | `demsky.py` | Builds a ±160 km elevation mosaic (AWS Terrain Tiles z10) and computes the true skyline from the summit. |
| 3 | `fit3.py` | Solves bearing, horizon, tilt, sweep warp and camera position by fitting the photo skyline to the DEM skyline. The loop drift of 368 px is fixed from the self-overlap. |
| 4 | `sim.py`, `sim2.py` + `rotfit.py` | SIFT-match the extra photos against the pano, then solve each camera's 3D rotation and focal length (Kabsch). |
| 5 | `p1map.py` + `p1geo.py` | Matches pano1 to pano2 (5,900 points), then solves pano1's own geometry: per-column bearing with a roll/shear term, plus its horizon and focal length. The bearing error is 0.025°. |
| 6 | `compose.py` | Composites the real photos into an equirectangular band (+36° to −54°) with soft alpha edges, in priority order: pano2 (loop seam feathered), then pano1, then the Motueka photo. Each lower layer is colour-gain matched to pano2 in their overlap. pano1 and the Motueka photo extend real ground from −25° down to −27…−34°. |
| 7 | `final.py` | Fills sky to the zenith (fitted to the photo's own gradient) and quilts tussock texture only where no photo reaches. |
| 8 | `insert.py` | Projects the summit-crew photo in by its solved camera rotation (0.035° fit), with low-frequency colour matching. It is limited to a soft ellipse around the people and the knoll top (bearing 180.5–188.3°), because blending the photo's outer edge ghosted a far peak. |
| 9 | `seamcheck.py`, `verify.py` | Measured checks on the final image. Results:<br>• Column-step score: stitched max 1.29×, below the raw unstitched pano2's own 1.38×, so stitching adds no seam above natural texture.<br>• Narrowest blend-mask transition: 0.42°, so there are no hard edges.<br>• Summit-crew photo vs the pano underneath: 0.42 px median displacement over 454 features.<br>• All 50 blend-zone tiles at full resolution: detail ratio to the neighbouring real image 0.88–1.15, median 1.01, so no blur and no seam.<br>• Synthetic content in any pixel the viewer can show: 0. The viewer's lowest view is −25°, which pano2 covers completely. |

Key numbers from the fit:
- pano2 circumference: 18,032 px
- bearing at its left edge: 347.3°
- camera position: 41.6075°S 172.9429°E, about 640 m ESE of the gazetteer point
- skyline match: median 0.2°
- summit-crew photo: 0.035° median error, focal length 7,002 px

All four photos are used. pano2 is the 360° base. pano1 and `looking at mot.jpg` add real ground below pano2's frame, and `Looking south I think with us in it.jpg` puts the summit crew on the knoll. The two extra photos also open full size from the photo markers.

## Calibration (round 2)

The phone sweep's bearing isn't uniform, so a single rotation can't line it up. The viewer maps band coordinates to true directions with `CAL` (Fourier series in band bearing), solved from:

1. **Landmark placements.** 12 landmarks were dragged onto the real places in the calibration build (`dist/calibrate.html`, which saves to the artifact database).
2. **The far skyline.** `skymatch.py` matched it in 3° windows against the true skyline from the trig (`sky2.py`), searching around the landmark-based prior.

`combfit3.py` fits the bearing series robustly, together with a small camera offset from the trig. The result is E +15.5 m, N −7.4 m, and peaks and spot heights land within ±0.6° (towns ±2°, since a town is a broad target). `elevfit.py` solves tilt and eye height: 1674.4 m, consistent with the 14 m model plus its ~7 m datum offset from the Topo50 heights. `seamfix.py` blends exposure and the far-ridge height across the sweep's wrap at band 348.9°. The route (`route.py`) is georeferenced from the Topo50 map on its spot heights, snapped to the crest, and its visibility is checked from the calibrated eye.
