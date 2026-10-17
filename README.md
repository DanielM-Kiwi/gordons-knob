# Gordons Knob Panorama

Interactive 360° view from the summit of Gordons Knob (1,592 m, Gordon Range, Mt Richmond Forest Park), with animated POI markers and a live topo minimap that shows where you're looking.

- `index.html` is the standalone page. Open it directly; it needs no server and has no build step.
- `dist/artifact.html` has the same content without the document wrapper, for publishing as a Claude artifact.

## Stack

- Raw WebGL 2 equirectangular viewer: one fragment shader with mipmapped, seam-safe sampling.
- An embedded 600 × 600 elevation grid (200 m cells, ±60 km) from [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/), which are open data. It drives:
  - the placeholder panorama, raycast from the summit with aerial perspective, vegetation zones and Red Hills colouring
  - the topo map: hypsometric tint, hillshade and marching-squares contours (50 m, with 250 m index lines)
  - POI placement and line-of-sight checks, which mark a peak as hidden when a nearer ridge blocks it
- Fonts: Archivo and IBM Plex Mono (Google Fonts).

## Using your own photo

Edit the `PANORAMA` config near the top of `src/panorama.html`:

```js
const PANORAMA = { image: "gordons-knob.jpg", centreBearing: 180, horizonOffset: 0 };
```

- `image`: an equirectangular (2:1) photo, as a URL or a `data:` URI. For the artifact, embed it as a `data:` URI or upload it as an artifact asset.
- `centreBearing`: the true bearing at the horizontal centre of the photo. Adjust it until the markers sit on their peaks.
- `horizonOffset`: degrees to correct a horizon that isn't level.

POIs live in the `POIS` array (lat/lon, height, note). Their positions are approximate.

## Rebuild

```sh
python3 build.py                                  # assemble index.html + dist/artifact.html
python3 build_dem.py <tile_cache_dir> data/dem-600-200m.b64   # re-fetch the elevation grid
```
