"""Assemble the standalone page (index.html) and the artifact body (dist/artifact.html).

Inlines the elevation grid and the stitched panorama + photos from assets/ as data URIs,
so both outputs are single self-contained files.
"""
import base64, pathlib
root = pathlib.Path(__file__).parent

def data_uri(path, mime="image/jpeg"):
    return f"data:{mime};base64," + base64.b64encode((root / path).read_bytes()).decode()

body = (root / "src/panorama.html").read_text()
for token, value in {
    "__DEM_B64__": (root / "data/dem-600-200m.b64").read_text().strip(),
    "__PANO_A__": data_uri("assets/pano-a.webp", "image/webp"),
    "__PANO_B__": data_uri("assets/pano-b.webp", "image/webp"),
    "__CREW_PATCH__": data_uri("assets/crew-patch.webp", "image/webp"),
    "__PHOTO_CREW__": data_uri("assets/summit-crew.jpg"),
    "__PHOTO_MOT__": data_uri("assets/toward-motueka.jpg"),
    "__ROUTE__": (root / "data/route.json").read_text().strip(),
}.items():
    assert token in body, token
    body = body.replace(token, value)

# calibration build: same page plus the calibration panel, saving to the artifact database
seam = root / "data"
cal = body.replace("__CALIBRATE__", "1").replace("<title>Gordons Knob Panorama</title>", "<title>Gordons Knob Calibration</title>", 1) + (root / "src/calibrate.html").read_text()     .replace("__SEAM_A__", "data:image/webp;base64," + base64.b64encode((seam / "seam-a.webp").read_bytes()).decode())     .replace("__SEAM_B__", "data:image/webp;base64," + base64.b64encode((seam / "seam-b.webp").read_bytes()).decode())
body = body.replace("__CALIBRATE__", "0")
(root / "dist").mkdir(exist_ok=True)
(root / "dist/calibrate.html").write_text(cal)
(root / "dist/artifact.html").write_text(body)
(root / "index.html").write_text(
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
    + body + "\n</body>\n</html>\n")
print(f"ok  {len(body) / 1e6:.1f} MB")
