"""Assemble the standalone page (index.html) and the artifact body (dist/artifact.html)."""
import pathlib
root = pathlib.Path(__file__).parent
body = (root / "src/panorama.html").read_text().replace("__DEM_B64__", (root / "data/dem-600-200m.b64").read_text().strip())
(root / "dist").mkdir(exist_ok=True)
(root / "dist/artifact.html").write_text(body)
(root / "index.html").write_text(
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
    + body + "\n</body>\n</html>\n")
print("ok", len(body))
