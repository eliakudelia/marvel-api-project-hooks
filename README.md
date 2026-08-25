# scanify

Makes a PDF look like it left the computer and came back. Two modes:

- **scan** — the page as a flatbed scanner returns it: grain, skew, warm paper,
  uneven lamp, dust and compression artefacts. PDF in, PDF out.
- **photo** — the printed sheet lying on a lit surface, shot with a phone:
  perspective, a cast shadow, lens softness and sensor noise. PDF in, images out.

Everything happens locally; nothing is uploaded.

## Install

```bash
pip install -e .
```

Needs Python 3.10+, PyMuPDF, Pillow and NumPy.

## Browser interface

```bash
pip install -e ".[ui]"
scanify-ui
```

Opens `http://127.0.0.1:8000/` — drop a PDF in, pick a preset, drag the sliders
and watch page one update, then download the finished file. The server listens
on the loopback interface only and keeps uploads in a temporary directory that
is wiped when you stop it. Previews render at 120 dpi so a slider feels
responsive; the downloaded PDF uses whatever resolution you selected.

Switch between **Скан** and **Фото** at the top; each mode keeps its own preset
and slider positions while you flip between them.

`scanify-ui --port 9000` moves it, `--no-browser` stops it opening a window.

## Command line

```bash
scanify report.pdf -o scanned.pdf                       # everyday office scanner
scanify report.pdf -o old.pdf --preset worn --seed 7    # yellowed and dusty
scanify report.pdf -o fax.pdf --preset fax --pages 1-3  # 1-bit, first three pages
scanify report.pdf -o out.pdf --preview page1.png       # peek at page 1 as an image
```

The output keeps the original page dimensions, so it still prints correctly.

### Presets

| preset | what it looks like |
| --- | --- |
| `clean` | modern office scanner at 300 dpi, barely any artefacts |
| `office` | the default: light grain, small skew, warm paper |
| `book` | shadow along the spine, a bowed page, softer focus |
| `worn` | yellowed and mottled, dust specks, visible skew |
| `photocopy` | crushed contrast, heavy toner, grey blotches |
| `fax` | 1-bit dithered at 150 dpi, harsh and ragged |

### Tuning

`--preset` picks a starting point; `--set NAME=VALUE` overrides any single knob
on top of it. `--list-settings` prints them all with their current values.

```bash
scanify in.pdf -o out.pdf --preset worn --set rotate=2.0 --set dust=0.0
scanify in.pdf -o out.pdf --set paper_tint=1.0,0.94,0.86   # stronger yellow
scanify in.pdf -o out.pdf --preset office --list-settings
```

Common knobs: `rotate` (max skew in degrees), `noise`, `blur`, `bleed` (how far
ink spreads), `vignette`, `fold_shadow`, `paper_blotch`, `dust`, `contrast`,
`black_point`, `quality` (JPEG), `dpi`, `mode` (`color` / `gray` / `bw`).

Distances are expressed at 200 dpi and rescale automatically, so a preset looks
the same whatever `--dpi` you choose.

### Reproducibility

Without `--seed` every run differs. With one, a given page always comes out
identical — useful for tests and for regenerating a file you liked.

## Photo mode

```bash
scanify-photo report.pdf -o photo.jpg                    # page 1, on a light desk
scanify-photo report.pdf -o shot.jpg --preset wood --seed 7
scanify-photo report.pdf -o shots/ --pages 1-3           # page-01.jpg, page-02.jpg …
scanify-photo report.pdf -o album.pdf                    # every shot in one PDF
```

`-o` decides the shape of the output: a directory (or a trailing slash) collects
numbered files, a `.pdf` name collects the photos into one document, and a plain
image name is used as-is for a single page and numbered for several.

| preset | what it looks like |
| --- | --- |
| `table` | a pale table, shot almost square on |
| `desk` | the default: a light desk, daylight from the upper left |
| `wood` | warm wooden surface, stronger grain and shadow |
| `handheld` | close, tilted, softer and noisier — a quick phone grab |
| `overhead` | flat and even, barely any perspective |
| `evening` | warm low light, long shadow, darker frame |

`--set` works the same way as in scan mode, over the photo settings: `margin`,
`tilt`, `perspective`, `surface`, `light_angle`, `light_falloff`, `shadow`,
`contact`, `curl`, `edge_light`, `defocus`, `chromatic`, `warmth`, `width`.

## How it works

The page is rendered to a bitmap, then nine stages run in a fixed order, each a
plain array operation in `scanify/effects.py`:

| stage | what it does |
| --- | --- |
| `bleed` | dark strokes creep outwards, the way toner does on fibre |
| `paper` | multiplies by the reflectance of a real sheet: tint, stains, grain |
| `geometry` | rotation, offset, page bow and ripple, in one resample |
| `lighting` | corner falloff, optional spine shadow, exposure drift |
| `optics` | defocus blur |
| `tone` | gamma, contrast, and the lifted blacks a scan always has |
| `noise` | gaussian sensor noise |
| `specks` | dust on the platen, hairline scratches on the glass |
| `colorize` | grayscale, or a dithered 1-bit conversion |

Pages are then re-encoded (JPEG, or lossless PNG for 1-bit) and assembled into a
new PDF at the original page size.

Photo mode reuses the `bleed` and `paper` stages — a printed sheet is a printed
sheet — and then composes a scene in `scanify/photo.py`: the sheet's four corners
are placed in the frame with a keystone and a tilt, a homography warps the page
onto that quadrilateral, a soft-edged mask composites it over a lit surface, and
a cast shadow is built from the same quadrilateral offset away from the light.
Camera effects finish the job: lateral colour fringing, corner defocus, vignette,
white balance and noise.

Every photo length is a fraction of the frame width, so `--width` changes the
size of the image without changing the composition.

## Library use

```python
from scanify import build, build_photo, photograph_pdf, scanify_pdf

scanify_pdf("in.pdf", "out.pdf", build("worn", seed=42), pages=[0, 1])

shots = photograph_pdf("in.pdf", build_photo("wood", seed=42), pages=[0])
shots[0].save("photo.jpg")
```

`process_image(img, settings, rng, skip=["specks"])` runs the same pipeline over
a single Pillow image and lets you switch individual stages off.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
