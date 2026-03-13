"""
update_docs.py — Rewrite markdown cells and fix inline comments in code-06
of MagicScanRevised.ipynb to accurately reflect the current implementation.
"""

import json, ast, textwrap, sys

NOTEBOOK_PATH = r'C:\PDTSource\MagicScan\MagicScanRevised.ipynb'

# ---------------------------------------------------------------------------
# New cell sources
# ---------------------------------------------------------------------------

MD_05 = """\
---
## Step 1 — Card Detection

### Why the original homography approach failed

`CardOrient.ipynb` used **ORB feature matching + homography** to align a photo against a template card. Two problems:

**Problem 1 — Wrong tool.**
Homography via feature matching answers *"These two images show the same object — what transform maps one to the other?"*
It requires a reference photo of the *exact same card* you're scanning. For a general scanner that needs to work on any card, that would require a reference library for every card you own.

**Problem 2 — A silent bug in the keypoint matching code.**
```python
# CardOrient.ipynb, cell 12 — ORIGINAL (broken):
for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.queryIdx].pt   # BUG: should be match.trainIdx
```
Both arrays use `queryIdx`, so both pull coordinates from *image 1*.
The computed homography maps image 1 → image 1 (an identity transform), not image 2 → image 1.
This is why the warped output looked the same as the input — the transform was a no-op.

**Problem 3 — Bad ORB parameters.**
`scaleFactor=2` is far too large (normal: 1.2–1.5; this controls scale-space resolution).
`patchSize=2` is far too small: each ORB descriptor summarises a 2×2 pixel patch — almost no information content (the default is 31×31).

---
### Why the first revised version (contour + approxPolyDP + order_points) also failed

The previous version of this notebook used `findContours` → `approxPolyDP` → `order_points` → `getPerspectiveTransform`.
That approach broke for these particular test images:

> **The test images show cards rotated ≈ 90° on the table.**
> When a card is rotated nearly 90°, its four corners form a diamond pattern.
> Two corners end up with nearly identical `x + y` sums, and two others with nearly identical `x − y` differences — the exact values `order_points` uses to assign TL/TR/BR/BL.
> NumPy's `argmin` picks the first tie-match arbitrarily, so **the same corner can be assigned to two output positions**.
> The resulting perspective transform is degenerate — it maps parts of the image to the same output location, which is why all three cards returned an empty OCR string.

---
### Why the second revised version (Canny edges + minAreaRect) also failed

After switching to `minAreaRect` (which eliminates the corner-ordering problem entirely), the detection was driven by Canny edge detection. This exposed a different failure:

**The card outer border is dark cardstock sitting on a dark table.**
Canny finds edges by looking for sharp intensity changes. A dark border against a dark background produces very little contrast — Canny misses the card outline almost entirely.

Instead, Canny latched onto the *art box* inner border, which has high contrast against the bright artwork and white frame inside the card. The art box contour is roughly 60% of the card's area — but its aspect ratio is approximately 1.40:1, nearly the same as the full card's 1.397:1 ratio. The aspect-ratio filter that was supposed to reject non-card shapes let the art box through every time.

---
### The fix: threshold-based blob detection + minAreaRect

**Why thresholding works where Canny failed.**
Magic cards have large white areas: the title bar, type-line bar, and rules-text box cover most of the card face. Otsu's method computes the threshold that best separates two pixel populations (bright card interior vs. dark background) automatically, with no manual tuning. The result is a clean white mask over the card face and near-black everywhere else.

**Filling the art-box hole.**
The one dark region *inside* the card face is the artwork. After Otsu thresholding, this appears as a hole in the white mask. A 40×40-pixel morphological close operation (dilation followed by erosion) fills that hole, leaving a single solid white blob that covers the entire card face.

**Contour selection: closest ratio, not first qualifying.**
`findContours` on the closed binary mask can still return multiple blobs — reflections, stray bright patches on the table, etc. Rather than picking the largest blob or the first one above a size threshold, the code computes `minAreaRect` for every candidate and selects the one whose aspect ratio is **closest to 1.397** (the true MTG card ratio of 88 mm ÷ 63 mm). This makes the selector robust against large false positives that happen to pass the size filter.

**Three bugs discovered and fixed in the rotation/crop pipeline.**

*Bug 1 — Angle normalisation.*
`minAreaRect` returns angle in the range (−90°, 0°]. It defines "width" as the dimension measured along the direction closest to horizontal. When a card is nearly portrait (long axis close to vertical), `minAreaRect` reports the short side as width and assigns an angle close to −90°. Naively rotating by −(−75°) = +75° would fling the card 75° CCW — far from straight. The fix: if `angle < −45°`, add 90° and swap `w`/`h`. This maps the angle to the small tilt of the long axis from vertical, always in (−45°, 0°].

*Bug 2 — Rotation sign.*
`cv2.getRotationMatrix2D(center, angle, 1.0)` uses OpenCV's convention where positive angle means counter-clockwise rotation (Y-axis points down). After normalisation, the angle value already carries the correct sign for the correction needed. Passing `angle` directly — not `−angle` — is correct. Using `−angle` doubles the tilt instead of removing it.

*Bug 3 — Crop via box-point transform, not centre ± half-size.*
An earlier crop used `center ± rect_w/2` and `center ± rect_h/2`. After the `w`/`h` swap in Bug 1's fix, those dimensions may no longer correspond to the image axes. The correct approach: transform the four `minAreaRect` corner points through the same rotation matrix `M`, then take the axis-aligned bounding box of those transformed corners. This is exact regardless of angle or dimension swap.

---
> **Simple version:** Imagine you put a playing card on a dark table and take a photo. The computer needs to figure out exactly where the card is. The first two approaches failed — one tried to match the card to a reference photo (which you might not have), and the other tried to find the card's four corners precisely (but this breaks when the card is tilted sideways, because two corners end up so close together that the algorithm picks the same one twice).
>
> The approach that actually works: Magic cards have a lot of white space on their face (the text boxes, title bar, etc.). We use a technique called Otsu thresholding — the computer automatically finds the brightness level that best separates the white card from the dark background. We fill in the dark artwork region with a "morphological close" (think of it as smearing white paint to fill gaps). Now we have a solid white blob shaped like the card. We fit the tightest possible rotated rectangle to that blob, which gives us the card's angle and size directly. Finally, we rotate the photo to un-tilt the card, and carefully crop it out using the exact transformed corner positions.
"""

MD_08 = """\
---
## Step 2 — Orientation Correction

After `detect_card` the raw crop may be in landscape orientation, and the card content could be upright, upside-down, or 90° off — we don't know yet.

**The pipeline tries all four 90° rotations (in `scan_card`).**
For each candidate rotation (0°, 90° CW, 180°, 270° CCW), it forces the result to portrait (CCW if still landscape), then extracts the title region and runs OCR. The first rotation that produces a Scryfall match wins; if none match, the one with highest OCR confidence is kept.

**Why rotate the raw crop, not the portrait card?**
A previous version applied rotations *after* `ensure_portrait`. The problem: `ensure_portrait` makes the card portrait, but `_try_orientation` then applies a 90° rotation — turning that portrait card back into landscape. The TITLE_* constants (e.g. `TITLE_Y1 = 0.038`) are calibrated for **portrait** cards. Extracting 3.8% from the top of a *landscape* card lands in the artwork, not the title strip.

The fix: rotate the **raw detect_card output** first, then force portrait. Every candidate that enters `_try_orientation` is portrait with the title in the expected fractional position near the top.

For example, for a card photographed 90° sideways:
- Raw crop: landscape, title at the **left** end.
- 90° CW rotation: portrait, title now at the **top** — title strip lands at the right fractional y-coordinates.
- Remaining rotations (180°, 270° CCW, 0°) cover the other three cases.

---
> **Simple version:** Even after cutting the card out of the photo, we don't know which way is "up." The card could be right-side-up, upside-down, or turned sideways in either direction — four possibilities. The pipeline just tries all four, runs a text search on the title each time, and picks the one that finds a real card name on the Scryfall database. The key insight is that we apply each rotation *before* forcing the image into portrait mode, because all our "where's the title?" measurements assume a portrait card. If we rotated after going portrait, we'd accidentally put the card back into landscape.
"""

MD_11 = """\
---
## Step 3 — Region Extraction & Crop Diagnostic

With the card in a consistent portrait orientation, we crop the title bar and the collector info strip using the fractional constants defined above.

**The diagnostic plot below is the most important debugging tool in this notebook.**
It draws the crop regions on top of each card image and saves those images to `output/04_crop_regions/`. If the green (title) or blue (set-code) boxes are in the wrong place, adjust the `TITLE_*` and `SETCODE_*` constants in the constants cell and re-run.

---
> **Simple version:** Now that the card is upright, we need to cut out just the title bar (top of the card) and the collector info strip (bottom). We describe these regions as fractions of the card's height and width — e.g., "the title bar starts 3.8% from the top and ends 9.5% from the top" — so the measurements work regardless of the photo's resolution. The diagnostic images let you visually check that the green and blue boxes are landing in the right places before you trust any OCR results downstream.
"""

MD_14 = """\
---
## Step 4 — OCR Preprocessing

### What was wrong with the original

```python
# CardTest.ipynb — ORIGINAL:
flag, thresh = cv2.threshold(title, 150, 255, cv2.THRESH_BINARY)
```

**Problem 1 — Fixed threshold.**
A threshold of 150 assumes consistent, even lighting across the whole image. Photos taken on a table have shadows, hot spots, and colour cast from room lighting. Text in a shadowed area might have pixel intensities of 80–100, well below 150, so those characters are wiped out of the binary image entirely.

**Problem 2 — No upscaling.**
The hardcoded-pixel title crop was roughly 16 pixels tall. Tesseract and most neural OCR models degrade noticeably on text smaller than ~30–40 px. Larger input is better.

**Problem 3 — No denoising.**
Photographic grain and JPEG compression produce intensity variations that, after binarisation, become salt-and-pepper noise around letter edges — extra dots and gaps that OCR reads as parts of characters.

### The fix
1. **Denoise first**, then threshold (removes grain before it can become noise pixels).
2. **Adaptive thresholding** — each pixel's threshold is computed from its local neighbourhood, compensating for shadows and highlights automatically.
3. **3× cubic upscale** before passing to the OCR engine.

---
> **Simple version:** Before we hand the title image to the OCR engine, we clean it up in three steps. First, we remove photographic grain using a denoising filter — otherwise the grain turns into specks that the OCR reads as parts of letters. Second, instead of one fixed brightness cutoff for the whole image (which fails in shadows), we use adaptive thresholding: each tiny patch of the image gets its own cutoff based on its local neighbours, so shadowed text and brightly lit text both become clean black-on-white. Third, we enlarge the title strip to 3× its size using a high-quality resize, because OCR engines work much better on larger text — the extra resolution gives them more detail to work with.
"""

MD_18 = """\
---
## Step 5 — OCR with EasyOCR

### Why EasyOCR instead of Tesseract

Tesseract is built on statistical language models (LSTM in its current mode). It works well on standard printed text but is fragile on decorative fonts. The Magic card title font (Beleren Bold / Matrix Bold) is legible but distinctive enough to confuse Tesseract's English model, even with the community-trained MTG dataset used in the original project.

Training Tesseract properly requires a large labelled dataset and significant tuning of `--psm` (page segmentation mode) and `--oem` (engine mode) flags.

**EasyOCR** uses a CRNN (Convolutional Recurrent Neural Network) trained on a much wider variety of fonts and image conditions. Its convolutional layers learn font-agnostic visual features, making it significantly more tolerant of stylised text — with no MTG-specific training.

**Tesseract comparison (optional):**
```python
import pytesseract
# --psm 7 = single line of text  ← important for a title-bar crop
# --psm 8 = single word
# --oem 3 = LSTM engine (most accurate)
text = pytesseract.image_to_string(prep, config='--psm 7 --oem 3')
```
The `--psm 7` flag is critical. Without it, Tesseract tries to parse the whole page layout inside a tiny title-bar crop and produces garbage.

---
> **Simple version:** There are two main OCR tools here — Tesseract and EasyOCR. Tesseract is the old reliable: great for normal book-style text, but it gets confused by Magic's custom stylised fonts. EasyOCR uses a neural network that was trained on a huge variety of fonts, so it handles decorative text much better without any special Magic-specific training. Think of Tesseract as a reader who only ever studied standard textbooks, versus EasyOCR as someone who grew up reading everything — signs, logos, handwriting — and learned to recognise letterforms from context rather than memorised templates.
"""

MD_21 = """\
---
## Step 6 — Scryfall Validation & Fuzzy Search

This step is entirely new — the original project had no equivalent.

OCR output is rarely perfect. Even with a better engine and preprocessing, the title might come back as *"Fearlezs Pup"* or *"Gods Hall Guardian"* (missing apostrophe). Without correction, near-misses count as failures.

Scryfall's `/cards/named?fuzzy=` endpoint uses **edit distance** to find the closest card name to the input. It handles minor typos, substituted characters, and dropped punctuation well. Adding this single correction step can recover a large fraction of "almost right" OCR results.

Scryfall also returns rich metadata — confirmed set code, collector number, type line, rarity — which can supplement or cross-check what we OCR from the bottom strip.

---
> **Simple version:** OCR is never perfect — it might read "Fearless Pup" as "Fearlezs Pup" or drop an apostrophe from a card name. Instead of calling that a failure, we send whatever the OCR returned to Scryfall's fuzzy search API. Scryfall (the comprehensive Magic card database) uses edit-distance matching — basically counting how many single-character changes it takes to turn our OCR result into a real card name — and returns the closest match. If it's close enough, we get back the confirmed card name plus all its metadata for free: set code, collector number, rarity, and more.
"""

MD_23 = """\
---
## Step 7 — Full Pipeline

All steps wired into `scan_card()`.

**Four-orientation retry:**
The pipeline tries four candidate orientations of the raw `detect_card` output: 0°, 90° CW, 180°, and 270° CCW. For each candidate, `_force_portrait` ensures the image is portrait, then `_try_orientation` extracts the title region, runs OCR, and queries Scryfall. The first orientation that produces a successful Scryfall match is returned immediately. If no orientation matches Scryfall, the candidate with the highest OCR confidence score is kept as a best-effort result.

---
> **Simple version:** This is where all the pieces come together. For each photo, `scan_card` tries four possible "right-side-up" orientations for the card, runs the full pipeline on each (portrait correction → title crop → OCR → Scryfall lookup), and returns the first result that Scryfall recognises as a real card name. If Scryfall doesn't recognise any of the four, it returns whichever attempt the OCR engine was most confident about. It's a systematic try-everything approach that trades a bit of speed for robustness.
"""

# ---------------------------------------------------------------------------
# New code-06 source — same logic, just updated comment blocks and title fix
# ---------------------------------------------------------------------------

# We'll do targeted string replacements in the code-06 source rather than
# rewriting the whole cell, to minimise risk of introducing bugs.

OLD_COMMENT_BLOCK = """\
        # ------------------------------------------------------------------
        # minAreaRect: fit the tightest rotated bounding rectangle.
        #
        # OLD APPROACH used approxPolyDP to look for exactly 4 corners, then
        # relied on order_points (sum/diff trick) to label them TL/TR/BR/BL.
        # That broke for ~90°-rotated cards where two corners have nearly
        # identical sums/differences (see Step 1 markdown).
        #
        # NEW APPROACH: minAreaRect gives us centre + angle + dimensions
        # directly, with no corner-ordering step at all.
        # ------------------------------------------------------------------"""

NEW_COMMENT_BLOCK = """\
        # minAreaRect fits the tightest rotated rectangle to this bright-
        # region contour, giving centre + (width, height) + angle directly.
        # No corner-ordering needed — the angle tells us how much to rotate."""

OLD_TITLE_LINE = "        axes[1].set_title(f'Full image after -{angle:.1f}° rotation')"
NEW_TITLE_LINE = "        axes[1].set_title(f'Full image after {angle:.1f}° correction')"

# ---------------------------------------------------------------------------
# Apply changes
# ---------------------------------------------------------------------------

def main():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    cell_map = {cell['id']: cell for cell in cells}

    # --- Markdown cells ---
    updates = {
        'md-05': MD_05,
        'md-08': MD_08,
        'md-11': MD_11,
        'md-14': MD_14,
        'md-18': MD_18,
        'md-21': MD_21,
        'md-23': MD_23,
    }

    for cell_id, new_source in updates.items():
        if cell_id not in cell_map:
            print(f'ERROR: cell {cell_id} not found', file=sys.stderr)
            sys.exit(1)
        cell_map[cell_id]['source'] = [new_source]
        print(f'Updated {cell_id}')

    # --- code-06: targeted comment replacement ---
    code06 = cell_map.get('code-06')
    if code06 is None:
        print('ERROR: code-06 not found', file=sys.stderr)
        sys.exit(1)

    source = ''.join(code06['source'])

    if OLD_COMMENT_BLOCK not in source:
        print('ERROR: old comment block not found in code-06', file=sys.stderr)
        sys.exit(1)
    source = source.replace(OLD_COMMENT_BLOCK, NEW_COMMENT_BLOCK, 1)
    print('Replaced comment block in code-06')

    if OLD_TITLE_LINE not in source:
        print('ERROR: old title line not found in code-06', file=sys.stderr)
        sys.exit(1)
    source = source.replace(OLD_TITLE_LINE, NEW_TITLE_LINE, 1)
    print('Replaced title line in code-06')

    # Validate the updated code-06 parses as valid Python
    try:
        ast.parse(source)
        print('ast.parse: code-06 is valid Python')
    except SyntaxError as e:
        print(f'ERROR: code-06 syntax error after edit: {e}', file=sys.stderr)
        sys.exit(1)

    code06['source'] = [source]

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print('Done.')

if __name__ == '__main__':
    main()
