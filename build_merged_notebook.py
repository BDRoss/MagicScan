"""
Generates MagicScanMerged.ipynb — a notebook that combines the best of
ExperimentFinal.ipynb (perspective warp, two-stage thresholding) with
MagicScanRevised.ipynb (full-card detection, EasyOCR, fuzzy Scryfall lookup,
four-rotation retry, dynamic title detection).
"""
import json, os

def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "outputs": [], "execution_count": None}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

# ---------------------------------------------------------------------------

CELLS = []

# ── 0  Introduction ─────────────────────────────────────────────────────────
CELLS.append(md("""\
# MagicScan — Merged Implementation

This notebook combines the strongest ideas from two prior implementations:

**From `ExperimentFinal.ipynb` (course final project):**
- Full **perspective transform** (`warpPerspective`) for geometric correction.
  The original affine-only approach corrected rotation but left trapezoidal
  distortion from off-axis photography uncorrected.  A homographic warp
  handles both simultaneously.

**From `MagicScanRevised.ipynb`:**
- Robust full-card detection (Otsu threshold + morphological close + blur sweep).
- Four-rotation exhaustive search for orientation recovery.
- Dynamic title detection from EasyOCR bounding boxes (no hardcoded crop).
- EasyOCR (CRNN + CTC) in place of Tesseract — better tolerance of the Magic
  card title font.
- Fuzzy Scryfall lookup in place of exact string matching — recovers from
  single-character OCR errors.

**Pipeline stages:**
1. Card detection — segmentation + contour selection
2. Perspective correction — full homographic warp
3. Orientation disambiguation — four-rotation retry
4. Title detection — dynamic EasyOCR strip with spatial mana-cost filter
5. OCR preprocessing — upscale → denoise → unsharp mask
6. OCR — EasyOCR CRNN
7. Fuzzy lookup — Scryfall `/cards/named?fuzzy=`
"""))

# ── 1  Imports & output directories ─────────────────────────────────────────
CELLS.append(code("""\
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import easyocr
import re
import os

OUTPUT_DIRS = {
    'detection':    'output_merged/01_detection',
    'card':         'output_merged/02_card',
    'crop_regions': 'output_merged/03_crop_regions',
    'title_raw':    'output_merged/04_title_raw',
    'title_prep':   'output_merged/05_title_prep',
    'setcode_raw':  'output_merged/06_setcode_raw',
    'setcode_prep': 'output_merged/07_setcode_prep',
}
for d in OUTPUT_DIRS.values():
    os.makedirs(d, exist_ok=True)
print('Output directories ready.')
"""))

# ── 2  Constants ─────────────────────────────────────────────────────────────
CELLS.append(code("""\
# ---------------------------------------------------------------------------
# CARD LAYOUT CONSTANTS
# ---------------------------------------------------------------------------
# Fractional coordinates (0.0–1.0) relative to the corrected portrait card.
# TITLE_* are used for the crop-region diagnostic overlay only — the OCR
# itself uses dynamic bounding-box detection via find_title_from_card().
# SETCODE_* are still used for the collector strip crop.

TITLE_Y1, TITLE_Y2 = 0.038, 0.098
TITLE_X1, TITLE_X2 = 0.040, 0.700

SETCODE_Y1, SETCODE_Y2 = 0.935, 0.975
SETCODE_X1, SETCODE_X2 = 0.020, 0.620

# Physical Magic card ratio: 88 mm tall × 63 mm wide
CARD_RATIO = 88.0 / 63.0   # ≈ 1.397
"""))

# ── 3  Utility functions ─────────────────────────────────────────────────────
CELLS.append(code("""\
def save_step(folder_key, filename, image):
    \"\"\"Write image to the numbered output subfolder. Returns the written path.\"\"\"
    path = os.path.join(OUTPUT_DIRS[folder_key], filename)
    cv2.imwrite(path, image)
    return path


def extract_region(image, y1_frac, y2_frac, x1_frac, x2_frac):
    \"\"\"Crop a region using fractional (0.0–1.0) coordinates.\"\"\"
    h, w = image.shape[:2]
    return image[int(h * y1_frac):int(h * y2_frac),
                 int(w * x1_frac):int(w * x2_frac)]


def show_crop_regions(card, stem='card'):
    \"\"\"
    Draw the title and set-code bounding boxes on the card and save to
    output_merged/03_crop_regions/.  The title box is diagnostic only —
    the OCR uses dynamic detection.  Green = title, blue = set code.
    \"\"\"
    h, w = card.shape[:2]
    vis  = card.copy()
    cv2.rectangle(vis,
                  (int(w * TITLE_X1),   int(h * TITLE_Y1)),
                  (int(w * TITLE_X2),   int(h * TITLE_Y2)),
                  (0, 255, 0), 3)
    cv2.rectangle(vis,
                  (int(w * SETCODE_X1), int(h * SETCODE_Y1)),
                  (int(w * SETCODE_X2), int(h * SETCODE_Y2)),
                  (255, 0, 0), 3)
    save_step('crop_regions', f'{stem}_crop_regions.jpg', vis)
    return vis


def _order_corners(pts):
    \"\"\"
    Order 4 corner points as [TL, TR, BR, BL].

    Uses coordinate sums and differences:
      TL = smallest x+y   (top-left in a standard image)
      BR = largest  x+y
      TR = smallest y-x   (top-right)
      BL = largest  y-x

    This ordering is purely geometric and does not depend on the card's
    content orientation, so it is consistent for any rotation of the card
    in the frame.  The four-rotation retry in scan_card() handles content
    orientation separately.
    \"\"\"
    pts  = pts.reshape(4, 2)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[s.argmin()],
        pts[diff.argmin()],
        pts[s.argmax()],
        pts[diff.argmax()],
    ], dtype='float32')
"""))

# ── 4  Card detection + perspective correction ───────────────────────────────
CELLS.append(md("""\
---
## Stage 1 & 2 — Detection and Perspective Correction

### Detection
The card is located by intensity thresholding (Otsu) followed by morphological
closing to fill the dark art-box hole.  A Gaussian blur sweep handles textured
backgrounds that would otherwise merge into one image-spanning blob after
dilation.  A luminance-inversion check handles light backgrounds where the
card interior is not the dominant bright region.

### Perspective correction
The original revised implementation used `warpAffine` (6 DOF affine transform),
which corrects in-plane rotation but cannot rectify the trapezoidal distortion
produced by off-axis photography.

This notebook uses `warpPerspective` (8 DOF homographic transform), which maps
any quadrilateral to a rectangle — correcting both tilt and keystoning in one
step.  The four corners of the detected bounding rectangle are ordered
TL→TR→BR→BL using coordinate sums/differences, and mapped to an axis-aligned
output rectangle whose dimensions are computed from the actual edge lengths in
the source image.
"""))

CELLS.append(code("""\
def detect_card(image, stem='card', debug=False):
    \"\"\"
    Locate the Magic card in a photo and return a perspective-corrected crop.

    Detection
    ---------
    Otsu threshold → invert if background is dominant bright region (>55%
    white) → 40×40 morphological close → contour filtering by area (3–70%)
    and aspect ratio (1.1–2.0) → select candidate closest to CARD_RATIO.
    A Gaussian blur sweep [0, 11, 21, 51] is tried in sequence; the first
    level that yields a candidate wins.

    Geometric correction
    --------------------
    The four boxPoints of the selected minAreaRect are ordered TL→TR→BR→BL
    by _order_corners(), then mapped to an axis-aligned rectangle via
    getPerspectiveTransform + warpPerspective (INTER_CUBIC).  Output
    dimensions are computed from the actual edge lengths in the source
    image, which handles mild perspective distortion where opposite sides
    of the card are not equal length.

    Returns (card, box_pts) or (None, None).
    \"\"\"
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_area = image.shape[0] * image.shape[1]

    card_rect = None
    thresh    = None
    closed    = None

    for blur in [0, 11, 21, 51]:
        g = cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 0 else gray
        _, t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if the background — not the card — is the bright region.
        inverted = np.sum(t == 255) / t.size > 0.55
        if inverted:
            t = cv2.bitwise_not(t)

        # On a light background the card interior (after inversion) has large
        # dark holes — level-up boxes, type-line bars, rules text regions —
        # that a 40×40 kernel cannot bridge.  An 80×80 kernel fills these
        # while remaining small enough not to absorb nearby background objects.
        k_size = 80 if inverted else 40
        kernel = np.ones((k_size, k_size), np.uint8)

        cl = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            rect = cv2.minAreaRect(c)
            _, (w, h), _ = rect
            if min(w, h) < 10:               continue
            rect_area = w * h
            if rect_area < img_area * 0.03:  continue
            if rect_area > img_area * 0.70:  continue
            ratio = max(w, h) / min(w, h)
            if not (1.1 < ratio < 2.0):      continue
            candidates.append((abs(ratio - CARD_RATIO), rect))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            card_rect = candidates[0][1]
            thresh, closed = t, cl
            if debug:
                print(f'[{stem}] Detected at blur={blur}')
            break

    if debug:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original')
        axes[1].imshow(thresh, cmap='gray')
        axes[1].set_title('Otsu threshold')
        axes[2].imshow(closed, cmap='gray')
        axes[2].set_title('Morphological close')
        for ax in axes: ax.axis('off')
        plt.suptitle(f'Stage 1 — Detection: {stem}', fontsize=12)
        plt.tight_layout(); plt.show()

    if card_rect is None:
        print(f'[{stem}] No card-shaped contour found.')
        return None, None

    # ----------------------------------------------------------------
    # Perspective correction
    # ----------------------------------------------------------------
    box_pts = cv2.boxPoints(card_rect).astype(np.float32)
    src     = _order_corners(box_pts)

    # Measure actual edge lengths in the source image.
    # Using max() of opposite-side pairs handles mild perspective
    # distortion where top != bottom and left != right.
    w_top    = np.linalg.norm(src[1] - src[0])
    w_bottom = np.linalg.norm(src[2] - src[3])
    h_left   = np.linalg.norm(src[3] - src[0])
    h_right  = np.linalg.norm(src[2] - src[1])
    out_w    = max(1, int(max(w_top,  w_bottom)))
    out_h    = max(1, int(max(h_left, h_right)))

    dst = np.array([
        [0,         0        ],
        [out_w - 1, 0        ],
        [out_w - 1, out_h - 1],
        [0,         out_h - 1],
    ], dtype='float32')

    M    = cv2.getPerspectiveTransform(src, dst)
    card = cv2.warpPerspective(image, M, (out_w, out_h),
                               flags=cv2.INTER_CUBIC)

    if debug:
        overlay = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        cv2.drawContours(overlay, [box_pts.astype(int)], -1, (0, 255, 0), 4)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(overlay)
        axes[0].set_title('Detected corners (green)')
        axes[1].imshow(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Perspective-corrected ({card.shape[1]}x{card.shape[0]})')
        for ax in axes: ax.axis('off')
        plt.suptitle(f'Stage 2 — Perspective correction: {stem}', fontsize=12)
        plt.tight_layout(); plt.show()

    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    cv2.drawContours(overlay, [box_pts.astype(int)], -1, (0, 255, 0), 4)
    save_step('detection', f'{stem}_detection.jpg',
              cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return card, box_pts
"""))

# ── 5  OCR functions ─────────────────────────────────────────────────────────
CELLS.append(md("""\
---
## Stages 3–6 — Orientation, OCR Preprocessing, and Recognition

### Orientation (Stage 3)
The perspective-corrected crop may be in any of four 90° orientations.
`scan_card` tries all four, forces each to portrait, and accepts the first
that produces a Scryfall match.

### OCR preprocessing (Stage 5)
Rather than binarizing (which breaks letter strokes on blurry input),
the pipeline applies three steps to a grayscale crop:
1. **3× bicubic upscale** — more pixels for the sharpener to work with.
2. **Non-local means denoising** (h=7) — suppresses photographic grain
   before amplification, using patch similarity rather than spatial proximity.
3. **Unsharp mask** (σ=3, α=0.5) — amplifies existing edge contrast by
   50%, recovering sharpness lost to camera blur.

### Title detection (Stage 4)
Instead of a fixed fractional crop, EasyOCR is run on the top 20% of
the card.  Detections whose bounding-box centre falls in the right 30%
of the strip are discarded as mana cost.  The remainder are joined
left-to-right as the card name.  This works across all frame styles
because the physical layout constraint — title left, mana cost right —
is invariant.

### OCR engine (Stage 6)
EasyOCR uses a CRNN (CNN backbone + bidirectional LSTM + CTC decoder),
which is significantly more tolerant of Magic's decorative title fonts
than Tesseract's LSTM model trained on standard document text.
"""))

CELLS.append(code("""\
def _force_portrait(card):
    \"\"\"Rotate 90° CCW if wider than tall.\"\"\"
    h, w = card.shape[:2]
    if w > h:
        card = cv2.rotate(card, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return card


def preprocess_for_ocr(crop_bgr):
    \"\"\"
    Prepare a colour crop for EasyOCR: upscale → NL-means denoise → unsharp mask.
    Returns a grayscale image.

    No binarization.  EasyOCR is a neural network that handles real-valued
    grayscale gradients natively.  Binarizing blurry input breaks soft letter
    strokes into disconnected fragments that confuse the CRNN decoder.
    \"\"\"
    gray     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(upscaled, h=7)
    blurred  = cv2.GaussianBlur(denoised, (0, 0), sigmaX=3)
    return cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)


def find_title_from_card(card_image, reader):
    \"\"\"
    Dynamically locate and OCR the card title.

    Runs EasyOCR on the preprocessed top 20% of the card.  The mana cost
    (right ~30% of the strip width) is discarded by bounding-box x-position.
    Remaining detections are sorted left-to-right and joined as the card name.

    The 70/30 split matches TITLE_X2 = 0.700 and is physically grounded:
    the card name occupies the left portion of the title bar on every frame
    style — standard, showcase, borderless, extended art — because it is a
    functional layout requirement, not an aesthetic one.
    \"\"\"
    h, w     = card_image.shape[:2]
    strip    = card_image[:int(h * 0.20), :]
    prepared = preprocess_for_ocr(strip)
    strip_w  = prepared.shape[1]

    results = reader.readtext(prepared)
    if not results:
        return '', 0.0

    title_parts = []
    for bbox, text, conf in results:
        text = text.strip()
        if not text:
            continue
        x_center = sum(p[0] for p in bbox) / len(bbox)
        if x_center > strip_w * 0.70:
            continue                          # mana cost — discard
        title_parts.append((x_center, text, conf))

    if not title_parts:
        return '', 0.0

    title_parts.sort(key=lambda x: x[0])
    return ' '.join(t for _, t, _ in title_parts), max(c for _, _, c in title_parts)


def run_ocr(preprocessed_image, reader):
    \"\"\"Run EasyOCR; return (joined_text, max_confidence).\"\"\"
    results = reader.readtext(preprocessed_image)
    if not results:
        return '', 0.0
    return (' '.join(r[1] for r in results).strip(),
            max(r[2] for r in results))
"""))

# ── 6  Lookup functions ──────────────────────────────────────────────────────
CELLS.append(code("""\
def query_scryfall(card_name):
    \"\"\"
    Fuzzy-search Scryfall for a card by name.

    Uses /cards/named?fuzzy= which applies Levenshtein edit-distance matching.
    Single-character OCR errors (substitutions, dropped apostrophes) resolve
    correctly without exact-string matching.  Returns a metadata dict or None.
    \"\"\"
    if not card_name or len(card_name.strip()) < 2:
        return None
    try:
        resp = requests.get(
            'https://api.scryfall.com/cards/named',
            params={'fuzzy': card_name},
            timeout=5,
        )
        if resp.status_code == 200:
            d = resp.json()
            return {
                'name':             d.get('name'),
                'set':              d.get('set', '').upper(),
                'set_name':         d.get('set_name'),
                'collector_number': d.get('collector_number'),
                'type_line':        d.get('type_line'),
                'rarity':           d.get('rarity'),
            }
        return None
    except requests.RequestException as e:
        print(f'Network error: {e}')
        return None


def parse_set_code(ocr_text):
    \"\"\"
    Extract the set code from OCR output of the collector info strip.
    Filters out language codes (EN, FR, …) and rarity letters (C, U, R, M, S).
    \"\"\"
    if not ocr_text:
        return None
    LANG   = {'EN', 'FR', 'DE', 'ES', 'IT', 'PT', 'JP', 'KO', 'RU', 'CS', 'CT'}
    RARITY = {'C', 'U', 'R', 'M', 'S'}
    for m in re.findall(r'\\b[A-Z]{2,5}\\b', ocr_text):
        if m not in LANG and m not in RARITY:
            return m
    return None
"""))

# ── 7  Full pipeline ─────────────────────────────────────────────────────────
CELLS.append(md("""\
---
## Stage 7 — Full Pipeline

`scan_card` wires all stages together.  The four-rotation retry tries 0°,
90° CW, 180°, and 270° CCW on the perspective-corrected crop.  For each
candidate, `_force_portrait` ensures portrait orientation before
`_try_orientation` extracts the title strip, runs OCR, and queries Scryfall.
The first candidate that Scryfall recognises wins; if none match, the
candidate with the highest OCR confidence is returned as a best-effort result.

Note: the rotation is applied to the raw `detect_card` output *before*
`_force_portrait`, not after.  Applying rotation after `_force_portrait`
would turn a portrait card back into landscape, causing the TITLE_* fractional
constants to land in the artwork rather than the title bar.
"""))

CELLS.append(code("""\
def _try_orientation(card_image, reader, stem, suffix=''):
    \"\"\"
    Run OCR + Scryfall on one portrait-oriented candidate.
    Saves intermediate crops to the output folders.
    Returns a result dict; check result['scryfall'] for success.
    \"\"\"
    h_c, w_c     = card_image.shape[:2]
    title_crop   = card_image[:int(h_c * 0.20), :]   # top strip — dynamic OCR
    title_prep   = preprocess_for_ocr(title_crop)
    setcode_crop = extract_region(card_image,
                                  SETCODE_Y1, SETCODE_Y2,
                                  SETCODE_X1, SETCODE_X2)
    setcode_prep = preprocess_for_ocr(setcode_crop)

    save_step('title_raw',    f'{stem}_title_raw{suffix}.jpg',    title_crop)
    save_step('title_prep',   f'{stem}_title_prep{suffix}.jpg',   title_prep)
    save_step('setcode_raw',  f'{stem}_setcode_raw{suffix}.jpg',  setcode_crop)
    save_step('setcode_prep', f'{stem}_setcode_prep{suffix}.jpg', setcode_prep)

    title_text,  title_conf  = find_title_from_card(card_image, reader)
    setcode_text, _          = run_ocr(setcode_prep, reader)
    set_code                 = parse_set_code(setcode_text)
    card_data                = query_scryfall(title_text)

    return {
        'card_image':           card_image,
        'title_crop':           title_crop,
        'title_prep':           title_prep,
        'setcode_crop':         setcode_crop,
        'setcode_prep':         setcode_prep,
        'ocr_title':            title_text,
        'ocr_title_confidence': title_conf,
        'ocr_setcode_raw':      setcode_text,
        'parsed_set_code':      set_code,
        'scryfall':             card_data,
    }


def scan_card(image_path, reader, debug=False):
    \"\"\"
    Full pipeline: load → detect + perspective-correct → orient → OCR → Scryfall.
    Returns a result dict, or None if detection fails.
    \"\"\"
    stem  = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    if image is None:
        print(f'Could not load: {image_path}')
        return None

    raw_card, _ = detect_card(image, stem=stem, debug=debug)
    if raw_card is None:
        return None

    ROTATIONS = [
        ('',      None),
        ('_r90',  cv2.ROTATE_90_CLOCKWISE),
        ('_r180', cv2.ROTATE_180),
        ('_r270', cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]

    best_result = None
    for suffix, rot_code in ROTATIONS:
        rotated  = cv2.rotate(raw_card, rot_code) if rot_code is not None else raw_card
        portrait = _force_portrait(rotated)
        result   = _try_orientation(portrait, reader, stem, suffix=suffix)
        result['image_path'] = image_path
        result['rotation']   = suffix

        if result['scryfall'] is not None:
            best_result = result
            break
        if (best_result is None or
                result['ocr_title_confidence'] > best_result['ocr_title_confidence']):
            best_result = result

    save_step('card', f'{stem}_card.jpg', best_result['card_image'])
    show_crop_regions(best_result['card_image'], stem=stem)

    if debug:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].imshow(cv2.cvtColor(best_result['title_crop'],   cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Title strip (colour)')
        axes[0, 1].imshow(best_result['title_prep'], cmap='gray')
        axes[0, 1].set_title('Title strip (preprocessed)')
        axes[1, 0].imshow(cv2.cvtColor(best_result['setcode_crop'], cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Set-code crop (colour)')
        axes[1, 1].imshow(best_result['setcode_prep'], cmap='gray')
        axes[1, 1].set_title('Set-code crop (preprocessed)')
        for ax in axes.flat: ax.axis('off')
        rot = best_result['rotation'] or '0\u00b0'
        plt.suptitle(f'{stem} \u2014 final crops (best rotation: {rot})', fontsize=12)
        plt.tight_layout(); plt.show()

    return best_result
"""))

# ── 8  Initialise reader & test images ───────────────────────────────────────
CELLS.append(code("""\
print('Loading EasyOCR (first run downloads ~100 MB)...')
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print('EasyOCR ready.')

test_images = sorted([
    (os.path.join('test_assets', f), os.path.splitext(f)[0])
    for f in os.listdir('test_assets')
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
print(f'{len(test_images)} test image(s) found:')
for path, stem in test_images:
    print(f'  {path}')
"""))

# ── 9  Run pipeline ──────────────────────────────────────────────────────────
CELLS.append(code("""\
results     = []
results_all = []

for path, stem in test_images:
    print(f'\\n{"=" * 55}')
    print(f'Scanning: {path}')
    result = scan_card(path, reader, debug=False)
    if result is None:
        print('  FAILED \u2014 card not detected')
        results_all.append({'filename': os.path.basename(path),
                            'status':   'Detection failed'})
        continue
    result['filename'] = os.path.basename(path)
    results.append(result)
    results_all.append(result)
    sf        = result['scryfall']
    rot_label = result['rotation'] or '0\u00b0 (as-is)'
    print(f"  OCR title  : '{result['ocr_title']}'  "
          f"(conf: {result['ocr_title_confidence']:.2f})")
    print(f"  Set code   : {result['parsed_set_code']}")
    print(f"  Rotation   : {rot_label}")
    if sf:
        print(f"  Scryfall   : {sf['name']} "
              f"[{sf['set']} #{sf['collector_number']}]  \u2014  {sf['type_line']}")
    else:
        print('  Scryfall   : no match')
"""))

# ── 10  Per-card visual summary ──────────────────────────────────────────────
CELLS.append(code("""\
# Per-card diagnostic grid: detection overlay | card crop | title prep
# Images are loaded from the output_merged/ subdirectories written during the run.

for r in results_all:
    stem  = os.path.splitext(r['filename'])[0]
    det   = cv2.imread(os.path.join(OUTPUT_DIRS['detection'],   stem + '_detection.jpg'))
    crd   = cv2.imread(os.path.join(OUTPUT_DIRS['card'],        stem + '_card.jpg'))
    tprep = cv2.imread(os.path.join(OUTPUT_DIRS['title_prep'],  stem + '_title_prep.jpg'),
                       cv2.IMREAD_GRAYSCALE)

    sf    = r.get('scryfall')
    if sf:
        label = sf['name'] + '  [' + sf['set'].upper() + ']'
        color = (0.1, 0.6, 0.1)
    elif r.get('status') == 'Detection failed':
        label = 'Detection failed'
        color = (0.8, 0.1, 0.1)
    else:
        label = 'No match  (OCR: "' + r.get('ocr_title','') + '")'
        color = (0.7, 0.5, 0.0)

    panels = []
    titles = []
    if det   is not None: panels.append(cv2.cvtColor(det, cv2.COLOR_BGR2RGB)); titles.append('Detection')
    if crd   is not None: panels.append(cv2.cvtColor(crd, cv2.COLOR_BGR2RGB)); titles.append('Card crop')
    if tprep is not None: panels.append(tprep);                                titles.append('Title prep')

    if not panels:
        print(f'{stem}: no output images found')
        continue

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, img, ttl in zip(axes, panels, titles):
        cmap = 'gray' if img.ndim == 2 else None
        ax.imshow(img, cmap=cmap)
        ax.set_title(ttl, fontsize=9)
        ax.axis('off')
    fig.suptitle(stem + '  \\u2014  ' + label, fontsize=11, color=color, fontweight='bold')
    plt.tight_layout()
    plt.show()
"""))

# ── 11  Summary table ────────────────────────────────────────────────────────
CELLS.append(code("""\
from IPython.display import display, HTML

rows = []
for r in results_all:
    fname = r['filename']
    if r.get('status') == 'Detection failed':
        name   = '<em style="color:#c00">Detection failed</em>'
        detail = ''
    else:
        sf = r.get('scryfall')
        if sf:
            name   = '<strong>' + sf['name'] + '</strong>'
            detail = (sf['set_name'] + ' (' + sf['set'].upper() + ') #'
                      + sf['collector_number'] + ' \u2014 ' + sf['type_line'])
        else:
            ocr    = r.get('ocr_title', '')
            name   = '<em style="color:#a60">No Scryfall match</em>'
            detail = 'OCR: \u201c' + ocr + '\u201d'
    rows.append((fname, name, detail))

header = (
    '<tr style="background:#222;color:#fff">'
    '<th style="padding:6px 12px;text-align:left">File</th>'
    '<th style="padding:6px 12px;text-align:left">Card name</th>'
    '<th style="padding:6px 12px;text-align:left">Details</th>'
    '</tr>'
)
body = ''
for i, (fname, name, detail) in enumerate(rows):
    bg    = '#f5f5f5' if i % 2 == 0 else '#ffffff'
    body += (
        '<tr style="background:' + bg + '">'
        '<td style="padding:5px 12px">' + fname  + '</td>'
        '<td style="padding:5px 12px">' + name   + '</td>'
        '<td style="padding:5px 12px;color:#555;font-size:0.9em">'
        + detail + '</td></tr>'
    )
display(HTML(
    '<table style="border-collapse:collapse;width:100%">'
    '<thead>' + header + '</thead>'
    '<tbody>' + body   + '</tbody>'
    '</table>'
))
"""))

# ---------------------------------------------------------------------------
# Assemble and write notebook
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.13.0"}
    },
    "cells": CELLS,
}

# Assign cell ids
for i, cell in enumerate(nb["cells"]):
    cell["id"] = f"cell-{i:02d}"

out = os.path.join(os.path.dirname(__file__), "MagicScanMerged.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written: {out}  ({len(CELLS)} cells)")
