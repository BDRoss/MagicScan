# MagicScan — Original vs. Merged: Algorithm Comparison

**Course level:** CSCI 3000 / Image Processing
**Prerequisite assumed knowledge:** digital image fundamentals, basic probability and statistics, neural networks at an introductory level.

---

## Context

The original working implementation is `FinalProject/ExperimentFinal.ipynb`, submitted as a course final project. It achieved a 28% success rate (7 of 25 test images) and represents the genuine baseline for this comparison. Three earlier attempts in the main repository (`CardOrient.ipynb`, `SquaringTheCard.ipynb`, `CardTest.ipynb`) never produced working output and are not used as the baseline here.

The current implementation is `MagicScanMerged.ipynb`, which combines the strongest algorithmic ideas from the original and from an intermediate revised version (`MagicScanRevised.ipynb`).

---

## The Central Architectural Difference

Before going stage by stage, it is worth naming the single biggest structural difference between the two pipelines.

The **original** targets the **title bar directly**. It runs edge detection on the full photo, finds all rectangular contours in the scene, and filters them by the title bar's own aspect ratio — approximately 11.5:1 (wide and very short). The full card is never detected or cropped. The pipeline jumps straight from the raw photo to a warped, isolated title bar.

The **merged** targets the **full card first**. It detects the card as an object in the scene, corrects its perspective, crops it out, and *then* extracts the title region from within that crop.

Both approaches are legitimate design choices with different tradeoff profiles, which the stage-by-stage comparison makes concrete.

---

## Pipeline Comparison Table

| Stage | Original | Merged |
|---|---|---|
| Segmentation target | Title bar (aspect ratio ≈ 11.5) | Full card (aspect ratio ≈ 1.40) |
| Edge / region finding | Adaptive threshold → Laplacian → morphological clean | Otsu threshold → luminance inversion → Gaussian blur sweep → morphological close |
| Geometric correction | Full perspective transform (homography) | Full perspective transform (homography) |
| Background variation | None | Luminance inversion (light bg) + Gaussian blur sweep (textured bg) |
| Orientation handling | None | Four-rotation exhaustive search |
| Title localisation | Implicit (the detected contour *is* the title bar) | Dynamic top-strip with spatial mana-cost filter |
| OCR preprocessing | Binary threshold (Otsu) + morphological open | 3× bicubic upscale → NL-means denoise → unsharp mask |
| OCR engine | Tesseract (LSTM) | EasyOCR (CRNN + CTC) |
| Lookup / validation | Exact string match via mtgsdk | Fuzzy edit-distance search via Scryfall |
| Success rate (test set) | 28% (7 / 25) | Higher on development set; not formally measured on the original 25-image set |

---

## Stage 1 — Finding the Region of Interest

### Original: Hunt for the Title Bar

The original treats card identification as a **title bar detection problem** rather than a card detection problem. Its reasoning is direct: you want to read the card name, the card name lives in the title bar, so find the title bar.

The processing chain is:

1. Convert to grayscale.
2. Apply **adaptive mean thresholding** with a 13×13 neighbourhood and a constant offset of 2. Unlike Otsu's global threshold, adaptive thresholding computes a separate threshold for each pixel based on the mean intensity of its local neighbourhood. This compensates for uneven illumination across the image.
3. Apply the **Laplacian operator** with a 13×13 kernel. The Laplacian is a second-order derivative filter that produces large responses at locations of rapid curvature change — i.e., edges and corners. Applying it to the already-thresholded image finds edges within the binary structure rather than in the raw intensity surface, which tends to produce cleaner, more closed contours around rectangular features.
4. Apply a **morphological close** then **open** (both with a 3×3 kernel) to close small gaps and remove noise.

Contours are filtered by area (1,000–1,000,000 pixels) then by aspect ratio (within 1 unit of 11.47, derived from pixel measurements of the title bar: 734 wide, 64 tall).

### Merged: Find the Full Card

The merged pipeline identifies the full card in the scene using **intensity thresholding** rather than edge detection. The core observation is that a Magic card's interior is substantially brighter than most photographed surfaces, producing a bimodal intensity histogram.

**Otsu's method** finds the threshold that maximally separates these two peaks. A 40×40 (or 80×80 on light backgrounds) **morphological close** bridges the dark artwork region that splits the bright interior into disconnected blobs.

The merged pipeline also handles two failure modes the original does not:
- **Light backgrounds**, where Otsu finds the background rather than the card. If more than 55% of the post-threshold image is white, the binary result is inverted.
- **Textured backgrounds**, where fine detail produces enough scattered bright pixels to merge into one image-spanning blob after the close. A **Gaussian blur sweep** at four progressively stronger levels suppresses texture before thresholding.

### Comparison

The original's approach is more elegant in concept: skip the full card entirely and go straight for the text you want. However, the title bar is physically small — perhaps 60–150 pixels tall in a typical phone photo — which makes contour detection at that scale fragile. Any lighting condition or compression artifact that disrupts the title bar's edge continuity causes the contour to be missed or broken.

The merged approach targets a much larger object (the full card occupies 20–50% of the image), making detection significantly more robust. The luminance inversion and blur sweep extend operation to backgrounds the original would fail on entirely, at the cost of adding a cropping step between detection and OCR.

---

## Stage 2 — Geometric Correction

### Original: Full Perspective Transform

For each candidate contour that passes the area filter, the original computes a `minAreaRect` and extracts its four corners as `boxPoints`. It then constructs a **perspective transform** matrix using `getPerspectiveTransform` and applies it with `warpPerspective`. This corrects both in-plane rotation and the converging-lines distortion from off-axis photography simultaneously.

### Merged: Full Perspective Transform

The merged pipeline also uses `getPerspectiveTransform` + `warpPerspective`. The four corners are ordered TL→TR→BR→BL using coordinate sums and differences — a geometric ordering that is independent of the card's content orientation. Output dimensions are computed from actual source edge lengths rather than the `minAreaRect` size, which handles mild distortion where opposite sides of the card are not equal length.

### Comparison

Both pipelines use the correct tool for this problem. The earlier intermediate implementation (`MagicScanRevised.ipynb`) used `warpAffine` (affine transform, 6 DOF), which corrected rotation but not keystone distortion. The merged implementation restores the original's geometric correctness while pairing it with the more robust full-card detection approach.

---

## Stage 3 — Aspect Ratio Filtering

### Original: Filter for the Title Bar

The original checks whether the contour's aspect ratio is within 1 unit of **11.47**. This is a very tight filter. The primary failure mode is the type line, which is also a wide, flat rectangle with a similar aspect ratio. The keyword filter in the validation step ("Creature", "Artifact", "Enchantment", etc.) handles this by detecting when Tesseract has read type-line text — an indirect workaround for a geometric ambiguity.

### Merged: Filter for the Full Card

The merged pipeline filters for aspect ratios between 1.1 and 2.0. The card ratio (1.397) is unique enough in the scene that false positives are uncommon. The type line and title bar shapes do not appear in this range.

---

## Stage 4 — Orientation Handling

### Original: None

The original makes no attempt to handle an upside-down or sideways card. A card photographed in landscape orientation, or flipped, fails with no recovery path.

### Merged: Four-Rotation Exhaustive Search

The merged pipeline tries all four 90° rotations of the corrected crop (0°, 90° CW, 180°, 270° CCW), forces each to portrait orientation, and accepts the first rotation that produces a successful Scryfall match. This handles any orientation the camera was held at.

---

## Stage 5 — OCR Preprocessing

### Original: Binary Threshold + Morphological Open

After cropping the title bar region (full height, left 80% of width to exclude the mana cost), the original converts to grayscale and applies a **binary threshold with Otsu** at a base value of 150. The image is then inverted, a morphological **open** is applied with an 11×11 kernel, and the image is inverted back. The result is a black-text-on-white-background binary image passed to Tesseract.

When the crop is correct, the result is genuinely clean. The problem is that the pipeline to reach that crop is fragile.

### Merged: Upscale → Denoise → Sharpen

The merged pipeline never binarises. It applies:

1. **3× bicubic upscaling** — gives subsequent operations more pixel data.
2. **Non-local means denoising** — suppresses photographic grain while preserving letter edge contrast.
3. **Unsharp masking** — amplifies existing high-frequency edge content by 50%.

The rationale for avoiding binarisation is that EasyOCR operates on real-valued feature maps internally. On blurry input, binarisation breaks soft letter strokes into disconnected fragments.

---

## Stage 6 — OCR Engine

### Original: Tesseract

Tesseract uses an LSTM network but is trained primarily on standard printed documents. Magic card title fonts — Beleren Bold, Matrix Bold — fall outside the statistical distribution of Tesseract's typical training data. It frequently confuses similarly-shaped characters (l/1/I, O/0, rn/m) in these fonts.

### Merged: EasyOCR (CRNN + CTC)

EasyOCR uses a CNN backbone that extracts spatial features, a bidirectional LSTM for contextual dependencies, and a CTC decoder for character alignment. Its training corpus spans many font styles and scene text conditions; Magic card fonts fall within this distribution. EasyOCR also returns bounding box coordinates, which the pipeline uses to spatially filter the mana cost from the title strip.

---

## Stage 7 — Lookup and Validation

### Original: Exact String Match

After Tesseract returns a string, the original queries the mtgsdk API and checks whether `card[0].name == text` exactly. If the OCR output differs by even one character, the lookup returns None. This is the single largest contributor to the 28% success rate — the pipeline produces recognisable output on many cards, but Tesseract's character-level errors convert near-misses into hard failures.

### Merged: Fuzzy Edit-Distance Search

The merged pipeline queries Scryfall's `/cards/named?fuzzy=` endpoint, which applies Levenshtein edit-distance matching. "Fearlezs Pup" (edit distance 1) and "Gods Hall Guardian" (edit distance 1) both resolve correctly. Replacing exact matching with fuzzy matching likely accounts for more improvement in success rate than any single change to the image processing pipeline.

---

## What the Original Got Right

**Geometric correctness.** The original's use of `warpPerspective` is the proper tool for rectifying obliquely-photographed cards. The merged implementation preserves this.

**Directness.** Skipping full card detection and targeting the title bar immediately is an elegant design. Fewer pipeline stages mean fewer things to go wrong, and the intermediate images show that the title bar crops are genuinely clean when the contour detection succeeds.

**Two-stage threshold logic.** Using adaptive thresholding to find edges (global structure) and then Otsu thresholding on the warped crop (local structure) is thoughtful — different thresholding strategies are appropriate at different stages of the pipeline.

---

## What the Merged Implementation Improved

**Robustness of segmentation.** Targeting the full card (a large, salient object) is more reliable than targeting the title bar (small, fragile at typical resolutions). The luminance inversion and blur sweep extend operation to backgrounds the original would fail on entirely.

**Background independence.** The original has no mechanism for light or textured backgrounds. The merged pipeline handles both.

**OCR accuracy.** The CRNN architecture is genuinely better suited to the Magic card title font than Tesseract's LSTM model with its standard English training.

**Fuzzy matching.** The single highest-impact change in the entire pipeline. Near-miss OCR results become correct identifications instead of silent failures.

**Orientation recovery.** The four-rotation retry handles cards photographed at any orientation.

---

## What Neither Handles Well

**Severe blur.** When defocus or motion blur is large enough that the blur radius exceeds character stroke width, adjacent letter strokes merge at the pixel level. No post-processing algorithm can recover this information. This is a physical limitation of the input, not an algorithmic gap.

**Title detection strip size.** The merged pipeline's "dynamic" title detection runs EasyOCR on the top 20% of the card. After upscaling, this produces a very large image where the actual title text occupies a small fraction of the area. EasyOCR's text detector performs poorly under these conditions. A two-pass approach — detect the title bar bounding box first, then OCR on a tight crop — is the correct solution.

**Parameter overfitting.** Several detection parameters (kernel sizes, blur sweep values, inversion threshold) were tuned on a small development set of 8 images. Their generality beyond this set is not validated.

**Foil cards.** Metallic surfaces produce specular reflections that corrupt both thresholding and OCR. Neither pipeline addresses this.

**Dark-bordered showcase and borderless frames.** These have very little bright interior area, challenging the merged pipeline's Otsu-based detection. The original, which targets the title bar by its own edge geometry, is theoretically more robust to frame style variation.

**Type line ambiguity in the original.** The type line and title bar have similar aspect ratios, causing the original to frequently find the type line first. Its keyword blocklist workaround is incomplete.

---

## Summary

The original and merged pipelines are genuinely different architectural choices. The original is more direct (no full-card detection stage), clean in execution when it succeeds, and — after the merge — both share the same geometric correctness (perspective warp). The original fails primarily because its segmentation target is too small and fragile, its OCR engine struggles with the card font, and its exact-string validation turns OCR near-misses into hard failures.

The merged pipeline is more robust at the detection and orientation stages, uses a better-matched OCR engine, handles arbitrary camera orientations, and recovers gracefully from partial OCR errors via fuzzy matching. Its main remaining gaps relative to a fully general solution are the title detection strip size problem and the presence of hardcoded parameters tuned to a small test set.

The most instructive path forward is a ground-up redesign that targets the card's **black border** as the invariant detection feature — present on every card regardless of frame style or background — combined with a two-pass title extraction strategy that first localises the title bar and then runs OCR on a tight crop.
