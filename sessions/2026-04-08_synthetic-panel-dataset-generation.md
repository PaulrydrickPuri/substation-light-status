# Session Log — 2026-04-08
**Project:** Substation Light Panel Status Detection
**Working directory:** `/Users/paulrydrickpuri/Documents/Claude/Projects/lightpanelstatus`
**Model:** Claude Sonnet 4.6

---

## Summary

This session established the full data pipeline for a YOLO-based electrical switchgear panel light status detector. Starting from a single cowork instruction markdown spec, a Python script was written to generate 300 synthetic CCTV-style panel images with balanced YOLO annotations, the dataset was reorganised into a COCO-style train/val/test split, and a real labelled image from the actual deployment camera was duplicated 50 times into the training set. ML training configuration was recommended and revised twice — once after reviewing the real CCTV image to account for the sim-to-real domain gap, and again after confirming the camera is physically angled. The session closed by creating a reusable `session-github-journal` Claude Code skill for future progress keepsakes.

---

## What Was Built

### 1. Synthetic Panel Dataset Generator
- **Script:** `generate_panel_dataset.py`
- **Location:** `/Users/paulrydrickpuri/Documents/Claude/Projects/lightpanelstatus/`
- **Output:** `~/Desktop/panel_dataset/images/` (300 × PNG) + `~/Desktop/panel_dataset/labels/` (300 × TXT)
- Simulates a 3-bay ABB electrical switchgear cabinet viewed from a CCTV camera
- Canvas size: 780 × 460 px
- Each image contains two active light panels (LEFT bay: radius 13px, CENTER bay: radius 15px)
- 7 light slots per bay; slot 5 (LAMP TEST P/B) excluded from annotation
- 6 labelled slots per bay: red, green, yellow, yellow, blue, blue
- Each bulb drawn with 5-layer bezel + radial glow (ON) or faint highlight (OFF)
- CCTV overlay: randomised timestamp `2026-04-08 HH:MM:SS` (hour 08–17) top-left, `HOLOWITS` top-right
- Per-image augmentation: ±6% brightness shift, 0–4% warm tint
- Built using only `Pillow`, `os`, `random`, `math`

### 2. YOLO Annotation Files
- 14 lines per `.txt` file: 2 `light_panel` bboxes + 12 individual light bboxes (6 per bay)
- YOLO normalised format: `<class_id> <cx> <cy> <w> <h>` (6 decimal places)
- Bounding box half-size = bulb_radius + 5 (square bbox per bulb)

### 3. Balanced Class Pool (CRITICAL design)
- For each of the 6 labelled positions: list of 300 values — exactly 150 ON + 150 OFF, independently shuffled
- Separate pools for LEFT bay and CENTER bay
- Guarantees exact class balance across 300 images

### 4. COCO-Style Dataset Split
- **Script:** `convert_to_coco_splits.py`
- **Output:** `~/Desktop/panel_dataset_coco/`
- Split mirrors reference COCO carplate dataset ratio (~80/10/10)
- train: 240 images+labels | val: 30 | test: 30
- Shuffle seed: 42 (reproducible)

### 5. `data.yaml`
- **Location:** `~/Desktop/panel_dataset_coco/data.yaml`
- Modelled on retail detection reference yaml format
- `nc: 9`, class names: `[light_panel, red_on, red_off, green_on, green_off, yellow_on, yellow_off, blue_on, blue_off]`
- Relative paths: `../train/images`, `../val/images`, `../test/images`

### 6. Real Image Integration into VisionSamurai Dataset
- **Target dataset:** `(YOLO)light panel status model v1(08Apr2026-15_47_15)`
- Source: 1 real labelled CCTV image (`8372c656-594f-4b59-978c-92337e35f2f4.png`) from `val/`
- 50 UUID-named copies written to `train/images/` + `train/labels/`
- 1 UUID-named copy written to `test/images/` + `test/labels/`
- UUID format matches VisionSamurai naming convention

### 7. `session-github-journal` Claude Code Skill
- **Location:** `~/.claude/plugins/cache/claude-plugins-official/skill-creator/unknown/skills/session-github-journal/SKILL.md`
- Reusable skill: reviews session, writes structured markdown log, pushes to GitHub
- Output format: `sessions/YYYY-MM-DD_<slug>.md` + `PROGRESS.md` index

---

## Key Decisions & Rationale

| Decision | Choice Made | Why |
|---|---|---|
| Dataset size | 300 synthetic images | Balanced coverage: 150 ON + 150 OFF per class per bay = exact balance without over-collecting |
| Class balance method | Pre-shuffled balanced pools per position | Guarantees exact counts at generation time rather than relying on random chance |
| COCO split ratio | 240 / 30 / 30 (80/10/10) | Mirrors reference carplate COCO dataset (3556/437/388) — consistent with existing workflow |
| Real images → train only | All 50 real copies go to train | With only 50 images, every real domain anchor belongs in training; val/test remain synthetic |
| Perspective augmentation | ✅ Kept in training config | Real camera is physically angled left; synthetic images are frontal — perspective aug bridges the gap |
| Saturation augmentation | ❌ Excluded | Colour is the class discriminator (red_on vs green_on); desaturating an ON light can make it look OFF |
| HSV hue shift | ≤ ±3° only | Larger hue shifts risk red→orange or green→yellow confusion which maps to wrong class labels |
| Flip L/R | ❌ Excluded | Panel bay positions are spatially fixed — flipping moves center bay to left side, breaking spatial annotations |
| Architecture | yolov8m (upgraded from yolov8s) | Domain gap between synthetic flat circles and real 3D bulbs with reflections requires stronger feature extraction |
| LR Scheduler | WarmupCosineLR | Smoother decay better suited for fine-tuning on small mixed dataset; avoids harsh drops of MultiStepLR |

---

## Findings & Observations

1. **Class balance is exactly met**: Across 300 images × 2 bays = 600 panel instances: `light_panel=600`, `red_on=300`, `red_off=300`, `green_on=300`, `green_off=300`, `yellow_on=600`, `yellow_off=600`, `blue_on=600`, `blue_off=600`. Balance check: PASSED.

2. **Domain gap is significant**: The real CCTV image shows physically 3D bulbs with lens reflections, real shadows, and perspective distortion. Synthetic images are flat circles on a flat canvas. This is a textbook sim-to-real transfer problem.

3. **"Overfitting to the view" is desirable here**: Since the deployment camera is physically fixed to the wall and will always produce the same framing, training exclusively on this view is correct. The model should memorise the exact pixel geometry of the bulb positions. The only variation at inference time is which lights are ON/OFF.

4. **The 300 synthetic + 50 real split has a clear division of labour**: Synthetic covers *what to detect* (all ON/OFF combinations); real covers *what it looks like* (actual camera optics, panel texture, 3D appearance). This is a sound sim-to-real design.

5. **Real images are mostly static (one light state)**: The 50 real duplicates are all identical frames — they don't provide state variation. This is acceptable because the 300 synthetic images handle all state combinations. The real images serve purely as domain appearance anchors.

6. **Pillow font size ≥ 8px required**: Initial script used font size 6px which caused `OSError: division by zero` in Pillow's FreeType renderer. Fixed by increasing to 8px minimum.

7. **VisionSamurai dataset already had UUID naming**: The existing labelled image uses UUID v4 format. Duplication script used Python's `uuid.uuid4()` to match this convention exactly.

8. **YOLO label file has 14 lines per image**: 2 `light_panel` bboxes (class 0) + 12 individual light bboxes (6 per bay × 2 bays). Slot 5 (LAMP TEST P/B) is excluded from annotation per spec.

---

## Problems & Fixes

| Problem | Root Cause | Fix Applied |
|---|---|---|
| `OSError: division by zero` in Pillow | Font size set to 6px — below FreeType minimum render size | Changed label strip font from 6px to 8px |
| Glow effect needed alpha compositing | `ImageDraw` doesn't support per-pixel alpha; radial glow requires RGBA blending | Used `Image.new("RGBA")` + `alpha_composite()` for each glow layer |

---

## ML / Training Config (Recommended)

```
PROJECT  : Substation Light Panel Status Detection
TASK     : Object Detection (fixed-camera, sim-to-real)
DATASET  : ~350 total (300 synthetic + 50 real)

▸ MODEL
  Architecture : yolov8m
  Fine-Tune    : ✅ Yes (COCO pretrained)

▸ DATA SETTINGS
  Input Shape : 640
  Batch Size  : 8

▸ TRAINING SETTINGS
  Base LR : 0.0001
  Epochs  : 250
  Patience: 40

▸ SOLVER SETTINGS
  Warmup Epochs       : 7
  Evaluation Interval : 5
  Event Interval      : 5
  Checkpoint Interval : 10

▸ OPTIMIZER
  Type              : SGD
  Weight Decay      : 0.0001
  Nesterov Momentum : ✅
  Gradient Clipping : ✅

▸ LR SCHEDULER
  Type    : WarmupCosineLR
  Final LR: 0

▸ DATA SAMPLING
  Sampler : RepeatFactorTrainingSampler

▸ AUGMENTATION
  ✅ Brightness   (0.7–1.4,    55%)   CCTV lighting variance
  ✅ Contrast     (0.7–1.3,    55%)   camera exposure shifts
  ✅ Blur         (0.8–1.2,    50%)   CCTV defocus/compression
  ✅ Shadow       (50%)               indoor overhead shadows
  ✅ Scale        (-0.15–0.15, 35%)   camera distance variation
  ✅ Perspective  (0.05–0.12,  45%)   bridges flat synthetic → angled real camera
  ✅ Dropout      (0.05–0.1,   35%)   partial occlusion simulation
  ✅ Defocus      (0.8–1.2,    35%)   real lens depth-of-field
  ✅ Lighting     (0–180,      40%)   fluorescent overhead patches
  ✅ HSV          (Hue ±3, Sat ±10, Val ±20, 45%)

  ❌ Flip L/R, Flip U/D, Rain, Fog, Snow, Sunflare, Motion Blur,
     Saturation (standalone), Rotation > 5°

⚠️  COLOUR WARNING: HSV Hue ≤ ±3° — any more risks red→orange or
    green→yellow class confusion. Do NOT enable standalone Saturation.
```

---

## Files Created / Modified

| File | Location | Purpose |
|---|---|---|
| `generate_panel_dataset.py` | `.../lightpanelstatus/` | Generates 300 synthetic panel images + YOLO labels |
| `convert_to_coco_splits.py` | `.../lightpanelstatus/` | Reorganises flat dataset into train/val/test COCO splits |
| `panel_dataset/images/` | `~/Desktop/` | 300 synthetic PNG images (780×460px) |
| `panel_dataset/labels/` | `~/Desktop/` | 300 YOLO annotation TXT files (14 lines each) |
| `panel_dataset_coco/train/images/` | `~/Desktop/` | 240 training images |
| `panel_dataset_coco/train/labels/` | `~/Desktop/` | 240 training labels |
| `panel_dataset_coco/val/images/` | `~/Desktop/` | 30 validation images |
| `panel_dataset_coco/val/labels/` | `~/Desktop/` | 30 validation labels |
| `panel_dataset_coco/test/images/` | `~/Desktop/` | 30 test images |
| `panel_dataset_coco/test/labels/` | `~/Desktop/` | 30 test labels |
| `panel_dataset_coco/data.yaml` | `~/Desktop/` | YOLO dataset config (nc=9, 9 class names) |
| `train/images/` (×50 UUIDs) | `(YOLO)light panel status model v1.../` | 50 real image copies for domain anchoring |
| `train/labels/` (×50 UUIDs) | `(YOLO)light panel status model v1.../` | Matching annotation copies |
| `test/images/` (×1 UUID) | `(YOLO)light panel status model v1.../` | 1 real image for test split |
| `test/labels/` (×1 UUID) | `(YOLO)light panel status model v1.../` | Matching annotation |
| `session-github-journal/SKILL.md` | `~/.claude/plugins/.../skills/` | Reusable session logging skill |

---

## Next Steps / Open Questions

- [ ] Upload `panel_dataset_coco/` to VisionSamurai project for training
- [ ] Merge synthetic COCO dataset with real labelled dataset (50 real + 300 synthetic → combined train split)
- [ ] Train yolov8m with recommended config on VisionSamurai platform
- [ ] Evaluate on real CCTV stream — check if sim-to-real gap is handled by perspective/blur aug
- [ ] Collect more real images at different times of day (morning vs afternoon lighting) to diversify the 50 real anchor images
- [ ] Consider whether the right bay (partial, class 0 only) needs a separate annotation pass
- [ ] Validate that slot 5 (LAMP TEST P/B) exclusion is correct per site requirements
- [ ] Open question: Should ON vs OFF be inferred from position (always same slot = always same colour) or purely from pixel appearance? Current design assumes appearance-only — position is implicit in the bbox location.

---

## Session Metadata
- **Date:** 2026-04-08
- **Model:** Claude Sonnet 4.6 (`claude-sonnet-4-6`)
- **Working directory:** `/Users/paulrydrickpuri/Documents/Claude/Projects/lightpanelstatus`
- **Key packages used:** `Pillow`, `os`, `random`, `math`, `shutil`, `uuid`
- **Reference dataset:** `(COCO)Carplate classification(26Mar2026-17_13_00)(pt1)` — used for split ratio
- **VisionSamurai project:** `5129c149-2ae1-46b8-9bff-952f7bd8b729`
- **GitHub repo:** `https://github.com/PaulrydrickPuri/substation-light-status`
