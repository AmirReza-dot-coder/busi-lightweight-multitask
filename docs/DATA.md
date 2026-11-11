# Data and fixed split

Original BUSI: 780 images (normal 133, benign 437, malignant 210). The prepared cohort excludes 19 documented ambiguous/conflicting images and contains 761 images (normal 131, benign 427, malignant 203).

| Split | normal | benign | malignant | Total |
|---|---:|---:|---:|---:|
| Train | 91 | 281 | 127 | 499 |
| Validation | 16 | 64 | 35 | 115 |
| Historical test | 24 | 82 | 41 | 147 |

`split_manifest.csv` is a byte-for-byte copy from the closure export. SHA256:

```
cafadff2cb11e0eaebe0eb428bba3768a5c179f3702341c3021f314556ef2d3d
```

The cohort contains 582 similarity groups. No registered group crosses the three splits. Similarity groups are not verified patient identifiers. There are 117 test groups. Three seeds reuse the same split and images.

Expected raw structure: `<root>/<normal|benign|malignant>/images/<name>.png` and corresponding `masks/<name>_mask.png`. Multiple masks are merged by positive-region union; normal reference masks are empty. Raw data are external to the repository.

Class mapping: normal=0, benign=1, malignant=2. Images are resized to 224×224 with PIL bilinear interpolation; reference masks use nearest-neighbor. The historical loader computes/uses the saved training Z-score; the controlled ST/MTL study reverses that stored normalization and applies the MobileNetV2 [-1,1] preprocessing once inside its model. See per-study settings rather than applying both transformations twice.

Argmax selects the class. Hard masks use threshold 0.5. Positive Dice/IoU are averaged per lesion-containing image; normal false foreground area is reported separately. Test has 123 lesion-containing images; validation has 99.

For exact curation code, inspect `notebooks/01_BUSI_Data_Preparation_Clean.ipynb`; earlier manual steps are retained under `legacy/`. Recreating only the split percentages does not reproduce the fixed membership.
