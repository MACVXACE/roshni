# Pre-registration: Roshni v2, more cameras in training

**Written and committed before the v2 model was trained.** The Git commit that adds this file is the timestamp. Nothing below will be changed after v2's results are seen. Results that come out differently from what we hoped are reported in full.

## Why

The v1 model (trained on APTOS 2019 only, one Indian provider) ranked the unseen DeepDRiD hospital photos well (AUC 0.911). But at its frozen cut-off it caught only 56.1% of referable eyes there. Our reading: the cut-off learned on one source's cameras does not carry over to new cameras.

**Question:** does adding photos from many hospitals and camera models to training and validation make the frozen cut-off carry over better to hospitals the model has never seen?

## What changes (and nothing else)

- **Added data:** the OIA-DDR grading photos (Li et al. 2019): 13,673 photos from 147 hospitals in 23 provinces of China, taken with 42 camera models, graded 0 to 4 plus "ungradable". Licence: CC BY-NC-SA 4.0. Ungradable photos are excluded from grading.
- **Split:** OIA-DDR is split exactly as APTOS was. Near-duplicate photos (256-bit difference hash, at most 10 bits apart) are grouped. Then 7 stratified group folds with seed 0 give one test fold, one validation fold and five training folds. The APTOS split is the same as in v1.
- **Training:** APTOS training split plus OIA-DDR training split.
- **Validation:** APTOS validation plus OIA-DDR validation. The checkpoint (best validation AUC) and the referral cut-off (90% sensitivity) are chosen on this combined set, then frozen.
- **Unchanged:**
  - Model and training: EfficientNet-B0 with ImageNet weights, 384 px, 12 epochs, learning rate 3e-4, batch size 32, same augmentation, seed 0.
  - Targets: 90% sensitivity target, and the quality cut-off of 0.5.
  - Quality check: retrained the same way on DeepDRiD training-photo quality labels.
  - The decision rule.

## Tests (each scored once, after freezing)

1. **Primary: DeepDRiD (all 1,600 graded photos, 400 patients).** These hospitals were never used to train or to set the cut-off.
2. The 400-photo DeepDRiD validation workflow (refer / retake / rescreen, with the quality check).
3. The same 352 APTOS test photos as v1.
4. The OIA-DDR test split. This is new photos from the hospitals in training, so it is reported separately and is not external.
5. The same stress test as v1 (4 changes × 2 strengths).

**Safeguard:** every DeepDRiD photo is hash-checked against the OIA-DDR photos used for training and validation. If any near-copies exist, the primary result is also reported with them removed.

## Outcomes

- **Primary:** sensitivity and specificity on DeepDRiD at v2's frozen cut-off.
- **Secondary:**
  - DeepDRiD AUC;
  - referable eyes given an all-clear in the 400-photo workflow;
  - APTOS test AUC;
  - the stress test;
  - the same post-hoc local re-calibration analysis as v1 (`local_calibration.py`).

## Decision rule for the demo and the headline (fixed now)

v2 replaces v1 in the browser demo and the README headline **only if all five hold**:

1. DeepDRiD sensitivity at the frozen cut-off is **at least 66.1%** (v1's 56.1% plus 10 points).
2. DeepDRiD specificity is **at least 85.0%**.
3. APTOS test AUC is **at least 0.950**.
4. Referable eyes given an all-clear in the 400-photo workflow, with the quality check, are **at most 46** (v1's number).
5. Fewer than 1% of DeepDRiD photos (fewer than 16 of 1,600) have a near-copy among the OIA-DDR photos used for training or validation. Otherwise DeepDRiD is no longer unseen, so v2 cannot be promoted, whatever its numbers.

If any of these fails, v1 stays the headline and v2 is reported in full as a tested idea that did not meet its bar. Either way, both models' numbers are published side by side.

## What would not count as evidence

- **The OIA-DDR test result.** It is the same hospitals as training.
- **Any threshold chosen after looking at DeepDRiD.**
- **One more caution:** both OIA-DDR and DeepDRiD come from China. An improvement on DeepDRiD would show the value of camera variety, not that the model works in Pakistan. Testing on Pakistani clinic photos remains the next step.
