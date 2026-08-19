# Drift-Weighted-Graph-Attention-Networks-for-Botnet-Detection

All experiment scripts use a 70%/15%/15% train/validation/test split.

- Random-split experiments use two-stage stratified sampling with a fixed seed.
- The chronological experiment keeps three non-overlapping time ranges: the
  first 70% for training, the next 15% for validation, and the final 15% for
  testing.
- Missing-value statistics and feature scaling are fitted on the training set
  only, then applied to validation and test data.
