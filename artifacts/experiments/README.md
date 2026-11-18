# Experiment run notes

The run `comparison-20261118T050925Z` is retained for reproducibility but is superseded. Its first implementation treated candidate-miss observations as all-negative candidate sets during fit. The corrected `experiments.py` masks those observations out of classifier fitting and candidate-level loss, while keeping them in observation-level accuracy and candidate-miss reporting. Use the later timestamped comparison run for stage-4 results.
