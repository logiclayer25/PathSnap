"""Logistic-regression candidate scorer for PathSnap.

Run independently with the same feature generation, trip-grouped CV and metrics as
experiments.py: ``python logistic_regression.py``.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAME = 'logistic_regression'


def build_model(seed: int = 42):
    """Balanced linear probabilistic classifier; scaler is fit within each CV fold."""
    return make_pipeline(StandardScaler(), LogisticRegression(
        class_weight='balanced', max_iter=250, random_state=seed))


if __name__ == '__main__':
    from experiments import run_cli
    run_cli([MODEL_NAME])
