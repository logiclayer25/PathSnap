"""Histogram gradient-boosted candidate scorer for PathSnap."""
from sklearn.ensemble import HistGradientBoostingClassifier

MODEL_NAME = 'hist_gradient_boosting'


def build_model(seed: int = 42):
    """Histogram boosting configuration used in the five-model comparison."""
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08,
        max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)


if __name__ == '__main__':
    from experiments import run_cli
    run_cli([MODEL_NAME])
