"""Gaussian Naive Bayes candidate scorer for PathSnap."""
from sklearn.naive_bayes import GaussianNB

MODEL_NAME = 'gaussian_naive_bayes'


def build_model(seed: int = 42):
    """Gaussian likelihood model with a small variance floor for stable probabilities."""
    del seed
    return GaussianNB(var_smoothing=1e-8)


if __name__ == '__main__':
    from experiments import run_cli
    run_cli([MODEL_NAME])
