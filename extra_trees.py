"""Extra Trees ensemble candidate scorer for PathSnap."""
from sklearn.ensemble import ExtraTreesClassifier

MODEL_NAME = 'extra_trees'


def build_model(seed: int = 42):
    """Randomized tree ensemble; fixed size/depth controls runtime and complexity."""
    return ExtraTreesClassifier(n_estimators=40, max_depth=14, min_samples_leaf=5,
                                class_weight='balanced', n_jobs=-1, random_state=seed)


if __name__ == '__main__':
    from experiments import run_cli
    run_cli([MODEL_NAME])
