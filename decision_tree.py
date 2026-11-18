"""Decision-tree candidate scorer for PathSnap."""
from sklearn.tree import DecisionTreeClassifier

MODEL_NAME = 'decision_tree'


def build_model(seed: int = 42):
    """Capacity-limited, class-balanced tree with a fixed reproducible seed."""
    return DecisionTreeClassifier(max_leaf_nodes=64, min_samples_leaf=20,
                                  class_weight='balanced', random_state=seed)


if __name__ == '__main__':
    from experiments import run_cli
    run_cli([MODEL_NAME])
