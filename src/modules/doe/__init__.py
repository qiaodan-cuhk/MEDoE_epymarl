from .joint_mlp_class import JointMLPClassifier
from .mlp_class import MLPClassifier

REGISTRY = {}

REGISTRY["mlp"] = MLPClassifier
REGISTRY["joint_mlp"] = JointMLPClassifier


def doe_classifier_config_loader(cfg, ids):
    type = cfg.type
    registry = {
        "MLP": MLPClassifier,
        }
    cls = registry[type]
    if not hasattr(cls, "from_config"):
        raise NotImplementedError(f"There is no from_config method defined for {type}")
    else:
        return cls.from_config(ids, cfg)
