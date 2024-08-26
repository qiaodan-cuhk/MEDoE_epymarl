from .joint_mlp_class import JointMLPClassifier
from .mlp_class import MLPClassifier
from .llm_class import LLMlassifier

REGISTRY = {}

REGISTRY["mlp"] = MLPClassifier
REGISTRY["joint_mlp"] = JointMLPClassifier
REGISTRY["llm_mlp"] = LLMlassifier


def doe_classifier_config_loader(cfg, ids):
    type = cfg.type
    cls = REGISTRY[type]
    if not hasattr(cls, "from_config"):
        raise NotImplementedError(f"There is no from_config method defined for {type}")
    else:
        return cls.from_config(ids, cfg)
