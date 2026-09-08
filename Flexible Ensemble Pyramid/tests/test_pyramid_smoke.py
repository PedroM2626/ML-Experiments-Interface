"""Smoke tests for Flexible Ensemble Pyramid (fast, no full training)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flexible_ensemble_pyramid as fep


def test_clean_tweet_removes_url_mentions():
    out = fep.clean_tweet("Hello @user check http://x.com #Tag!")
    assert "http" not in out
    assert "@user" not in out
    assert "hello" in out
    assert fep.clean_tweet(None) == ""
    assert fep.clean_tweet(123) == ""


def test_available_base_models_subset():
    models = fep.get_available_base_models()
    for expected in ["lr", "svc", "nb", "ridge", "rf", "et", "ada"]:
        assert expected in models


def test_get_model_factory_light():
    for name in ["lr", "nb", "rf"]:
        m = fep.get_model(name, seed=42, jitter=False)
        assert hasattr(m, "fit")
        assert hasattr(m, "predict")


def test_set_seed_reproducible():
    fep.set_seed(123)
    import numpy as np

    a = np.random.rand(3)
    fep.set_seed(123)
    b = np.random.rand(3)
    assert (a == b).all()


def test_rl_suggest_models_tmp(tmp_path):
    rl = fep.RLMetaLearner(knowledge_path=str(tmp_path / "rl.json"), epsilon=1.0)
    models = fep.get_available_base_models()
    picked = rl.suggest_models(1, models, min_p=2, max_p=3)
    assert 2 <= len(picked) <= 3
    assert set(picked) <= set(models)


def test_rl_update_knowledge_tmp(tmp_path):
    path = tmp_path / "rl.json"
    rl = fep.RLMetaLearner(knowledge_path=str(path), epsilon=0.0)
    rl.update_knowledge(
        [{"layer": 1, "model": "lr_L1", "f1": 0.8, "accuracy": 0.8, "precision": 0.8, "recall": 0.8}]
    )
    assert path.exists()
    assert rl.knowledge["runs"] == 1
    assert "1" in rl.knowledge["layer_stats"]


def test_nas_controller_init():
    nas = fep.NASController(population_size=4, generations=2)
    assert nas.population_size == 4
    assert nas.generations == 2
