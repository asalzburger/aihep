import numpy as np

from flavor_tagging.evaluate import evaluate_model
from flavor_tagging.train import load_checkpoint, save_checkpoint, train_model


def test_train_model_runs_and_learns_the_synthetic_signal(synthetic_reco_run):
    tracks, clusters = synthetic_reco_run
    model, preprocessing, history = train_model(
        tracks, clusters, epochs=30, batch_size=16, val_fraction=0.3, lr=5e-3, device="cpu", seed=0
    )

    assert len(history.train_loss) == 30
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.val_accuracy[-1] > 0.85  # trivially-separable synthetic signal
    assert preprocessing["n_track_slots"] > 0
    assert len(preprocessing["feature_names"]) == model.n_features


def test_save_and_load_checkpoint_round_trip(synthetic_reco_run, tmp_path):
    tracks, clusters = synthetic_reco_run
    model, preprocessing, _history = train_model(tracks, clusters, epochs=2, batch_size=16, device="cpu", seed=0)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, model, preprocessing)

    loaded_model, loaded_preprocessing = load_checkpoint(checkpoint_path, device="cpu")

    assert loaded_model.n_features == model.n_features
    assert loaded_preprocessing["n_track_slots"] == preprocessing["n_track_slots"]
    assert loaded_preprocessing["feature_names"] == preprocessing["feature_names"]
    np.testing.assert_allclose(loaded_preprocessing["mean"], preprocessing["mean"])
    np.testing.assert_allclose(loaded_preprocessing["std"], preprocessing["std"])


def test_evaluate_model_generalizes_to_an_independent_dataset(synthetic_reco_run, independent_synthetic_reco_run):
    train_tracks, train_clusters = synthetic_reco_run
    test_tracks, test_clusters = independent_synthetic_reco_run

    model, preprocessing, _history = train_model(
        train_tracks, train_clusters, epochs=30, batch_size=16, val_fraction=0.3, lr=5e-3, device="cpu", seed=0
    )
    result = evaluate_model(model, preprocessing, test_tracks, test_clusters)

    assert result["confusion_matrix"].shape == (2, 2)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert result["accuracy"] > 0.8  # same clean signal, independently sampled
    assert len(result["score"]) == len(result["dataset"].is_b_jet)
