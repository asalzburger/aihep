from multiplicity.evaluate import evaluate_model
from multiplicity.train import load_checkpoint, save_checkpoint, train_model


def test_train_model_runs_and_improves(synthetic_run):
    # lr bumped up from the CLI default (1e-3): MSE-toward-sigmoid-output
    # starts near a "predict the mean" plateau (all scores ~0.5, i.e.
    # always "2 particles"), and this tiny/easy dataset doesn't need many
    # epochs, just enough gradient steps to break out of it.
    model, matrix_shape, history = train_model(
        [synthetic_run], fmt="arrow", epochs=60, batch_size=16, val_fraction=0.3, lr=5e-3, device="cpu", seed=0
    )

    assert matrix_shape == (4, 2)  # 3-particle cluster spans 3 pixels -> 3+1
    assert len(history.train_loss) == 60
    # loss should trend down on this trivially-separable synthetic signal
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.val_accuracy[-1] > 0.34  # better than a naive one-class baseline (1/3)


def test_save_and_load_checkpoint_round_trip(synthetic_run, tmp_path):
    model, matrix_shape, _history = train_model(
        [synthetic_run], fmt="arrow", epochs=2, batch_size=16, device="cpu", seed=0
    )
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, model, matrix_shape)

    loaded_model, loaded_shape = load_checkpoint(checkpoint_path, device="cpu")

    assert loaded_shape == matrix_shape
    assert loaded_model.input_shape == model.input_shape


def test_evaluate_model_produces_roc_and_confusion_matrix(synthetic_run):
    model, matrix_shape, _history = train_model(
        [synthetic_run], fmt="arrow", epochs=15, batch_size=16, device="cpu", seed=0
    )
    result = evaluate_model(model, matrix_shape, [synthetic_run], fmt="arrow")

    assert result["confusion_matrix"].shape == (3, 3)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert set(result["roc"].keys()) == {1, 2, 3}
    for _fpr, _tpr, roc_auc in result["roc"].values():
        assert 0.0 <= roc_auc <= 1.0
