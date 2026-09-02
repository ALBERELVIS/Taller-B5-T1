"""The downstream problem: a fixed CNN that flags stress windows.

This is the model whose error tells us whether synthetic data was worth
generating. The professor's rule is strict and we follow it literally: **one**
architecture, chosen once on real data, then retrained unchanged for every
real/synthetic mix. If the architecture moved between runs the comparison would
measure architecture search, not data quality.

Deliberate choice: no class weighting. Re-balancing the minority class is
exactly the job we are asking the synthetic data to do, so building a class
weight into the loss would hide the effect we are trying to measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config
from .keras_setup import keras, set_seeds
from .layers import MagnitudeFeatures


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------
def build_classifier(
    input_shape: tuple[int, int],
    seed: int = config.SEED,
    variant: str = config.DOWNSTREAM_VARIANT,
) -> keras.Model:
    """Build a candidate classifier.

    ``variant`` is only used by notebook 02, where we search for a valid
    architecture on real data. Once chosen, ``config.DOWNSTREAM_VARIANT`` freezes
    it and every sweep run uses exactly the same network.

    The convolutional skeleton follows the professor's best regressor ("CNN 2":
    three Conv1D + MaxPooling blocks, then a dense head), with a sigmoid output
    and dropout because our training sets get as small as 500 windows.
    """
    set_seeds(seed)
    inputs = keras.layers.Input(shape=input_shape)

    if variant in ("cnn_mag", "cnn_mag_bn", "cnn_mag_gap"):
        x = MagnitudeFeatures()(inputs)
    else:
        x = inputs

    # "cnn_prof" reproduces the professor's CNN 2 literally (valid padding). The
    # other variants use same padding, which keeps the sequence long enough for
    # three pooling stages and turned out to matter a lot here.
    padding = "valid" if variant == "cnn_prof" else "same"
    use_bn = variant == "cnn_mag_bn"
    for filters in (64, 128, 128):
        x = keras.layers.Conv1D(filters, 3, activation=None, padding=padding)(x)
        if use_bn:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling1D(2)(x)

    if variant == "cnn_mag_gap":
        x = keras.layers.GlobalAveragePooling1D()(x)
    else:
        x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(100, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name=f"stress_{variant}")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = config.SEED,
    epochs: int = config.DOWNSTREAM_EPOCHS,
    batch_size: int = config.DOWNSTREAM_BATCH,
    patience: int = config.DOWNSTREAM_PATIENCE,
    variant: str = config.DOWNSTREAM_VARIANT,
    verbose: int = 0,
) -> tuple[keras.Model, dict]:
    """Train and return ``(model, history)``.

    Early stopping watches validation PR-AUC, never training loss, and restores
    the best weights. The validation set is 100% real in every single call --
    synthetic samples only ever enter ``X_train``.
    """
    set_seeds(seed)
    model = build_classifier(X_train.shape[1:], seed=seed, variant=variant)
    stop = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=patience, restore_best_weights=True
    )
    hist = model.fit(
        X_train,
        y_train.astype("float32"),
        validation_data=(X_val, y_val.astype("float32")),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[stop],
        verbose=verbose,
        shuffle=True,
    )
    return model, {k: [float(v) for v in vals] for k, vals in hist.history.items()}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
@dataclass
class Metrics:
    pr_auc: float
    roc_auc: float
    f1: float
    precision: float
    recall: float
    threshold: float
    base_rate: float
    lift: float = field(init=False)

    def __post_init__(self):
        # PR-AUC is only meaningful relative to the base rate: a random ranker
        # scores exactly the base rate, so the lift is what we actually report.
        self.lift = self.pr_auc / self.base_rate if self.base_rate > 0 else float("nan")

    def as_dict(self) -> dict:
        return {
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "threshold": self.threshold,
            "base_rate": self.base_rate,
            "lift": self.lift,
        }


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1, chosen on validation and then frozen.

    Choosing it on the test set would be a textbook way of inflating results.
    """
    candidates = np.unique(np.quantile(scores, np.linspace(0.50, 0.999, 120)))
    f1s = [f1_score(y_true, (scores >= t).astype(int), zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(f1s))])


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Metrics:
    y_pred = (scores >= threshold).astype(int)
    return Metrics(
        pr_auc=float(average_precision_score(y_true, scores)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        threshold=float(threshold),
        base_rate=float(np.mean(y_true)),
    )


def predict_scores(model: keras.Model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    return model.predict(X, batch_size=batch_size, verbose=0).ravel()


def evaluate_model(
    model: keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[Metrics, Metrics, np.ndarray]:
    """Full evaluation: tune the threshold on val, report both val and test."""
    s_val = predict_scores(model, X_val)
    s_test = predict_scores(model, X_test)
    thr = best_threshold(y_val, s_val)
    return (
        evaluate_scores(y_val, s_val, thr),
        evaluate_scores(y_test, s_test, thr),
        s_test,
    )


# --------------------------------------------------------------------------
# Reference models (notebook 02)
# --------------------------------------------------------------------------
def volatility_score(X: np.ndarray) -> np.ndarray:
    """Classic financial baseline: recent realised volatility of the portfolio.

    Volatility clusters, so 'it has been turbulent lately' is already a decent
    stress predictor. Any model that cannot beat this is not earning its keep.
    """
    market = X.mean(axis=2)
    return market.std(axis=1)


def drawdown_score(X: np.ndarray) -> np.ndarray:
    """Second baseline: how far the portfolio already is below its window peak."""
    market = X.mean(axis=2)
    level = np.exp(np.cumsum(market, axis=1))
    peak = np.maximum.accumulate(level, axis=1)
    return -(level[:, -1] / peak[:, -1] - 1.0)


def logistic_baseline(X_train, y_train, X_val, X_test, seed: int = config.SEED):
    """Flattened-window logistic regression, the professor's 'linear model' slot."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.05, random_state=seed),
    )
    pipe.fit(X_train.reshape(len(X_train), -1), y_train)
    return (
        pipe.predict_proba(X_val.reshape(len(X_val), -1))[:, 1],
        pipe.predict_proba(X_test.reshape(len(X_test), -1))[:, 1],
    )
