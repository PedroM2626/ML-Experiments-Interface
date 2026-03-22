"""
Deep Learning module — sklearn-compatible wrappers for MLP and Keras/TensorFlow models.
"""

import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")

# ── Keras availability ─────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    _KERAS_AVAILABLE = True
    # Suppress TF log spam
    tf.get_logger().setLevel("ERROR")
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
except ImportError:
    _KERAS_AVAILABLE = False
    tf = None
    keras = None


# ── MLP wrappers (re-export with sensible default labels) ──────────────────────

def make_mlp_classifier(**kwargs):
    """Default `MLPClassifier` with good AutoML defaults."""
    defaults = dict(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    defaults.update(kwargs)
    return MLPClassifier(**defaults)


def make_mlp_regressor(**kwargs):
    """Default `MLPRegressor` with good AutoML defaults."""
    defaults = dict(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    defaults.update(kwargs)
    return MLPRegressor(**defaults)


# ── Keras Wrappers ─────────────────────────────────────────────────────────────

def _build_keras_model(
    input_dim: int,
    output_dim: int,
    task_type: str,
    units: int = 128,
    hidden_layers: int = 2,
    dropout: float = 0.3,
    optimizer: str = "adam",
):
    """Build a dense Keras model for classification or regression."""
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))

    for i in range(hidden_layers):
        layer_units = units // (2 ** i) if i > 0 else units
        model.add(keras.layers.Dense(max(16, layer_units), activation="relu"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout))

    if task_type == "classification":
        if output_dim == 2:
            model.add(keras.layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        else:
            model.add(keras.layers.Dense(output_dim, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
            metrics = ["accuracy"]
    else:
        model.add(keras.layers.Dense(1, activation="linear"))
        loss = "mse"
        metrics = ["mae"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


class KerasClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible Keras classifier.
    Falls back to MLPClassifier if TensorFlow is not available.
    """

    def __init__(
        self,
        units: int = 128,
        hidden_layers: int = 2,
        dropout: float = 0.3,
        optimizer: str = "adam",
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.units = units
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._fallback = None
        self._model = None
        self._le = None

    def fit(self, X, y):
        if not _KERAS_AVAILABLE:
            self._fallback = make_mlp_classifier(random_state=self.random_state)
            self._fallback.fit(X, y)
            return self

        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self.classes_ = self._le.classes_
        n_classes = len(self.classes_)

        X = np.array(X, dtype=np.float32)
        self._model = _build_keras_model(
            input_dim=X.shape[1],
            output_dim=n_classes,
            task_type="classification",
            units=self.units,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            optimizer=self.optimizer,
        )
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
        ]
        self._model.fit(
            X, y_enc,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict(self, X):
        if self._fallback:
            return self._fallback.predict(X)
        X = np.array(X, dtype=np.float32)
        proba = self.predict_proba(X)
        return self._le.inverse_transform(proba.argmax(axis=1))

    def predict_proba(self, X):
        if self._fallback:
            return self._fallback.predict_proba(X)
        X = np.array(X, dtype=np.float32)
        raw = self._model.predict(X, verbose=0)
        n_classes = len(self.classes_)
        if n_classes == 2:
            neg = 1.0 - raw
            return np.hstack([neg, raw])
        return raw


class KerasRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible Keras regressor.
    Falls back to MLPRegressor if TensorFlow is not available.
    """

    def __init__(
        self,
        units: int = 128,
        hidden_layers: int = 2,
        dropout: float = 0.3,
        optimizer: str = "adam",
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.units = units
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._fallback = None
        self._model = None

    def fit(self, X, y):
        if not _KERAS_AVAILABLE:
            self._fallback = make_mlp_regressor(random_state=self.random_state)
            self._fallback.fit(X, y)
            return self

        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        self._model = _build_keras_model(
            input_dim=X.shape[1],
            output_dim=1,
            task_type="regression",
            units=self.units,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            optimizer=self.optimizer,
        )
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
        ]
        self._model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict(self, X):
        if self._fallback:
            return self._fallback.predict(X)
        X = np.array(X, dtype=np.float32)
        return self._model.predict(X, verbose=0).ravel()
