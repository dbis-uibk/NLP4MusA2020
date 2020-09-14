import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_curve
from sklearn.preprocessing import normalize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, AlphaDropout

class SimpleGenreNN(BaseEstimator, RegressorMixin):
    def __init__(self, batch_size=2, epochs=10, dense_sizes=(100,), dropout_rate=0.1):
        self.batch_size = batch_size
        self.epochs = epochs
        self.dense_sizes = dense_sizes
        self.dropout_rate = dropout_rate

    def _create_model(self, X, y):
        # Input layer.
        inp = Input(shape=(X.shape[1],))

        # Dense layers.
        layer = inp
        for size in self.dense_sizes:
            layer = Dense(size, activation="selu", kernel_initializer="lecun_normal")(layer)
            layer = AlphaDropout(self.dropout_rate)(layer)

        # Output layer.
        out = Dense(y.shape[1], activation="sigmoid")(layer)

        # Create the model.
        self.model = Model(inputs=inp, outputs=out)
        self.model.compile(optimizer="adam", loss="binary_crossentropy")

        # Print summary.
        self.model.summary()

    def fit(self, X, y):
        # Create the model.
        self._create_model(X, y)

        # Fit the model.
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)

    def predict(self, X):
        # Make predictions.
        y  = self.model.predict(X)

        # Convert probabilities to labels.
        res = np.zeros(y.shape)
        res[y > 0.5] = 1

        return res
