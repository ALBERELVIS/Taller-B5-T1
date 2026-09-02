"""Custom Keras layers shared by the classifier, the GAN and the AR model.

There is a single idea here and it bit us three separate times, so it is worth
stating once, clearly.

A dense or convolutional layer computes ``W x + b`` followed by a ReLU. That is
a *first-order* operation: it can add, subtract and threshold its inputs, but it
cannot multiply two of them together. So it cannot compute a variance, a
volatility or any other second moment, unless the network is deep enough and
wide enough to approximate a square piecewise -- which is a lot to ask from a
small model trained on a few hundred windows.

Every part of this project happens to need second moments:

* the **classifier** has to recognise turbulent regimes, and turbulence is a
  variance. Its first version lost to a one-line realised-volatility baseline;
* the **discriminator** of the GAN has to notice that the generated windows are
  five times more volatile than the real ones, and it could not, so the
  generator was free to diverge;
* the **autoregressive** model has to predict a conditional scale from recent
  history.

Appending ``|x|`` to the input solves it for all three at zero parameter cost
and with no learning involved, so it cannot overfit or leak.
"""

from __future__ import annotations

from .keras_setup import keras


@keras.saving.register_keras_serializable(package="b5t1")
class MagnitudeFeatures(keras.layers.Layer):
    """Concatenate ``|x|`` to ``x`` along the last axis.

    Works for both ``(batch, features)`` and ``(batch, steps, features)``.
    """

    def call(self, x):
        return keras.ops.concatenate([x, keras.ops.abs(x)], axis=-1)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], input_shape[-1] * 2)
