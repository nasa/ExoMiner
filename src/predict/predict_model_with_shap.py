import tensorflow as tf

class ExoMinerProba(tf.keras.Model):
    def __init__(self, exominer_model):
        super().__init__()
        self.m = exominer_model                # original model

    @tf.function  # keeps it fast & graph-compatible
    def call(self, inputs, training=False):
        # return only the predicted probabilities/logits
        probs, _ = self.m(inputs, training=training)
        return probs

