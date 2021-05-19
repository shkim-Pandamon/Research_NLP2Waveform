import tensorflow as tf
import numpy as no

class ModelCreator(object):
    def __init__(self):
        self.output_node = 4
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.metric = ['accuracy']

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            # tf.keras.layers.LSTM(64),
            tf.keras.layers.LSTM(64, return_sequences = True),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(self.output_node),
            tf.keras.layers.Softmax()
        ])
        model.compile(loss=self.loss,
                    optimizer=self.optimizer,
                    metrics=self.metric)
        return model