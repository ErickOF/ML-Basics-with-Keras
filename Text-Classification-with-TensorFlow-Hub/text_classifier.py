import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 0.005


# Download the IMDB dataset
# Split the training set into 60% and 40%, so we'll end up with 15,000
# examples for training, 10,000 examples for validation and 25,000
# examples for testing.
train_data, validation_data, test_data = tfds.load(
    name='imdb_reviews', 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)


# Explore the data
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))


# Build the model
# Pre training model
embedding = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])
print(hub_layer)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()


# Loss function and optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, amsgrad=True)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


# Train the model
history = model.fit(train_data.shuffle(10000).batch(BATCH_SIZE),
                    validation_data=validation_data.batch(BATCH_SIZE),
                    validation_steps=1,
                    epochs=EPOCHS,
                    verbose=1)


# Evaluate the model
results = model.evaluate(test_data.batch(BATCH_SIZE),
                         verbose=2)

for name, value in zip(model.metrics_names, results):
    print('%s: %.3f' % (name, value))


