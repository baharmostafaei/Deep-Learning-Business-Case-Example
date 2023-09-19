import tensorflow as tf
import numpy as np

npz = np.load('Audiobooks_data_train.npz')
train_inputs = npz['inputs'].astype(np.float32)
train_targets = npz['targets'].astype(np.int64)

npz = np.load('Audiobooks_data_validation.npz')
validation_inputs = npz['inputs'].astype(np.float32)
validation_targets = npz['targets'].astype(np.int64)

npz = np.load('Audiobooks_data_test.npz')
test_inputs = npz['inputs'].astype(np.float32)
test_targets = npz['targets'].astype(np.int64)

inputs_size = 10
output_size = 2
hidden_layer_size = 250

model = tf.keras.Sequential([
                            tf.keras.layers.Dense(inputs_size, activation='relu'), 
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

batchs = 50
max_epoch = 100

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

model.fit(train_inputs,
          train_targets, 
          batch_size= batchs, 
          epochs= max_epoch,
          callbacks=[early_stopping],
          validation_data=(validation_inputs, validation_targets),
          verbose= 2)


test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)


