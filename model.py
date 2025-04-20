# data - https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Constants
BATCH_SIZE = 32
IMG_SIZE = (384, 384)
IMG_SHAPE = IMG_SIZE + (3,)
BUFFER = 10000

# Load datasets OUTSIDE the scope
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "/kaggle/input/brain-tumor-mri-dataset/Training",
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='training',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    "/kaggle/input/brain-tumor-mri-dataset/Training",
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='validation',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "/kaggle/input/brain-tumor-mri-dataset/Testing",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

class_names = train_dataset.class_names

# Prefetch OUTSIDE the scope
train_dataset = train_dataset.shuffle(BUFFER).prefetch(tf.data.AUTOTUNE).map(lambda image,label: (image,tf.one_hot(label, depth=len(class_names))))
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE).map(lambda image, label: (image, tf.one_hot(label, depth=len(class_names))))
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE).map(lambda image, label: (image, tf.one_hot(label, depth=len(class_names))))

base_model = tf.keras.applications.EfficientNetV2S(
    include_top=False,
    input_shape=IMG_SHAPE,
    weights='imagenet',
    pooling='max'
) # by default takes input images with [0-255] values and has inbuilt rescale to [0-1]
base_model.trainable = False

checkpoint =  tf.keras.callbacks.ModelCheckpoint(filepath = "model.keras", 
                                                 monitor='val_loss', 
                                                 verbose=0, 
                                                 mode='auto', 
                                                 save_best_only=True)

model = tf.keras.Sequential([
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(5e-4),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = [tf.keras.metrics.CategoricalAccuracy(),
               tf.keras.metrics.Precision(),
               tf.keras.metrics.Recall()]
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    # callbacks = [checkpoint]
)

plt.plot(history.history['loss'], marker='x', color='red', label='loss')
plt.plot(history.history['val_loss'], marker='.', color='blue', label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['categorical_accuracy'], marker='x', color='red', label='acc')
plt.plot(history.history['val_categorical_accuracy'], marker='.', color='blue', label='val_acc')
plt.legend()
plt.show()

print(len(base_model.layers))

# Note: this needs to be done only after we did the above initial training 
# otherwise our new dense layer and classifyer layer will hinder the base_model layers performan
# model = tf.keras.models.load_model("/kaggle/working/model.keras")
fine_tune_from_layer = 450 # 514-450 = 64, meaning only 64 layers of the base model will be unfrozen
base_model.trainable = True

for layer in base_model.layers[:fine_tune_from_layer]:
    layer.trainable = False

base_model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(),
   metrics = [tf.keras.metrics.CategoricalAccuracy(),
               tf.keras.metrics.Precision(),
               tf.keras.metrics.Recall()]
)

history = model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs = 15
)

plt.plot(history.history['loss'], marker='x', color='red', label='loss')
plt.plot(history.history['val_loss'], marker='.', color='blue', label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['categorical_accuracy'], marker='x', color='red', label='acc')
plt.plot(history.history['val_categorical_accuracy'], marker='.', color='blue', label='val_acc')
plt.legend()
plt.show()

plt.plot(history.history['recall_3'], marker='x', color='red', label='recall')
plt.plot(history.history['val_recall_3'], marker='.', color='blue', label='val_recall')
plt.legend()
plt.show()

plt.plot(history.history['precision_3'], marker='x', color='red', label='precision')
plt.plot(history.history['val_precision_3'], marker='.', color='blue', label='val_precision')
plt.legend()
plt.show()

test_loss, test_acc, test_precision, test_recall = model.evaluate(test_dataset, verbose = 0)
print(f"Test Data Loss: {test_loss:.4f}")
print(f"Test Data Accuracy: {test_acc:.4f}")
print(f"Test Data Precision: {test_precision:.4f}")
print(f"Test Data Recall: {test_recall:.4f}")


model.save('mri-image-classifier.keras')