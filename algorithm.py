import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
from tensorflow.keras.activations import relu


dataset_url = "dataset/train"

img_height= 100
img_width = 100
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory( 
    dataset_url, 
    validation_split=0.2, 
    subset= 'training', 
    seed = 256, 
    image_size=(img_height,img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_url,
  validation_split=0.2,
  subset="validation",
  seed=256,
  image_size=(img_height,img_width),
  batch_size=batch_size
)



class_names = train_ds.class_names
print(class_names)


plt.figure(figsize=(12, 12))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
    AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(buffer_size=batch_size).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().shuffle(buffer_size=batch_size).prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.5),
])

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)), #rescale to be between 0 and 1
  data_augmentation,
  layers.Conv2D(16, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),  
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(),  
  layers.Flatten(),
   layers.Dense(256, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

epochs=20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

test_url = "dataset/test"

test_ds = tf.keras.utils.image_dataset_from_directory( 
    test_url, 
    seed = 256, 
    image_size=(img_height,img_width),
    shuffle=False #No shuffling for classification report
)

test_images, test_labels = tuple(zip(*test_ds))

predictions = model.predict(test_ds)
score = tf.nn.softmax(predictions)
results = model.evaluate(test_ds)
print("Test loss, test acc:", results)

y_test = np.concatenate(test_labels) 
y_pred = np.array([np.argmax(s) for s in score])

import pickle
pickle_out=open('history.pkl','wb')
pickle.dump(model,pickle_out)
pickle_out.close()