import os, sys, time
import csv
import math, random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import time
from tensorflow.keras.optimizers import Adam
from pathlib import Path
import pathlib
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters import rank
from skimage import io, color, exposure, transform
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard
import keras.preprocessing.image 




from keras.preprocessing.image import ImageDataGenerator


scale = 0.1

data_dir = '/path/folder/archive/'
data_dir_train = '/path/folder/archive/reduced/train'
data_dir_val = '/path/folder/archive/reduced/validation'


emotions = ['sad','happy','anger','neutral']



batch_size = 20






# Set the size of the images
img_size = (48, 48)

# Create an image generator for training with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values between 0 and 1
    shear_range=0.2,         # Shear effect
    zoom_range=0.2,          # Zoom effect
    horizontal_flip=True,    # Horizontal flip
    
)

# Create an image generator for validation without data augmentation
val_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation images
train_generator = train_datagen.flow_from_directory(
    data_dir_train,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'  # Use 'categorical' for categorical labels
)

validation_generator = val_datagen.flow_from_directory(
    data_dir_val,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)








#####################################################################


# Prepare callbacks



# It's possible to save the model each epoch or at each improvement. The model can be saved completely or partially. 
# For full format we can use HDF5 format


from datetime import datetime

# Create directories
run_dir = '/path/folder/archive/reduced/'
os.makedirs(os.path.join(run_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)

# TensorBoard Callback
log_dir = os.path.join(run_dir, "logs", "tb_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# ModelCheckpoint Callback - Save the best model based on validation metric
bestmodel_checkpoint_dir = os.path.join(run_dir, "models", "best-model.h5")
bestmodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=bestmodel_checkpoint_dir,
                                                         verbose=0,
                                                         monitor='val_accuracy',  # Use the validation metric
                                                         save_best_only=True)

# ModelCheckpoint Callback - Save the model at each epoch
checkpoint_dir = os.path.join(run_dir, "models", "model-{epoch:04d}.h5")
savemodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, verbose=0)

# Display the command to run TensorBoard
tensorboard_command = f'tensorboard --logdir {os.path.abspath(log_dir)}'
print(f'To run TensorBoard, use the following command:\n{tensorboard_command}')



#####################################################################





train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
)



val_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_val,
    validation_split = 0.2,
    subset = "validation",
    seed = 42,
    image_size = img_size,
    batch_size = batch_size,
)



class_names = train_data.class_names
print(class_names)

# Get a batch of images and labels from the train_data
for images, labels in train_data.take(1):
    # Display the first 4 images
    plt.figure(figsize=(10, 10))
    for i in range(4):
        ax = plt.subplot(1, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])  # Convert categorical labels to index
        plt.axis("off")

plt.show()



# The model

num_class = 4

model = tf.keras.Sequential([

    keras.layers.experimental.preprocessing.Rescaling(1./255),
    keras.layers.Conv2D(128, 4, activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64, 4, activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(32, 4, activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.2),


    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(num_class,activation='softmax')


])

# Create an Adam optimizer with the desired learning rate
custom_optimizer = Adam(learning_rate=0.0005)  



model.compile(optimizer=custom_optimizer,
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])


history = model.fit(train_data,
                    validation_data = val_data,
                    epochs = 30,
                    batch_size = batch_size,
                    verbose = 1,
                    callbacks=[tensorboard_callback,bestmodel_callback,savemodel_callback])

print(history)


model.summary()





score = model.evaluate(val_data, verbose = 0)



print('Test loss: {:5.4f}'.format(score[0]))
print('Test accuracy: {:5.4f}'.format(score[1]))


# Retrieving training and validation metrics
train_accuracy = history.history['accuracy']
train_loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Visualizing the evolution of accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Visualizing the evolution of loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()



# Load a single image from the validation set
sample_image, sample_label = next(iter(validation_generator))

# Expand dimensions for prediction
sample_image = np.expand_dims(sample_image[0], axis=0)

# Make prediction
predictions = model.predict(sample_image)


# Get the predicted class and probabilities
predicted_class = np.argmax(predictions)
predicted_probabilities = predictions[0]

print(predicted_probabilities)

# Get the actual class
actual_class = np.argmax(sample_label[0])

# Visualize the results
plt.imshow(np.squeeze(sample_image))
plt.title(f'Predicted Class: {class_names[predicted_class]}, Actual Class: {class_names[actual_class]}')
plt.axis('off')
plt.show()
