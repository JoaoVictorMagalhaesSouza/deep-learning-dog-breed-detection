#%%
import pandas as pd
import numpy as np
from src.data_loading import LoadImages
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#%%
DEFAULT_SIZE = 256

# Carregar dados de teste
teste = LoadImages(
    folder_path="data/testing_pessoal/",
    labels_path="data/",
    default_size=DEFAULT_SIZE
)
# %%
images, labels = teste.load_images()
# %%
X = np.array(images) / 255.0

# %% Load model

model = tf.keras.models.load_model("best_model.h5")

# %%
input_shape = (DEFAULT_SIZE, DEFAULT_SIZE, 3)



predictions = model.predict(X)

#%%
predictions = np.argmax(predictions, axis=1)

# %% Load classes
import json
with open('classes.json', 'r') as f:
    classes = json.load(f)

#Print images with classes
for i in range(len(images)):
    plt.title(classes[predictions[i]])
    plt.imshow(images[i])
    plt.show()
#%%
