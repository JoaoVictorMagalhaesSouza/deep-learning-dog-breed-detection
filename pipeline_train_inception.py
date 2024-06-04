#%%
import pandas as pd
import numpy as np
import cv2
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
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.layers import MaxPooling2D
DEFAULT_SIZE = 256



print("Lendo imagens de treino...")
# Carregar dados de treino
teste = LoadImages(
    folder_path="data/train/",
    labels_path="data/",
    default_size=DEFAULT_SIZE,
    qtde_images=0.6
)

# Carregar imagens e rótulos
images, labels = teste.load_images()
print(f"Quantidade de imagens: {len(images)}")

# Pré-processar imagens
process = True

X = np.array(images) / 255.0

# View image 0 segmented
plt.imshow(X[0])
plt.show()
#%%
# Codificar rótulos
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
y = labels

num_classes = y.shape[1]

print(f"Quantidade de classes: {num_classes}")
# Dividir os dados em conjuntos de treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Quantidade de imagens de treino: {len(X_train)}")
print(f"Quantidade de imagens de val: {len(X_val)}")
print(y_train.shape)

# Definir a arquitetura do modelo usando InceptionV3
input_shape = (DEFAULT_SIZE, DEFAULT_SIZE, 3)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

x = base_model.output
x = GlobalAveragePooling2D()(x)
#MaxPooling2D(pool_size=(2, 2))(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Configurar Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.5,
    height_shift_range=0.5,
    shear_range=0.5,
    zoom_range=0.5,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Configurar callbacks
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

# Treinar o modelo
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=150,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping]
)

# Plotar gráficos de acurácia
loss = history.history['accuracy']
val_loss = history.history['val_accuracy']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'y', label='Treinamento')
plt.plot(epochs, val_loss, 'r', label='Validação')

plt.title("Treinamento versus Validação")
plt.xlabel("Épocas")
plt.ylabel("Acurácia Global")
plt.legend()
plt.show()

# Avaliar o modelo
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Acurácia: {accuracy:.2f}")

# Descongelar algumas camadas do modelo base para fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar novamente o modelo com fine-tuning
history_fine = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping]
)

# Plotar gráficos de acurácia após fine-tuning
loss_fine = history_fine.history['accuracy']
val_loss_fine = history_fine.history['val_accuracy']

epochs_fine = range(1, len(loss_fine) + 1)

plt.plot(epochs_fine, loss_fine, 'y', label='Treinamento')
plt.plot(epochs_fine, val_loss_fine, 'r', label='Validação')

plt.title("Treinamento versus Validação após Fine-Tuning")
plt.xlabel("Épocas")
plt.ylabel("Acurácia Global")
plt.legend()
plt.show()

# Avaliar o modelo após fine-tuning
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Acurácia após fine-tuning: {accuracy:.2f}")

# %%
