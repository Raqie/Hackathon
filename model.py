import numpy as np
import pandas as pd
from keras import regularizers, optimizers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
!/usr/local/bin/python

df = pd.read_csv(
    "D:/Programming/Hackathon/skyhacks_hackathon_dataset/training_labels.csv", delimiter=',')

df * 1
print(df)
columns = ["Amusement park", "Animals", "Bench", "Building", "Castle", "Cave", "Church", "City", "Cross", "Cultural institution", "Food", "Footpath", "Forest", "Furniture", "Grass", "Graveyard", "Lake", "Landscape", "Mine",
           "Monument", "Motor vehicle", "Mountains", "Museum", "Open-air museum", "Park", "Person", "Plants", "Reservoir", "River", "Road", "Rocks", "Snow", "Sport", "Sports facility", "Stairs", "Trees", "Watercraft", "Windows"]
rows = df.Name.to_string(index=False)  # wszystkie nazwy plików jpg

datagen = ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(rescale=1./255.)
train_generator = datagen.flow_from_dataframe(
    dataframe=df[:1800],
    directory="D:/Programming/Hackathon/skyhacks_hackathon_dataset/training_images",
    x_col="Name",
    y_col=columns,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(100, 100))

valid_generator = test_datagen.flow_from_dataframe(
    dataframe=df[1800:1900],
    directory="D:/Programming/Hackathon/skyhacks_hackathon_dataset/training_images",
    x_col="Name",
    y_col=columns,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(100, 100))
test_generator = test_datagen.flow_from_dataframe(
    dataframe=df[1900:2000],
    directory="D:/Programming/Hackathon/skyhacks_hackathon_dataset/training_images",
    x_col="Name",
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(100, 100))

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(100, 100, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(columns), activation='sigmoid'))
model.compile(optimizers.Adam(lr=0.0001, decay=1e-6),
              loss="binary_crossentropy", metrics=["accuracy"])

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
                    )
model.save("TrainedModel.h5")
