from keras.models import Sequential, load_model
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

df = pd.read_csv(
    "D:/Programming/Hackathon/skyhacks_hackathon_dataset/training_labels.csv", delimiter=',')

df1 = df * 1
print(df1)
columns = ["Amusement park", "Animals", "Bench", "Building", "Castle", "Cave", "Church", "City", "Cross", "Cultural institution", "Food", "Footpath", "Forest", "Furniture", "Grass", "Graveyard", "Lake", "Landscape", "Mine",
           "Monument", "Motor vehicle", "Mountains", "Museum", "Open-air museum", "Park", "Person", "Plants", "Reservoir", "River", "Road", "Rocks", "Snow", "Sport", "Sports facility", "Stairs", "Trees", "Watercraft", "Windows"]
# rows = df.Name.to_string(index=False)  # wszystkie nazwy plików jpg

datagen = ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=df[1900:2000],
    directory="D:/Programming/Hackathon/skyhacks_hackathon_dataset/training_images",
    x_col="Name",
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(100, 100))

STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
model = load_model("TrainedModel.h5")
test_generator.reset()
pred = model.predict_generator(test_generator,
                               steps=STEP_SIZE_TEST,
                               verbose=1)

pred_bool = (pred > 0.5)
predictions = pred_bool.astype(int)
# print(len(predictions))
results = pd.DataFrame(predictions, columns=columns)
results["Name"] = test_generator.filenames
#ordered_cols = [rows]+columns
# results = results[ordered_cols]  # To get the same column order
results.set_index("Name")
results.to_csv("results.csv", sep=";", index=False)
rescsv = pd.read_csv("results.csv")
col_name = "Name"
first_col = rescsv.pop(col_name)
rescsv.insert(0, col_name, first_col)
rescsv.to_csv("results.csv")
