from tensorflow.keras.applications.mobilenet import preprocess_input
from keras.models import load_model
import numpy as np
import pandas as pd
import os
import cv2


cap = cv2.VideoCapture(
    "D:/Programming/Hackathon/Radom  żeromskiego.mp4")


class CountingImg():
    def __init__(self):
        super().__init__()

    def countFPS(self):
        count = 0
        success = True
        columns = ["Amusement park", "Animals", "Bench", "Building", "Castle", "Cave", "Church", "City", "Cross", "Cultural institution", "Food", "Footpath", "Forest", "Furniture", "Grass", "Graveyard", "Lake", "Landscape", "Mine",
                   "Monument", "Motor vehicle", "Mountains", "Museum", "Open-air museum", "Park", "Person", "Plants", "Reservoir", "River", "Road", "Rocks", "Snow", "Sport", "Sports facility", "Stairs", "Trees", "Watercraft", "Windows"]
        model = load_model("Mobilenet_448_building_model.07-0.93.h5")
        pred_list = []
        if(cap.isOpened() == False):
            print("error opening video stream or file")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        while success:
            success, image = cap.read()
            count += 1
            if count % 100 == 0:
                print("count = ", count)
                cv2.imwrite("frame%d.jpg" % count, image)
                print(image.shape)
                image_resized = cv2.resize(image, (100, 100))
                print(image_resized.shape)
                print('successfully writen frame')

                img = np.asarray(image_resized)
                img_exp_dim = np.expand_dims(img, axis=0)
                print("img_exp_dim = ", img_exp_dim.shape)
                pred = model.predict(img_exp_dim)
                pred_bool = (pred > 0.5)
                #print("pred_bool =", pred_bool)
                pred_list.append([])
                for idx, column in enumerate(columns):
                    if pred_bool[0][idx]:
                        print("idx =", idx, "pred_bool =", pred_bool[0][idx])
                        pred_list[-1].append((column, count))

        cap.release()
        cv2.destroyAllWindows()
        df = pd.DataFrame(pred_list)
        df.to_csv("pred_list_Radom.csv", sep=';')


CountingImg.countFPS(self=CountingImg)
