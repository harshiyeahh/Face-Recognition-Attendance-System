import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []

label_map = {}

current_id = 0

for person_name in os.listdir(path):

    person_path = os.path.join(path, person_name)

    label_map[current_id] = person_name

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        face_img = Image.open(image_path).convert('L')

        image_np = np.array(face_img, 'uint8')

        faces.append(image_np)

        ids.append(current_id)

    current_id += 1

recognizer.train(faces, np.array(ids))

recognizer.save('trainer/trainer.yml')

np.save('trainer/labels.npy', label_map)

print("Model Trained Successfully")