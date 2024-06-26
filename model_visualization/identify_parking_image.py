import sys
import os
import torch
import random
import json
import cv2
from PIL import Image
import csv
import numpy as np
from torchvision import transforms

PATH_TO_SPLIT_FILES = '../data/splits'
PATH_TO_IMAGES = '../data/datasets'
PATH_TO_FULL_IMAGES = '../data/full_images'
PATH_TO_MODELS = '../data/models'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_DIR = '../data/predicted_images'

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


def perspective_transform(image: np.ndarray, coordinates: list) -> np.ndarray:
    x1, y1 = coordinates[0]
    x2, y2 = coordinates[1]
    x3, y3 = coordinates[2]
    x4, y4 = coordinates[3]

    new_coordinates = np.array([[0, 0], [0, IMAGE_HEIGHT], [IMAGE_WIDTH, IMAGE_HEIGHT], [IMAGE_WIDTH, 0]], np.float32)
    old_coordinates = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.float32)

    matrix = cv2.getPerspectiveTransform(old_coordinates, new_coordinates)
    transformed_image = cv2.warpPerspective(image, matrix, (IMAGE_WIDTH, IMAGE_HEIGHT))

    return transformed_image


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    image = image / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = torch.tensor(image).permute(2, 0, 1).float()

    return image


def identify_acpds() -> None:
    model_path = "../data/models/80_20_split/acpds/all/5_epochs/mobilenet.pth"

    model_state_dict = torch.load(model_path, map_location=device)

    model = test_model.load_model(model_path)
    model.load_state_dict(model_state_dict)
    model.eval()

    full_images_path = os.path.join(PATH_TO_FULL_IMAGES, 'acpds', 'images')
    annotations_path = os.path.join(PATH_TO_FULL_IMAGES, 'acpds', 'int_markup')

    random_int = random.randint(0, len(os.listdir(full_images_path)) - 1)
    one_image = os.listdir(full_images_path)[random_int]
    one_annotation = os.listdir(annotations_path)[random_int]

    all_locations = []

    with open(os.path.join(annotations_path, one_annotation)) as f:
        json_annotation = json.load(f)

    for coordinate in json_annotation['lots']:
        print(coordinate['label'], coordinate['coordinates'])
        all_locations.append((coordinate['coordinates'], coordinate['label']))

    parking_lot = cv2.imread(os.path.join(full_images_path, one_image))

    for location in all_locations:
        coordinates = location[0]
        label = location[1]
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        x3, y3 = coordinates[2]
        x4, y4 = coordinates[3]

        transformed_image = perspective_transform(parking_lot, coordinates)

        transformed_image = preprocess_image(transformed_image)

        prediction = model(transformed_image.unsqueeze(0).to(device))

        prediction = torch.argmax(prediction, dim=1).item()

        if prediction == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.line(parking_lot, (x1, y1), (x2, y2), color, 3)
        cv2.line(parking_lot, (x2, y2), (x3, y3), color, 3)
        cv2.line(parking_lot, (x3, y3), (x4, y4), color, 3)
        cv2.line(parking_lot, (x4, y4), (x1, y1), color, 3)

        cv2.putText(parking_lot, f"True: {label}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(parking_lot, f"Predicted: {prediction}", (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

    parking_lot = cv2.resize(parking_lot, (1000, 800))
    cv2.imshow('image', parking_lot)
    cv2.waitKey(0)

    if not os.path.exists(os.path.join(SAVE_DIR, 'acpds')):
        os.makedirs(os.path.join(SAVE_DIR, 'acpds'))

    cv2.imwrite(os.path.join(SAVE_DIR, 'acpds', one_image), parking_lot)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.path.append('../')
    import model_visualization.test_model as test_model

    identify_acpds()
