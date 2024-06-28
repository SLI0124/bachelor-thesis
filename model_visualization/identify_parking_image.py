import sys
import os
import time

import torch
import random
import json
import cv2
import csv
import numpy as np
import xml.etree.ElementTree as ET

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

    time_detections = []

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
        all_locations.append((coordinate['coordinates'], coordinate['label']))

    parking_lot = cv2.imread(os.path.join(full_images_path, one_image))

    for location in all_locations:
        coordinates = location[0]
        label = location[1]
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        x3, y3 = coordinates[2]
        x4, y4 = coordinates[3]

        start_time = time.time()
        transformed_image = perspective_transform(parking_lot, coordinates)

        transformed_image = preprocess_image(transformed_image)

        prediction = model(transformed_image.unsqueeze(0).to(device))

        prediction = torch.argmax(prediction, dim=1).item()
        end_time = time.time()
        time_detections.append(end_time - start_time)

        if prediction == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.line(parking_lot, (x1, y1), (x2, y2), color, 3)
        cv2.line(parking_lot, (x2, y2), (x3, y3), color, 3)
        cv2.line(parking_lot, (x3, y3), (x4, y4), color, 3)
        cv2.line(parking_lot, (x4, y4), (x1, y1), color, 3)

    print(f"Average time for detection: {sum(time_detections) / len(all_locations)}, "
          f"Total time: {sum(time_detections)}")

    parking_lot = cv2.resize(parking_lot, (1000, 800))
    cv2.imshow('image', parking_lot)
    cv2.waitKey(0)

    if not os.path.exists(os.path.join(SAVE_DIR, 'acpds')):
        os.makedirs(os.path.join(SAVE_DIR, 'acpds'))

    cv2.imwrite(os.path.join(SAVE_DIR, 'acpds', one_image), parking_lot)
    cv2.destroyAllWindows()


def identify_cnr_park_ext() -> None:
    model_path = "../data/models/80_20_split/cnr/cnr_park_ext/5_epochs/shufflenet.pth"

    all_locations = []
    full_images_path = os.path.join(PATH_TO_FULL_IMAGES, 'cnr', 'FULL_IMAGE_1000x750')
    for root, dirs, files in os.walk(full_images_path):
        for file in files:
            if file.endswith('.jpg'):
                all_locations.append(os.path.join(root, file))

    all_locations = [loc.replace("\\", "/") for loc in all_locations]

    random_int = random.randint(0, len(all_locations) - 1)
    one_image = all_locations[random_int]
    camera = one_image.split('/')[7]

    csv_path = os.path.join(PATH_TO_FULL_IMAGES, 'cnr')
    csv_file = os.path.join(csv_path, camera + '.csv')

    if not os.path.isfile(csv_file):
        print(f"Error: CSV file {csv_file} does not exist.")
        return

    rows = []
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue
            rows.append(row)

    whole_parking_lot = cv2.imread(one_image)
    # reshape it to 2592x1944 (original size downscaled due to privacy reasons)
    whole_parking_lot = cv2.resize(whole_parking_lot, (2592, 1944))

    model_state_dict = torch.load(model_path, map_location=device)
    model = test_model.load_model(model_path)
    model.load_state_dict(model_state_dict)
    model.eval()

    total_time = 0

    # each row is represented as [id, x1, y1, width, height]
    for row in rows:
        x1 = int(row[1])
        y1 = int(row[2])
        width = int(row[3])
        height = int(row[4])

        start_time = time.time()
        parking_spot = whole_parking_lot[y1:y1 + height, x1:x1 + width]

        parking_spot = cv2.resize(parking_spot, (IMAGE_WIDTH, IMAGE_HEIGHT))

        parking_spot = preprocess_image(parking_spot)

        prediction = model(parking_spot.unsqueeze(0).to(device))

        prediction = torch.argmax(prediction, dim=1).item()
        end_time = time.time()

        if prediction == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(whole_parking_lot, (x1, y1), (x1 + width, y1 + height), color, 3)

    print(f"Average time for detection: {total_time / len(rows)}, Total time: {total_time}")

    whole_parking_lot = cv2.resize(whole_parking_lot, (1000, 800))
    cv2.imshow('image', whole_parking_lot)
    cv2.waitKey(0)

    if not os.path.exists(os.path.join(SAVE_DIR, 'cnr')):
        os.makedirs(os.path.join(SAVE_DIR, 'cnr'))

    cv2.imwrite(os.path.join(SAVE_DIR, 'cnr', camera + '.jpg'), whole_parking_lot)
    cv2.destroyAllWindows()


def identify_spkl() -> None:
    model_path = "../data/models/80_20_split/spkl/all/5_epochs/mobilenet.pth"

    time_detections = []

    model_state_dict = torch.load(model_path, map_location=device)

    model = test_model.load_model(model_path)
    model.load_state_dict(model_state_dict)
    model.eval()

    full_images_path = os.path.join(PATH_TO_FULL_IMAGES, 'spkl', 'images')
    annotations_path = os.path.join(PATH_TO_FULL_IMAGES, 'spkl', 'int_markup')

    random_int = random.randint(0, len(os.listdir(full_images_path)) - 1)
    one_image = os.listdir(full_images_path)[random_int]
    one_annotation = os.listdir(annotations_path)[random_int]

    all_locations = []

    with open(os.path.join(annotations_path, one_annotation)) as f:
        json_annotation = json.load(f)

    for coordinate in json_annotation['lots']:
        all_locations.append((coordinate['coordinates'], coordinate['label']))

    parking_lot = cv2.imread(os.path.join(full_images_path, one_image))

    for location in all_locations:
        coordinates = location[0]
        label = location[1]
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        x3, y3 = coordinates[2]
        x4, y4 = coordinates[3]

        start_time = time.time()
        transformed_image = perspective_transform(parking_lot, coordinates)

        transformed_image = preprocess_image(transformed_image)

        prediction = model(transformed_image.unsqueeze(0).to(device))

        prediction = torch.argmax(prediction, dim=1).item()
        end_time = time.time()
        time_detections.append(end_time - start_time)

        if prediction == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.line(parking_lot, (x1, y1), (x2, y2), color, 3)
        cv2.line(parking_lot, (x2, y2), (x3, y3), color, 3)
        cv2.line(parking_lot, (x3, y3), (x4, y4), color, 3)
        cv2.line(parking_lot, (x4, y4), (x1, y1), color, 3)

    print(f"Average time for detection: {sum(time_detections) / len(all_locations)}, "
          f"Total time: {sum(time_detections)}")

    parking_lot = cv2.resize(parking_lot, (1000, 800))
    cv2.imshow('image', parking_lot)
    cv2.waitKey(0)

    if not os.path.exists(os.path.join(SAVE_DIR, 'spkl')):
        os.makedirs(os.path.join(SAVE_DIR, 'spkl'))

    cv2.imwrite(os.path.join(SAVE_DIR, 'spkl', one_image), parking_lot)
    cv2.destroyAllWindows()


def element_to_dict(element):
    elem_dict = {'tag': element.tag, 'attributes': element.attrib, 'children': []}
    for child in element:
        elem_dict['children'].append(element_to_dict(child))
    return elem_dict


def identify_pklot() -> None:
    model_path = "../data/models/80_20_split/pklot/all/5_epochs/mobilenet.pth"

    time_detections = []

    model_state_dict = torch.load(model_path, map_location=device)

    model = test_model.load_model(model_path)
    model.load_state_dict(model_state_dict)
    model.eval()

    all_images = []

    # print all files in the directory
    full_images_path = os.path.join(PATH_TO_FULL_IMAGES, 'pklot')
    # print every file
    for root, dirs, files in os.walk(full_images_path):
        for file in files:
            if file.endswith('.jpg'):
                one_image = os.path.join(root, file)
                all_images.append(one_image)
                break

    random_int = random.randint(0, len(all_images) - 1)
    one_image = all_images[random_int]
    one_annotation = one_image.replace('.jpg', '.xml')

    tree = ET.parse(one_annotation)
    root = tree.getroot()

    spaces = []
    for space in root.findall('space'):
        space_dict = element_to_dict(space)
        spaces.append(space_dict)

    parking_lot = cv2.imread(one_image)

    for space in spaces:
        contour = space['children'][1]

        points = []
        for point in contour['children']:
            x = int(point['attributes']['x'])
            y = int(point['attributes']['y'])
            points.append((x, y))

        start_time = time.time()
        transformed_image = perspective_transform(parking_lot, points)

        transformed_image = preprocess_image(transformed_image)

        prediction = model(transformed_image.unsqueeze(0).to(device))

        prediction = torch.argmax(prediction, dim=1).item()
        end_time = time.time()
        time_detections.append(end_time - start_time)

        if prediction == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        for i in range(len(points)):
            cv2.line(parking_lot, points[i], points[(i + 1) % len(points)], color, 3)

    print(f"Average time for detection: {sum(time_detections) / len(spaces)}, "
          f"Total time: {sum(time_detections)}")

    parking_lot = cv2.resize(parking_lot, (1000, 800))
    cv2.imshow('image', parking_lot)
    cv2.waitKey(0)

    if not os.path.exists(os.path.join(SAVE_DIR, 'pklot')):
        os.makedirs(os.path.join(SAVE_DIR, 'pklot'))

    cv2.imwrite(os.path.join(SAVE_DIR, 'pklot', one_image.split('/')[-1]), parking_lot)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.path.append('../')
    import model_visualization.test_models as test_model

    # identify_acpds()
    # identify_cnr_park_ext()
    # identify_spkl()
    # identify_pklot()
