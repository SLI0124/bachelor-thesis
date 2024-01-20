import cv2
import numpy as np
import os
import sys

IMAGE_SIZE = 200
INPUT_FOLDER_PATH = "data/photos/school_dataset/"
OUTPUT_FOLDER_PATH = "data/output/raw/"

# list to store the points during click event, which are used to create parking spots in the first phase
list_of_points = []

# list to store the points after parsing from the file, which are used to create parking spots in the second phase
list_of_parsed_points = []

# reading the image
img = cv2.imread('data/photos/school_dataset/test7.jpg', 1)


# 1. Identify parking spots in the image by manually clicking on the corners of the parking spot
def click_event(event, x, y, flags, param):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 2, (0, 0, 255), 2)
        list_of_points.append([x, y])
        print(len(list_of_points))
        cv2.imshow('image', img)


# 2. Extract parking spots from the image I've clicked and save them to the file
def save_points_to_file(point_list: list):
    with open("data/points2.txt", "w") as f:
        for item in point_list:
            f.write("%s\n" % item)


# 3. Load parking spots from the file and save them to the list
def load_points_from_file():
    temp_list = []
    return_list = []
    count = 0
    with open("data/points.txt", "r") as f:
        for line in f:
            # format of one point on one line is [63, 862]
            x, y = line[1:-2].split(", ")
            x = int(x)
            y = int(y)
            temp_list.append(x)
            temp_list.append(y)

            if count == 3:
                # we want to save the points in the format [x1, y1, x2, y2, x3, y3, x4, y4] for each parking spot
                return_list.append(temp_list)
                temp_list = []
                count = 0
            else:
                count += 1

    return return_list


# 4. small little helper function to display parking spots on the image
def display_parking_spots(input_list: list):
    for spot in input_list:
        x1, y1, x2, y2, x3, y3, x4, y4 = spot
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), 2)
        cv2.line(img, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.line(img, (x4, y4), (x1, y1), (0, 255, 0), 2)
        cv2.imshow('image', img)


def modify_image_using_perspective(input_file_path: str, spot: list):
    local_img = cv2.imread(input_file_path, 1)
    x1, y1, x2, y2, x3, y3, x4, y4 = spot
    # use perspective transform to get the parking spot
    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[0, 0], [IMAGE_SIZE, 0], [IMAGE_SIZE, IMAGE_SIZE], [0, IMAGE_SIZE]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(local_img, matrix, (IMAGE_SIZE, IMAGE_SIZE))
    # rotate 90 to get the correct orientation
    result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return result


# 5. Extract parking spots from the image I've clicked and save them to the file
def extract_and_save_all_parking_spots(input_directory_path: str):
    # get directory of all images
    return_list = []
    print(f"input directory path: {input_directory_path}\n")

    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)
        print(f"created output folder: {OUTPUT_FOLDER_PATH}\n")

    for filename in os.listdir(input_directory_path):
        if filename.endswith(".jpg"):
            return_list.append(filename)
            print("=====================================")
            print(f"processing file: {filename}")
            # this will be called on one image in folder,
            # which will go through all the parking spots and save them to the output folder
            save_all_parking_spots_to_individual_file(f"{input_directory_path}/{filename}", OUTPUT_FOLDER_PATH)
            print(f"finished processing file: {filename}\n")

    print(f"finished processing all files in directory: {input_directory_path}\n")


# 6. Extract parking spots from the image I've clicked and save each parking spot to the output folder
def save_all_parking_spots_to_individual_file(input_file_path: str, output_folder_path: str):
    """
    This will be called on one image in folder
    This will go through all the parking spots and save them to the output folder
    :return: None
    """
    counter = 0
    for spot in list_of_parsed_points:
        # use perspective transform to get the parking spot
        result = modify_image_using_perspective(input_file_path, spot)
        # save the image
        file_name = input_file_path.split("/")[-1]
        cv2.imwrite(f"{output_folder_path}/{file_name}{counter}.jpg", result)
        counter += 1


def parse_points_from_file():
    for spot in load_points_from_file():
        x1, y1, x2, y2, x3, y3, x4, y4 = spot
        list_of_parsed_points.append([x1, y1, x2, y2, x3, y3, x4, y4])


def modify_image_using_fisheye():
    pass


def main(args):
    # list to store the points after parsing from the file, which are used to create parking spots in the second phase
    # parse each line representing one parking spot as [63, 862], [152, 697], etc.
    # to the format [x1, y1, x2, y2, x3, y3, x4, y4] for each parking spot
    parse_points_from_file()

    # displaying the image
    # cv2.imshow('image', img)

    # setting mouse handler for the image, using to identify parking spots in the first phase
    # cv2.setMouseCallback('image', click_event)

    # load points from file and show them
    my_points = load_points_from_file()
    # display_parking_spots(my_points)

    # print(my_points)
    print(f"number of parking spots: {len(my_points)}")

    extract_and_save_all_parking_spots(INPUT_FOLDER_PATH)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()

    # save points to file
    # save_points_to_file(list_of_points)


if __name__ == "__main__":
    main(sys.argv[1:])
