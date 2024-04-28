import os
import random
import shutil


def shuffle_images(image_list):
    random.shuffle(image_list)
    return image_list


def split_dataset_into_train_test(directory, train_ratio=0.8):
    train_ratio_str = str(int(train_ratio * 100))
    test_ratio_str = str(100 - int(train_ratio * 100))

    output_directory = os.path.join(directory, train_ratio_str + "_" + test_ratio_str)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    else:
        print("Directory already exists.")
        return

    train_directory = os.path.join(output_directory, "train")
    test_directory = os.path.join(output_directory, "test")

    if not os.path.exists(train_directory):
        os.makedirs(train_directory)
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    all_directory = os.path.join(directory, "all")

    busy_images = os.listdir(os.path.join(all_directory, "Busy"))
    free_images = os.listdir(os.path.join(all_directory, "Free"))
    busy_images = shuffle_images(busy_images)
    free_images = shuffle_images(free_images)

    train_busy_images = busy_images[:int(len(busy_images) * train_ratio)]
    test_busy_images = busy_images[int(len(busy_images) * train_ratio):]

    train_free_images = free_images[:int(len(free_images) * train_ratio)]
    test_free_images = free_images[int(len(free_images) * train_ratio):]

    train_busy_directory = os.path.join(train_directory, "Busy")
    test_busy_directory = os.path.join(test_directory, "Busy")
    train_free_directory = os.path.join(train_directory, "Free")
    test_free_directory = os.path.join(test_directory, "Free")

    os.makedirs(train_busy_directory, exist_ok=True)
    os.makedirs(test_busy_directory, exist_ok=True)
    os.makedirs(train_free_directory, exist_ok=True)
    os.makedirs(test_free_directory, exist_ok=True)

    print("Copying images for train busy")
    for image in train_busy_images:
        shutil.copy(os.path.join(all_directory, "Busy", image), os.path.join(train_busy_directory, image))
    print("Copying images for test busy")
    for image in test_busy_images:
        shutil.copy(os.path.join(all_directory, "Busy", image), os.path.join(test_busy_directory, image))

    print("Copying images for train free")
    for image in train_free_images:
        shutil.copy(os.path.join(all_directory, "Free", image), os.path.join(train_free_directory, image))
    print("Copying images for test free")
    for image in test_free_images:
        shutil.copy(os.path.join(all_directory, "Free", image), os.path.join(test_free_directory, image))


def main():
    directory = "../data/datasets/"

    datasets = ['acmps', 'acpds', 'cnrpark', 'cnrpark_ext', 'pklot_all', 'pklot_puc', 'pklot_ufpr04', 'pklot_ufpr05',
                'spkl']

    for current_directory in os.listdir(directory):
        if current_directory in datasets:
            print(current_directory + "\n" + "-" * 10)

            current_directory_path = os.path.join(directory, current_directory)

            print("Splitting dataset into 80-20")
            split_dataset_into_train_test(current_directory_path, train_ratio=0.8)
            print()
            print("Splitting dataset into 50-50")
            split_dataset_into_train_test(current_directory_path, train_ratio=0.5)
            print()

            print("*****" * 10)


if __name__ == "__main__":
    main()
