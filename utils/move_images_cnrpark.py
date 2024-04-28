import os
import shutil


def main():
    path = "../data/datasets/cnr/"

    cnrpark_path = path + "CNRPark-Patches-150x150/"
    cnrpark_ext = path + "CNR-EXT-Patches-150x150/"

    cnr_park_all = "../data/datasets/cnrpark/all/"
    cnr_park_ext_all = "../data/datasets/cnrpark_ext/all/"

    print("Working on CNRPark-Patches-150x150")

    if not os.path.exists(cnr_park_all):
        os.makedirs(cnr_park_all)
    else:
        raise Exception("Directory already exists.")

    for directory in os.listdir(cnrpark_path):
        for label in os.listdir(cnrpark_path + directory):
            label_directory = os.path.join(cnrpark_path, directory, label)
            for filename in os.listdir(label_directory):
                if not os.path.exists(os.path.join(cnr_park_all, label)):
                    os.makedirs(os.path.join(cnr_park_all, label))
                shutil.copy2(os.path.join(label_directory, filename),
                             os.path.join(cnr_park_all, label, filename))

    print("Moved all images from CNRPark-Patches-150x150 to cnrpark/all")

    print("\n**********************************\n")

    print("Working on CNR-EXT-Patches-150x150")

    if not os.path.exists(cnr_park_ext_all):
        os.makedirs(cnr_park_ext_all)
    else:
        raise Exception("Directory already exists.")

    file_all_labels = cnrpark_ext + "LABELS/all.txt"
    labels = []
    with open(file_all_labels, "r") as f:
        for line in f:
            labels.append(line.strip())

    cnrpark_ext_patches = cnrpark_ext + "PATCHES/"
    free_images = []
    busy_images = []
    for label in labels:
        label, status = label.split(" ")
        if status == "0":
            free_images.append(label)
        else:
            busy_images.append(label)

    if not os.path.exists(os.path.join(cnr_park_ext_all, "Free")):
        os.makedirs(os.path.join(cnr_park_ext_all, "Free"))

    if not os.path.exists(os.path.join(cnr_park_ext_all, "Busy")):
        os.makedirs(os.path.join(cnr_park_ext_all, "Busy"))

    for image in free_images:
        image_name = image.split("/")[-1]
        shutil.copy2(os.path.join(cnrpark_ext_patches, image), os.path.join(cnr_park_ext_all, "Free", image_name))

    for image in busy_images:
        image_name = image.split("/")[-1]
        shutil.copy2(os.path.join(cnrpark_ext_patches, image), os.path.join(cnr_park_ext_all, "Busy", image_name))

    print("Moved all images from CNR-EXT-Patches-150x150 to cnrpark_ext/all")


if __name__ == "__main__":
    main()
