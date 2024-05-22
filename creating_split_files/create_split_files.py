import os
import random
from typing import List, Dict

SPLIT_DIR = '../data/splits'


def get_image_paths(local_directory: str, status_labels: Dict[str, int]) -> Dict[str, List[str]]:
    images = {label: os.listdir(os.path.join(local_directory, label)) for label in status_labels}
    formatted_images = {
        label: [
            str(os.path.join(local_directory, label, image)).replace('\\', '/') + f' {status_labels[label]}' for image
            in images[label]
        ] for label in status_labels
    }
    return {label: ['/'.join(image.split('/')[3:]) for image in formatted_images[label]] for label in status_labels}


def save_images_to_file(split_directory: str, filename: str, images: List[str]) -> None:
    if not os.path.exists(split_directory):
        os.makedirs(split_directory)

    with open(os.path.join(split_directory, filename), 'w') as f:
        for image in images:
            f.write(image + '\n')


def process_and_save_acmps_acpds_spkl(directory: str, dataset_name: str) -> None:
    local_directory = os.path.join(directory, 'patch_markup')
    if dataset_name == 'acmps':
        local_directory = os.path.join(local_directory, 'classes')

    status_labels = {'Busy': 1, 'Free': 0}
    images = get_image_paths(local_directory, status_labels)

    for label in images:
        random.shuffle(images[label])

    all_images = images['Busy'] + images['Free']
    save_images_to_file(os.path.join(SPLIT_DIR, dataset_name), 'all.txt', all_images)


def process_and_save_pklot(directory: str, dataset_name: str) -> None:
    local_directory = os.path.join(directory, 'PKLotSegmented')
    split_directory = os.path.join(SPLIT_DIR, dataset_name)

    all_images = []
    for folder in ['PUC', 'UFPR04', 'UFPR05']:
        folder_images = []
        for weather in ['Cloudy', 'Rainy', 'Sunny']:
            for date in os.listdir(os.path.join(local_directory, folder, weather)):
                for status in ['Empty', 'Occupied']:
                    status_label = 1 if status == 'Occupied' else 0
                    status_path = os.path.join(local_directory, folder, weather, date, status)
                    if os.path.exists(status_path):
                        images = os.listdir(status_path)
                        formatted_images = [
                            str(os.path.join(local_directory,
                                             folder,
                                             weather,
                                             date,
                                             status,
                                             image)).replace('\\', '/') + f' {status_label}'
                            for image in images
                        ]
                        formatted_images = ['/'.join(image.split('/')[3:]) for image in formatted_images]
                        random.shuffle(formatted_images)
                        folder_images.extend(formatted_images)

        save_images_to_file(split_directory, f'{folder.lower()}_all.txt', folder_images)
        all_images.extend(folder_images)

    save_images_to_file(split_directory, 'all.txt', all_images)


def process_and_save_cnr(directory: str, dataset_name: str) -> None:
    """
    CNRPark-Patches-150x150 only appends path to the image, not the actual label.
    cnr/CNRPark-Patches-150x150/A/busy/20150703_1555_25.jpg
    cnr/CNRPark-Patches-150x150/B/busy/20150708_1135_45.jpg
    cnr/CNRPark-Patches-150x150/A/busy/20150703_1130_50.jpg
    cnr/CNRPark-Patches-150x150/A/busy/20150703_1430_14.jpg
    cnr/CNRPark-Patches-150x150/A/busy/20150703_0905_2.jpg
    cnr/CNRPark-Patches-150x150/A/busy/20150703_1240_49.jpg
    cnr/CNRPark-Patches-150x150/B/busy/20150708_1615_17.jpg
    """

    cnr_ext_file = os.path.join(directory, 'CNR-EXT-Patches-150x150', 'LABELS', 'all.txt')
    with open(cnr_ext_file, 'r') as f:
        lines = f.readlines()

    formatted_lines = [
        str(os.path.join(directory, 'CNR-EXT-Patches-150x150', 'PATCHES', line.strip())).replace('\\', '/') for line in
        lines
    ]
    formatted_lines = ['/'.join(line.split('/')[3:]) for line in formatted_lines]
    random.shuffle(formatted_lines)

    split_directory = os.path.join(SPLIT_DIR, dataset_name)
    save_images_to_file(split_directory, 'cnr_park_ext_all.txt', formatted_lines)

    cnr_park_busy = []
    cnr_park_free = []
    for folder in ['A', 'B']:
        for status in ['busy', 'free']:
            status_path = os.path.join(directory, 'CNRPark-Patches-150x150', folder, status)
            images = os.listdir(status_path)
            formatted_images = [
                str(os.path.join(status_path, image)).replace('\\', '/') for image in images
            ]
            formatted_images = ['/'.join(image.split('/')[3:]) for image in formatted_images]
            if status == 'busy':
                # cnr_park_busy.extend(formatted_images)
                # add the path and label to the list
                cnr_park_busy.extend([f'{image} 1' for image in formatted_images])
            else:
                # cnr_park_free.extend(formatted_images)
                # add the path and label to the list
                cnr_park_free.extend([f'{image} 0' for image in formatted_images])

    random.shuffle(cnr_park_busy)
    random.shuffle(cnr_park_free)

    save_images_to_file(split_directory, 'cnr_park_all.txt', cnr_park_busy + cnr_park_free)

    all_cnr_images = cnr_park_busy + cnr_park_free + formatted_lines
    save_images_to_file(split_directory, 'all.txt', all_cnr_images)


def main() -> None:
    directory = '../data/datasets/'
    dataset_processors = {
        'acmps': process_and_save_acmps_acpds_spkl,
        'acpds': process_and_save_acmps_acpds_spkl,
        'spkl': process_and_save_acmps_acpds_spkl,
        'cnr': process_and_save_cnr,
        'pklot': process_and_save_pklot,
    }

    for current_directory in os.listdir(directory):
        print(f'Processing {current_directory}')
        dataset_directory = os.path.join(directory, current_directory)
        processor = dataset_processors.get(current_directory)
        if processor:
            processor(dataset_directory, current_directory)
        else:
            print(f'{current_directory} is not a recognized dataset')

        print('Done with processing')

    print('\nDone with all datasets')


if __name__ == '__main__':
    main()
