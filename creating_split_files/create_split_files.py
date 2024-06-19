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

    process_weather_pklot(dataset_name)


def process_and_save_cnr(directory: str, dataset_name: str) -> None:
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
                cnr_park_busy.extend([f'{image} 1' for image in formatted_images])
            else:
                cnr_park_free.extend([f'{image} 0' for image in formatted_images])

    random.shuffle(cnr_park_busy)
    random.shuffle(cnr_park_free)

    save_images_to_file(split_directory, 'cnr_park_all.txt', cnr_park_busy + cnr_park_free)

    all_cnr_images = cnr_park_busy + cnr_park_free + formatted_lines
    save_images_to_file(split_directory, 'all.txt', all_cnr_images)

    process_weather_cnr_ext(directory, dataset_name)


def process_weather_cnr_ext(directory: str, dataset_name: str) -> None:
    cnr_ext_all_file = os.path.join(SPLIT_DIR, dataset_name, 'cnr_park_ext_all.txt')
    with open(cnr_ext_all_file, 'r') as f:
        lines = f.readlines()

    formatted_lines = [
        str(os.path.join(directory, 'CNR-EXT-Patches-150x150', 'PATCHES', line.strip())).replace('\\', '/') for line in
        lines
    ]

    sunny_images = [line.replace('../data/datasets/', '', 1) for line in formatted_lines if 'SUNNY' in line]
    rainy_images = [line.replace('../data/datasets/', '', 1) for line in formatted_lines if 'RAINY' in line]
    cloudy_images = [line.replace('../data/datasets/', '', 1) for line in formatted_lines if 'OVERCAST' in line]

    split_directory = os.path.join(SPLIT_DIR, dataset_name)
    save_images_to_file(split_directory, 'cnr_park_ext_sunny.txt', sunny_images)
    save_images_to_file(split_directory, 'cnr_park_ext_rainy.txt', rainy_images)
    save_images_to_file(split_directory, 'cnr_park_ext_cloudy.txt', cloudy_images)

    print('Done processing CNR Park EXT weather images')


def get_list_of_weather_images(directory: str) -> List[str]:
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                images.append(os.path.join(root, file))
    return images


def process_weather_pklot(dataset_name: str) -> None:
    split_directory = os.path.join(SPLIT_DIR, dataset_name)
    file_names = ['all', 'puc', 'ufpr04', 'ufpr05']
    weather_types = ['Rainy', 'Sunny', 'Cloudy']

    for file_name in file_names:
        file_path = os.path.join(split_directory, f'{file_name}_all.txt')
        with open(file_path, 'r') as f:
            images = f.readlines()
            images = [image.strip() for image in images]

        for weather_type in weather_types:
            filtered_images = [image for image in images if weather_type in image]
            save_images_to_file(split_directory, f'{file_name.lower()}_{weather_type.lower()}.txt', filtered_images)

    print('Done processing PKLot weather images')


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
