import os
import random
import argparse

SPLIT_DIR = '../data/splits'


def save_images_to_file(split_directory: str, images: list, split_ratio: int, prefix: str) -> None:
    if not os.path.exists(split_directory):
        os.makedirs(split_directory)

    random.shuffle(images)

    train_test_split = int(len(images) * split_ratio / 100)
    train_images = images[:train_test_split]

    test_images = images[train_test_split:]
    test_split = int(len(test_images) / 2)
    validation_images = test_images[:test_split]
    test_images = test_images[test_split:]

    with open(os.path.join(split_directory, f'{prefix}_train_{split_ratio}_{100 - split_ratio}.txt'), 'w') as f:
        for image in train_images:
            f.write(image + '\n')

    with open(os.path.join(split_directory, f'{prefix}_valid_{split_ratio}_{100 - split_ratio}.txt'), 'w') as f:
        for image in validation_images:
            f.write(image + '\n')

    with open(os.path.join(split_directory, f'{prefix}_test_{split_ratio}_{100 - split_ratio}.txt'), 'w') as f:
        for image in test_images:
            f.write(image + '\n')


def process_dataset(dataset: str, split_ratio: int) -> None:
    print(f'Processing {dataset}')

    specific_folders = {
        'pklot': ['all.txt', 'puc_all.txt', 'ufpr04_all.txt', 'ufpr05_all.txt', 'all_rainy.txt', 'all_sunny.txt',
                  'all_cloudy.txt', 'puc_rainy.txt', 'puc_sunny.txt', 'puc_cloudy.txt', 'ufpr04_rainy.txt',
                  'ufpr04_sunny.txt', 'ufpr04_cloudy.txt', 'ufpr05_rainy.txt', 'ufpr05_sunny.txt', 'ufpr05_cloudy.txt'],
        'cnr': ['all.txt', 'cnr_park_all.txt', 'cnr_park_ext_all.txt', 'cnr_park_ext_sunny.txt',
                'cnr_park_ext_rainy.txt', 'cnr_park_ext_cloudy.txt']
    }

    # Process specific folders (e.g., cnr_park, cnr_park_ext)
    if dataset in specific_folders:
        for folder in specific_folders[dataset]:
            prefix = folder.split('.')[0]
            with open(os.path.join(SPLIT_DIR, dataset, folder), 'r') as f:
                images = f.readlines()
                images = [image.strip() for image in images]
                save_images_to_file(os.path.join(SPLIT_DIR, dataset), images, split_ratio, prefix)

    # Process other datasets (e.g., acmps, acpds, spkl)
    with open(os.path.join(SPLIT_DIR, dataset, 'all.txt'), 'r') as f:
        images = f.readlines()
        images = [image.strip() for image in images]
        save_images_to_file(os.path.join(SPLIT_DIR, dataset), images, split_ratio, 'all')

    print()

    print(f'Done processing {dataset}')


def main(split_ratio: int) -> None:
    datasets = os.listdir(SPLIT_DIR)

    for dataset in datasets:
        if dataset in ['CNRParkAB', 'CNRPark-EXT', 'Combined', 'pklot_download']:
            continue
        else:
            process_dataset(dataset, split_ratio)

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process split ratio.')
    parser.add_argument('--split_ratio', type=int, required=True, help='Split ratio for train and test datasets')
    args = parser.parse_args()
    print(f'Split ratio: {args.split_ratio}\n')
    main(args.split_ratio)
