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
        'pklot': ['puc_all.txt', 'ufpr04_all.txt', 'ufpr05_all.txt'],
        'cnr': ['cnr_park_all.txt', 'cnr_park_ext_all.txt']
    }

    if dataset in specific_folders:
        for folder in specific_folders[dataset]:
            prefix = '_'.join(folder.split('_')[:-1])  # get the prefix (e.g., cnr_park, cnr_park_ext)
            with open(os.path.join(SPLIT_DIR, dataset, folder), 'r') as f:
                images = f.readlines()
                images = [image.strip() for image in images]
                save_images_to_file(os.path.join(SPLIT_DIR, dataset), images, split_ratio, prefix)

    # Process 'all.txt' for both 'pklot' and 'cnr'
    with open(os.path.join(SPLIT_DIR, dataset, 'all.txt'), 'r') as f:
        images = f.readlines()
        images = [image.strip() for image in images]
        save_images_to_file(os.path.join(SPLIT_DIR, dataset), images, split_ratio, 'all')

    print(f'Done processing {dataset}')


def main(split_ratio: int):
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
    print(f'Split ratio: {args.split_ratio}')
    main(args.split_ratio)
