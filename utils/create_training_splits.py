import os
import random
import argparse

SPLIT_DIR = '../data/splits'


def save_images_to_file(split_directory: str, dataset: str, images: list, split_ratio: float) -> None:
    if not os.path.exists(split_directory):
        os.makedirs(split_directory)

    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    training_images = images[:split_index]
    testing_images = images[split_index:]

    training_ratio = int(split_ratio)
    testing_ratio = 100 - training_ratio

    with open(os.path.join(split_directory, f'train_{training_ratio}_{testing_ratio}.txt'), 'w') as f:
        for image in training_images:
            f.write(image + '\n')

    with open(os.path.join(split_directory, f'test_{training_ratio}_{testing_ratio}.txt'), 'w') as f:
        for image in testing_images:
            f.write(image + '\n')


def main(split_ratio):
    datasets = os.listdir(SPLIT_DIR)

    for dataset in datasets:
        print(f'Processing {dataset}')
        with open(os.path.join(SPLIT_DIR, dataset, 'all.txt'), 'r') as f:
            images = f.readlines()
            images = [image.strip() for image in images]
            save_images_to_file(os.path.join(SPLIT_DIR, dataset), 'all.txt', images, split_ratio)
        print(f'Done processing {dataset}')

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process split ratio.')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Split ratio for train and test datasets')
    args = parser.parse_args()
    print(f'Split ratio: {args.split_ratio}')
    main(args.split_ratio)
