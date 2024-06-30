import os

PATH_TO_RESULTS = '../data/model_results/'

# DATASETS = ['cnr_cnr_park', 'cnr_cnr_park_ext', 'pklot_all', 'spkl_all', 'acpds_all']  # major datasets

# DATASETS = ['pklot_all', 'pklot_all_cloudy', 'pklot_all_rainy', 'pklot_all_sunny',
#             'pklot_ufpr04', 'pklot_ufpr04_cloudy', 'pklot_ufpr04_rainy', 'pklot_ufpr04_sunny',
#             'pklot_ufpr05', 'pklot_ufpr05_cloudy', 'pklot_ufpr05_rainy',
#             'pklot_ufpr05_sunny']  # pklot specific datasets


# DATASETS = ['pklot_all', 'pklot_all_rainy', 'pklot_all_sunny', 'pklot_all_cloudy', 'pklot_puc', 'pklot_puc_rainy',
#             'pklot_puc_sunny', 'pklot_puc_cloudy', 'pklot_ufpr04', 'pklot_ufpr04_rainy', 'pklot_ufpr04_sunny',
#             'pklot_ufpr04_cloudy', 'pklot_ufpr05', 'pklot_ufpr05_rainy', 'pklot_ufpr05_sunny',
#             'pklot_ufpr05_cloudy']  # pklot specific datasets

DATASETS = ['cnr_cnr_park_ext', 'cnr_cnr_park_ext_cloudy', 'cnr_cnr_park_ext_rainy',
            'cnr_cnr_park_ext_sunny']  # weather cnr specific datasets

# DATASETS = ['pklot_all', 'pklot_all_cloudy', 'pklot_all_rainy', 'pklot_all_sunny']

# DATASETS = ['pklot_all', 'pklot_puc', 'pklot_ufpr04', 'pklot_ufpr05']  # pklot specific datasets


# MODELS = ['mobilenet', 'squeezenet', 'shufflenet']
MODELS = ['mobilenet']
EPOCHS = 5


def main():
    index = 0
    total = len(DATASETS) * len(MODELS) * len(DATASETS)
    for train_dataset in DATASETS:
        for test_dataset in DATASETS:
            for model in MODELS:
                index += 1
                text_path = f'{PATH_TO_RESULTS}{train_dataset}/{test_dataset}_{model}_5.txt'

                # load the text file
                with open(text_path, 'r') as f:
                    lines = f.readlines()
                    lines = [line.strip() for line in lines]

                    # print(f'{index}/{total} models displayed')
                    # format this into some readable format
                    # print(f'Train Model: {lines[0].split(": ")[1]}')
                    # print(f'Train Dataset: {lines[1].split(": ")[1]}')
                    print(f'Train View: {lines[2].split(": ")[1].split("_")[-1]}')
                    # print(f'Train Epochs: {lines[3].split(": ")[1]}')
                    # print(f'\tTest Dataset: {lines[4].split(": ")[1]}')
                    print(f'\tTest View: {lines[5].split(": ")[1].split("_")[-1]}')
                    # print(f'Accuracy: {float(lines[6].split(": ")[1]):.5f}')
                    # print accuracy in percentage
                    print(f'Accuracy: {float(lines[6].split(": ")[1]) * 100:.3f}%')
                    # print(f'F1: {lines[7].split(": ")[1]}')
                    # print(f'ROC AUC: {lines[8].split(": ")[1]}')
                    # print(f'True Positives: {lines[9].split(": ")[1]}')
                    # print(f'False Positives: {lines[10].split(": ")[1]}')
                    # print(f'True Negatives: {lines[11].split(": ")[1]}')
                    # print(f'False Negatives: {lines[12].split(": ")[1]}')
                    # print(f'Confusion Matrix:')
                    # print(f'{lines[13]}')
                    # print(f'{lines[14]}')
                    # print(f'{lines[15]}')
                    print('*' * 50)


if __name__ == "__main__":
    main()
