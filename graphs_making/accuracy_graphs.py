import os

import matplotlib.pyplot as plt

DATASETS = ['cnr_cnr_park', 'cnr_cnr_park_ext', 'pklot_all', 'spkl_all', 'acpds_all']  # major datasets
DATASET_NAMES = ['CNRPark', 'CNRPark-Ext', 'PKLot', 'SPKL', 'ACDS']

# DATASETS = ['pklot_all', 'pklot_puc', 'pklot_ufpr04', 'pklot_ufpr05']  # pklot specific datasets
# DATASET_NAMES = ['PKLot', 'PKLot PUC', 'PKLot UFPR04', 'PKLot UFPR05']

# DATASETS = ['pklot_all', 'pklot_all_cloudy', 'pklot_all_rainy', 'pklot_all_sunny']
# DATASET_NAMES = ['PKLot', 'PKLot Cloudy', 'PKLot Rainy', 'PKLot Sunny']

# DATASETS = ['pklot_puc', 'pklot_puc_cloudy', 'pklot_puc_rainy', 'pklot_puc_sunny']
# DATASET_NAMES = ['PKLot PUC', 'PKLot PUC Cloudy', 'PKLot PUC Rainy', 'PKLot PUC Sunny']

# DATASETS = ['pklot_ufpr04', 'pklot_ufpr04_cloudy', 'pklot_ufpr04_rainy', 'pklot_ufpr04_sunny']
# DATASET_NAMES = ['PKLot UFPR04', 'PKLot UFPR04 Cloudy', 'PKLot UFPR04 Rainy', 'PKLot UFPR04 Sunny']

# DATASETS = ['pklot_ufpr05', 'pklot_ufpr05_cloudy', 'pklot_ufpr05_rainy', 'pklot_ufpr05_sunny']
# DATASET_NAMES = ['PKLot UFPR05', 'PKLot UFPR05 Cloudy', 'PKLot UFPR05 Rainy', 'PKLot UFPR05 Sunny']

PATH_TO_RESULTS = '../data/model_results/'
MODELS = ['mobilenet', 'squeezenet', 'shufflenet']
SAVE_PATH = '../data/graphs/plotting_accuracy/'


def main():
    index = 0
    total = len(DATASETS) * len(MODELS)
    dataset_results = {model: [] for model in MODELS}
    dataset_names = []

    for dataset in DATASETS:
        dataset_names.append(dataset)
        for model in MODELS:
            index += 1
            text_path = f'{PATH_TO_RESULTS}{dataset}/{dataset}_{model}_5.txt'

            # load the text file
            with open(text_path, 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]

                # extract accuracy (assuming it's the 7th line as per your print statements)
                accuracy = float(lines[6].split(": ")[1])

                # Append accuracy to the corresponding model list
                dataset_results[model].append(round(accuracy, 4))

                print(f'{index}/{total} models displayed')
                # format this into some readable format
                print(f'Train Model: {lines[0].split(": ")[1]}')
                print(f'Train Dataset: {lines[1].split(": ")[1]}')
                print(f'Train View: {lines[2].split(": ")[1]}')
                print(f'Train Epochs: {lines[3].split(": ")[1]}')
                print(f'Test Dataset: {lines[4].split(": ")[1]}')
                print(f'Test View: {lines[5].split(": ")[1]}')
                print(f'Accuracy: {lines[6].split(": ")[1]}')
                print(f'F1: {lines[7].split(": ")[1]}')
                print(f'ROC AUC: {lines[8].split(": ")[1]}')
                print(f'True Positives: {lines[9].split(": ")[1]}')
                print(f'False Positives: {lines[10].split(": ")[1]}')
                print(f'True Negatives: {lines[11].split(": ")[1]}')
                print(f'False Negatives: {lines[12].split(": ")[1]}')
                print(f'Confusion Matrix:')
                print(f'{lines[13]}')
                print(f'{lines[14]}')
                print(f'{lines[15]}')

                print('*' * 75)

    # Prepare data for plotting
    models = list(dataset_results.keys())
    num_models = len(models)
    dataset_names = ['CNRPark', 'CNRPark-Ext', 'PKLot', 'SPKL', 'ACDS']
    bar_width = 0.3  # Width of the bars

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        accuracies = [acc * 100 for acc in dataset_results[model]]
        x_positions = [j + i * bar_width for j in range(len(DATASETS))]
        ax.bar(x_positions, accuracies, bar_width, label=model)

    ax.set_xlabel('Datasety')
    ax.set_ylabel('Přesnost')
    ax.set_title('Přesnost modelů na jednotlivých počasích datasetech')
    ax.set_xticks([j + bar_width * (num_models - 1) / 2 for j in range(len(DATASETS))])
    ax.set_xticklabels(dataset_names)
    ax.legend(loc='lower right')

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    plt.savefig(f'{SAVE_PATH}accuracy_plot_all.png')
    plt.show()


if __name__ == '__main__':
    main()
