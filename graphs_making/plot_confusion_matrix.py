import os

import matplotlib.pyplot as plt

SAVE_PATH = '../data/graphs/plotting_confusion_matrix/'
PATH_TO_RESULTS = '../data/model_results/'

DATASETS = ['spkl_all', 'cnr_cnr_park', 'acpds_all']
MODELS = ['mobilenet', 'squeezenet', 'shufflenet']


def main():
    index = 0
    total = len(DATASETS) * len(MODELS)
    for dataset in DATASETS:
        for model in MODELS:
            index += 1
            text_path = f'{PATH_TO_RESULTS}{dataset}/{dataset}_{model}_5.txt'

            # load the text file
            with open(text_path, 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]

                print(f'{index}/{total} models displayed')
                # format this into some readable format
                print(f'Train Model: {lines[0].split(": ")[1]}')

                if not lines[1].split(": ")[1] in ['cnr', 'spkl']:
                    continue

                train_dataset = lines[1].split(": ")[1]
                print(f'Train Dataset: {train_dataset}')
                train_view = lines[2].split(": ")[1]
                print(f'Train View: {train_view}')

                train_epochs = lines[3].split(": ")[1]
                print(f'Train Epochs: {train_epochs}')

                test_dataset = lines[4].split(": ")[1]
                print(f'Test Dataset: {test_dataset}')

                test_view = lines[5].split(": ")[1]
                print(f'Test View: {test_view}')

                accuracy = float(lines[6].split(": ")[1])
                print(f'Accuracy: {accuracy}')

                f1 = float(lines[7].split(": ")[1])
                print(f'F1: {f1}')

                roc_auc = float(lines[8].split(": ")[1])
                print(f'ROC AUC: {roc_auc}')

                true_positives = int(lines[9].split(": ")[1])
                print(f'True Positives: {true_positives}')

                false_positives = int(lines[10].split(": ")[1])
                print(f'False Positives: {false_positives}')

                true_negatives = int(lines[11].split(": ")[1])
                print(f'True Negatives: {true_negatives}')

                false_negatives = int(lines[12].split(": ")[1])
                print(f'False Negatives: {false_negatives}')

                print(f'Confusion Matrix:')
                print(f'{lines[13]}')
                print(f'{lines[14]}')
                print(f'{lines[15]}')
                print('*' * 75)

                tp = int(lines[9].split(": ")[1])
                fp = int(lines[10].split(": ")[1])
                tn = int(lines[11].split(": ")[1])
                fn = int(lines[12].split(": ")[1])

                cm = [[tn, fp], [fn, tp]]

                # Plot confusion matrix with numbers
                fig, ax = plt.subplots()
                cax = ax.matshow(cm, cmap='Blues')
                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, str(cm[i][j]), va='center', ha='center')
                # plt.title(f'Matice záměn pro {test_dataset}_{test_view}/{train_dataset}_{train_view}_{model}')

                plt.xlabel('Předpověď')
                plt.ylabel('Skutečnost')
                plt.xticks([0, 1], ['Negativní', 'Pozitivní'])
                plt.yticks([0, 1], ['Negativní', 'Pozitivní'])
                plt.colorbar(cax)

                save_path = f'{SAVE_PATH}{test_dataset}_{test_view}/{train_dataset}_{train_view}_{model}.png'
                if not os.path.exists(f'{SAVE_PATH}{test_dataset}_{test_view}/'):
                    os.makedirs(f'{SAVE_PATH}{test_dataset}_{test_view}/')
                # plt.savefig(save_path)
                plt.show()


if __name__ == "__main__":
    main()
