import matplotlib.pyplot as plt
import re
import os

LOG_DIR = '../data/logs/80_20_split/'
OUTPUT_DIR = '../data/graphs/model_visualization/'


def get_log_files(log_dir):
    return [os.path.join(root, file) for root, dirs, files in os.walk(log_dir) for file in files if
            file.endswith('.txt')]


def extract_values_from_log_file(log_file_path):
    metrics = ['Train Loss', 'Valid Loss', 'Train Acc', 'Valid Acc', 'Train F1', 'Valid F1', 'Train ROC AUC',
               'Valid ROC AUC']
    values = {metric: [] for metric in metrics}
    with open(log_file_path, 'r') as file:
        for line in file:
            for metric in metrics:
                if metric in line and 'Batch' not in line:
                    match = re.search(f'{metric}: ([0-9.]+)', line)
                    if match:
                        values[metric].append(float(match.group(1)))
    return values


def translate_metric_to_czech(metric):
    translations = {
        'Train Loss': 'Trénovací chyba',
        'Valid Loss': 'Validační chyba',
        'Train Acc': 'Trénovací přesnost',
        'Valid Acc': 'Validační přesnost',
        'Train F1': 'Trénovací F1 skóre',
        'Valid F1': 'Validační F1 skóre',
        'Train ROC AUC': 'Trénovací ROC AUC',
        'Valid ROC AUC': 'Validační ROC AUC',
        'Acc': 'Přesnost',
        'F1': 'F1 skóre',
        'ROC AUC': 'ROC AUC skóre',
        'Loss': 'Chyba'
    }
    return translations.get(metric, metric)


def plot_metrics(values, output_path, dataset_name, dataset_view, model_name):
    os.makedirs(output_path, exist_ok=True)
    metrics_pairs = [('Train Loss', 'Valid Loss'), ('Train Acc', 'Valid Acc'), ('Train F1', 'Valid F1'),
                     ('Train ROC AUC', 'Valid ROC AUC')]
    for train_metric, valid_metric in metrics_pairs:
        # Plot training metric
        plt.figure()
        plt.plot(values[train_metric], label=translate_metric_to_czech(train_metric), marker='o')
        plt.xlabel('Epocha')
        plt.ylabel(translate_metric_to_czech(train_metric))
        plt.title(f'{translate_metric_to_czech(train_metric)} - {dataset_name} - {dataset_view} - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_path, f'{train_metric.lower().replace(" ", "_")}.png'))
        plt.close()

        # Plot validation metric
        plt.figure()
        plt.plot(values[valid_metric], label=translate_metric_to_czech(valid_metric), marker='o')
        plt.xlabel('Epocha')
        plt.ylabel(translate_metric_to_czech(valid_metric))
        plt.title(f'{translate_metric_to_czech(valid_metric)} - {dataset_name} - {dataset_view} - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_path, f'{valid_metric.lower().replace(" ", "_")}.png'))
        plt.close()

        # Plot both training and validation metrics together
        plt.figure()
        plt.plot(values[train_metric], label=translate_metric_to_czech(train_metric), marker='o')
        plt.plot(values[valid_metric], label=translate_metric_to_czech(valid_metric), marker='x')
        plt.xlabel('Epocha')
        plt.ylabel(
            translate_metric_to_czech(train_metric.split()[1]))  # Use metric name without "Train" or "Valid" prefix
        plt.title(
            f'{translate_metric_to_czech(train_metric.split()[1])} - {dataset_name} - {dataset_view} - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_path,
                                 f'{train_metric.lower().replace(" ", "_")}_and_{valid_metric.lower().replace(" ", "_")}.png'))
        plt.close()


def main():
    log_files = get_log_files(LOG_DIR)
    # replace all \ with / in the log file path to avoid errors with Windows paths
    log_files = [log_file.replace('\\', '/') for log_file in log_files]
    for log_file_path in log_files:
        dataset_name = log_file_path.split('/')[4]
        dataset_view = log_file_path.split('/')[5]
        epochs = log_file_path.split('/')[6].split('_')[0]
        model_name = log_file_path.split('/')[7].split('.')[0]
        values = extract_values_from_log_file(log_file_path)
        output_path = os.path.join(OUTPUT_DIR, dataset_name, dataset_view, f"{epochs}_epochs", model_name)
        plot_metrics(values, output_path, dataset_name, dataset_view, model_name)


if __name__ == '__main__':
    main()
