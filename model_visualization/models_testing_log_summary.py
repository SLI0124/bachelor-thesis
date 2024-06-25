import os
import re
from typing import Dict, Tuple, List

LOG_PATH = "../data/model_results/"


def get_all_log_files(log_path: str) -> List[str]:
    all_files = []
    for root, _, files in os.walk(log_path):
        for file in files:
            if file.endswith(".txt"):
                all_files.append(os.path.join(root, file))
    return all_files


def parse_log_file(file_path: str) -> Dict[str, str]:
    with open(file_path, 'r') as file:
        content = file.read()

    log_data = {
        "train_model": re.search(r'Train Model: (.+)', content).group(1),
        "train_dataset": re.search(r'Train Dataset: (.+)', content).group(1),
        "train_view": re.search(r'Train View: (.+)', content).group(1),
        "test_dataset": re.search(r'Test Dataset: (.+)', content).group(1),
        "test_view": re.search(r'Test View: (.+)', content).group(1),
        "accuracy": re.search(r'Accuracy: (.+)', content).group(1),
        "f1": re.search(r'F1: (.+)', content).group(1),
        "roc_auc": re.search(r'ROC AUC: (.+)', content).group(1),
        "true_positives": re.search(r'True Positives: (.+)', content).group(1),
        "false_positives": re.search(r'False Positives: (.+)', content).group(1),
        "true_negatives": re.search(r'True Negatives: (.+)', content).group(1),
        "false_negatives": re.search(r'False Negatives: (.+)', content).group(1),
    }

    return log_data


def collect_results(log_files: List[str]) -> Dict[Tuple[Tuple[str, str, str], Tuple[str, str]], Dict[str, str]]:
    results = {}
    for file in log_files:
        log_data = parse_log_file(file)
        key = ((log_data["train_dataset"], log_data["train_view"], log_data["train_model"]),
               (log_data["test_dataset"], log_data["test_view"]))
        results[key] = {
            "accuracy": log_data["accuracy"],
            "f1": log_data["f1"],
            "roc_auc": log_data["roc_auc"],
            "true_positives": log_data["true_positives"],
            "false_positives": log_data["false_positives"],
            "true_negatives": log_data["true_negatives"],
            "false_negatives": log_data["false_negatives"]
        }
    return results


def print_worst_accuracies(results: Dict[Tuple[Tuple[str, str, str], Tuple[str, str]], Dict[str, str]],
                           count: int = 5) -> None:
    sorted_results = sorted(results.items(), key=lambda x: float(x[1]["accuracy"]))[:count]
    print("Worst 5 accuracies:")
    for i, (key, value) in enumerate(sorted_results, 1):
        print(f"{i}. Train Dataset: {key[0][0]}, Train View: {key[0][1]}, Train Model: {key[0][2]}, "
              f"Test Dataset: {key[1][0]}, Test View: {key[1][1]}, Accuracy: {value['accuracy']}\n")


def print_best_accuracies(results: Dict[Tuple[Tuple[str, str, str], Tuple[str, str]], Dict[str, str]],
                          count: int = 5) -> None:
    sorted_results = sorted(results.items(), key=lambda x: float(x[1]["accuracy"]), reverse=True)[:count]
    print("Best 5 accuracies:")
    for i, (key, value) in enumerate(sorted_results, 1):
        print(f"{i}. Train Dataset: {key[0][0]}, Train View: {key[0][1]}, Train Model: {key[0][2]}, "
              f"Test Dataset: {key[1][0]}, Test View: {key[1][1]}, Accuracy: {value['accuracy']}\n")


def main() -> None:
    log_files = get_all_log_files(LOG_PATH)
    results = collect_results(log_files)
    print_worst_accuracies(results)
    print_best_accuracies(results)


if __name__ == "__main__":
    main()
