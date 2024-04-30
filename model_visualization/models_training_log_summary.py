import os
import re
import os

LOG_PATH = "../data/logs/"
SPLIT_80_20 = "80_20_split/"
SPLIT_50_50 = "50_50_split/"


def get_logs():
    logs = []
    for root, dirs, files in os.walk(LOG_PATH):
        for file in files:
            if file.endswith(".txt"):
                logs.append(os.path.join(root, file))
    return logs


def get_summary(log):
    total_loss = 0
    total_train_acc = 0
    count = 0

    with open(log, "r") as file:
        print(f"Log: {log}")
        lines = file.readlines()
        for line in lines:
            if "Train Loss" in line:
                loss = float(re.search(r"Train Loss: (\d+\.\d+)", line).group(1))
                total_loss += loss
                count += 1
            if "Train Acc" in line:
                train_acc = float(re.search(r"Train Acc: (\d+\.\d+)", line).group(1))
                total_train_acc += train_acc
            if "Total training time" in line:
                training_time = float(re.search(r"Total training time: (\d+\.\d+)", line).group(1))

    avg_loss = total_loss / count if count > 0 else 0
    avg_train_acc = total_train_acc / count if count > 0 else 0

    print(f"Average Train Loss: {avg_loss}")
    print(f"Average Train Acc: {avg_train_acc}")
    print(f"Total training time: {training_time}")


if __name__ == "__main__":
    logs = get_logs()
    for log in logs:
        get_summary(log)
        print()
