# Informace o skriptu pro trénování modelu na datasetu PKLot

Tento skript slouží k trénování modelu na datasetu PKLot pomocí frameworku PyTorch.

## Parametry:

- `--dataset`: Určuje dataset, který se má použít.
    - **Možné hodnoty:** 'puc', 'ufpr04', 'ufpr05', 'all'
    - **Výchozí hodnota:** žádná (povinný parametr)
    - **Příklady:**
      ```bash
      python train_model.py --dataset puc --model alexnet --epochs 10 --weights default --eighty_twenty_split True
      python train_model.py --dataset ufpr04 --model squeezenet --epochs 20 --weights none --eighty_twenty_split False
      ```

- `--model`: Určuje architekturu modelu, která se má použít.
    - **Možné hodnoty:** 'alexnet', 'mobilenet', 'squeezenet'
    - **Výchozí hodnota:** žádná (povinný parametr)
    - **Příklady:**
      ```bash
      python train_model.py --dataset puc --model alexnet --epochs 10 --weights default --eighty_twenty_split True
      python train_model.py --dataset ufpr04 --model squeezenet --epochs 20 --weights none --eighty_twenty_split False
      ```

- `--epochs`: Určuje počet epoch pro trénování modelu.
    - **Možné hodnoty:** Celé číslo (např. 10, 20, 50)
    - **Výchozí hodnota:** žádná (povinný parametr)
    - **Příklady:**
      ```bash
      python train_model.py --dataset puc --model alexnet --epochs 10 --weights default --eighty_twenty_split True
      python train_model.py --dataset ufpr04 --model squeezenet --epochs 20 --weights none --eighty_twenty_split False
      ```

- `--eighty_twenty_split`: Určuje, zda se má použít rozdělení 80/20 pro trénování a testování.
    - **Možné hodnoty:** True, False
    - **Výchozí hodnota:** žádná (povinný parametr)
    - **Příklady:**
      ```bash
      python train_model.py --dataset puc --model alexnet --epochs 10 --weights default --eighty_twenty_split True
      python train_model.py --dataset ufpr04 --model squeezenet --epochs 20 --weights none --eighty_twenty_split False
      ```

- `--weights`: Určuje váhy, které se mají použít pro model.
    - **Možné hodnoty:** 'default', 'none'
    - **Výchozí hodnota:** žádná (povinný parametr)
    - **Příklady:**
      ```bash
      python train_model.py --dataset puc --model alexnet --epochs 10 --weights default --eighty_twenty_split True
      python train_model.py --dataset ufpr04 --model squeezenet --epochs 20 --weights none --eighty_twenty_split False
      ```

