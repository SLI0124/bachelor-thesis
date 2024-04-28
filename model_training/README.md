# Návod k použití trénování modelu

## Parametry vstupu

- `--dataset`: Název datasetu, který chcete použít. Vyberte mezi:
    - `pklot`
    - `cnrpark`
    - `acmps`
    - `acpds`
    - `spkl`

- `--camera_view`:
    - pro dataset PKLot vyberte mezi `puc`, `ufpr04`, `ufpr05` a `all`
    - pro CNRPark vyberte mezi `cnrpark` (kamera A a B), `cnrpark_ext` (kamera 0-9 a cnrpark) a `all`
    - pro ACPDS vyberte `all`
    - pro ACPMS vyberte `all`
    - pro SPKL vyberte `all`

- `--model`: Název modelu, který chcete použít. Vyberte mezi:
    - `alexnet`
    - `mobilenet`
    - `squeezenet`
    - `shufflenet`

- `--train_split`: Procento datasetu, které chcete použít pro trénování. Zbytek bude použit pro testování. Výchozí
  hodnota je 80. Pokud byste chtěli použít jiné procento, spustě [skript](../utils/split_datasets.py) pro rozdělení
  datasetu.
    - 80 - 80% trénovacích dat, 20% testovacích dat
    - 50 - 50% trénovacích dat, 50% testovacích dat

## Příklad použití

```bash
python model_training.py --dataset pklot --camera_view puc --model alexnet --train_split 80
python model_training.py --dataset pklot --camera_view ufpr04 --model mobilenet --train_split 50
python model_training.py --dataset pklot --camera_view ufpr05 --model squeezenet --train_split 80
python model_training.py --dataset pklot --camera_view all --model shufflenet --train_split 80
```

```bash
python model_training.py --dataset cnrpark --camera_view cnrpark --model alexnet --train_split 80
python model_training.py --dataset cnrpark --camera_view cnrparkext --model mobilenet --train_split 50
python model_training.py --dataset cnrpark --camera_view all --model squeezenet --train_split 80
```

```bash
python model_training.py --dataset acmps --camera_view all --model mobilenet --train_split 50
```

```bash
python model_training.py --dataset acpds --camera_view all --model shufflenet --train_split 80
```

```bash
python model_training.py --dataset spkl --camera_view all --model alexnet --train_split 50
```

## Výstup

- Výstupe je logovací soubor, který obsahuje informace o průběhu trénování modelu.
- Model je uložen v adresáři `../data/models/{dataset_name}/{camera_view}/{model_name}/{train_split}/best_model.pth`
- Výstupní soubor obsahuje informace o trénování modelu, jako je ztráta, přesnost, matice záměn a další.

Pokud chcete vizualizovat více metrik, jako je ztráta, přesnost, matice záměn atd., můžete nakouknot do složky a jejího
obsahu.
Je zde [popisující skript](../model_visualization/README.md) pro vizualizaci modelu.




