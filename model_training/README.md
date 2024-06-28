# Návod k použití trénování modelu

## Parametry vstupu

- `--dataset`: Název datasetu, který chcete použít. Vyberte mezi:
    - `pklot`
    - `cnrpark`
    - `acpds`
    - `spkl`

- `--camera_view`:
    - pro PKLot vyberte mezi:
        - `all`, `all_cloudy`, `all_sunny`, `all_rainy`,
        - `puc`, `puc_cloudy`, `puc_sunny`, `puc_rainy`,
        - `ufpr04_cloudy`, `ufpr04_sunny`, `ufpr04_rainy`,
        - `ufpr05_cloudy`, `ufpr05_sunny`, `ufpr05_rainy`
    - pro CNRPark vyberte mezi:
        - `cnr_park` (camera A a B), `cnr_park_ext` (camera 0-9), `all` (cnr_park a cnr_park_ext),
        - `cnr_park_ext_cloudy`, `cnr_park_ext_sunny`, `cnr_park_ext_rainy`
    - pro ACPDS vyberte `all`
    - pro SPKL vyberte `all`

- `--model`: Název modelu, který chcete použít. Vyberte mezi:
    - `mobilenet`
    - `squeezenet`
    - `shufflenet`

- `--train_split`: Procento datasetu, které chcete použít pro trénování. Zbytek bude použit pro testování. Dvě základní
  možnosti, které jsou nastaveny výchozí hodnotou:
    - 80 - 80% trénovacích dat, 20% testovacích dat
    - 50 - 50% trénovacích dat, 50% testovacích dat

  Pokud byste chtěli použít jiné procento, spustě [skript](../creating_split_files/create_training_splits.py) s
  vašim vlastním parametrem `--split_ratio`.
  ```bash
  cd ../create_training_splits
  python3 create_training_splits.py --split_ratio 70
  ```

- `--k_fold`: Počet složek pro křížovou validaci. Výchozí hodnota je 5.
- `--num_epochs`: Počet epoch pro trénování modelu. Výchozí hodnota je 10.

Nejde použít `--k_fold` a `--train_split` zároveň. Pokud chcete použít křížovou validaci, nastavte `--k_fold` na
požadovaný počet složek. Pokud chcete použít trénovací dělení, nastavte `--train_split` na požadované procento.

## Příklad použití

```bash
python3 model_training.py --dataset acpds --camera_view all --model mobilenet --train_split 80
```

```bash
python3 model_training.py --dataset cnr --camera_view cnr_park_ext_sunny --model mobilenet --k_fold 5 --num_epochs 5 
```

```bash
python model_training.py --dataset pklot --camera_view all_rainy --model mobilenet --k_fold 5 --num_epochs 5
```

## Výstup

- Výstupem je logovací soubor, který obsahuje informace o průběhu trénování modelu. Všechny logy jsou uloženy v
  této [složce](../data/logs), kde jsou rozděleny podle datasetu, pohledu kamery, modelu a trénovacího dělení.
- Model je uložen podobně jako logy, ale v [složce](../data/models).
- Výstupní soubor obsahuje informace o trénování modelu, jako je ztráta, přesnost, matice záměn a další.

Pokud chcete vizualizovat více metrik, jako je ztráta, přesnost, matice záměn atd., můžete nakouknot do složky a jejího
obsahu.
Je zde [popisující skript](../model_visualization/README.md) pro vizualizaci modelu.
