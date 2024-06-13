# Parking Lot Occupancy Detection

### Úvod

Tento projekt se zabývá detekcí obsazenosti parkovacích míst na základě obrazových dat. Tento problém je řešen pomocí
konvolučních neuronových sítí. Výsledkem je model, textový výstup procesu učení a vyhodnocení.

## Datasety

Pro trénování a testování modelu byly použity datasety:

- [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)
- [CNRPARK a CNRPark+EXT](http://cnrpark.it/)
- [Action Camera Parking Lot Dataset](https://github.com/martin-marek/parking-space-occupancy)
- [ACPDS](https://github.com/Eighonet/parking-research)
- [ACMPS](https://github.com/Eighonet/parking-research)

# Instalace

### GPU podpora

Pokud máte k dispozici graficjou kartu Nvidia, můžete využít GPU podporu pro trénování modelu.
Pro instalaci je potřeba mít nainstalovaný CUDA Toolkit a CuDNN, nebo preferovaně nastavit Anaconda prostředí,
další informace naleznete na [stránkách PyTorch](https://pytorch.org/get-started/locally/).

## Instalace závislostí

```
pip install -r requirements.txt
```

# Stáhnutí a příprava datasetů

## Stáhnutí

Pro stáhnutí a přípravu datasetů je potřeba spustit [skript prepare_dataset](prepare_dataset.bash). Tento skript
stáhne všechny datasety, extrahuje je, vytvoří složky s obrázky a anotacemi, vytvoří split soubory s všemi obrázky
a anotacemi a nakonec rozdělí datasety na trénovací a testovací data. V základu je nastaveno rozdělení dat na 80-20% a
50-50%.

```bash
bash prepare_dataset.bash
```

## Rozdělení datasetů

Dalším krokem je rozdělení datasetů na trénovací a testovací data. Tento krok je nezbytný, a proto již byl proveden
v předchozím kroku v skriptu [prepare_dataset](prepare_dataset.bash). Výsledkem jsou soubory s obrázky a anotacemi
rozdělené na trénovací a testovací data. Pro trénování a testování modelu byly datasety rozděleny s následujícími
poměry:

- 80% trénovacích a 20% testovacích dat
- 50% trénovacích a 50% testovacích dat

```bash
cd creating_split_files
python3 create_split_files.py
python3 create_training_splits.py --split_ratio 80
python3 create_training_splits.py --split_ratio 50
```

Pokud chcete změnit poměr, můžete upravit parametr **_train_ratio_** funkce `split_dataset_into_train_test`
ve skriptu `split_dataset.py`. Například pro změnu poměru na 70-30%:

```bash
python3 split_dataset_into_train_test(dataset, train_ratio=70)
```

# Spuštění

## Trénování modelů

Pokud chcete trénovat model, podívejte se na soubor `model_training.py` ve složce [model_training](model_training).
Zde je již připravený další [README soubor](model_training/README.md), který obsahuje parametry pro trénování modelu a
nějaké další příklady použití. Příklad jednoho z nich:

```bash
cd model_training
python3 model_training.py --dataset cnr --camera_view cnr_park_ext --model mobilenet --k_fold 5 --num_epochs 5
```

## evaluace modelů, vyhodnocení a vizualizace výsledků

TODO

## Vyhodnocení modelů

TODO

## Vizualizace výsledků

TODO