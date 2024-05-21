# Parking Lot Occupancy Detection

### Úvod

Tento projekt se zabývá detekcí obsazenosti parkovacích míst na základě obrazových dat. Tento problém je řešen pomocí
konvolučních neuronových sítí. Výsledkem je model, textový výstup procesu učení a vyhodnocení.

## Datasety

Pro trénování a testování modelu byly použity datasety:

- [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)
- [CNRPark+EXT](http://cnrpark.it/)
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

Pro stáhnutí a přípravu datasetů je potřeba spustit [skript download_dataset](prepare_dataset.bash). Tento skript
stáhne všechny datasety a připraví je pro trénování modelu.

```
bash download_dataset.bash
```

## Rozdělení datasetů

Dále je potřeba připravit datasety pro trénování a testování modelu.
První skript vytvoří soubory s všemi obrázky a anotacemi, které budou následně použity pro trénování a testování modelu.
Druhý skript rozdělí datasety na trénovací a testovací data a má jediný parametr, kterým je poměr rozdělení
trénvacíchdat. Výchozí hodnota je 80%. Základními hodnotami, které jsou použity v této práci, jsou 80% a 50%.

```
cd utils
python prepare_datasets.py
```

Pro trénování a testování modelu byly datasety rozděleny na trénovací a testovací data. Výchozí poměr je rozdělen na:

- 80% trénovacích a 20% testovacích dat
- 50% trénovacích a 50% testovacích dat

Pokud chcete změnit poměr, můžete upravit parametr **_train_ratio_** funkce `split_dataset_into_train_test`
ve skriptu `split_dataset.py`.

# Spuštění

## Trénování modelů

Pokud chcete trénovat model, podívejte se na soubor `model_training.py` ve složce *model_training*.
Existuje [README soubor](model_training/README.md), který obsahuje parametry pro trénování modelu a nějaké
další příklady použití. Příklad jednoho z nich:

```

python model_training.py --dataset pklot --camera_view all --model shufflenet --train_split 80

```

## evaluace modelů, vyhodnocení a vizualizace výsledků

TODO

## Vyhodnocení modelů

TODO

## Vizualizace výsledků

TODO