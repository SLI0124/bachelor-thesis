# Parking Lot Occupancy Detection

## Úvod

Tento projekt je zaměřen na detekci obsazenosti parkovacích míst. Využívá se k tomu kamera, která snímá parkoviště a na
základě snímků se určuje obsazenost parkovacích míst.

## Datasety

Pro trénování a testování modelu byly použity datasety:

- [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)
- [CNRPark+EXT](http://cnrpark.it/)

# Instalace

## GPU podpora

Pokud máte k dispozici Nvidia GPU, můžete využít GPU podporu pro trénování modelu. Pro instalaci je potřeba mít
nainstalovaný
CUDA Toolkit a CuDNN, nebo preferovaně nastavit Anaconda prostředí, další informace naleznete
na [stránkách PyTorch](https://pytorch.org/get-started/locally/).

```
pip install -r requirements.txt
```

# Stáhnutí a příprava datasetů

```
bash prepare_datasets.sh
```

```
python utils/move_images_pklot.py
```

# Spuštění

Tento skript spustí evaluaci modelů, vyhodnocení a vizualizaci výsledků.

```
python main.py
```

## Trénování modelů

Pokud chcete trénovat model, podívejte se na soubor `model_training.py` ve složce *model_training*.
Existuje [textový soubor](model_training/README.md), který obsahuje parametry pro trénování modelu a nějaké
další příklady.

```
python model_training.py
```

```
python model_training.py --dataset puc --model_name resnet18 --batch_size 32 --epochs 10
```

## Vyhodnocení modelů

TODO

## Vizualizace výsledků

TODO