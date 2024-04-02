# Parking Lot Occupancy Detection

## Úvod

Tento projekt je zaměřen na detekci obsazenosti parkovacích míst. Využívá se k tomu kamera, která snímá parkoviště a na
základě snímků se určuje obsazenost parkovacích míst.

## Datasety

Pro trénování a testování modelu byly použity datasety:

- [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)
- [CNRPark+EXT](http://cnrpark.it/)

# Instalace

```
pip install -r requirements.txt
```

# Stáhnutí a příprava datasetů

TODO

```
bash prepare_datasets.sh
```

```
python utils/move_images_pklot.py
```

# Spuštění

Tento skript spustí evaluaci modelů na všech datasetech.

```
python main.py
```