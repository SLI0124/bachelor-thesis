# Vizualizace modelů

Tento adresář obsahuje skripty pro vizualizaci modelů. Skripty jsou určeny pro zpracování logů z testování modelů
a vizualizaci výsledků testování. Pokud tyto soubory neexistují, odkazuji sena
adresář [model_training](../model_training) a [návod](../model_training/README.md) na jejich vytvoření. Spuštěním se
získají modely, na kterých je možné provést testování a vizualizaci.

## Obsah

- [identify_parking_image.py](identify_parking_image.py) - Skript pro identifikaci jedné fotografie z ACPDS a
  CNRPark-EXT datasetu. Skript načte model, dataset a provede predikci všech míst na fotografii. Výsledný obrázek s
  barevnými obdélníky kolem míst, kde se nachází parkující vozidla, je uložen do složky `data/predicted_images`.
- [incorrectly_identified.py](incorrectly_identified.py) - Skript pro identifikaci chybně identifikovaných míst na
  fotografii. Skript načte model, dataset a provede predikci. Prních 25 chybně identifikovaných míst je uloženo do
  složky `data/incorrectly_identified_images`.
- [models_testing_log_summary.py](models_testing_log_summary.py) - Skript pro zpracování logů z testování modelů. Na
  základě logů vytvoří tabulku s výsledky testování modelů a uloží ji do souboru `data/models_testing_summary.csv`
  a `data/models_testing_summary.txt`.
- [plot_roc_auc.py](plot_roc_auc.py) - Skript pro vykreslení ROC křivek a AUC pro jednotlivé modely. Na základě logů
  vytvoří grafy a uloží je do složky `data/graphs/roc_auc`.
- [test_models.py](test_models.py) - Skript pro testování modelů. Skript načte všechny modely a všechny datasety a
  provede testování. Výsledky testování jsou uloženy do složky `data/testing_logs`.
- [print_testing_logs.py](print_testing_logs.py) - Skript pro zpracování logů z trénování modelů. Na základě logů
  vytvoří seznam s výsledky trénování modelů. Dataset a model se musí náležitě odkomentovat.
- [visualize_training_metrics.py](visualize_training_metrics.py) - Skript pro zpracování logů z trénování modelů. Na
  základě logů vytvoří grafy s vývojem metrik během trénování a uloží je do složky `data/graphs/training_metrics`.
  Obsahuje testovací a validační metriky a grafy pro jednotlivé epochy. 