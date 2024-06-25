# Vizualizace modelů

Tento adresář obsahuje skripty pro vizualizaci modelů. Skripty jsou určeny pro zpracování logů z testování modelů
a vizualizaci výsledků testování. Pokud tyto soubory neexistují, na to se odkazuje v
adresáři [model_training](../model_training) a [návod](../model_training/README.md) na jejich vytvoření. Spuštěním se
získají modely, na kterých je možné provést testování.

## Obsah

- [models_testing_log_summary.py](models_testing_log_summary.py) - Skript pro zpracování logů z testování modelů.
  Skript zpracuje logy z testování modelů a vytvoří soubor s shrnutím výsledků testování. Poskytuje informace o
  úspěšnosti modelů, průměrné úspěšnosti, průměrné úspěšnosti pro jednotlivé třídy a další informace.
- [test_model.py](test_model.py) - Skript pro testování modelů. Skript hromadně načte cestu ke všem modelům v adresáři
  a datasetů a provede testování modelů. Výsledky testování jsou uloženy do logů.
- [visualize_training_metrics.py](visualize_training_metrics.py) - Skript pro vizualizaci metrik trénování modelů.
  Skript načte logy z trénování modelů a vytvoří grafy z metrik trénování. Grafy zobrazují průběh metrik trénování

