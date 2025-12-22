# MLflow CI Workflow

Project ini merupakan implementasi workflow Continuous Integration (CI)
menggunakan MLflow Project untuk melakukan retraining model machine learning
secara otomatis ketika terjadi trigger (push) pada repository GitHub.

## Struktur Project

- MLProject/
  - modelling.py : Script training model
  - conda.yaml : Environment MLflow
  - MLProject : Konfigurasi MLflow Project
  - dataset_preprocessing.csv : Dataset siap latih
  - README.md

## Cara Menjalankan Secara Lokal

```bash
mlflow run .
```
