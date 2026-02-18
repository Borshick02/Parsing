- `x_data.npy` — encoded feature matrix
- `y_data.npy` — target vector
## Запуск
Перейди в корень репозитория:

```bash
cd path/to/hh_chain_project
Запустить

```bash
python app path/to/hh.csv --target salary
python app path/to/hh.csv --target "ЗП"
```
Результат: рядом с hh.csv будут созданы файлы:

x_data.npy — матрица признаков (shape: [N, 6])

y_data.npy — целевая переменная (зарплата, shape: [N])
## What the pipeline does

1. **LoadCsvHandler** — reads CSV (tries UTF-8, falls back to CP1251)
2. **BasicCleanHandler** — drops duplicates, normalizes column names, fills missing values
3. **SplitTargetHandler** — separates target column (heuristic defaults, fallback to last column)
4. **EncodeCategoricalsHandler** — one-hot encodes categorical features
5. **SaveNpyHandler** — saves `x_data.npy` and `y_data.npy`

# Regression laba
код лежит в https://github.com/Borshick02/HH_salary.git Проект лежит в hh_salary_model/ установка зависимостей


# HH Level Classifier 
код лежит в https://github.com/Borshick02/hh_Classifier.git

