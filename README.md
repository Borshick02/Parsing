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


## Regression laba
Проект лежит в hh_salary_model/
установка зависимостей
```bash
cd hh_salary_model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

обучение модели
```bash
python scripts/train.py ../x_data.npy ../y_data.npy
```
После этого веса сохраняются в:

hh_salary_model/resources/model.npz

Вывести список зарплат в stdout (в рублях, float)
```bash
python app ../x_data.npy
```

