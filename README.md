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
код лежит в https://github.com/Borshick02/HH_salary.git
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
# HH Level Classifier 
код лежит в https://github.com/Borshick02/hh_Classifier.git
## Что делает проект

1) **Фильтрует** резюме IT-разработчиков из `hh.csv`  
2) **Размечает уровень (label)** по правилам (ключевые слова в названии + fallback по опыту)  
3) **Строит график баланса классов**  
4) **Обучает модель классификации** (baseline: LogisticRegression + TF-IDF + OneHot + числовые признаки)  
5) **Сохраняет артефакты**:
   - `reports/class_balance.png`
   - `reports/classification_report.txt`
   - `resources/model.joblib`
   - `resources/meta.json`
Устнавка
 ```bash
cd hh_it_level_classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Вывол результатов
```bash
python app path/to/hh.csv --mode prepare
python app path/to/hh.csv --mode train
python app path/to/hh.csv --mode eval
```
Как устроена разметка (labels)

Разметка уровня делается в src/hh_it_level_classifier/labels.py:

Сначала проверяем ключевые слова в названии должности (token-based; phrase: team lead/tech lead -> senior)

Если ключевых слов нет — fallback по опыту:

< 2 лет → junior

2–6 лет → middle

>= 6 лет → senior

Это PoC-разметка: она не идеальна, но показывает, что идея работает.
Признаки (features)

Используются:

Числовые: age, salary_rub, exp_years

Категориальные: city, employment, schedule

Текст: skills_text → TF-IDF (uni/bi-grams)

Модель: LogisticRegression в Pipeline:

numeric: imputer + scaler

categorical: imputer + onehot

text: selector + TF-IDF

classifier: LogisticRegression

Middle хуже размечается: в названиях резюме часто нет слова “middle”, поэтому метка часто ставится по опыту → появляется шум.

Признаки пересекаются: по зарплате/навыкам/опыту мидлы похожи и на джунов, и на сеньоров → модель чаще “уверенно” относит к соседним классам.
