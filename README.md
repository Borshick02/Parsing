- `x_data.npy` — encoded feature matrix
- `y_data.npy` — target vector


Run from the project root:

```bash
python app path/to/hh.csv
```

Optional: target column explicitly:

```bash
python app path/to/hh.csv --target salary
```

## What the pipeline does

1. **LoadCsvHandler** — reads CSV (tries UTF-8, falls back to CP1251)
2. **BasicCleanHandler** — drops duplicates, normalizes column names, fills missing values
3. **SplitTargetHandler** — separates target column (heuristic defaults, fallback to last column)
4. **EncodeCategoricalsHandler** — one-hot encodes categorical features
5. **SaveNpyHandler** — saves `x_data.npy` and `y_data.npy`

