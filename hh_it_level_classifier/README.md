# hh_it_level_classifier (PoC)

Proof of Concept: выделяем резюме IT-разработчиков из hh.csv, автоматически размечаем уровень (junior/middle/senior) и обучаем классификатор.  
Артефакты: график баланса классов и classification_report.

## 1) Setup (macOS)
```bash
cd hh_it_level_classifier
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
