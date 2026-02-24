# 📧 Spam Email Detection

A robust, modular machine-learning system for classifying emails as **Spam** or **Ham**.
Includes a training pipeline with multiple model comparison, CLI tools, and a Streamlit web UI
for single-email and batch `.mbox` classification.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  CSV / MBOX  │────▶│   Ingestion  │────▶│  Validation &    │────▶│  TF-IDF       │
│  Data Source │     │  (pandas)    │     │  Cleaning (bs4)  │     │  Vectorizer   │
└─────────────┘     └──────────────┘     └──────────────────┘     └───────┬───────┘
                                                                          │
                    ┌──────────────────────────────────────────────────────┘
                    ▼
          ┌──────────────────┐
          │  Model Training  │  LinearSVC · LogisticRegression
          │  + GridSearchCV  │  DecisionTree · RandomForest
          └────────┬─────────┘
                   │
      ┌────────────┼────────────┐
      ▼            ▼            ▼
 best_model   vectorizer    metrics.json
  .joblib      .joblib      (outputs/)
      │            │
      └─────┬──────┘
            ▼
   ┌─────────────────┐
   │   Prediction     │◀── Streamlit UI  /  CLI
   │   Pipeline       │
   └─────────────────┘
```

## Features

- **Multiple models**: LinearSVC, Logistic Regression, Decision Tree, Random Forest
- **Hyperparameter tuning**: GridSearchCV on SVM & Logistic Regression
- **Reproducible**: fixed random seeds, deterministic splits, timestamped artifacts
- **Batch processing**: upload `.mbox` files → get a CSV report with per-email predictions
- **Streamlit UI**: two-tab interface (single email & batch)
- **Structured logging**: rotating log files in `logs/`
- **Modular code**: clean separation of config, components, pipelines, and utilities

## Project Structure

```
Spam-Email-Detection/
├── app.py                          # Streamlit web UI
├── main.py                         # CLI entrypoint (train / predict / batch)
├── requirements.txt
├── pyproject.toml
├── Makefile
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   ├── dataset/                    # Place your full dataset.csv here
│   └── sample/sample_dataset.csv   # Tiny 20-row sample (included)
├── outputs/
│   ├── models/                     # Saved model .joblib files
│   ├── vectorizers/                # Saved TF-IDF vectorizer
│   └── metrics/                    # Training metrics JSON
├── logs/                           # Rotating log files
├── src/
│   ├── config/config.py            # Central configuration
│   ├── components/
│   │   ├── data_ingestion.py       # CSV loading
│   │   ├── data_validation.py      # Schema & label validation
│   │   ├── data_transformation.py  # Text cleaning + TF-IDF
│   │   └── model_trainer.py        # Training, tuning, evaluation
│   ├── pipeline/
│   │   ├── training_pipeline.py    # End-to-end training
│   │   └── prediction_pipeline.py  # Inference (single + batch)
│   └── utils/
│       ├── logger.py               # Structured logging
│       ├── common.py               # Artifact I/O, text cleaning, math
│       └── mbox_parser.py          # Robust .mbox parsing
└── tests/
    ├── test_preprocessing.py       # Unit tests for cleaning & validation
    └── test_inference_smoke.py     # E2E smoke tests
```

## Setup

### Prerequisites

- Python 3.10+

### Install dependencies

**With pip:**
```bash
pip install -r requirements.txt
```

**With uv:**
```bash
uv pip install -r requirements.txt
```

### Prepare data

The repo includes a 20-row sample at `data/sample/sample_dataset.csv` so everything works out of the box.

For a real dataset, place a CSV at `data/dataset/dataset.csv` with these columns:

| Column | Description |
|--------|-------------|
| `text` | Email body (raw text or HTML) |
| `label` | `spam` or `ham` (case-insensitive) |

**Recommended datasets** (not included due to size):
- [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)
- [Enron Spam Dataset](http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html)
- [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## Usage

### Train a model

```bash
# Using the default dataset path (data/dataset/dataset.csv, falls back to sample)
python main.py train

# Using a specific CSV
python main.py train path/to/your/dataset.csv

# Or via Makefile
make train-sample
```

### Classify a single email (CLI)

```bash
python main.py predict "Congratulations! You won a free iPhone!"
```

### Batch classify an .mbox file (CLI)

```bash
python main.py batch path/to/emails.mbox -o results.csv
```

### Launch the Streamlit UI

```bash
streamlit run app.py

# Or via Makefile
make app
```

The UI has two tabs:
1. **Single Email** — paste text, click Predict, see label + confidence
2. **Batch MBOX** — upload `.mbox`, process, preview results, download CSV

### Run tests

```bash
# With pytest (recommended)
python -m pytest tests/ -v

# Without pytest
python -c "
from tests.test_preprocessing import TestCleanText, TestValidateData
from tests.test_inference_smoke import TestInferenceSmoke
for cls in [TestCleanText, TestValidateData, TestInferenceSmoke]:
    obj = cls()
    for name in sorted(dir(obj)):
        if name.startswith('test_'):
            getattr(obj, name)()
            print(f'  PASS: {cls.__name__}.{name}')
print('All tests passed!')
"

# Or via Makefile
make test
```

## How It Works

### Preprocessing
1. Strip HTML tags (BeautifulSoup4 + lxml)
2. Remove common email header lines (From:, To:, Subject:, etc.)
3. Collapse whitespace and lowercase
4. TF-IDF vectorization (configurable n-gram range, max features, min_df)

### Training
1. Stratified train/test split (80/20, seed=42)
2. GridSearchCV (5-fold) for LinearSVC and LogisticRegression
3. Standard fit for DecisionTree and RandomForest
4. Champion selected by best F1 score on the spam class
5. Model, vectorizer, and metrics saved with timestamps

### Inference confidence scores
- Models with `predict_proba` (LogisticRegression, RandomForest, DecisionTree): probability used directly
- Models without it (LinearSVC): `decision_function` output mapped through a sigmoid to [0, 1]

## Configuration

All tunables live in `src/config/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `tfidf.ngram_range` | `(1, 2)` | Unigrams + bigrams |
| `tfidf.max_features` | `20000` | Vocabulary cap |
| `tfidf.min_df` | `2` | Minimum document frequency |
| `training.test_size` | `0.2` | Hold-out fraction |
| `training.random_seed` | `42` | Global seed |
| `training.cv_folds` | `5` | Cross-validation folds |

## Troubleshooting

### "Artifact not found" / "Please run the training pipeline first"
You need to train before predicting:
```bash
python main.py train
```

### Import errors (`ModuleNotFoundError: No module named 'src'`)
Run all commands from the project root directory:
```bash
cd Spam-Email-Detection
python main.py train
```

### Low accuracy on sample dataset
The included 20-row sample is just for smoke-testing. Use a real dataset (thousands of emails) for meaningful results.

### Streamlit won't start
Ensure streamlit is installed: `pip install streamlit`

## Roadmap

- [ ] Add character-level and subword n-gram features
- [ ] Integrate a lightweight transformer (DistilBERT) as an optional model
- [ ] Add SHAP / LIME explanations for predictions
- [ ] Docker container for one-command deployment
- [ ] REST API via FastAPI
- [ ] MLflow experiment tracking
- [ ] Active learning loop for user-flagged misclassifications

## License

[MIT](LICENSE)
