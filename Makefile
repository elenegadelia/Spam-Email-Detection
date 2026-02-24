.PHONY: install train app test clean

install:
	pip install -r requirements.txt

train:
	python -m src.pipeline.training_pipeline

train-sample:
	python main.py train data/sample/sample_dataset.csv

app:
	streamlit run app.py

test:
	python -m pytest tests/ -v

clean:
	rm -rf outputs/models/*.joblib outputs/vectorizers/*.joblib outputs/metrics/*.json logs/*.log
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
