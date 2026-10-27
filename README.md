# Synthetic Data Factory

Synthetic Data Factory generates tabular synthetic data from an uploaded CSV
and automatically reports how useful and how risky the generated dataset may
be. It is an MVP for the Meta AI hackathon.

## MVP flow

1. Upload a CSV file.
2. Inspect its schema, missing values, duplicates, and basic statistics.
3. Generate synthetic rows with an SDV Gaussian Copula synthesizer.
4. Review an automated trust report:
   - statistical similarity of distributions and correlations;
   - downstream ML utility for an optional target column;
   - heuristic privacy risk based on exact matches and nearest neighbors.
5. Ask Meta Muse Spark 1.3 to explain the dataset profile and recommend next
   steps.
6. Download the generated CSV.

## Local setup

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add a Meta Model API key to `.env`:

```dotenv
MODEL_API_KEY=replace_with_your_api_key
```

The Meta integration uses the OpenAI-compatible Meta Model API with
`muse-spark-1.3`. The canonical configuration is documented in the
[Meta Model API cookbook](https://github.com/meta-models/meta-model-cookbook/tree/main/01_api_fundamentals).

## Run the application

Run this command from the repository root:

```bash
streamlit run frontend/streamlit_app.py
```

## Run tests

```bash
pytest -q
```

## Interpreting the report

- Higher **Statistical Similarity** is better.
- Higher **ML Utility** is better.
- Lower **Privacy Risk** is better.

The privacy score is an MVP screening heuristic. Synthetic data does not
automatically guarantee anonymity, differential privacy, or protection from
membership-inference and re-identification attacks.
