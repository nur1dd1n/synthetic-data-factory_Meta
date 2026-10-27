# Synthetic Data Factory

Synthetic Data Factory generates tabular synthetic data from an uploaded CSV
and automatically reports how useful and how risky the generated dataset may
be. It is an MVP for the Meta AI hackathon.

## MVP flow

1. Upload a CSV file, or select the built-in invented transaction demo.
2. Inspect its schema, missing values, duplicates, and basic statistics.
3. Optionally exclude identifier columns and select an ML target and task.
   Generate synthetic rows with an SDV Gaussian Copula synthesizer.
4. Review an automated trust report:
   - statistical similarity of distributions and correlations;
   - downstream ML utility for an optional target column;
   - heuristic privacy risk based on exact matches and nearest neighbors.
5. Ask Meta Muse Spark 1.3 to explain the dataset profile and recommend next
   steps.
6. Compare distribution charts and correlation matrices, then download the
   generated CSV and an aggregate report as JSON or Markdown.

## Local setup

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env
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
python -m streamlit run frontend/streamlit_app.py
```

## Run tests

```bash
python -m pytest -q
```

## Interpreting the report

- Higher **Statistical Similarity** is better.
- Higher **ML Utility** is better.
- Lower **Copying-risk heuristic** indicates fewer copying/proximity signals.

The privacy score is an MVP screening heuristic. Synthetic data does not
automatically guarantee anonymity, differential privacy, or protection from
membership-inference and re-identification attacks.

## Demo in 90 seconds

1. Select **Try invented transaction demo data**. All 300 rows are invented;
   this is not a real fraud benchmark or real customer data.
2. Select target **fraud**, task **classification**, and generate 300 rows.
3. Inspect distributions for **amount** and **channel**, correlations, and the
   real-trained versus synthetic-trained macro F1 scores.
4. Inspect exact matches, proximity, and recommendations. Download the report.

Meta AI is optional. Generation, diagnostics and export run locally without an
API key. The existing Meta AI button sends the dataset profile to Meta; uploaded
row previews are not sent by that code. Profiles can still contain sensitive
column names and aggregate statistics.

## Evaluation protocol and limits

When a target is selected, rows without a target and exact duplicate rows are
excluded from the ML cohort. A fixed 25% real holdout is reserved **before** the
synthesizer is fitted. Classification uses a stratified split. The synthesizer
and the real baseline only receive the training portion. The synthetic-trained
model and the real-trained model are evaluated on the same real holdout.
Similarity and copying diagnostics compare against the generator's training
portion. The report records training, test, and excluded-row counts.

Classification uses macro F1 so minority classes contribute equally. Missing
synthetic classes remain visible as poorer performance. Regression reports MAE
in target units. Relative utility is synthetic F1 / real F1, or real MAE /
synthetic MAE, multiplied by 100. The display caps at 100; the JSON retains the
uncapped ratio. Undefined baselines show no relative score. A high ratio can
still hide weak absolute performance: inspect both raw model scores.

This is a single random split for independent tabular rows. Related subjects,
future information and time series require a group or temporal split, which is
not implemented here. The split-before-fitting protocol follows the
[scikit-learn data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Exact copying is measured over all generated rows and all original training
rows. Distance diagnostics sample up to 500 synthetic queries, but search every
original training row in bounded memory blocks. Numeric distances are range
normalized; categorical differences count as 0 or 1. The copying score and its
thresholds are uncalibrated heuristics, not a disclosure probability. Identifier
value overlap is separate and is not incorporated into that score.

An unavailable diagnostic does not discard generated data. Changing the file,
excluded columns, row count, target or task clears the prior result. JSON and
Markdown exports include aggregate results, settings, column names and errors;
they do not include source records. Review exports before sharing.

The local UI caps input at 10 MB, 100,000 rows and 50 columns, and output at
100,000 rows. These are MVP resource limits, not a runtime guarantee. There are
no accounts, saved dataset history, differential privacy, or completed FastAPI
endpoints. The API skeleton is unchanged.
