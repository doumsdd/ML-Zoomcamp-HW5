# Homework 5: Deploying Machine Learning Models - Solutions & Reasoning

**Dataset**: course_lead_scoring_2026.csv (5,000 rows)
**Model Pipeline**: DictVectorizer + LogisticRegression(solver="liblinear", C=1.0)
**Artifact Checksum**: pipeline.bin SHA256 = 1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1
**Scikit-Learn Version**: 1.7.2
**Docker Base Image**: python:3.11.15-slim-bookworm

---

## Setup

### Environment

- uv: 0.13.5
- Python: 3.11.15
- scikit-learn: 1.7.2
- fastapi: 0.119.1
- uvicorn: 0.38.0

Installed dependencies with `uv sync --locked`. The `--locked` flag ensures we use the exact versions pinned in `uv.lock`. No regeneration of the lockfile was performed.

---

## Question 1 - Environment Check

**Task**: Run `uv --version` and record it.

**Answer: `uv 0.13.5`

This is a setup verification only - your local `uv` version is not graded. It confirms that the toolchain works so subsequent commands (`uv sync`, `uv run`) will function correctly.

---

## Question 2 - Locked Dependency (Multiple Choice)

**Task**: Identify the scikit-learn version from `pyproject.toml` or `uv.lock`.

Options: 1.6.1 | 1.7.2 | 1.8.0 | 2.0.0

### Reasoning

In `pyproject.toml`, line 9 declares:

    dependencies = [
        ...
        "scikit-learn==1.7.2",
        ...
    ]

The same version is locked in `uv.lock`. Both files are consistent. No lockfile regeneration was done.

### Answer

**1.7.2**

---

## Question 3 - Load Model & Predict (Free Response)

**Task**: Load `pipeline.bin` via pickle, call `predict_proba` on a specific lead, report the conversion probability rounded to 3 decimal places.

Lead input:

    {
        "lead_source": "paid_ads",
        "industry": "technology",
        "employment_status": "employed",
        "location": "north_america",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0,
        "interaction_count": 4,
        "lead_score": 0.41
    }

### Reasoning

The pipeline consists of two stages:

1. DictVectorizer - transforms categorical features into one-hot encoded columns and scales numerical features. Missing values are filled using defaults from `feature_defaults.json` (training-set median).
2. LogisticRegression - computes the logit and converts it to a probability via the sigmoid function.

The `normalize_record()` function in `model.py` applies the same imputation policy used during training:
- Categorical features: missing/empty values -> "NA"
- Numerical features: missing/empty -> median from defaults JSON

For this lead, all fields are present, so normalization is straightforward.

Code executed:

    import pickle
    from model import normalize_record

    with open("pipeline.bin", "rb") as f:
        pipeline = pickle.load(f)

    record = normalize_record(lead_dict)
    probability = float(pipeline.predict_proba([record])[0, 1])
    print(round(probability, 3))

Result:

    Raw probability : 0.5329944186432507
    Rounded to 3 dp : 0.533
    Conversion      : True (>= 0.5)

### Answer

**0.533**

---

## Question 4 - Serve the Model via API (Free Response)

**Task**: Start the FastAPI server on port 9696, send a POST request to `/predict`, report the conversion probability.

Lead input:

    {
        "lead_source": "organic_search",
        "industry": "technology",
        "employment_status": "employed",
        "location": "europe",
        "number_of_courses_viewed": 4,
        "annual_income": 80304.0,
        "interaction_count": 7,
        "lead_score": 0.74
    }

### Reasoning

The `predict.py` module defines a FastAPI app with two endpoints:

- GET `/health` - returns status and model checksums
- POST `/predict` - accepts a JSON body matching the Lead Pydantic model, normalizes it, runs inference, and returns conversion_probability + conversion bool.

The server was started with `uv run uvicorn predict:app --host 0.0.0.0 --port 9696`. A curl-style request sent the test lead to `http://localhost:9696/predict`.

Result:

    {"conversion_probability": 0.769799, "conversion": true}

Rounded to 3 decimal places: **0.770**

### Answer

**0.770**

---

## Question 5 - Container Configuration (Multiple Choice)

**Task**: Identify the Python base-image tag declared in the Dockerfile.

Options:
- python:3.9-slim-bullseye
- python:3.11.15-slim-bookworm
- python:3.12-slim-bookworm
- python:3.13.10-slim-bookworm

### Reasoning

Line 1 of the `Dockerfile`:

    FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

This is the exact, pinned image digest for deterministic builds. The file also installs `uv 0.10.11` from a multi-stage build, runs `uv sync --locked --no-dev` to install dependencies, copies artifacts, and starts `uvicorn predict:app` on port 9696.

### Answer

**python:3.11.15-slim-bookworm**

---

## Question 6 - Run the Container (Free Response)

**Task**: Build and run the Docker container, send the same prediction request as Q4, verify health check, report conversion probability.

Steps:

1. **Build the image**: `docker build -t zoomcamp-model:2026-hw5 .` (~957 MB, built successfully)
2. **Run the container**: `docker run --rm -d -p 9696:9696 --name hw5-container zoomcamp-model:2026-hw5`
3. **Health check**: `curl -s http://localhost:9696/health` returned `{"status":"ok","model_sha256":"1646bbdcd38d..."}` which matches `model_metadata.json`
4. **Prediction request**: Same payload as Q4 sent to `http://localhost:9696/predict`
5. **Smoke test**: Ran `q6_test.py` inside the container - assertions passed

Result:

    {"conversion_probability": 0.769799, "conversion": true}

Rounded to 3 decimal places: **0.770**

This matches Q4 exactly, confirming the model produces deterministic results whether run locally or inside the container.

### Answer

**0.770**

---

## Summary Table

| Q | Type | Answer |
|---|------|--------|
| 1 | Setup | uv 0.13.5, Python 3.11.15 |
| 2 | Multiple choice | 1.7.2 |
| 3 | Free response | 0.533 |
| 4 | Free response (API) | 0.770 |
| 5 | Multiple choice | python:3.11.15-slim-bookworm |
| 6 | Free response (Container) | 0.770 |

---

## Notes

- All probabilities were rounded to 3 decimal places as instructed.
- The grader accepts an absolute error of at most 0.005.
- The model checksum (`pipeline.bin` SHA256) matches between `model_metadata.json` and the `/health` endpoint, confirming artifact integrity.
- Q4 and Q6 return identical probabilities because the container runs the exact same frozen pipeline artifact.
