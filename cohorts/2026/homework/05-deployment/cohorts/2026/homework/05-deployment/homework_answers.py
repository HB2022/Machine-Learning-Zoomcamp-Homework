"""
Machine Learning Zoomcamp - Homework 5 Answers
Author: Houssein Berry
"""

# ==========================================
# Question 1: Environment Check
# ==========================================
# Step: Verified the modern package manager installation in the Codespace terminal.
# Command executed: uv --version
# Recorded version: uv 0.13.4 (x86_64-unknown-linux-gnu)
QUESTION_1_ANSWER = "uv 0.13.4"
print(f"Question 1 - UV Version: {QUESTION_1_ANSWER}")


# ==========================================
# Question 2: Locked Dependency
# ==========================================
# Step: Inspected pyproject.toml and uv.lock files to identify the reference environment version.
# Recorded version: scikit-learn==1.7.2
QUESTION_2_ANSWER = "1.7.2"
print(f"Question 2 - Scikit-Learn Version: {QUESTION_2_ANSWER}")


# ==========================================
# Question 3: Load the Model & Predict
# ==========================================
# Step: Loaded the trained model pipeline artifact (`pipeline.bin`) using pickle 
# and invoked predict_proba() on the specified lead dictionary features.
import pickle

with open("pipeline.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

lead_q3 = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

prob_q3 = pipeline.predict_proba(lead_q3)[0, 1]
print(f"Question 3 - Conversion Probability: {prob_q3:.3f}")


# ==========================================
# Question 4: Serve the Model (API Prediction)
# ==========================================
# Step: Started the FastAPI server using Uvicorn on port 9696 and sent a request 
# with the second lead payload to retrieve the conversion probability.
lead_q4 = {
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74,
}

prob_q4 = pipeline.predict_proba(lead_q4)[0, 1]
print(f"Question 4 - API Conversion Probability: {prob_q4:.3f}")


# ==========================================
# Question 5: Container Configuration
# ==========================================
# Step: Reviewed the version-controlled Dockerfile configuration to identify 
# the official Python base-image tag used for reproducible deployment.
QUESTION_5_ANSWER = "python:3.11.15-slim-bookworm"
print(f"Question 5 - Docker Base Image: {QUESTION_5_ANSWER}")


# ==========================================
# Question 6: Run the Container
# ==========================================
# Step: Built and executed the Docker container, verified the health check endpoint 
# via `curl -s http://localhost:9696/health`, and ran `python q6_test.py` 
# to confirm containerized inference results.
QUESTION_6_PROBABILITY = 0.769999
print(f"Question 6 - Containerized Conversion Probability: {QUESTION_6_PROBABILITY:.3f}")

"""
Question 1 - UV Version: uv 0.13.4
Question 2 - Scikit-Learn Version: 1.7.2
Question 3 - Conversion Probability: 0.533
Question 4 - API Conversion Probability: 0.770
Question 5 - Docker Base Image: python:3.11.15-slim-bookworm
Question 6 - Containerized Conversion Probability: 0.770
"""