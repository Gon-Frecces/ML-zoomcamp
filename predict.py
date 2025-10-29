import pickle

# Load the saved pipeline
with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Input record
record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Predict probability of conversion
pred = pipeline.predict_proba([record])[0, 1]

print(f"Probability of conversion: {pred:.3f}")


