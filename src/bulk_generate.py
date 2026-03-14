"""
Bulk synthetic data generator.
Sends 200 randomized applications through the FastAPI /predict endpoint.
Run with: python scripts/bulk_generate.py
"""

import requests
import numpy as np
from faker import Faker
from datetime import date
import time

fake = Faker("en_IN")

API_URL = "http://localhost:8000"

EDUCATIONS = [
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary"
]

FAMILY_STATUSES = [
    "Married",
    "Single / not married",
    "Civil marriage",
    "Separated"
]

INCOMES = [15000, 22500, 30000, 45000, 67500, 90000, 112500, 135000, 180000, 225000]


def random_application(risk_profile="low"):
    """
    risk_profile:
        low    — high EXT scores, stable employment, low loan ratio
        medium — moderate EXT scores, mixed profile
        high   — low EXT scores, short employment, high loan ratio
    """
    dob = fake.date_of_birth(minimum_age=21, maximum_age=65)
    owns_car = np.random.choice(["Y", "N"], p=[0.4, 0.6])
    income = int(np.random.choice(INCOMES))

    if risk_profile == "low":
        ext1 = round(float(np.random.uniform(0.6, 0.95)), 4)
        ext2 = round(float(np.random.uniform(0.6, 0.95)), 4)
        ext3 = round(float(np.random.uniform(0.6, 0.95)), 4)
        loan = int(income * np.random.uniform(0.5, 2.0))
        annuity = int(loan / np.random.uniform(20, 40))
        employment_years = int(np.random.randint(5, 30))

    elif risk_profile == "medium":
        ext1 = round(float(np.random.uniform(0.3, 0.6)), 4)
        ext2 = round(float(np.random.uniform(0.3, 0.6)), 4)
        ext3 = round(float(np.random.uniform(0.3, 0.6)), 4)
        loan = int(income * np.random.uniform(2.0, 5.0))
        annuity = int(loan / np.random.uniform(10, 20))
        employment_years = int(np.random.randint(1, 10))

    else:  # high
        ext1 = round(float(np.random.uniform(0.05, 0.3)), 4)
        ext2 = round(float(np.random.uniform(0.05, 0.3)), 4)
        ext3 = round(float(np.random.uniform(0.05, 0.3)), 4)
        loan = int(income * np.random.uniform(5.0, 10.0))
        annuity = int(loan / np.random.uniform(5, 10))
        employment_years = int(np.random.randint(0, 3))

    annuity = max(1000, min(annuity, 200000))
    loan    = max(1000, min(loan, 1000000))

    return {
        "full_name":        fake.name(),
        "email":            fake.email(),
        "phone":            fake.phone_number()[:15],
        "address":          fake.address().replace("\n", ", "),
        "date_of_birth":    str(dob),
        "gender":           np.random.choice(["M", "F"]),
        "income":           income,
        "loan_amount":      loan,
        "annuity":          annuity,
        "employment_years": employment_years,
        "family_members":   int(np.random.randint(1, 7)),
        "owns_car":         owns_car,
        "owns_house":       np.random.choice(["Y", "N"], p=[0.55, 0.45]),
        "education":        np.random.choice(EDUCATIONS, p=[0.55, 0.25, 0.12, 0.08]),
        "family_status":    np.random.choice(FAMILY_STATUSES, p=[0.55, 0.25, 0.12, 0.08]),
        "ext_source_1":     ext1,
        "ext_source_2":     ext2,
        "ext_source_3":     ext3,
    }


def run(n=200):

    print(f"Generating {n} synthetic applications via FastAPI...\n")

    success = 0
    failed  = 0

    # generate a realistic mix: 50% low, 30% medium, 20% high
    profiles = (
        ["low"] * int(n * 0.5) +
        ["medium"] * int(n * 0.3) +
        ["high"] * int(n * 0.2)
    )
    np.random.shuffle(profiles)

    for i, profile in enumerate(profiles, 1):
        payload = random_application(profile)

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            print(f"[{i:>3}/{n}] {payload['full_name']:<25} → {result['risk_label']:<12} ({result['stacked_score']:.2%})")
            success += 1

        except Exception as e:
            print(f"[{i:>3}/{n}] FAILED — {e}")
            failed += 1

        # small delay to avoid hammering the API
        time.sleep(0.1)

    print(f"\nDone! {success} inserted, {failed} failed.")


if __name__ == "__main__":
    run(200)
