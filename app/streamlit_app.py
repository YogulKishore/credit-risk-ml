import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import requests
from src.db import fetch_all_applications, create_tables

API_URL = "http://localhost:8000"

# ensure table exists


st.set_page_config(page_title="Credit Risk System", layout="wide")

# ------------------------------------------------
# Pages
# ------------------------------------------------

page = st.sidebar.radio("Navigation", ["Predict", "History", "Dashboard"])


# ================================================
# PAGE 1 — PREDICT
# ================================================

if page == "Predict":

    st.title("Credit Risk Prediction System")
    st.write("Fill in the applicant details below to estimate default risk.")

    # ------------------------------------------------
    # Randomize helper
    # ------------------------------------------------

    def randomize():
        from faker import Faker
        fake = Faker("en_IN")

        st.session_state["full_name"]        = fake.name()
        st.session_state["email"]            = fake.email()
        st.session_state["phone"]            = fake.phone_number()[:15]
        st.session_state["address"]          = fake.address().replace("\n", ", ")
        st.session_state["date_of_birth"]    = fake.date_of_birth(minimum_age=21, maximum_age=65)
        st.session_state["gender"]           = np.random.choice(["M", "F"])
        # age derived from date_of_birth — no separate state needed
        st.session_state["income"]           = int(np.random.choice([25000, 45000, 67500, 90000, 135000, 180000]))
        st.session_state["credit"]           = int(np.random.randint(50000, 600000))
        st.session_state["annuity"]          = int(np.random.randint(5000, 50000))
        st.session_state["employment_years"] = int(np.random.randint(0, 20))
        st.session_state["family_members"]   = int(np.random.randint(1, 6))
        st.session_state["owns_car"]         = np.random.choice(["Y", "N"])
        st.session_state["owns_house"]       = np.random.choice(["Y", "N"])
        st.session_state["education"]        = np.random.choice([
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary"
        ])
        st.session_state["family_status"]    = np.random.choice([
            "Married", "Single / not married", "Civil marriage", "Separated"
        ])
        st.session_state["ext1"] = round(float(np.random.uniform(0.1, 0.9)), 2)
        st.session_state["ext2"] = round(float(np.random.uniform(0.1, 0.9)), 2)
        st.session_state["ext3"] = round(float(np.random.uniform(0.1, 0.9)), 2)

    if st.button("🎲 Randomize"):
        randomize()

    st.divider()

    # ------------------------------------------------
    # Personal Details
    # ------------------------------------------------

    st.subheader("Personal Details")

    p1, p2 = st.columns(2)

    with p1:
        full_name = st.text_input("Full Name", value=st.session_state.get("full_name", ""))
        email     = st.text_input("Email", value=st.session_state.get("email", ""))
        phone     = st.text_input("Phone Number", value=st.session_state.get("phone", ""))

    with p2:
        address       = st.text_input("Address", value=st.session_state.get("address", ""))
        date_of_birth = st.date_input(
            "Date of Birth",
            value=st.session_state.get("date_of_birth", date(1990, 1, 1)),
            min_value=date(1940, 1, 1),
            max_value=date(2005, 12, 31)
        )
        age = (date.today() - date_of_birth).days // 365
        st.caption(f"Derived age: {age} years")

    st.divider()

    # ------------------------------------------------
    # Loan Details
    # ------------------------------------------------

    st.subheader("Loan Application Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["M", "F"],
            index=["M", "F"].index(st.session_state.get("gender", "M"))
        )

        income = st.number_input(
            "Annual Income",
            min_value=10000, max_value=500000,
            value=st.session_state.get("income", 50000)
        )

        credit = st.number_input(
            "Loan Amount",
            min_value=1000, max_value=1000000,
            value=st.session_state.get("credit", 150000)
        )

        employment_years = st.slider(
            "Employment Years", 0, 40,
            st.session_state.get("employment_years", 5)
        )

    with col2:
        annuity = st.number_input(
            "Loan Annuity",
            min_value=1000, max_value=200000,
            value=st.session_state.get("annuity", 20000)
        )

        family_members = st.slider(
            "Family Members", 1, 10,
            st.session_state.get("family_members", 3)
        )

        owns_car = st.selectbox(
            "Owns Car", ["Y", "N"],
            index=["Y", "N"].index(st.session_state.get("owns_car", "N"))
        )

        owns_house = st.selectbox(
            "Owns House", ["Y", "N"],
            index=["Y", "N"].index(st.session_state.get("owns_house", "N"))
        )

        education = st.selectbox(
            "Education",
            ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"],
            index=["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"]
                  .index(st.session_state.get("education", "Secondary / secondary special"))
        )

        family_status = st.selectbox(
            "Family Status",
            ["Married", "Single / not married", "Civil marriage", "Separated"],
            index=["Married", "Single / not married", "Civil marriage", "Separated"]
                  .index(st.session_state.get("family_status", "Married"))
        )

    st.subheader("External Credit Scores")

    col3, col4, col5 = st.columns(3)

    with col3:
        ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, float(st.session_state.get("ext1", 0.5)))
    with col4:
        ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, float(st.session_state.get("ext2", 0.5)))
    with col5:
        ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, float(st.session_state.get("ext3", 0.5)))

    st.divider()

    # ------------------------------------------------
    # Predict button
    # ------------------------------------------------

    if st.button("Predict Default Risk", type="primary", use_container_width=True):

        if not full_name.strip():
            st.warning("Please enter the applicant's full name before predicting.")
        else:

            car_age = np.random.randint(0, 20) if owns_car == "Y" else None

            app_df = pd.DataFrame({
                "SK_ID_CURR":          [999999],
                "CODE_GENDER":         [gender],
                "DAYS_BIRTH":          [-age * 365],
                "DAYS_EMPLOYED":       [-employment_years * 365],
                "AMT_INCOME_TOTAL":    [income],
                "AMT_CREDIT":          [credit],
                "AMT_ANNUITY":         [annuity],
                "CNT_FAM_MEMBERS":     [family_members],
                "FLAG_OWN_CAR":        [owns_car],
                "FLAG_OWN_REALTY":     [owns_house],
                "NAME_EDUCATION_TYPE": [education],
                "NAME_FAMILY_STATUS":  [family_status],
                "OWN_CAR_AGE":         [car_age],
                "EXT_SOURCE_1":        [ext1],
                "EXT_SOURCE_2":        [ext2],
                "EXT_SOURCE_3":        [ext3],
            })

            log = []
            log.append(f"[1] Raw input received for applicant: {full_name}")
            log.append(f"[2] Built application DataFrame — {app_df.shape[1]} raw columns")

            try:

                payload = {
                    "full_name":        full_name,
                    "email":            email,
                    "phone":            phone,
                    "address":          address,
                    "date_of_birth":    str(date_of_birth),
                    "gender":           gender,
                    "income":           income,
                    "loan_amount":      credit,
                    "annuity":          annuity,
                    "employment_years": employment_years,
                    "family_members":   family_members,
                    "owns_car":         owns_car,
                    "owns_house":       owns_house,
                    "education":        education,
                    "family_status":    family_status,
                    "ext_source_1":     ext1,
                    "ext_source_2":     ext2,
                    "ext_source_3":     ext3,
                }

                log.append("[1] Sending request to FastAPI at POST /predict")
                response = requests.post(f"{API_URL}/predict", json=payload)
                response.raise_for_status()
                result = response.json()

                risk = result["stacked_score"]

                log.append("[2] API received request and built application DataFrame")
                log.append("[3] Applied preprocess_application — binary flags, age, ratios, EXT_SOURCE_MEAN computed")
                log.append("[4] Filled missing features with NaN (bureau, POS, installment data not provided)")
                log.append("[5] Coerced object columns to numeric for LGB/XGB pipeline")
                log.append("[6] LightGBM pipeline: median imputation → predict_proba")
                log.append("[7] XGBoost pipeline: median imputation → predict_proba")
                log.append("[8] CatBoost: cat feature NaN → 'Missing' string → predict_proba")
                log.append("[9] Stacking model: logistic regression on [lgb, xgb, cat] outputs")
                log.append(f"[10] Final stacked probability: {risk:.4f}")
                log.append(f"[11] Saved to database — application_id: {result['application_id']}")

                engineered = {
                    "credit_income_ratio": round(credit / income, 4),
                    "annuity_income_ratio": round(annuity / income, 4),
                    "credit_annuity_ratio": round(credit / annuity, 4),
                    "income_per_person":    round(income / family_members, 2),
                    "ext_source_mean":      round((ext1 + ext2 + ext3) / 3, 4),
                }

                # ------------------------------------------------
                # Results
                # ------------------------------------------------

                st.subheader("Prediction Result")

                st.metric("Default Probability", f"{risk:.2%}")

                if risk < 0.2:
                    st.success("Low Risk Applicant")
                elif risk < 0.5:
                    st.warning("Medium Risk Applicant")
                else:
                    st.error("High Risk Applicant")

                st.subheader("Individual Model Scores")

                m1, m2, m3 = st.columns(3)
                m1.metric("LightGBM", f"{result['lgb']:.2%}")
                m2.metric("XGBoost",  f"{result['xgb']:.2%}")
                m3.metric("CatBoost", f"{result['cat']:.2%}")

                st.divider()

                # ------------------------------------------------
                # Feature values used
                # ------------------------------------------------

                st.subheader("Feature Values Used")

                feat_col1, feat_col2 = st.columns(2)

                with feat_col1:
                    st.markdown("**Raw Inputs**")
                    raw_df = pd.DataFrame({
                        "Feature": ["Date of Birth", "Derived Age", "Gender", "Income", "Loan Amount", "Annuity",
                                    "Employment Years", "Family Members", "Owns Car",
                                    "Owns House", "Education", "Family Status",
                                    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
                        "Value":   [str(date_of_birth), f"{age} years", gender, f"₹{income:,}", f"₹{credit:,}", f"₹{annuity:,}",
                                    str(employment_years), str(family_members), owns_car,
                                    owns_house, education, family_status,
                                    str(ext1), str(ext2), str(ext3)]
                    })
                    st.dataframe(raw_df, hide_index=True, use_container_width=True)

                with feat_col2:
                    st.markdown("**Engineered Features**")
                    eng_df = pd.DataFrame({
                        "Feature": list(engineered.keys()),
                        "Value":   list(engineered.values())
                    })
                    st.dataframe(eng_df, hide_index=True, use_container_width=True)

                st.divider()

                # ------------------------------------------------
                # Processing log
                # ------------------------------------------------

                st.subheader("Processing Log")
                for entry in log:
                    st.text(entry)

                # ------------------------------------------------
                # Save to Postgres
                # ------------------------------------------------

                personal = {
                    "full_name":     full_name,
                    "email":         email,
                    "phone":         phone,
                    "address":       address,
                    "date_of_birth": date_of_birth,
                }

                inputs = {
                    "gender":           gender,
                    "age":              age,
                    "income":           income,
                    "loan_amount":      credit,
                    "annuity":          annuity,
                    "employment_years": employment_years,
                    "family_members":   family_members,
                    "owns_car":         owns_car,
                    "owns_house":       owns_house,
                    "education":        education,
                    "family_status":    family_status,
                    "ext_source_1":     ext1,
                    "ext_source_2":     ext2,
                    "ext_source_3":     ext3,
                }

                st.success(f"Submission saved — Application ID: {result['application_id']}")

            except Exception as e:
                st.error("Prediction failed.")
                st.write(e)
                import traceback
                st.code(traceback.format_exc())


# ================================================
# PAGE 2 — HISTORY
# ================================================

elif page == "History":

    st.title("Submission History")

    try:
        response = requests.get(f"{API_URL}/applications")
        response.raise_for_status()
        df = pd.DataFrame(response.json())

        if df.empty:
            st.info("No submissions yet.")
        else:
            st.write(f"Total submissions: **{len(df)}**")

            def colour_risk(val):
                if val == "Low Risk":
                    return "background-color: #d4edda; color: #155724"
                elif val == "Medium Risk":
                    return "background-color: #fff3cd; color: #856404"
                else:
                    return "background-color: #f8d7da; color: #721c24"

            styled = df.style.applymap(colour_risk, subset=["risk_label"])

            st.dataframe(styled, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error("Could not load history.")
        st.write(e)


# ================================================
# PAGE 3 — DASHBOARD
# ================================================

elif page == "Dashboard":

    st.title("Dashboard")

    try:
        resp0 = requests.get(f"{API_URL}/applications")
        resp0.raise_for_status()
        df = pd.DataFrame(resp0.json())

        if df.empty:
            st.info("No submissions yet.")
        else:
            import plotly.express as px
            import plotly.graph_objects as go
            from src.db import get_connection
            import pandas as pd

            # load full data with extra columns for breakdowns
            conn = get_connection()
            full_df = pd.read_sql("""
                SELECT submitted_at, full_name, gender, age, income, loan_amount,
                       education, family_status, owns_car, owns_house,
                       stacked_score, lgb_score, xgb_score, cat_score, risk_label
                FROM applications
                ORDER BY submitted_at ASC
            """, conn)
            conn.close()

            full_df["submitted_at"] = pd.to_datetime(full_df["submitted_at"])
            full_df["date"] = full_df["submitted_at"].dt.date

            # ------------------------------------------------
            # Top metrics
            # ------------------------------------------------
            st.subheader("Overview")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Applications", len(full_df))
            m2.metric("High Risk", len(full_df[full_df["risk_label"] == "High Risk"]))
            m3.metric("Medium Risk", len(full_df[full_df["risk_label"] == "Medium Risk"]))
            m4.metric("Low Risk", len(full_df[full_df["risk_label"] == "Low Risk"]))

            st.divider()

            # ------------------------------------------------
            # Row 1: Risk distribution + Score histogram
            # ------------------------------------------------
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Risk Distribution")
                risk_counts = full_df["risk_label"].value_counts().reset_index()
                risk_counts.columns = ["Risk Label", "Count"]
                colors = {"Low Risk": "#28a745", "Medium Risk": "#ffc107", "High Risk": "#dc3545"}
                fig = px.pie(
                    risk_counts, names="Risk Label", values="Count",
                    color="Risk Label", color_discrete_map=colors,
                    hole=0.4
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Stacked Score Distribution")
                fig = px.histogram(
                    full_df, x="stacked_score", nbins=30,
                    color_discrete_sequence=["#636EFA"],
                    labels={"stacked_score": "Default Probability"}
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # ------------------------------------------------
            # Row 2: Model score comparison + Scores over time
            # ------------------------------------------------
            col3, col4 = st.columns(2)

            with col3:
                st.subheader("Model Score Comparison")
                fig = go.Figure()
                for model, color in [("lgb_score", "#00CC96"), ("xgb_score", "#AB63FA"), ("cat_score", "#FFA15A"), ("stacked_score", "#636EFA")]:
                    fig.add_trace(go.Box(y=full_df[model], name=model.replace("_score", "").upper(), marker_color=color))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                st.subheader("Average Score Over Time")
                daily = full_df.groupby("date")["stacked_score"].mean().reset_index()
                fig = px.line(
                    daily, x="date", y="stacked_score",
                    labels={"stacked_score": "Avg Default Probability", "date": "Date"},
                    color_discrete_sequence=["#636EFA"]
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # ------------------------------------------------
            # Row 3: Risk by education + Risk by gender
            # ------------------------------------------------
            col5, col6 = st.columns(2)

            with col5:
                st.subheader("Risk by Education")
                edu = full_df.groupby(["education", "risk_label"]).size().reset_index(name="count")
                fig = px.bar(
                    edu, x="education", y="count", color="risk_label",
                    color_discrete_map={"Low Risk": "#28a745", "Medium Risk": "#ffc107", "High Risk": "#dc3545"},
                    barmode="stack",
                    labels={"education": "Education", "count": "Count"}
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", xaxis_tickangle=-20)
                st.plotly_chart(fig, use_container_width=True)

            with col6:
                st.subheader("Risk by Gender")
                gen = full_df.groupby(["gender", "risk_label"]).size().reset_index(name="count")
                fig = px.bar(
                    gen, x="gender", y="count", color="risk_label",
                    color_discrete_map={"Low Risk": "#28a745", "Medium Risk": "#ffc107", "High Risk": "#dc3545"},
                    barmode="group",
                    labels={"gender": "Gender", "count": "Count"}
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # ------------------------------------------------
            # Row 4: Income vs score scatter + Risk by family status
            # ------------------------------------------------
            col7, col8 = st.columns(2)

            with col7:
                st.subheader("Income vs Default Probability")
                fig = px.scatter(
                    full_df, x="income", y="stacked_score",
                    color="risk_label",
                    color_discrete_map={"Low Risk": "#28a745", "Medium Risk": "#ffc107", "High Risk": "#dc3545"},
                    labels={"income": "Annual Income", "stacked_score": "Default Probability"},
                    opacity=0.7
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

            with col8:
                st.subheader("Risk by Family Status")
                fam = full_df.groupby(["family_status", "risk_label"]).size().reset_index(name="count")
                fig = px.bar(
                    fam, x="family_status", y="count", color="risk_label",
                    color_discrete_map={"Low Risk": "#28a745", "Medium Risk": "#ffc107", "High Risk": "#dc3545"},
                    barmode="stack",
                    labels={"family_status": "Family Status", "count": "Count"}
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", xaxis_tickangle=-20)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("Could not load dashboard.")
        st.write(e)
        import traceback
        st.code(traceback.format_exc())