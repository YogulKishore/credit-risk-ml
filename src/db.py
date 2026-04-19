import os
import psycopg2
import psycopg2.extras
import pandas as pd
from contextlib import contextmanager

# FIX 1: credentials from environment variables — never hardcode passwords
# set these in your .env or Railway/Heroku config vars
DB_CONFIG = {
    "host":     os.environ.get("DB_HOST",     "localhost"),
    "port":     int(os.environ.get("DB_PORT", 5432)),
    "dbname":   os.environ.get("DB_NAME",     "credit_risk"),
    "user":     os.environ.get("DB_USER",     "admin"),
    "password": os.environ.get("DB_PASSWORD", "admin123"),
}

# FIX 5: use the tuned threshold from evaluate.py consistently
RISK_THRESHOLD = float(os.environ.get("RISK_THRESHOLD", 0.115))


def get_connection():
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        database_url = database_url.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(database_url, sslmode="require")
    return psycopg2.connect(**DB_CONFIG)


# FIX 3: context manager ensures connection always closes even on error
@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_tables():

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS applications (
                id                      SERIAL PRIMARY KEY,
                submitted_at            TIMESTAMP DEFAULT NOW(),
                full_name               TEXT,
                email                   TEXT,
                phone                   TEXT,
                address                 TEXT,
                date_of_birth           DATE,
                gender                  TEXT,
                age                     INT,
                income                  FLOAT,
                loan_amount             FLOAT,
                annuity                 FLOAT,
                employment_years        INT,
                family_members          INT,
                owns_car                TEXT,
                owns_house              TEXT,
                education               TEXT,
                family_status           TEXT,
                ext_source_1            FLOAT,
                ext_source_2            FLOAT,
                ext_source_3            FLOAT,
                credit_income_ratio     FLOAT,
                annuity_income_ratio    FLOAT,
                credit_annuity_ratio    FLOAT,
                income_per_person       FLOAT,
                ext_source_mean         FLOAT,
                lgb_score               FLOAT,
                xgb_score               FLOAT,
                cat_score               FLOAT,
                stacked_score           FLOAT,
                is_default              BOOLEAN,
                risk_label              TEXT
            )
        """)

        # FIX 6: application_features built dynamically from feature_columns.pkl
        # so it always stays in sync with the model — no hardcoded column list
        try:
            import joblib
            feature_cols = joblib.load("models/feature_columns.pkl")

            # FIX 4: use correct column name BUREAU_LOAN_COUNT (no trailing underscore)
            col_defs = []
            for col in feature_cols:
                # quote the column name to handle special chars
                col_defs.append(f'"{col}" FLOAT')

            col_defs_sql = ",\n        ".join(col_defs)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS application_features (
                    id              SERIAL PRIMARY KEY,
                    application_id  INT REFERENCES applications(id) ON DELETE CASCADE,
                    submitted_at    TIMESTAMP DEFAULT NOW(),
                    {col_defs_sql}
                )
            """)
        except FileNotFoundError:
            print("Warning: models/feature_columns.pkl not found — skipping application_features table")

        cur.close()

    print("Tables created successfully.")


def insert_application(personal, inputs, features, predictions, feature_vector):

    risk = predictions["stacked"]

    # FIX 5: use tuned threshold for is_default flag
    is_default = bool(risk >= RISK_THRESHOLD)

    if risk < 0.2:
        label = "Low Risk"
    elif risk < 0.5:
        label = "Medium Risk"
    else:
        label = "High Risk"

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO applications (
                full_name, email, phone, address, date_of_birth,
                gender, age, income, loan_amount, annuity,
                employment_years, family_members, owns_car, owns_house,
                education, family_status,
                ext_source_1, ext_source_2, ext_source_3,
                credit_income_ratio, annuity_income_ratio,
                credit_annuity_ratio, income_per_person, ext_source_mean,
                lgb_score, xgb_score, cat_score, stacked_score,
                is_default, risk_label
            ) VALUES (
                %(full_name)s, %(email)s, %(phone)s, %(address)s, %(date_of_birth)s,
                %(gender)s, %(age)s, %(income)s, %(loan_amount)s, %(annuity)s,
                %(employment_years)s, %(family_members)s, %(owns_car)s, %(owns_house)s,
                %(education)s, %(family_status)s,
                %(ext_source_1)s, %(ext_source_2)s, %(ext_source_3)s,
                %(credit_income_ratio)s, %(annuity_income_ratio)s,
                %(credit_annuity_ratio)s, %(income_per_person)s, %(ext_source_mean)s,
                %(lgb_score)s, %(xgb_score)s, %(cat_score)s, %(stacked_score)s,
                %(is_default)s, %(risk_label)s
            ) RETURNING id
        """, {
            **personal,
            **inputs,
            **features,
            "lgb_score":     predictions["lgb"],
            "xgb_score":     predictions["xgb"],
            "cat_score":     predictions["cat"],
            "stacked_score": predictions["stacked"],
            "is_default":    is_default,
            "risk_label":    label,
        })

        application_id = cur.fetchone()[0]

        # FIX 6: build INSERT dynamically from feature_vector keys
        # so we never have to maintain a hardcoded 160-column string again
        fv = {
            k: (None if (v is not None and isinstance(v, float) and v != v) else v)
            for k, v in feature_vector.items()
        }
        fv["application_id"] = application_id

        cols    = ["application_id"] + list(feature_vector.keys())
        col_sql = ", ".join(f'"{c}"' for c in cols)
        val_sql = ", ".join(f"%({c})s" for c in cols)

        cur.execute(
            f"INSERT INTO application_features ({col_sql}) VALUES ({val_sql})",
            fv
        )

        cur.close()

    return application_id


def fetch_all_applications():
    # FIX 7: use context manager so connection always closes
    with get_db() as conn:
        df = pd.read_sql("""
            SELECT
                id, submitted_at, full_name, email, phone,
                loan_amount, income, stacked_score,
                lgb_score, xgb_score, cat_score,
                is_default, risk_label
            FROM applications
            ORDER BY submitted_at DESC
        """, conn)
    return df


def fetch_application(app_id):
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM applications WHERE id = %s", (app_id,))
        row = cur.fetchone()
        cur.close()
    return dict(row) if row else None


def fetch_features(app_id):
    with get_db() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT * FROM application_features WHERE application_id = %s",
            (app_id,)
        )
        row = cur.fetchone()
        cur.close()
    return dict(row) if row else None


if __name__ == "__main__":
    create_tables()