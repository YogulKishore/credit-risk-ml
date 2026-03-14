import os
import psycopg2
import psycopg2.extras
import pandas as pd

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "credit_risk",
    "user": "admin",
    "password": "admin123"
}


def get_connection():
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        database_url = database_url.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(database_url, sslmode="require")
    return psycopg2.connect(**DB_CONFIG)


def create_tables():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id                  SERIAL PRIMARY KEY,
            submitted_at        TIMESTAMP DEFAULT NOW(),
            full_name           TEXT,
            email               TEXT,
            phone               TEXT,
            address             TEXT,
            date_of_birth       DATE,
            gender              TEXT,
            age                 INT,
            income              FLOAT,
            loan_amount         FLOAT,
            annuity             FLOAT,
            employment_years    INT,
            family_members      INT,
            owns_car            TEXT,
            owns_house          TEXT,
            education           TEXT,
            family_status       TEXT,
            ext_source_1        FLOAT,
            ext_source_2        FLOAT,
            ext_source_3        FLOAT,
            credit_income_ratio     FLOAT,
            annuity_income_ratio    FLOAT,
            credit_annuity_ratio    FLOAT,
            income_per_person       FLOAT,
            ext_source_mean         FLOAT,
            lgb_score           FLOAT,
            xgb_score           FLOAT,
            cat_score           FLOAT,
            stacked_score       FLOAT,
            risk_label          TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS application_features (
            id                  SERIAL PRIMARY KEY,
            application_id      INT REFERENCES applications(id) ON DELETE CASCADE,
            submitted_at        TIMESTAMP DEFAULT NOW(),
        "NAME_CONTRACT_TYPE" TEXT,
        "CODE_GENDER" TEXT,
        "FLAG_OWN_CAR" FLOAT,
        "FLAG_OWN_REALTY" FLOAT,
        "CNT_CHILDREN" FLOAT,
        "AMT_INCOME_TOTAL" FLOAT,
        "AMT_CREDIT" FLOAT,
        "AMT_ANNUITY" FLOAT,
        "AMT_GOODS_PRICE" FLOAT,
        "NAME_TYPE_SUITE" TEXT,
        "NAME_INCOME_TYPE" TEXT,
        "NAME_EDUCATION_TYPE" TEXT,
        "NAME_FAMILY_STATUS" TEXT,
        "NAME_HOUSING_TYPE" TEXT,
        "REGION_POPULATION_RELATIVE" FLOAT,
        "DAYS_BIRTH" FLOAT,
        "DAYS_EMPLOYED" FLOAT,
        "DAYS_REGISTRATION" FLOAT,
        "DAYS_ID_PUBLISH" FLOAT,
        "OWN_CAR_AGE" FLOAT,
        "FLAG_MOBIL" FLOAT,
        "FLAG_EMP_PHONE" FLOAT,
        "FLAG_WORK_PHONE" FLOAT,
        "FLAG_CONT_MOBILE" FLOAT,
        "FLAG_PHONE" FLOAT,
        "FLAG_EMAIL" FLOAT,
        "OCCUPATION_TYPE" TEXT,
        "CNT_FAM_MEMBERS" FLOAT,
        "REGION_RATING_CLIENT" FLOAT,
        "REGION_RATING_CLIENT_W_CITY" FLOAT,
        "WEEKDAY_APPR_PROCESS_START" TEXT,
        "HOUR_APPR_PROCESS_START" FLOAT,
        "REG_REGION_NOT_LIVE_REGION" FLOAT,
        "REG_REGION_NOT_WORK_REGION" FLOAT,
        "LIVE_REGION_NOT_WORK_REGION" FLOAT,
        "REG_CITY_NOT_LIVE_CITY" FLOAT,
        "REG_CITY_NOT_WORK_CITY" FLOAT,
        "LIVE_CITY_NOT_WORK_CITY" FLOAT,
        "ORGANIZATION_TYPE" TEXT,
        "EXT_SOURCE_1" FLOAT,
        "EXT_SOURCE_2" FLOAT,
        "EXT_SOURCE_3" FLOAT,
        "APARTMENTS_AVG" FLOAT,
        "BASEMENTAREA_AVG" FLOAT,
        "YEARS_BEGINEXPLUATATION_AVG" FLOAT,
        "YEARS_BUILD_AVG" FLOAT,
        "COMMONAREA_AVG" FLOAT,
        "ELEVATORS_AVG" FLOAT,
        "ENTRANCES_AVG" FLOAT,
        "FLOORSMAX_AVG" FLOAT,
        "FLOORSMIN_AVG" FLOAT,
        "LANDAREA_AVG" FLOAT,
        "LIVINGAPARTMENTS_AVG" FLOAT,
        "LIVINGAREA_AVG" FLOAT,
        "NONLIVINGAPARTMENTS_AVG" FLOAT,
        "NONLIVINGAREA_AVG" FLOAT,
        "BASEMENTAREA_MODE" FLOAT,
        "YEARS_BEGINEXPLUATATION_MODE" FLOAT,
        "YEARS_BUILD_MODE" FLOAT,
        "COMMONAREA_MODE" FLOAT,
        "ELEVATORS_MODE" FLOAT,
        "ENTRANCES_MODE" FLOAT,
        "FLOORSMAX_MODE" FLOAT,
        "FLOORSMIN_MODE" FLOAT,
        "LANDAREA_MODE" FLOAT,
        "LIVINGAPARTMENTS_MODE" FLOAT,
        "LIVINGAREA_MODE" FLOAT,
        "NONLIVINGAPARTMENTS_MODE" FLOAT,
        "NONLIVINGAREA_MODE" FLOAT,
        "BASEMENTAREA_MEDI" FLOAT,
        "YEARS_BEGINEXPLUATATION_MEDI" FLOAT,
        "YEARS_BUILD_MEDI" FLOAT,
        "COMMONAREA_MEDI" FLOAT,
        "ELEVATORS_MEDI" FLOAT,
        "ENTRANCES_MEDI" FLOAT,
        "FLOORSMAX_MEDI" FLOAT,
        "FLOORSMIN_MEDI" FLOAT,
        "LANDAREA_MEDI" FLOAT,
        "LIVINGAPARTMENTS_MEDI" FLOAT,
        "LIVINGAREA_MEDI" FLOAT,
        "NONLIVINGAPARTMENTS_MEDI" FLOAT,
        "NONLIVINGAREA_MEDI" FLOAT,
        "FONDKAPREMONT_MODE" TEXT,
        "HOUSETYPE_MODE" TEXT,
        "TOTALAREA_MODE" FLOAT,
        "WALLSMATERIAL_MODE" TEXT,
        "EMERGENCYSTATE_MODE" TEXT,
        "OBS_30_CNT_SOCIAL_CIRCLE" FLOAT,
        "DEF_30_CNT_SOCIAL_CIRCLE" FLOAT,
        "OBS_60_CNT_SOCIAL_CIRCLE" FLOAT,
        "DEF_60_CNT_SOCIAL_CIRCLE" FLOAT,
        "DAYS_LAST_PHONE_CHANGE" FLOAT,
        "FLAG_DOCUMENT_2" FLOAT,
        "FLAG_DOCUMENT_3" FLOAT,
        "FLAG_DOCUMENT_4" FLOAT,
        "FLAG_DOCUMENT_5" FLOAT,
        "FLAG_DOCUMENT_6" FLOAT,
        "FLAG_DOCUMENT_7" FLOAT,
        "FLAG_DOCUMENT_8" FLOAT,
        "FLAG_DOCUMENT_9" FLOAT,
        "FLAG_DOCUMENT_10" FLOAT,
        "FLAG_DOCUMENT_11" FLOAT,
        "FLAG_DOCUMENT_12" FLOAT,
        "FLAG_DOCUMENT_13" FLOAT,
        "FLAG_DOCUMENT_14" FLOAT,
        "FLAG_DOCUMENT_15" FLOAT,
        "FLAG_DOCUMENT_16" FLOAT,
        "FLAG_DOCUMENT_17" FLOAT,
        "FLAG_DOCUMENT_18" FLOAT,
        "FLAG_DOCUMENT_19" FLOAT,
        "FLAG_DOCUMENT_20" FLOAT,
        "FLAG_DOCUMENT_21" FLOAT,
        "AMT_REQ_CREDIT_BUREAU_HOUR" FLOAT,
        "AMT_REQ_CREDIT_BUREAU_DAY" FLOAT,
        "AMT_REQ_CREDIT_BUREAU_WEEK" FLOAT,
        "AMT_REQ_CREDIT_BUREAU_MON" FLOAT,
        "AMT_REQ_CREDIT_BUREAU_QRT" FLOAT,
        "AMT_REQ_CREDIT_BUREAU_YEAR" FLOAT,
        "AGE" FLOAT,
        "DAYS_EMPLOYED_ANOM" FLOAT,
        "CREDIT_INCOME_RATIO" FLOAT,
        "ANNUITY_INCOME_RATIO" FLOAT,
        "CREDIT_ANNUITY_RATIO" FLOAT,
        "INCOME_PER_PERSON" FLOAT,
        "EXT_SOURCE_MEAN" FLOAT,
        "DAYS_CREDIT_MEAN" FLOAT,
        "DAYS_CREDIT_MIN" FLOAT,
        "DAYS_CREDIT_MAX" FLOAT,
        "AMT_CREDIT_SUM_SUM" FLOAT,
        "AMT_CREDIT_SUM_MEAN" FLOAT,
        "AMT_CREDIT_SUM_DEBT_SUM" FLOAT,
        "AMT_CREDIT_SUM_DEBT_MEAN" FLOAT,
        "AMT_CREDIT_SUM_OVERDUE_SUM" FLOAT,
        "BUREAU_LOAN_COUNT_" FLOAT,
        "ACTIVE_LOAN_COUNT" FLOAT,
        "STATUS_NUM_MAX_MAX" FLOAT,
        "STATUS_NUM_MAX_MEAN" FLOAT,
        "MONTHS_BALANCE_MIN_MIN" FLOAT,
        "AMT_APPLICATION_MEAN" FLOAT,
        "AMT_APPLICATION_MAX" FLOAT,
        "AMT_CREDIT_MEAN" FLOAT,
        "AMT_CREDIT_MAX" FLOAT,
        "AMT_ANNUITY_MEAN" FLOAT,
        "DAYS_DECISION_MEAN" FLOAT,
        "DAYS_DECISION_MIN" FLOAT,
        "PREV_APP_COUNT" FLOAT,
        "APPROVED_COUNT" FLOAT,
        "REFUSED_COUNT" FLOAT,
        "MONTHS_BALANCE_MIN" FLOAT,
        "MONTHS_BALANCE_MAX" FLOAT,
        "CNT_INSTALMENT_MEAN" FLOAT,
        "CNT_INSTALMENT_MAX" FLOAT,
        "CNT_INSTALMENT_FUTURE_MEAN" FLOAT,
        "SK_DPD_MAX_x" FLOAT,
        "SK_DPD_MEAN_x" FLOAT,
        "SK_DPD_DEF_MAX" FLOAT,
        "PAYMENT_DELAY_MEAN" FLOAT,
        "PAYMENT_DELAY_MAX" FLOAT,
        "AMT_PAYMENT_SUM" FLOAT,
        "AMT_PAYMENT_MEAN" FLOAT,
        "AMT_INSTALMENT_SUM" FLOAT,
        "AMT_INSTALMENT_MEAN" FLOAT,
        "AMT_BALANCE_MEAN" FLOAT,
        "AMT_BALANCE_MAX" FLOAT,
        "AMT_CREDIT_LIMIT_ACTUAL_MEAN" FLOAT,
        "SK_DPD_MAX_y" FLOAT,
        "SK_DPD_MEAN_y" FLOAT
        )
    """)

    conn.commit()
    cur.close()
    conn.close()


def insert_application(personal, inputs, features, predictions, feature_vector):

    conn = get_connection()
    cur = conn.cursor()

    risk = predictions["stacked"]
    if risk < 0.2:
        label = "Low Risk"
    elif risk < 0.5:
        label = "Medium Risk"
    else:
        label = "High Risk"

    cur.execute("""
        INSERT INTO applications (
            full_name, email, phone, address, date_of_birth,
            gender, age, income, loan_amount, annuity,
            employment_years, family_members, owns_car, owns_house,
            education, family_status,
            ext_source_1, ext_source_2, ext_source_3,
            credit_income_ratio, annuity_income_ratio,
            credit_annuity_ratio, income_per_person, ext_source_mean,
            lgb_score, xgb_score, cat_score, stacked_score, risk_label
        ) VALUES (
            %(full_name)s, %(email)s, %(phone)s, %(address)s, %(date_of_birth)s,
            %(gender)s, %(age)s, %(income)s, %(loan_amount)s, %(annuity)s,
            %(employment_years)s, %(family_members)s, %(owns_car)s, %(owns_house)s,
            %(education)s, %(family_status)s,
            %(ext_source_1)s, %(ext_source_2)s, %(ext_source_3)s,
            %(credit_income_ratio)s, %(annuity_income_ratio)s,
            %(credit_annuity_ratio)s, %(income_per_person)s, %(ext_source_mean)s,
            %(lgb_score)s, %(xgb_score)s, %(cat_score)s, %(stacked_score)s, %(risk_label)s
        ) RETURNING id
    """, {
        **personal,
        **inputs,
        **features,
        "lgb_score":     predictions["lgb"],
        "xgb_score":     predictions["xgb"],
        "cat_score":     predictions["cat"],
        "stacked_score": predictions["stacked"],
        "risk_label":    label,
    })

    application_id = cur.fetchone()[0]

    # insert full feature vector
    fv = {k: (None if (isinstance(v, float) and v != v) else v) for k, v in feature_vector.items()}
    fv["application_id"] = application_id

    cur.execute("""
        INSERT INTO application_features (application_id, "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE", "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START", "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY", "ORGANIZATION_TYPE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMAX_MODE", "FLOORSMIN_MODE", "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI", "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI", "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "TOTALAREA_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE", "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "DAYS_LAST_PHONE_CHANGE", "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16", "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21", "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR", "AGE", "DAYS_EMPLOYED_ANOM", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_ANNUITY_RATIO", "INCOME_PER_PERSON", "EXT_SOURCE_MEAN", "DAYS_CREDIT_MEAN", "DAYS_CREDIT_MIN", "DAYS_CREDIT_MAX", "AMT_CREDIT_SUM_SUM", "AMT_CREDIT_SUM_MEAN", "AMT_CREDIT_SUM_DEBT_SUM", "AMT_CREDIT_SUM_DEBT_MEAN", "AMT_CREDIT_SUM_OVERDUE_SUM", "BUREAU_LOAN_COUNT_", "ACTIVE_LOAN_COUNT", "STATUS_NUM_MAX_MAX", "STATUS_NUM_MAX_MEAN", "MONTHS_BALANCE_MIN_MIN", "AMT_APPLICATION_MEAN", "AMT_APPLICATION_MAX", "AMT_CREDIT_MEAN", "AMT_CREDIT_MAX", "AMT_ANNUITY_MEAN", "DAYS_DECISION_MEAN", "DAYS_DECISION_MIN", "PREV_APP_COUNT", "APPROVED_COUNT", "REFUSED_COUNT", "MONTHS_BALANCE_MIN", "MONTHS_BALANCE_MAX", "CNT_INSTALMENT_MEAN", "CNT_INSTALMENT_MAX", "CNT_INSTALMENT_FUTURE_MEAN", "SK_DPD_MAX_x", "SK_DPD_MEAN_x", "SK_DPD_DEF_MAX", "PAYMENT_DELAY_MEAN", "PAYMENT_DELAY_MAX", "AMT_PAYMENT_SUM", "AMT_PAYMENT_MEAN", "AMT_INSTALMENT_SUM", "AMT_INSTALMENT_MEAN", "AMT_BALANCE_MEAN", "AMT_BALANCE_MAX", "AMT_CREDIT_LIMIT_ACTUAL_MEAN", "SK_DPD_MAX_y", "SK_DPD_MEAN_y")
        VALUES (%(application_id)s, %(NAME_CONTRACT_TYPE)s, %(CODE_GENDER)s, %(FLAG_OWN_CAR)s, %(FLAG_OWN_REALTY)s, %(CNT_CHILDREN)s, %(AMT_INCOME_TOTAL)s, %(AMT_CREDIT)s, %(AMT_ANNUITY)s, %(AMT_GOODS_PRICE)s, %(NAME_TYPE_SUITE)s, %(NAME_INCOME_TYPE)s, %(NAME_EDUCATION_TYPE)s, %(NAME_FAMILY_STATUS)s, %(NAME_HOUSING_TYPE)s, %(REGION_POPULATION_RELATIVE)s, %(DAYS_BIRTH)s, %(DAYS_EMPLOYED)s, %(DAYS_REGISTRATION)s, %(DAYS_ID_PUBLISH)s, %(OWN_CAR_AGE)s, %(FLAG_MOBIL)s, %(FLAG_EMP_PHONE)s, %(FLAG_WORK_PHONE)s, %(FLAG_CONT_MOBILE)s, %(FLAG_PHONE)s, %(FLAG_EMAIL)s, %(OCCUPATION_TYPE)s, %(CNT_FAM_MEMBERS)s, %(REGION_RATING_CLIENT)s, %(REGION_RATING_CLIENT_W_CITY)s, %(WEEKDAY_APPR_PROCESS_START)s, %(HOUR_APPR_PROCESS_START)s, %(REG_REGION_NOT_LIVE_REGION)s, %(REG_REGION_NOT_WORK_REGION)s, %(LIVE_REGION_NOT_WORK_REGION)s, %(REG_CITY_NOT_LIVE_CITY)s, %(REG_CITY_NOT_WORK_CITY)s, %(LIVE_CITY_NOT_WORK_CITY)s, %(ORGANIZATION_TYPE)s, %(EXT_SOURCE_1)s, %(EXT_SOURCE_2)s, %(EXT_SOURCE_3)s, %(APARTMENTS_AVG)s, %(BASEMENTAREA_AVG)s, %(YEARS_BEGINEXPLUATATION_AVG)s, %(YEARS_BUILD_AVG)s, %(COMMONAREA_AVG)s, %(ELEVATORS_AVG)s, %(ENTRANCES_AVG)s, %(FLOORSMAX_AVG)s, %(FLOORSMIN_AVG)s, %(LANDAREA_AVG)s, %(LIVINGAPARTMENTS_AVG)s, %(LIVINGAREA_AVG)s, %(NONLIVINGAPARTMENTS_AVG)s, %(NONLIVINGAREA_AVG)s, %(BASEMENTAREA_MODE)s, %(YEARS_BEGINEXPLUATATION_MODE)s, %(YEARS_BUILD_MODE)s, %(COMMONAREA_MODE)s, %(ELEVATORS_MODE)s, %(ENTRANCES_MODE)s, %(FLOORSMAX_MODE)s, %(FLOORSMIN_MODE)s, %(LANDAREA_MODE)s, %(LIVINGAPARTMENTS_MODE)s, %(LIVINGAREA_MODE)s, %(NONLIVINGAPARTMENTS_MODE)s, %(NONLIVINGAREA_MODE)s, %(BASEMENTAREA_MEDI)s, %(YEARS_BEGINEXPLUATATION_MEDI)s, %(YEARS_BUILD_MEDI)s, %(COMMONAREA_MEDI)s, %(ELEVATORS_MEDI)s, %(ENTRANCES_MEDI)s, %(FLOORSMAX_MEDI)s, %(FLOORSMIN_MEDI)s, %(LANDAREA_MEDI)s, %(LIVINGAPARTMENTS_MEDI)s, %(LIVINGAREA_MEDI)s, %(NONLIVINGAPARTMENTS_MEDI)s, %(NONLIVINGAREA_MEDI)s, %(FONDKAPREMONT_MODE)s, %(HOUSETYPE_MODE)s, %(TOTALAREA_MODE)s, %(WALLSMATERIAL_MODE)s, %(EMERGENCYSTATE_MODE)s, %(OBS_30_CNT_SOCIAL_CIRCLE)s, %(DEF_30_CNT_SOCIAL_CIRCLE)s, %(OBS_60_CNT_SOCIAL_CIRCLE)s, %(DEF_60_CNT_SOCIAL_CIRCLE)s, %(DAYS_LAST_PHONE_CHANGE)s, %(FLAG_DOCUMENT_2)s, %(FLAG_DOCUMENT_3)s, %(FLAG_DOCUMENT_4)s, %(FLAG_DOCUMENT_5)s, %(FLAG_DOCUMENT_6)s, %(FLAG_DOCUMENT_7)s, %(FLAG_DOCUMENT_8)s, %(FLAG_DOCUMENT_9)s, %(FLAG_DOCUMENT_10)s, %(FLAG_DOCUMENT_11)s, %(FLAG_DOCUMENT_12)s, %(FLAG_DOCUMENT_13)s, %(FLAG_DOCUMENT_14)s, %(FLAG_DOCUMENT_15)s, %(FLAG_DOCUMENT_16)s, %(FLAG_DOCUMENT_17)s, %(FLAG_DOCUMENT_18)s, %(FLAG_DOCUMENT_19)s, %(FLAG_DOCUMENT_20)s, %(FLAG_DOCUMENT_21)s, %(AMT_REQ_CREDIT_BUREAU_HOUR)s, %(AMT_REQ_CREDIT_BUREAU_DAY)s, %(AMT_REQ_CREDIT_BUREAU_WEEK)s, %(AMT_REQ_CREDIT_BUREAU_MON)s, %(AMT_REQ_CREDIT_BUREAU_QRT)s, %(AMT_REQ_CREDIT_BUREAU_YEAR)s, %(AGE)s, %(DAYS_EMPLOYED_ANOM)s, %(CREDIT_INCOME_RATIO)s, %(ANNUITY_INCOME_RATIO)s, %(CREDIT_ANNUITY_RATIO)s, %(INCOME_PER_PERSON)s, %(EXT_SOURCE_MEAN)s, %(DAYS_CREDIT_MEAN)s, %(DAYS_CREDIT_MIN)s, %(DAYS_CREDIT_MAX)s, %(AMT_CREDIT_SUM_SUM)s, %(AMT_CREDIT_SUM_MEAN)s, %(AMT_CREDIT_SUM_DEBT_SUM)s, %(AMT_CREDIT_SUM_DEBT_MEAN)s, %(AMT_CREDIT_SUM_OVERDUE_SUM)s, %(BUREAU_LOAN_COUNT_)s, %(ACTIVE_LOAN_COUNT)s, %(STATUS_NUM_MAX_MAX)s, %(STATUS_NUM_MAX_MEAN)s, %(MONTHS_BALANCE_MIN_MIN)s, %(AMT_APPLICATION_MEAN)s, %(AMT_APPLICATION_MAX)s, %(AMT_CREDIT_MEAN)s, %(AMT_CREDIT_MAX)s, %(AMT_ANNUITY_MEAN)s, %(DAYS_DECISION_MEAN)s, %(DAYS_DECISION_MIN)s, %(PREV_APP_COUNT)s, %(APPROVED_COUNT)s, %(REFUSED_COUNT)s, %(MONTHS_BALANCE_MIN)s, %(MONTHS_BALANCE_MAX)s, %(CNT_INSTALMENT_MEAN)s, %(CNT_INSTALMENT_MAX)s, %(CNT_INSTALMENT_FUTURE_MEAN)s, %(SK_DPD_MAX_x)s, %(SK_DPD_MEAN_x)s, %(SK_DPD_DEF_MAX)s, %(PAYMENT_DELAY_MEAN)s, %(PAYMENT_DELAY_MAX)s, %(AMT_PAYMENT_SUM)s, %(AMT_PAYMENT_MEAN)s, %(AMT_INSTALMENT_SUM)s, %(AMT_INSTALMENT_MEAN)s, %(AMT_BALANCE_MEAN)s, %(AMT_BALANCE_MAX)s, %(AMT_CREDIT_LIMIT_ACTUAL_MEAN)s, %(SK_DPD_MAX_y)s, %(SK_DPD_MEAN_y)s)
    """, fv)

    conn.commit()
    cur.close()
    conn.close()

    return application_id


def fetch_all_applications():

    conn = get_connection()

    df = pd.read_sql("""
        SELECT
            id, submitted_at, full_name, email, phone,
            loan_amount, income, stacked_score,
            lgb_score, xgb_score, cat_score, risk_label
        FROM applications
        ORDER BY submitted_at DESC
    """, conn)

    conn.close()
    return df


def fetch_application(app_id):

    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM applications WHERE id = %s", (app_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else None


def fetch_features(app_id):

    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM application_features WHERE application_id = %s", (app_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else None


if __name__ == "__main__":
    create_tables()
    print("Tables created.")