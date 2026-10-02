
      import streamlit as st
import pandas as pd
import joblib

# Load the saved model, scaler, feature list, and threshold
model = joblib.load("cervical_model.joblib")
scaler = joblib.load("cervical_scaler.joblib")
feature_names = joblib.load("cervical_features.joblib")
threshold = joblib.load("cervical_threshold.joblib")

st.set_page_config(page_title="Cervical Cancer Risk Screening Tool", page_icon="🩺")

st.title("🩺 Cervical Cancer Risk Screening Tool")

st.warning(
    "⚠️ **ML Demo — Not a medical diagnosis.** "
    "This tool estimates a risk pattern based on a research dataset. "
    "It does not replace a doctor, a Pap smear, or an HPV test. "
    "Please consult a healthcare provider for an actual diagnosis."
)

st.write("Please fill in the fields below as accurately as you can.")

# --- Demographics ---
st.subheader("Demographics")
age = st.number_input("Age", min_value=10, max_value=100, value=30, key="age")
partners = st.number_input("Number of sexual partners", min_value=0, max_value=50, value=1, key="partners")
first_intercourse = st.number_input("Age at first sexual intercourse", min_value=0, max_value=50, value=18, key="first_intercourse")
pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=20, value=0, key="pregnancies")

# --- Habits ---
st.subheader("Habits")
smokes = st.checkbox("Do you smoke?", key="smokes")
smokes_years = st.number_input("If yes, for how many years?", min_value=0.0, max_value=60.0, value=0.0, key="smokes_years")
smokes_packs = st.number_input("If yes, packs per year", min_value=0.0, max_value=50.0, value=0.0, key="smokes_packs")

# --- Contraceptive history ---
st.subheader("Contraceptive History")
hormonal = st.checkbox("Have you used hormonal contraceptives?", key="hormonal")
hormonal_years = st.number_input("If yes, for how many years?", min_value=0.0, max_value=40.0, value=0.0, key="hormonal_years")
iud = st.checkbox("Have you used an IUD?", key="iud")
iud_years = st.number_input("If yes, for how many years?", min_value=0.0, max_value=40.0, value=0.0, key="iud_years")

# --- STD history ---
st.subheader("STD History")
stds = st.checkbox("Have you ever been diagnosed with an STD?", key="stds")
stds_number = st.number_input("If yes, how many different STDs in total?", min_value=0, max_value=10, value=0, key="stds_number")

st.caption("Tick any that apply:")
condylomatosis = st.checkbox("Condylomatosis", key="condylomatosis")
cervical_condylomatosis = st.checkbox("Cervical condylomatosis", key="cervical_condylomatosis")
vaginal_condylomatosis = st.checkbox("Vaginal condylomatosis", key="vaginal_condylomatosis")
vulvo_perineal_condylomatosis = st.checkbox("Vulvo-perineal condylomatosis", key="vulvo_perineal_condylomatosis")
syphilis = st.checkbox("Syphilis", key="syphilis")
pid = st.checkbox("Pelvic inflammatory disease", key="pid")
genital_herpes = st.checkbox("Genital herpes", key="genital_herpes")
molluscum = st.checkbox("Molluscum contagiosum", key="molluscum")
aids = st.checkbox("AIDS", key="aids")
hiv = st.checkbox("HIV", key="hiv")
hep_b = st.checkbox("Hepatitis B", key="hep_b")
hpv = st.checkbox("HPV", key="hpv")
stds_diagnosis_count = st.number_input("Total number of STD diagnoses received", min_value=0, max_value=20, value=0, key="stds_diagnosis_count")

# --- Predict button ---
if st.button("Check Risk Pattern"):
    # Build input row in the EXACT order the model expects
    input_dict = {
        "Age": age,
        "Number of sexual partners": partners,
        "First sexual intercourse": first_intercourse,
        "Num of pregnancies": pregnancies,
        "Smokes": int(smokes),
        "Smokes (years)": smokes_years,
        "Smokes (packs/year)": smokes_packs,
        "Hormonal Contraceptives": int(hormonal),
        "Hormonal Contraceptives (years)": hormonal_years,
        "IUD": int(iud),
        "IUD (years)": iud_years,
        "STDs": int(stds),
        "STDs (number)": stds_number,
        "STDs:condylomatosis": int(condylomatosis),
        "STDs:cervical condylomatosis": int(cervical_condylomatosis),
        "STDs:vaginal condylomatosis": int(vaginal_condylomatosis),
        "STDs:vulvo-perineal condylomatosis": int(vulvo_perineal_condylomatosis),
        "STDs:syphilis": int(syphilis),
        "STDs:pelvic inflammatory disease": int(pid),
        "STDs:genital herpes": int(genital_herpes),
        "STDs:molluscum contagiosum": int(molluscum),
        "STDs:AIDS": int(aids),
        "STDs:HIV": int(hiv),
        "STDs:Hepatitis B": int(hep_b),
        "STDs:HPV": int(hpv),
        "STDs: Number of diagnosis": stds_diagnosis_count,
    }

    # Arrange in the exact column order the scaler/model expect
    input_df = pd.DataFrame([input_dict])[feature_names]

    # Scale and predict probability
    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")
    if probability >= threshold:
        st.error(
            f"**Pattern flagged as higher-risk** (model confidence: {probability:.0%}).\n\n"
            "This is **not a diagnosis**. Based on the factors entered, your pattern is "
            "similar to cases flagged in this dataset. Please consult a doctor or get screened to know for sure."
        )
    else:
        st.success(
            f"**Pattern flagged as lower-risk** (model confidence: {probability:.0%}).\n\n"
            "This is **not a diagnosis** and does not rule out cancer. Routine screening is still "
            "recommended regardless of this result."
        )

st.caption("This tool is for educational purposes only and is not a substitute for professional medical advice.")
