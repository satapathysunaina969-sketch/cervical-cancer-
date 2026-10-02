import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cervical Cancer Risk & India Dashboard", page_icon="🩺", layout="wide")

st.title("🩺 Cervical Cancer Risk Screening & India Dashboard")

tab1, tab2 = st.tabs(["Risk Predictor", "India Dashboard"])

# =========================================================
# TAB 1 — RISK PREDICTOR
# =========================================================
with tab1:
    model = joblib.load("cervical_model.joblib")
    scaler = joblib.load("cervical_scaler.joblib")
    feature_names = joblib.load("cervical_features.joblib")
    threshold = joblib.load("cervical_threshold.joblib")

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

        input_df = pd.DataFrame([input_dict])[feature_names]
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

# =========================================================
# TAB 2 — INDIA DASHBOARD
# =========================================================
with tab2:
    st.subheader("Cervical Cancer in India — State-wise Overview")
    st.caption(
        "Source: Indian Council of Medical Research — National Cancer Registry Programme "
        "(ICMR-NCRP), as reported to the Lok Sabha (Government of India). "
        "Figures are modelled estimates, not direct counts."
    )

    mortality_long = pd.read_csv("mortality_clean.csv")
    incidence_long = pd.read_csv("incidence_clean.csv")

    metric = st.radio("View:", ["Mortality (Deaths)", "Incidence (New Cases)"], horizontal=True)

    if metric == "Mortality (Deaths)":
        data = mortality_long
        value_col = "Deaths"
        color = "indianred"
        label = "Estimated Deaths"
    else:
        data = incidence_long
        value_col = "Cases"
        color = "steelblue"
        label = "Estimated New Cases"

    years_available = sorted(data["Year"].unique())
    selected_year = st.select_slider("Select year:", options=years_available, value=years_available[-1])

    col1, col2 = st.columns(2)

    # --- Top 10 states bar chart ---
    with col1:
        top10 = data[data["Year"] == selected_year].sort_values(value_col, ascending=False).head(10)
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.barh(top10["State"], top10[value_col], color=color)
        ax1.invert_yaxis()
        ax1.set_xlabel(f"{label} ({selected_year})")
        ax1.set_title(f"Top 10 States — {label} ({selected_year})")
        plt.tight_layout()
        st.pyplot(fig1)

    # --- National trend line ---
    with col2:
        national_trend = data.groupby("Year")[value_col].sum().reset_index()
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.plot(national_trend["Year"], national_trend[value_col], marker="o", color=color)
        ax2.set_xlabel("Year")
        ax2.set_ylabel(f"Total {label} (All States)")
        ax2.set_title(f"India — National Trend ({years_available[0]}–{years_available[-1]})")
        plt.tight_layout()
        st.pyplot(fig2)

    st.info(
        "📌 **Note on interpretation:** These are absolute counts, not rates. Larger, more populous "
        "states (e.g. Uttar Pradesh, Maharashtra) naturally show higher numbers. This reflects "
        "population size as much as underlying risk, and should not be read as a per-capita risk ranking."
    )

    with st.expander("View raw data table"):
        st.dataframe(data[data["Year"] == selected_year].sort_values(value_col, ascending=False), use_container_width=True)
        import joblib
import pandas as pd
import streamlit as st
from google import genai

# Page Config
st.set_page_config(
    page_title="Cervical Health AI Assistant", page_icon="🩺", layout="wide"
)

# 1. Initialize Gemini Client safely using Streamlit Secrets
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


client = get_gemini_client()

# 2. Load ML Model & Scaler
@st.cache_resource
def load_ml_pipeline():
    model = joblib.load("cervical_model.joblib")
    scaler = joblib.load("cervical_scaler.joblib")
    return model, scaler


try:
    model, scaler = load_ml_pipeline()
except Exception:
    model, scaler = None, None

st.title("🩺 Cervical Health AI Assistant")
st.caption(
    "Conversational AI for Cervical Cancer Risk Assessment & Epidemiological Insights"
)

# 3. Maintain Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your Cervical Health Assistant. You can describe your demographic/health background for a risk estimation, or ask about cervical cancer stats in India.",
        }
    ]

# Display prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Handle Chat Input
if user_prompt := st.chat_input("Type your message or health details here..."):
    # Display User Input
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate Response using Gemini 3.8 Flash
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=f"""
                    You are a supportive clinical information assistant specializing in cervical health and epidemiology education.
                    Respond directly and helpfully to: '{user_prompt}'.
                    Remind users gently that statistical risk tools do not replace clinical screening or medical professional advice.
                    """,
                )
                assistant_text = response.text
            except Exception as err:
                assistant_text = (
                    f"I encountered an issue processing your request: {err}"
                )

            st.markdown(assistant_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_text}
            )
