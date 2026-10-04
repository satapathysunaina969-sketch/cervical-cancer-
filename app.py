import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from google import genai
from streamlit_mic_recorder import speech_to_text

# Page Config
st.set_page_config(
    page_title="Cervical Cancer Risk & India Dashboard",
    page_icon="🩺",
    layout="wide",
)

# Initialize Gemini Client safely using Streamlit Secrets
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

try:
    client = get_gemini_client()
except Exception:
    client = None

st.title("🩺 Cervical Cancer Risk Screening & India Dashboard")

tab1, tab2 = st.tabs(["Risk Predictor & AI Assistant", "India Dashboard"])

# =========================================================
# TAB 1 — RISK PREDICTOR & AI ASSISTANT
# =========================================================
with tab1:
    try:
        model = joblib.load("cervical_model.joblib")
        scaler = joblib.load("cervical_scaler.joblib")
        feature_names = joblib.load("cervical_features.joblib")
        threshold = joblib.load("cervical_threshold.joblib")
    except Exception:
        model, scaler, feature_names, threshold = None, None, None, 0.5

    st.warning(
        "⚠️ **ML Demo — Not a medical diagnosis.** "
        "This tool estimates a risk pattern based on a research dataset. "
        "It does not replace a doctor, a Pap smear, or an HPV test. "
        "Please consult a healthcare provider for an actual diagnosis."
    )

    st.write("Please fill in the fields below as accurately as you can.")

    col_a, col_b = st.columns(2)

    with col_a:
        # --- Demographics & Sexual Activity ---
        st.subheader("Demographics & Sexual History")
        age = st.number_input("Age", min_value=10, max_value=100, value=30, key="age")

        sexually_active_choice = st.radio(
            "Have you ever been sexually active?",
            ["No", "Yes"],
            horizontal=True,
            key="sexually_active",
        )
        if sexually_active_choice == "Yes":
            partners = st.number_input("Number of sexual partners", min_value=1, max_value=50, value=1, key="partners")
            first_intercourse = st.number_input("Age at first sexual intercourse", min_value=10, max_value=50, value=18, key="first_intercourse")
        else:
            partners = 0
            first_intercourse = 0

        pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=20, value=0, key="pregnancies")

        # --- Habits ---
        st.subheader("Habits")
        smokes_choice = st.radio("Do you smoke?", ["No", "Yes"], horizontal=True, key="smokes_radio")
        smokes = True if smokes_choice == "Yes" else False

        if smokes:
            smokes_years = st.number_input("If yes, for how many years?", min_value=0.0, max_value=60.0, value=1.0, step=0.5, key="smokes_years")
            smokes_packs = st.number_input("If yes, packs per year", min_value=0.0, max_value=50.0, value=0.5, step=0.1, key="smokes_packs")
        else:
            smokes_years = 0.0
            smokes_packs = 0.0

    with col_b:
        # --- Contraceptive History ---
        st.subheader("Contraceptive History")
        hormonal_choice = st.radio("Have you used hormonal contraceptives?", ["No", "Yes"], horizontal=True, key="hormonal_radio")
        hormonal = True if hormonal_choice == "Yes" else False

        if hormonal:
            hormonal_years = st.number_input("If yes, for how many years?", min_value=0.0, max_value=40.0, value=1.0, step=0.5, key="hormonal_years")
        else:
            hormonal_years = 0.0

        iud_choice = st.radio("Have you used an IUD?", ["No", "Yes"], horizontal=True, key="iud_radio")
        iud = True if iud_choice == "Yes" else False

        if iud:
            iud_years = st.number_input("If yes, for how many years?", min_value=0.0, max_value=40.0, value=1.0, step=0.5, key="iud_years")
        else:
            iud_years = 0.0

        # --- STD History ---
        st.subheader("STD History")
        stds_choice = st.radio("Have you ever been diagnosed with an STD?", ["No", "Yes"], horizontal=True, key="stds_radio")
        stds = True if stds_choice == "Yes" else False

        if stds:
            stds_number = st.number_input("If yes, how many different STDs in total?", min_value=1, max_value=10, value=1, key="stds_number")
        else:
            stds_number = 0

    st.markdown("---")
    st.subheader("Medical & Gynecological History")

    # Gynecological, Endocrine & General Medical Conditions
    repro_diseases = st.multiselect(
        "Select any diagnosed reproductive, hormonal, or general medical conditions:",
        [
            "PCOD (Polycystic Ovarian Disease)",
            "PCOS (Polycystic Ovary Syndrome)",
            "Thyroid Disorder (Hypothyroidism / Hyperthyroidism)",
            "Diabetes / Pre-diabetes",
            "Asthma / Chronic Respiratory Condition",
            "Hypertension (High Blood Pressure)",
            "Hormonal Imbalance (e.g., Estrogen dominance)",
            "Endometriosis",
            "Cervical Polyps / Chronic Cervicitis",
            "Pelvic Inflammatory Disease (PID)",
            "Irregular Bleeding / Intermenstrual Spotting",
            "Uterine Fibroids",
            "None of the above",
        ],
        key="repro_diseases",
    )

    with st.expander("Specific STD Diagnoses (Tick any that apply)"):
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

    # --- Predict Button ---
    if st.button("Check Risk Pattern", type="primary"):
        if model and scaler and feature_names:
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

            st.subheader("Prediction Result")
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

    # ---------------------------------------------------------
    # INTEGRATED AI HEALTH ASSISTANT CHAT WITH VOICE INPUT
    # ---------------------------------------------------------
    st.divider()
    st.subheader("💬 Cervical Health AI Assistant")
    st.caption("Ask questions or speak using the microphone button below regarding symptoms, PCOD, Diabetes, Thyroid, or screening guidance.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your Cervical Health Assistant. You can type or tap the microphone button to speak your questions about symptoms, PCOD/PCOS, Thyroid, Diabetes, or cervical cancer screening.",
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Voice Input Recorder Widget
   # Language selector for voice input
lang_choice = st.radio(
    "Select speech language:",
    ["English (India)", "Hindi (हिन्दी)"],
    horizontal=True,
    key="voice_lang",
)
selected_lang = "en-IN" if lang_choice == "English (India)" else "hi-IN"

spoken_text = speech_to_text(
    start_prompt="Click to Speak 🎙️",
    stop_prompt="Stop & Transcribe ⏹️",
    language=selected_lang,
    use_container_width=False,
    key="voice_recorder",
)

    # Standard Chat Input Box
    typed_prompt = st.chat_input("Type your message or health details here...")

    # Determine which input prompt was submitted (Voice or Typed)
    user_prompt = spoken_text if spoken_text else typed_prompt

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            repro_summary = ", ".join(repro_diseases) if repro_diseases else "None selected"
            context_info = f"""
            User Profile Summary:
            - Age: {age}
            - Sexually Active: {sexually_active_choice}
            - Smoking: {smokes_choice} ({smokes_years} yrs, {smokes_packs} packs/yr)
            - Hormonal Contraceptives: {hormonal_choice} ({hormonal_years} yrs)
            - IUD Use: {iud_choice} ({iud_years} yrs)
            - STD History: {stds_choice} ({stds_number} total)
            - Gynecological & Medical Conditions: {repro_summary}
            """

            prompt_content = f"""
            You are a supportive clinical information assistant specializing in cervical health, general women's health, and epidemiology education.
            
            {context_info}
            
            User Question: '{user_prompt}'
            
            Provide a concise, empathetic, and clear response addressing their query. If the user reported conditions like PCOD, Diabetes, Thyroid issues, or Asthma, address how those interact with overall wellness, inflammation, or hormonal health where relevant.
            Remind users gently that statistical risk tools and AI do not replace clinical screening or medical professional advice.
            """

            def stream_generator():
                if not client:
                    yield "Gemini API client is not initialized. Please ensure your GEMINI_API_KEY is configured in Streamlit Secrets."
                    return

                try:
                    response_stream = client.models.generate_content_stream(
                        model="gemini-3.8-flash",
                        contents=prompt_content,
                    )
                    for chunk in response_stream:
                        if chunk.text:
                            yield chunk.text
                except Exception:
                    try:
                        response_stream = client.models.generate_content_stream(
                            model="gemini-3.5-flash-lite",
                            contents=prompt_content,
                        )
                        for chunk in response_stream:
                            if chunk.text:
                                yield chunk.text
                    except Exception as e:
                        yield f"I encountered an issue generating a response: {e}"

            assistant_text = st.write_stream(stream_generator)
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})


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

    try:
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
    except Exception as e:
        st.error(f"Unable to load India Dashboard data files (mortality_clean.csv / incidence_clean.csv): {e}")
