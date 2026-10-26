
import csv
import io
from datetime import datetime
from pathlib import Path

import joblib
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Spam Alert Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
[data-testid="stSidebar"] * {
    color: #f9fafb;
}
.hero {
    background: linear-gradient(120deg, #172554, #2563eb);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 22px;
}
.hero h1 {
    color: white;
    font-size: 36px;
}
.hero p {
    color: #dbeafe;
    font-size: 17px;
}
.metric {
    background: white;
    border: 1px solid #e5eaf2;
    border-radius: 14px;
    padding: 20px;
}
.metric-label {
    color: #64748b;
    font-size: 14px;
}
.metric-value {
    color: #111827;
    font-size: 28px;
    font-weight: 700;
}
.stButton > button {
    border-radius: 9px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)

# ==========================================
# SESSION STATE
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_message(message):
    if model is None:
        return None, "Trained model is not available."

    prediction = str(model.predict([message])[0]).strip().lower()

    if prediction == "spam":
        return "Spam", (
            "The trained machine-learning model classified "
            "this message as spam."
        )

    if prediction in ("ham", "not spam", "not_spam", "safe"):
        return "Not flagged", (
            "The trained machine-learning model classified "
            "this message as legitimate."
        )

    return None, f"Unexpected model label: {prediction}"

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("# 🛡️ SpamGuard")
    st.caption("Spam Alert Detection System")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔍 Detect Spam",
            "📜 History",
            "ℹ️ About",
        ],
    )

    st.divider()

    if model is not None:
        st.success("ML Model Loaded")
    else:
        st.warning("ML Model Not Loaded")

    st.caption("Python • Streamlit • Machine Learning")

# ==========================================
# DASHBOARD
# ==========================================
if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🛡️ Spam Alert Detection</h1>
        <p>Analyze suspicious SMS messages with machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    total = len(st.session_state.history)
    spam_count = sum(
        item["result"] == "Spam"
        for item in st.session_state.history
    )
    not_flagged = total - spam_count

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div class="metric">'
            f'<div class="metric-label">Messages Analyzed</div>'
            f'<div class="metric-value">{total}</div></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f'<div class="metric">'
            f'<div class="metric-label">Flagged as Spam</div>'
            f'<div class="metric-value">{spam_count}</div></div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f'<div class="metric">'
            f'<div class="metric-label">Not Flagged</div>'
            f'<div class="metric-value">{not_flagged}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("Welcome!")
    st.write(
        "Enter a message to check its classification, "
        "and review previous results in Detection History."
    )

    if model is not None:
        st.success("Trained ML model connected successfully.")
    elif model_error:
        st.error(f"Could not load model: {model_error}")
    else:
        st.warning(
            "Model file not found. Run "
            "`python train_model.py` first."
        )

    if st.button("🔍 Go to Spam Detection", type="primary"):
        st.session_state.open_detector = True
        st.rerun()

# ==========================================
# DETECT SPAM
# ==========================================
elif page == "🔍 Detect Spam":

    st.title("🔍 Spam Message Detection")
    st.write("Enter an SMS message below for classification.")

    message = st.text_area(
        "Enter your message",
        placeholder="Example: Congratulations! You have won a prize...",
        height=180,
        max_chars=10000,
        key="message_input",
    )

    if st.button(
        "🛡️ Check Spam",
        type="primary",
        use_container_width=True,
    ):
        if not message.strip():
            st.warning("Please enter a message first.")

        elif model is None:
            st.error(
                "Trained model is unavailable. "
                "Check that spam_model.pkl exists and "
                "run python train_model.py."
            )

        else:
            try:
                result, explanation = predict_message(message.strip())

                if result is None:
                    st.error(explanation)
                else:
                    record = {
                        "time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "message": message.strip(),
                        "result": result,
                        "explanation": explanation,
                    }

                    st.session_state.last_result = record
                    st.session_state.history.insert(0, record)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    result = st.session_state.last_result

    if result:
        st.divider()
        st.subheader("Detection Result")

        if result["result"] == "Spam":
            st.error("🚨 Potential Spam Detected")
        else:
            st.success("✅ Message Not Flagged as Spam")

        st.write("**Analyzed message:**")
        st.info(result["message"])

        st.write("**Model explanation:**")
        st.write(result["explanation"])

        st.caption(
            "This is a model prediction, not a guarantee of safety. "
            "Avoid clicking suspicious links or sharing passwords."
        )

# ==========================================
# DETECTION HISTORY
# ==========================================
elif page == "📜 History":

    st.title("📜 Detection History")

    history = st.session_state.history

    if not history:
        st.info("No detection records yet. Analyze a message first.")

    else:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Analyzed", len(history))

        with col2:
            st.metric(
                "Spam Detected",
                sum(item["result"] == "Spam" for item in history),
            )

        for index, item in enumerate(history, start=1):
            with st.expander(
                f"{index}. {item['result']} — {item['time']}"
            ):
                st.write("**Message:**")
                st.write(item["message"])
                st.write("**Result:**", item["result"])
                st.write("**Details:**", item["explanation"])

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            ["time", "result", "message", "explanation"]
        )

        for item in history:
            writer.writerow([
                item["time"],
                item["result"],
                item["message"],
                item["explanation"],
            ])

        st.download_button(
            "⬇️ Download History as CSV",
            data=output.getvalue(),
            file_name="spam_detection_history.csv",
            mime="text/csv",
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

# ==========================================
# ABOUT
# ==========================================
elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    ## Spam Alert Detection System

    This project classifies SMS messages using a trained
    machine-learning text classification pipeline.

    ### Technologies
    - Python
    - Streamlit
    - Scikit-learn
    - TF-IDF text features
    - Multinomial Naive Bayes

    ### How it works
    1. Enter an SMS message.
    2. The trained model processes the text.
    3. The model returns a spam or ham prediction.
    4. The result is added to session-based history.

    **Limitations:** Predictions depend on the training dataset.
    A message classified as legitimate is not guaranteed to be safe.
    """)

st.divider()
st.caption("Spam Alert Detection | Academic Project")
