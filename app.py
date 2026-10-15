import streamlit as st
from main import analyze_scam

# PAGE CONFIG
st.set_page_config(
    page_title="AI Scam Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# CUSTOM CSS
st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .risk-box {
        padding: 25px;
        border-radius: 12px;
        background-color: #f5f5f5;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# HEADER
st.markdown(
    '<div class="main-title">🛡️ AI Scam Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered scam risk assessment using Machine Learning, '
    'RAG and LLMs'
    '</div>',
    unsafe_allow_html=True
)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    input_type = st.selectbox(
        "Select input type",
        ["message", "job", "url"]
    )
    st.divider()
    st.markdown("### About")
    st.write(
        "This system analyzes suspicious messages, "
        "job postings and URLs."
    )

    st.write(
        "It combines machine-learning predictions "
        "with retrieved scam knowledge and an LLM "
        "to generate a final risk assessment."
    )

    st.divider()
    st.caption(
        "AI Scam Intelligence Platform"
    )

# MAIN INPUT AREA
st.markdown(
    '<div class="section-title">🔍 Analyze Suspicious Content</div>',
    unsafe_allow_html=True
)

if input_type == "message":
    st.info(
        "Analyze SMS, WhatsApp messages, emails, "
        "banking messages or other suspicious text."
    )
    user_input = st.text_area(
        "Paste the suspicious message",
        height=220,
        placeholder=(
            "Example:\n\n"
            "Your bank account will be blocked today. "
            "Verify your account immediately using the "
            "link below."
        )
    )
elif input_type == "job":
    st.info(
        "Analyze a suspicious job posting or job offer."
    )
    user_input = st.text_area(
        "Paste the job posting",
        height=300,
        placeholder=(
            "Example:\n\n"
            "Congratulations! You have been selected "
            "for a work-from-home job.\n\n"
            "Pay ₹500 registration fee to continue."
        )
    )
else:
    st.info(
        "Analyze a suspicious or potentially malicious URL."
    )
    user_input = st.text_input(
        "Enter the URL",
        placeholder=(
            "https://example.com/login"
        )
    )
    # ANALYZE BUTTON
st.write("")
analyze_button = st.button(
    "🔎 Analyze Scam",
    type="primary",
    use_container_width=True
)

# ANALYSIS
if analyze_button:
    if not user_input.strip():
        st.warning(
            "⚠️ Please enter something to analyze.")
    else:
        st.write("")
        with st.spinner(
                "Analyzing with ML + RAG + LLM..."):

            try:

                result = analyze_scam(
                    input_type,
                    user_input
                )

                # RESULT HEADER

                st.divider()
                st.markdown(
                    '<div class="section-title">'
                    '🚨 AI Scam Risk Assessment'
                    '</div>',
                    unsafe_allow_html=True
                )

                # DISPLAY RESULT

                st.markdown(
                    '<div class="risk-box">',
                    unsafe_allow_html=True
                )
                st.markdown(result)
                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                # PROCESS INFORMATION
                st.divider()
                st.subheader("🧠 How the analysis was generated")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Machine Learning", "✓ Used")
                    st.caption("Predicts the scam probability.")

                with col2:
                    st.metric("RAG", "✓ Used")
                    st.caption("Retrieves relevant scam knowledge.")
                with col3:
                    st.metric("LLM", "✓ Used")
                    st.caption("Generates the final explanation.")
            except Exception as e:
                st.error(
                    "❌ An error occurred while analyzing the input."
                )
                st.exception(e)

# FOOTER
st.divider()
st.caption(
    "⚠️ This system provides a scam risk assessment. "
    "It should not be treated as absolute proof that "
    "a message, job or URL is fraudulent."
)
# python -m streamlit run app.py
