import streamlit as st
from PIL import Image
from datetime import datetime
import pandas as pd

from utils import (
    load_model,
    predict_image,
    get_detected_conditions,
    generate_gradcam,
    create_gradcam_overlay
)

from config import DISEASES


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Multi-Disease Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = None

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

if "last_file" not in st.session_state:
    st.session_state.last_file = None

if "gradcam_overlay" not in st.session_state:
    st.session_state.gradcam_overlay = None

if "gradcam_disease" not in st.session_state:
    st.session_state.gradcam_disease = None

if "gradcam_index" not in st.session_state:
    st.session_state.gradcam_index = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🩺 AI Multi-Disease Diagnosis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning based Chest X-ray Analysis using DenseNet121'
    '</div>',
    unsafe_allow_html=True
)

st.warning(
    "⚠️ Educational and research prototype only. "
    "This system is NOT a medical diagnosis tool. "
    "Predictions should not be used for clinical decisions."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():
    return load_model()


try:

    model = get_model()

except Exception as e:

    st.error(
        f"Unable to load the trained model.\n\n{e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧠 Model Information")

    st.write("**Model:** DenseNet121")
    st.write("**Architecture:** Transfer Learning")
    st.write("**Input Size:** 224 × 224")
    st.write("**Output Classes:** 14")
    st.write("**Task:** Multi-Label Classification")
    st.write("**Activation:** Sigmoid")
    st.write("**Loss:** Binary Cross-Entropy")

    st.divider()

    st.subheader("📋 Supported Conditions")

    for disease in DISEASES:

        st.write(f"• {disease}")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("📤 Upload Chest X-ray")

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file:

    # --------------------------------------------------------
    # CREATE UNIQUE FILE IDENTIFIER
    # --------------------------------------------------------

    current_file = (
        f"{uploaded_file.name}_"
        f"{uploaded_file.size}"
    )


    # --------------------------------------------------------
    # CHECK WHETHER A NEW IMAGE WAS UPLOADED
    # --------------------------------------------------------

    if st.session_state.last_file != current_file:

        st.session_state.results = None

        st.session_state.analyzed = False

        st.session_state.gradcam_overlay = None

        st.session_state.gradcam_disease = None

        st.session_state.gradcam_index = None

        st.session_state.last_file = current_file


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as e:

        st.error(
            f"Unable to open the uploaded image: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    st.subheader("🖼️ Uploaded Image")

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        st.image(
            image,
            caption="Uploaded Chest X-ray",
            width="stretch"
        )

    with col2:

        st.info(
            "Image uploaded successfully.\n\n"
            "Click **Analyze X-ray** to generate predictions."
        )

        analyze = st.button(
            "🔍 Analyze X-ray",
            width="stretch"
        )


    # ========================================================
    # ANALYSIS BUTTON
    # ========================================================

    if analyze:

        with st.spinner(
            "Analyzing X-ray using DenseNet121..."
        ):

            try:

                prediction_results = predict_image(
                    model,
                    image
                )

                st.session_state.results = prediction_results

                st.session_state.analyzed = True

                # Reset Grad-CAM for new analysis

                st.session_state.gradcam_overlay = None

                st.session_state.gradcam_disease = None

                st.session_state.gradcam_index = None

            except Exception as e:

                st.session_state.results = None

                st.session_state.analyzed = False

                st.error(
                    f"Prediction failed: {e}"
                )


        if st.session_state.analyzed:

            st.success(
                "Analysis completed."
            )


    # ========================================================
    # SHOW RESULTS
    # ========================================================

    if (
        st.session_state.analyzed
        and st.session_state.results is not None
    ):

        # ----------------------------------------------------
        # RESTORE RESULTS FROM SESSION
        # ----------------------------------------------------

        results = st.session_state.results


        # ====================================================
        # DETECTED CONDITIONS
        # ====================================================

        st.header(
            "🩺 Detected Conditions"
        )

        try:

            detected = get_detected_conditions(
                results,
                threshold=0.50
            )

        except Exception:

            detected = []

            for result in results:

                if result["Probability"] >= 0.50:

                    detected.append(
                        result
                    )


        if detected:

            for result in detected:

                st.error(
                    f"**{result['Disease']}** — "
                    f"{result['Percentage']:.2f}%"
                )

        else:

            st.success(
                "No condition crossed the 50% prediction threshold."
            )


        # ====================================================
        # TOP PREDICTION
        # ====================================================

        top_prediction = results[0]

        st.header(
            "🎯 Top Prediction"
        )

        st.metric(
            label=top_prediction["Disease"],
            value=f"{top_prediction['Percentage']:.2f}%"
        )


        # ====================================================
        # PROBABILITY CHART
        # ====================================================

        st.header(
            "📊 Disease Probability"
        )

        chart_data = pd.DataFrame(
            {
                "Disease": [
                    result["Disease"]
                    for result in results
                ],

                "Probability (%)": [
                    result["Percentage"]
                    for result in results
                ]
            }
        )

        chart_data = chart_data.set_index(
            "Disease"
        )

        st.bar_chart(
            chart_data,
            width="stretch"
        )


        # ====================================================
        # DETAILED RESULTS
        # ====================================================

        st.header(
            "📋 Detailed Prediction Results"
        )

        for result in results:

            disease = result["Disease"]

            probability = float(
                result["Probability"]
            )

            percentage = float(
                result["Percentage"]
            )

            st.write(
                f"**{disease}** — "
                f"{percentage:.2f}%"
            )

            st.progress(
                min(
                    max(
                        probability,
                        0.0
                    ),
                    1.0
                )
            )


        # ====================================================
        # GRAD-CAM
        # ====================================================

        st.header(
            "🔥 Explainable AI — Grad-CAM"
        )

        st.write(
            "Grad-CAM highlights image regions that "
            "contributed to the selected prediction."
        )


        # ----------------------------------------------------
        # DISEASE SELECTION
        # ----------------------------------------------------

        disease_names = [
            result["Disease"]
            for result in results
        ]

        selected_disease = st.selectbox(
            "Select a disease to visualize",
            disease_names
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # FIND ORIGINAL MODEL CLASS INDEX
        # ----------------------------------------------------

        if selected_disease in DISEASES:

            selected_index = DISEASES.index(
                selected_disease
            )

        else:

            selected_index = 0


        # ----------------------------------------------------
        # GET SELECTED PROBABILITY
        # ----------------------------------------------------

        selected_probability = 0.0

        for result in results:

            if result["Disease"] == selected_disease:

                selected_probability = float(
                    result["Probability"]
                )

                break


        st.write(
            f"Selected condition: "
            f"**{selected_disease}**"
        )

        st.write(
            f"Prediction probability: "
            f"**{selected_probability * 100:.2f}%**"
        )


        # ====================================================
        # GENERATE GRAD-CAM BUTTON
        # ====================================================

        if st.button(
            "🔥 Generate Grad-CAM",
            width="stretch"
        ):

            with st.spinner(
                "Generating Grad-CAM visualization..."
            ):

                try:

                    heatmap = generate_gradcam(
                        model,
                        image,
                        selected_index
                    )


                    overlay = create_gradcam_overlay(
                        image,
                        heatmap
                    )


                    # ------------------------------------------------
                    # SAVE RESULT IN SESSION STATE
                    # ------------------------------------------------

                    st.session_state.gradcam_overlay = overlay

                    st.session_state.gradcam_disease = (
                        selected_disease
                    )

                    st.session_state.gradcam_index = (
                        selected_index
                    )


                except Exception as e:

                    st.error(
                        "Grad-CAM generation failed."
                    )

                    st.exception(e)


        # ====================================================
        # DISPLAY STORED GRAD-CAM
        # ====================================================

        if (
            st.session_state.gradcam_overlay is not None
            and st.session_state.gradcam_disease is not None
        ):

            st.subheader(
                "🔥 Grad-CAM Visualization"
            )

            cam_col1, cam_col2 = st.columns(
                2
            )


            with cam_col1:

                st.image(
                    image,
                    caption="Original X-ray",
                    width="stretch"
                )


            with cam_col2:

                st.image(
                    st.session_state.gradcam_overlay,
                    caption=(
                        "Grad-CAM: "
                        f"{st.session_state.gradcam_disease}"
                    ),
                    width="stretch"
                )


            st.success(
                f"Grad-CAM generated for "
                f"{st.session_state.gradcam_disease}."
            )


        # ====================================================
        # REPORT GENERATION
        # ====================================================

        st.header(
            "📄 Download Prediction Report"
        )

        report_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        report = []


        report.append(
            "AI MULTI-DISEASE DIAGNOSIS REPORT"
        )

        report.append(
            "=" * 50
        )

        report.append(
            f"Generated: {report_time}"
        )

        report.append(
            f"Image: {uploaded_file.name}"
        )

        report.append(
            "Model: DenseNet121"
        )

        report.append(
            "Task: Multi-Label Chest X-ray Classification"
        )

        report.append("")


        report.append(
            "DISCLAIMER:"
        )

        report.append(
            "This report is generated by an educational "
            "AI prototype and must not be used for "
            "medical diagnosis or clinical decisions."
        )

        report.append("")


        report.append(
            "PREDICTION RESULTS"
        )

        report.append(
            "-" * 50
        )


        for result in results:

            report.append(
                f"{result['Disease']}: "
                f"{result['Percentage']:.2f}%"
            )


        report.append("")


        report.append(
            "DETECTED CONDITIONS"
        )

        report.append(
            "-" * 50
        )


        if detected:

            for result in detected:

                report.append(
                    f"{result['Disease']}: "
                    f"{result['Percentage']:.2f}%"
                )

        else:

            report.append(
                "No condition crossed the 50% threshold."
            )


        report.append("")


        report.append(
            "TOP PREDICTION"
        )

        report.append(
            f"{top_prediction['Disease']} - "
            f"{top_prediction['Percentage']:.2f}%"
        )


        report_text = "\n".join(
            report
        )


        st.download_button(
            label="📄 Download Report",
            data=report_text,
            file_name="xray_prediction_report.txt",
            mime="text/plain",
            width="stretch"
        )


# ============================================================
# NO IMAGE
# ============================================================

else:

    st.info(
        "👆 Upload a chest X-ray image to begin."
    )


    st.header(
        "How It Works"
    )


    col1, col2, col3, col4 = st.columns(
        4
    )


    with col1:

        st.subheader("1️⃣ Upload")

        st.write(
            "Upload a PNG, JPG or JPEG chest X-ray."
        )


    with col2:

        st.subheader("2️⃣ Analyze")

        st.write(
            "DenseNet121 processes the image."
        )


    with col3:

        st.subheader("3️⃣ Predict")

        st.write(
            "The model estimates probabilities "
            "for 14 conditions."
        )


    with col4:

        st.subheader("4️⃣ Explain")

        st.write(
            "Grad-CAM helps visualize important "
            "image regions."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Multi-Disease Diagnosis • DenseNet121 • "
    "TensorFlow/Keras • Streamlit"
)