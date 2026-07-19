import os
import pandas as pd
import streamlit as st
from PIL import Image

from utils.predict import predict_severity
from utils.repair_priority import get_repair_priority
from utils.maintenance_cost import get_maintenance_cost

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Pothole Severity Classification",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------
# Custom Professional CSS
# ---------------------------------------------------

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
padding-left:3rem;
padding-right:3rem;
}

html,body,[class*="css"]{

font-family:'Segoe UI';

background:#f5f7fb;

}


/* Sidebar */

section[data-testid="stSidebar"]{

background:#182848;

}

section[data-testid="stSidebar"] *{

color:white;

}


/* Header */

.main-title{

background:linear-gradient(90deg,#4facfe,#00f2fe);

padding:25px;

border-radius:15px;

text-align:center;

color:white;

font-size:38px;

font-weight:bold;

box-shadow:0px 5px 15px rgba(0,0,0,.2);

}


/* Cards */

.card{

background:white;

padding:20px;

border-radius:15px;

box-shadow:0 8px 20px rgba(0,0,0,.08);

margin-bottom:20px;

}


/* Prediction */

.green{

background:#d4edda;

padding:18px;

border-radius:12px;

font-size:25px;

font-weight:bold;

text-align:center;

color:#155724;

}

.orange{

background:#fff3cd;

padding:18px;

border-radius:12px;

font-size:25px;

font-weight:bold;

text-align:center;

color:#856404;

}

.red{

background:#f8d7da;

padding:18px;

border-radius:12px;

font-size:25px;

font-weight:bold;

text-align:center;

color:#721c24;

}


/* Footer */

.footer{

text-align:center;

padding:25px;

font-size:15px;

color:gray;

}

</style>

""",unsafe_allow_html=True)



# ---------------------------------------------------
# Utility Functions
# ---------------------------------------------------

MODEL_RESULT_FILE="outputs/model_comparison.csv"


def load_model_results():

    if os.path.exists(MODEL_RESULT_FILE):

        return pd.read_csv(MODEL_RESULT_FILE)

    return None



def severity_card(severity):

    severity=severity.lower()

    if "major" in severity:

        st.markdown(

        f'<div class="red">{severity.upper()}</div>',

        unsafe_allow_html=True

        )

    elif "medium" in severity:

        st.markdown(

        f'<div class="orange">{severity.upper()}</div>',

        unsafe_allow_html=True

        )

    else:

        st.markdown(

        f'<div class="green">{severity.upper()}</div>',

        unsafe_allow_html=True

        )



def page_title():

    st.markdown(

    """

    <div class='main-title'>

    🛣️ AI Based Pothole Severity Classification System

    </div>

    """,

    unsafe_allow_html=True

    )



page_title()
# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------

st.sidebar.markdown("## 🛣️ Dashboard")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Project Overview",

        "📤 Predict Severity",

        "📊 Model Comparison",

        "ℹ️ About Project"

    ]

)

st.sidebar.markdown("---")

st.sidebar.info(

"""
**Project**

AI Based Pothole Severity Classification

Algorithms Used

• Random Forest

• XGBoost

• LightGBM

• Gradient Boosting

• Support Vector Machine

"""

)

st.sidebar.markdown("---")

st.sidebar.success("Project Status : Completed")



# ---------------------------------------------------
# Project Overview
# ---------------------------------------------------

if page == "🏠 Project Overview":

    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown(

        """

        <div class="card">

        <h2>Project Overview</h2>

        <p style="font-size:18px; text-align:justify;">

        This project classifies potholes into three severity levels using
        Machine Learning.

        The uploaded road image is processed, transformed into image
        features and then classified using the trained Support Vector
        Machine model.

        The system also estimates the repair priority and maintenance
        cost, helping authorities prioritize road maintenance.

        </p>

        </div>

        """,

        unsafe_allow_html=True

        )

    with col2:

        st.markdown(

        """

        <div class="card">

        <h3>Severity Classes</h3>

        <br>

        🟢 Minor Pothole

        <br><br>

        🟡 Medium Pothole

        <br><br>

        🔴 Major Pothole

        </div>

        """,

        unsafe_allow_html=True

        )



    st.markdown("<br>", unsafe_allow_html=True)



    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(

            "Dataset Images",

            "717"

        )



    with c2:

        st.metric(

            "Annotated Samples",

            "1887"

        )



    with c3:

        st.metric(

            "Best Model",

            "SVM"

        )



    st.markdown("<br>", unsafe_allow_html=True)



    st.markdown(

    """

    <div class="card">

    <h3>Machine Learning Models Used</h3>

    <ul style="font-size:18px;">

    <li>Random Forest</li>

    <li>XGBoost</li>

    <li>LightGBM</li>

    <li>Gradient Boosting</li>

    <li>Support Vector Machine</li>

    </ul>

    </div>

    """,

    unsafe_allow_html=True

    )
    # ---------------------------------------------------
# Prediction Page
# ---------------------------------------------------

elif page == "📤 Predict Severity":

    st.markdown(

    """

    <div class="card">

    <h2>Upload Road Image</h2>

    <p>

    Upload a pothole image to predict its severity level,
    repair priority and estimated maintenance cost.

    </p>

    </div>

    """,

    unsafe_allow_html=True

    )

    uploaded_file = st.file_uploader(

        "Choose an Image",

        type=["jpg", "jpeg", "png"]

    )

    if uploaded_file is not None:

        temp_path = "temp_prediction_image.jpg"

        with open(temp_path, "wb") as file:

            file.write(uploaded_file.getbuffer())

        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1.1, 1])

        with col1:

            st.markdown(

            """

            <div class="card">

            <h3>Uploaded Image</h3>

            </div>

            """,

            unsafe_allow_html=True

            )

            st.image(

                image,

                use_container_width=True

            )

        with col2:

            st.markdown(

            """

            <div class="card">

            <h3>Prediction Result</h3>

            </div>

            """,

            unsafe_allow_html=True

            )

            if st.button(

                "Predict Severity",

                use_container_width=True

            ):

                with st.spinner("Predicting..."):

                    severity = predict_severity(temp_path)

                    priority = get_repair_priority(severity)

                    cost = get_maintenance_cost(severity)
                st.markdown("### Predicted Severity")

                severity_card(severity)

                st.markdown("<br>", unsafe_allow_html=True)

                c1, c2 = st.columns(2)

                with c1:

                    st.success(

                        "Repair Priority\n\n{}".format(priority)

                    )

                with c2:

                    st.info(

                        "Estimated Cost\n\n₹ {}".format(cost)

                    )

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(

                """

                <div class="card">

                <h3>Prediction Summary</h3>

                </div>

                """,

                unsafe_allow_html=True

                )

                summary = pd.DataFrame({

                    "Attribute":[

                        "Predicted Severity",

                        "Repair Priority",

                        "Estimated Cost"

                    ],

                    "Result":[

                        severity,

                        priority,

                        "₹ {}".format(cost)

                    ]

                })

                st.table(summary)

        if os.path.exists(temp_path):

            os.remove(temp_path)
            # ---------------------------------------------------
# Model Comparison
# ---------------------------------------------------

elif page == "📊 Model Comparison":

    st.markdown(
    """
    <div class="card">
    <h2>Model Performance Comparison</h2>
    <p>
    Comparison of all machine learning models used for pothole severity
    classification.
    </p>
    </div>
    """,
    unsafe_allow_html=True
    )

    results = load_model_results()

    if results is not None:

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        best_model = results.sort_values(
            by="Accuracy",
            ascending=False
        ).iloc[0]

        st.success(
            "🏆 Best Performing Model : {} (Accuracy : {})".format(
                best_model["Model"],
                best_model["Accuracy"]
            )
        )

    else:

        st.warning("Model comparison file not found.")





# ---------------------------------------------------
# About Project
# ---------------------------------------------------

elif page == "ℹ️ About Project":

    st.markdown(
    """
    <div class="card">

    <h2>About This Project</h2>

    <p style="font-size:18px;text-align:justify;">

    This project focuses on automatic pothole severity
    classification using Machine Learning.

    Images are converted into numerical image features,
    reduced using Principal Component Analysis (PCA),
    and classified using multiple machine learning models.

    The final deployed model predicts pothole severity,
    repair priority and estimated maintenance cost.

    </p>

    </div>
    """,
    unsafe_allow_html=True
    )

    st.markdown("### Technologies Used")

    tech = pd.DataFrame({

        "Technology":[

            "Python",

            "OpenCV",

            "Pandas",

            "NumPy",

            "Scikit-Learn",

            "Streamlit"

        ]

    })

    st.table(tech)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
    """
    <div class="card">

    <h3>Machine Learning Pipeline</h3>

    Dataset

    ➜ Image Preprocessing

    ➜ Feature Extraction

    ➜ StandardScaler

    ➜ PCA

    ➜ SVM Classification

    ➜ Repair Priority

    ➜ Maintenance Cost

    </div>
    """,
    unsafe_allow_html=True
    )





# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
"""
<div class="footer">

AI Based Pothole Severity Classification System

<br>

Developed using Python, Streamlit and Machine Learning

</div>
""",
unsafe_allow_html=True
)