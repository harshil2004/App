import streamlit as st
import os
import streamlit as st  
from streamlit_option_menu import option_menu  
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import streamlit.runtime.scriptrunner as scriptrunner
import importlib.util       

# Disease Detection Main Import
disease_path = os.path.join(os.path.dirname(__file__), 'pr_1', 'diseaseDetect-main.py')
spec = importlib.util.spec_from_file_location("diseaseDetect-main", disease_path)
diseaseDetectmain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diseaseDetectmain)

# Specch Emotion Recognition Main Import 
speeh_path = os.path.join(os.path.dirname(__file__), 'pr_2', 'speechEmo_main.py')
spec = importlib.util.spec_from_file_location("speechEmo_main", speeh_path)
speechEmomain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speechEmomain)

# Price Prediction Main Import
HomePrice_path = os.path.join(os.path.dirname(__file__), 'pr_3', 'housePrice_main.py')
spec = importlib.util.spec_from_file_location("housePrice_main", HomePrice_path)
homePricemain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(homePricemain)

# Emotion Detection Main Import
sremo_path = os.path.join(os.path.dirname(__file__), 'pr_4', 'sremo.py')
spec = importlib.util.spec_from_file_location("sremo", sremo_path)
sremo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sremo)

# Sidebar for Main Navigation
with st.sidebar:
    st.title("My ML Projects")
    selected_project = option_menu("Select Project", ["Home", "Health Assistant", "Speech Recognizer", "House Price Prediction", "Emotion Detection"])

# Function to update query parameters
def update_query_params(project):
    st.experimental_set_query_params(project=project)

# Home Page
if selected_project == "Home":
    # Main Page Header
    st.title("🌟 Welcome to the Multi-Project Dashboard!")
    st.markdown(
        """
        Explore a collection of interactive projects built to showcase the power of data science, machine learning, and web development.
        
        Choose a project below to get started!
        """
    )
    st.write("---")

    st.header("🔬 Disease Detection")
    st.image("pr_1/static/img/diabetes.png", width=600, caption="Disease Detection")
    st.markdown(
        """
        This project provides a disease directory with detailed assessments for:
        - Diabetes
        - Heart Disease
        - Parkinson's Disease
        - Breast Cancer
        
        💡 Learn about symptoms and understand the impact of these diseases.
        """
    )
    if st.button("Go to Disease Detection"):
        st.query_params = {"project": "disease_detection"} 

    st.header("🎙️ Speech Emotion Recognition")
    st.image("pr_2/voice-recognition-technology-conceptual-image-39055870.webp", width=600, caption="Speech Emotion Recognition")
    st.markdown(
        """
        This project uses deep learning to detect emotions from speech audio files. Supported emotions include:
        - Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
        
        💡 Applications: Mental health monitoring, customer service, and virtual assistants.
        """
    )
    if st.button("Go to SER Model"):
        st.query_params = {"project": "ser_model"}

    st.write("---")

    # Project 3: Price Prediction System
    st.header("🏡 Bangalore Home Price Prediction")
    st.image("pr_3/hsp_img.jpg", width=600, caption="Bangalore Real Estate Price Prediction")
    st.markdown(
        """
        Estimate the price of homes in Bangalore based on:
        - Area (Square Feet)
        - Number of Bedrooms (BHK)
        - Number of Bathrooms
        - Location
        
        💡 Real-time estimates powered by machine learning.
        """
    )
    if st.button("Go to Price Prediction"):
        st.query_params = {"project": "price_prediction"}

    st.write("---")

    # About Section
    st.header("👨‍💻 About the Creator")
    st.markdown(
        """
        **Harshil Ameta** is a Data Science Intern with expertise in building machine learning models and creating interactive web applications.

        - 💼 [Connect on LinkedIn](https://www.linkedin.com/in/harshil-ameta-7a0552200?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
        - 📫 [GitHub Repository](https://github.com/harshil2004)
        - 📧 harshilameta03@gmail.com
        """
    )

# Project 1
elif selected_project == "Health Assistant":
    diseaseDetectmain.main()

# Project 2
elif selected_project == "Speech Recognizer":
    speechEmomain.main()

# Project 3
elif selected_project == "House Price Prediction":
    homePricemain.main()

elif selected_project == 'Emotion Detection':
    sremo.main()
