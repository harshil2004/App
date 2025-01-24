import json
import streamlit as st
import os
import pickle
import time
import streamlit as st  
from streamlit_option_menu import option_menu  
from PIL import Image  
import numpy as np
import librosa
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import streamlit.runtime.scriptrunner as scriptrunner

working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(f'{working_dir}/pr_1/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/pr_1/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/pr_1/saved_models/parkinsons_model.sav', 'rb'))
cancer_model = pickle.load(open(f'{working_dir}/pr_1/saved_models/model.pkl', 'rb'))
speech_model_path = os.path.join("pr_2", "best_model.keras")
speech_model = tf.keras.models.load_model(speech_model_path)

# Emotion mapping dictionary
emotion_mapping = {
    0: "😠 Angry",
    1: "🤢 Disgust",
    2: "😨 Fear",
    3: "😊 Happy",
    4: "😐 Neutral",
    5: "😢 Sad",
    6: "😲 Surprise"
}

# Function to preprocess audio
def preprocess_audio(audio_data):
    """Preprocess the audio data for your model."""
    audio_data = librosa.resample(audio_data, orig_sr=44100, target_sr=16000)  # Resample to 16kHz if needed
    audio_data = np.expand_dims(audio_data, axis=0)
    return audio_data

# Load location data and ML model
@st.cache_resource
def load_resources():
    with open(r"pr_3\columns.json", "r") as f:  
        data_columns = json.load(f)["data_columns"]
        locations = data_columns[3:]  

    with open(r"pr_3\banglore_home_prices_model.pickle", "rb") as f:
        model = pickle.load(f)
    return data_columns, locations, model

# Load resources
data_columns, locations, model = load_resources()

# Define the price prediction function
def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    # Create an input array with zeros, matching the number of model features
    x = np.zeros(len(data_columns))
    x[0] = sqft  # Set square feet
    x[1] = bath  # Set number of bathrooms
    x[2] = bhk   # Set number of BHK
    if loc_index >= 0:
        x[loc_index] = 1  # Set location one-hot encoding to 1

    # Predict the price using the model
    return round(model.predict([x])[0], 2)

# Sidebar for Main Navigation
with st.sidebar:
    st.title("My ML Projects")
    selected_project = option_menu("Select Project", ["Home", "Health Assistant", "Speech Recognizer", "House Price Prediction"])

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
    st.title("Health Assistant")
    with st.sidebar:    
        selected = option_menu('Disease Prediction System',
                            ['Home',
                                'Diabetes Prediction',
                                'Heart Disease Prediction',
                                'Parkinsons Prediction',
                                'Cancer Type Prediction',
                                'About Us'],
                            menu_icon='hospital-fill',
                            icons=['house', 'activity', 'heart', 'person', 'person'],
                            default_index=0)
    # Home Page
    if selected == 'Home':
        st.title('RiskAssess - Disease Index')
        st.markdown("This directory provides assessments for various diseases along with their symptoms.")

        diseases = {
            "1. Diabetes": {
                "description": "Diabetes is a disease that occurs when blood glucose is too high.",
                "symptoms": [
                    "Heart disease", "Stroke", "Kidney disease", "Eye problems", 
                    "Dental disease", "Nerve damage", "Foot problems"
                ],
                "image": r"pr_1/static/img/diabetes.png"
            },
            "2. Heart Disease": {
                "description": "Heart disease describes conditions affecting your heart.",
                "symptoms": [
                    "Chest pain or discomfort", "Shortness of breath", "Pain in the legs or arms", 
                    "Pain in the neck, jaw, or back"
                ],
                "image": r"pr_1/static/img/heart.png"
            },
            "3. Parkinson's Disease": {
                "description": "Parkinson's disease is a progressive disorder of the nervous system.",
                "symptoms": [
                    "Loss of balance", "Loss of speech", "Loss of movement", 
                    "Fatigue", "Difficulty concentrating"
                ],
                "image": r"pr_1/static/img/brain-anatomy.webp"
            },
            "4. Breast Cancer": {
                "description": "Breast cancer is cancer that develops from breast tissue.",
                "symptoms": [
                    "A breast lump or thickening", "Change in the size, shape, or appearance of a breast", 
                    "Changes to the skin over the breast", "A newly inverted nipple", 
                    "Peeling, scaling, crusting or flaking of the skin around the nipple", 
                    "Redness or pitting of the skin over your breast"
                ],
                "image": r"pr_1/static/img/breastcancer.jpg"
            }
        }

        for disease, details in diseases.items():
            st.header(disease)
            st.image(details["image"], caption=disease.split('.')[1].strip(), width=300)  # Set width for scaling
            st.write(details["description"])
            st.subheader("Symptoms")
            st.write(', '.join(details["symptoms"]))
            st.markdown("---")

    # Diabetes Prediction Page
    if selected == 'Diabetes Prediction':
        st.title('Diabetes Prediction using ML')

        # Input fields
        col1, col2, col3 = st.columns(3)
        with col1: Pregnancies = st.text_input('Number of Pregnancies')
        with col2: Glucose = st.text_input('Glucose Level')
        with col3: BloodPressure = st.text_input('Blood Pressure value')
        with col1: SkinThickness = st.text_input('Skin Thickness value')
        with col2: Insulin = st.text_input('Insulin Level')
        with col3: BMI = st.text_input('BMI value')
        with col1: DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        with col2: Age = st.text_input('Age of the Person')

        # Prediction
        diab_diagnosis = ''
        if st.button('Diabetes Test Result'):
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            user_input = [float(x) for x in user_input]
            diab_prediction = diabetes_model.predict([user_input])
            diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
        st.success(diab_diagnosis)

    # Heart Disease Prediction Page
    if selected == 'Heart Disease Prediction':
        st.title('Heart Disease Prediction using ML')

        col1, col2, col3 = st.columns(3)
        with col1: age = st.text_input('Age')
        with col2: sex = st.text_input('Sex')
        with col3: cp = st.text_input('Chest Pain types')
        with col1: trestbps = st.text_input('Resting Blood Pressure')
        with col2: chol = st.text_input('Serum Cholestoral in mg/dl')
        with col3: fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        with col1: restecg = st.text_input('Resting Electrocardiographic results')
        with col2: thalach = st.text_input('Maximum Heart Rate achieved')
        with col3: exang = st.text_input('Exercise Induced Angina')
        with col1: oldpeak = st.text_input('ST depression induced by exercise')
        with col2: slope = st.text_input('Slope of the peak exercise ST segment')
        with col3: ca = st.text_input('Major vessels colored by flourosopy')
        with col1: thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

        # Prediction
        heart_diagnosis = ''
        if st.button('Heart Disease Test Result'):
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            user_input = [float(x) for x in user_input]
            heart_prediction = heart_disease_model.predict([user_input])
            heart_diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'
        st.success(heart_diagnosis)

    # Parkinson's Prediction Page
    if selected == "Parkinsons Prediction":
        st.title("Parkinson's Disease Prediction using ML")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: fo = st.text_input('MDVP:Fo(Hz)')
        with col2: fhi = st.text_input('MDVP:Fhi(Hz)')
        with col3: flo = st.text_input('MDVP:Flo(Hz)')
        with col4: Jitter_percent = st.text_input('MDVP:Jitter(%)')
        with col5: Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        with col1: RAP = st.text_input('MDVP:RAP')
        with col2: PPQ = st.text_input('MDVP:PPQ')
        with col3: DDP = st.text_input('Jitter:DDP')
        with col4: Shimmer = st.text_input('MDVP:Shimmer')
        with col5: Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        with col1: APQ3 = st.text_input('Shimmer:APQ3')
        with col2: APQ5 = st.text_input('Shimmer:APQ5')
        with col3: APQ = st.text_input('MDVP:APQ')
        with col4: DDA = st.text_input('Shimmer:DDA')
        with col5: NHR = st.text_input('NHR')
        with col1: HNR = st.text_input('HNR')
        with col2: RPDE = st.text_input('RPDE')
        with col3: DFA = st.text_input('DFA')
        with col4: spread1 = st.text_input('spread1')
        with col5: spread2 = st.text_input('spread2')
        with col1: D2 = st.text_input('D2')
        with col2: PPE = st.text_input('PPE')

        # Prediction
        parkinsons_diagnosis = ''
        if st.button("Parkinson's Test Result"):
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                        APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            user_input = [float(x) for x in user_input]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
        st.success(parkinsons_diagnosis)

    # Cancer Prediction Page
    if selected == "Cancer Type Prediction":
        st.title("Cancer Type Prediction using ML")

        input_fields = {
            "clump_thickness": st.slider("Clump Thickness", 1, 10, 5),
            "uniform_cell_size": st.slider("Uniformity of Cell Size", 1, 10, 5),
            "uniform_cell_shape": st.slider("Uniformity of Cell Shape", 1, 10, 5),
            "marginal_adhesion": st.slider("Marginal Adhesion", 1, 10, 5),
            "single_epithelial_size": st.slider("Single Epithelial Cell Size", 1, 10, 5),
            "bare_nuclei": st.slider("Bare Nuclei", 1, 10, 5),
            "bland_chromatin": st.slider("Bland Chromatin", 1, 10, 5),
            "normal_nucleoli": st.slider("Normal Nucleoli", 1, 10, 5),
            "mitoses": st.slider("Mitoses", 1, 10, 5)
        }

        prediction_mapping = {
            0: "Benign", 1: "Malignant", 2: "Invasive", 3: "Ductal Carcinoma", 4: "High Risk"
        }

        if st.button("Predict"):
            input_data = [input_fields[key] for key in sorted(input_fields.keys())]
            try:
                prediction = cancer_model.predict([input_data])
                predicted_label = prediction_mapping.get(prediction[0], "Unknown Type")
                res_val = "a high risk of Breast Cancer" if prediction[0] == 4 else "a low risk of Breast Cancer"
                st.write(f"Predicted Cancer Type: {predicted_label}")
                st.write(f"Risk Assessment: {res_val}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    # Data for team members
    member_info = {
        "Harshil Ameta": {
            "role": "Lead Machine Learning Engineer",
            "image": r"pr_1/static/img/avatar.png"
        }
    }

    # Team members in a list
    team_members = list(member_info.keys())

    # About Us Page
    if selected == 'About Us':
        st.title("About Us")
        st.write("This is a Multiple Disease Prediction System developed by a team of students from the Department of Computer Science and Engineering, JECRC University, Jaipur.")
        st.write("Our team consists of:")

        for name, info in member_info.items():
            col1, col2, col3 = st.columns([1, 2, 1])  # Create a layout with columns
            with col2:  # The image will be in the center column
                st.image(info["image"], width=150)
            with col1:  # Text appears when hovered over or clicked on
                with st.expander(name):
                    st.write(info["role"])

# Project 2
elif selected_project == "Speech Recognizer":
    st.title("Speech Recognizer")

    # Secondary Navigation
    with st.sidebar:
        page = option_menu('SER Model',
                        ['Home', 'Model Tester', 'About', 'About Us'],
                        icons=['house', 'mic', 'record-circle', 'info-circle'],
                        default_index=0)

    # Home Page
    if page == "Home":
        st.title("Speech Emotion Recognition (SER) Model")
        st.image(r"pr_2/voice-recognition-technology-conceptual-image-39055870.webp", use_container_width =True)
        st.write("""
        Welcome to the **Speech Emotion Recognition (SER) Model Tester**!

        This app uses a deep learning model to detect emotions from speech audio files. The model can identify the following emotions:
        
        - 😠 Angry
        - 🤢 Disgust
        - 😨 Fear
        - 😊 Happy
        - 😐 Neutral
        - 😢 Sad
        - 😲 Surprise
        
        **Use Cases:**
        - Improving customer service by detecting emotions in call center conversations
        - Mental health monitoring through emotional state analysis
        - Enhancing human-computer interaction by recognizing user emotions
        - Real-time emotion detection in virtual assistants
        - Content creation for emotionally adaptive games or applications

        Upload an audio file on the **Model Tester** page to see the prediction in action!
        """)

    # Model Tester Page
    elif page == "Model Tester":
        st.title("Speech Recognition Model Tester")
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

        if audio_file is not None:
            audio_data, sample_rate = librosa.load(audio_file, sr=16000)
            processed_data = preprocess_audio(audio_data)
            prediction = speech_model.predict(processed_data)
            predicted_label = np.argmax(prediction)
            emotion_name = emotion_mapping[predicted_label]
            st.write("🎙️ **Prediction Result:** ", emotion_name)

    # About Page
    elif page == "About":
        st.title("About This App")
        st.write("""
        This app was created to demonstrate how a Speech Emotion Recognition model works.

        **Technologies Used:**
        - Python
        - Streamlit
        - TensorFlow
        - Librosa

        **Developer:** Harshil Ameta
        """)

    # Data for team members
    member_info = {
        "Harshil Ameta": {
            "role": "Data Analyst and Scientist",
            "image": r"pr_3/avatar.png"
        }
    }

    # Team members in a list
    team_members = list(member_info.keys())

    if page == 'About Us':
        st.title("About Us")
        st.write("This is a Multiple Disease Prediction System developed by a team of students from the Department of Computer Science and Engineering, JECRC University, Jaipur.")
        st.write("Our team consists of:")

        for name, info in member_info.items():
            col1, col2, col3 = st.columns([1, 2, 1])  # Create a layout with columns
            
            with col2:  # The image will be in the center column
                st.image(info["image"], width=150)

            with col1:  # Text appears when hovered over or clicked on
                with st.expander(name):
                    st.write(info["role"])

# Project 3
elif selected_project == "House Price Prediction":
    st.title("House Price Prediction")

    # Secondary Navigation
    with st.sidebar:
        option = option_menu('Price Prediction System',
                            ['Home',
                            'Price Prediction System',
                            'About Us'],
                            menu_icon='hospital-fill',
                            icons=['house', 'activity', 'heart', 'person', 'person'],
                            default_index=0)

    if option == "Home":
        # Set title and subtitle
        st.title("🏡 Bangalore Home Price Prediction")
        st.markdown("Welcome to the **Bangalore Home Price Prediction System**, a tool to help you estimate property prices in Bangalore!")

        # Add an image (replace with your preferred image URL or local file)
        st.image(
            r"pr_3/hsp_img.jpg",
            caption="Bangalore Real Estate",
            use_container_width=True,
        )

        # Project Information
        st.header("📋 Project Overview")
        st.markdown(
            """
            This project is designed to estimate the **price of homes in Bangalore** based on various features such as:
            - Total area in square feet
            - Number of bedrooms (BHK)
            - Number of bathrooms
            - Location of the property

            **Highlights**:
            - Predict prices for over 240 locations in Bangalore.
            - Utilizes a trained machine learning model.
            - Provides real-time estimates for user inputs.
            """
        )

        # Key Features
        st.header("✨ Key Features")
        features = [
            "Interactive and user-friendly interface.",
            "Accurate predictions based on historical data.",
            "Dynamic location-based price predictions.",
            "Optimized for quick and seamless user experience.",
        ]
        for feature in features:
            st.markdown(f"- {feature}")

        # How It Works
        st.header("🔍 How It Works")
        st.markdown(
            """
            1. Enter the total area of the property (in square feet).
            2. Select the number of bedrooms (BHK) and bathrooms.
            3. Choose a location from the dropdown menu.
            4. Click **Estimate Price** to get the predicted price of the property.
            """
        )

        # About the Creator
        st.header("👨‍💻 About the Creator")
        st.markdown(
            """
            This project was developed by **Harshil Ameta**, a Data Science Intern with experience in building machine learning models and creating interactive web applications.
            """
        )
        st.markdown("💡 [Connect on LinkedIn](https://www.linkedin.com/in/harshil-ameta-7a0552200?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)")

        # Footer
        st.write("---")
        st.markdown("🔗 **GitHub Repository:** [View Code](https://github.com/harshil2004)")
        st.markdown("📫 **Contact:** harshilameta03@gamil.com")

    elif option == "Price Prediction System":
        # Add an image (replace with your preferred image URL or local file)
        st.image(
            r"pr_3/bg_img.jpg",
            caption="Bangalore Real Estate",
            use_container_width=True,
        )

        # Area Input
        st.subheader("Area (Square Feet)")
        area = st.number_input("Enter area in square feet", min_value=1, value=1000, step=1)

        # BHK Input
        st.subheader("BHK")
        bhk = st.radio(
            "Select BHK",
            options=[1, 2, 3, 4, 5],
            index=1,
            horizontal=True,
        )

        # Bathrooms Input
        st.subheader("Bath")
        bathrooms = st.radio(
            "Select Number of Bathrooms",
            options=[1, 2, 3, 4, 5],
            index=1,
            horizontal=True,
        )

        # Location Dropdown
        st.subheader("Location")
        location = st.selectbox("Choose a Location", ["Select"] + locations, index=0)

        # Button to estimate price
        if st.button("Estimate Price"):
            if location != "Select":
                # Predict the price using the helper function
                estimated_price = get_estimated_price(location, area, bhk, bathrooms)
                st.success(f"The estimated price is: **₹{estimated_price:,.2f} Lakhs**")
            else:
                st.error("Please select a valid location.")
        
        # End custom content wrapper
        st.markdown("""</div>""", unsafe_allow_html=True)
    
    # Data for team members
    member_info = {
        "Harshil Ameta": {
            "role": "Data Analyst and Scientist",
            "image": r"pr_3/avatar.png"
        }
    }

    # Team members in a list
    team_members = list(member_info.keys())

    if option == 'About Us':
        st.title("About Us")
        st.write("This is a Multiple Disease Prediction System developed by a team of students from the Department of Computer Science and Engineering, JECRC University, Jaipur.")
        st.write("Our team consists of:")

        for name, info in member_info.items():
            col1, col2, col3 = st.columns([1, 2, 1])  # Create a layout with columns
            
            with col2:  # The image will be in the center column
                st.image(info["image"], width=150)

            with col1:  # Text appears when hovered over or clicked on
                with st.expander(name):
                    st.write(info["role"])
