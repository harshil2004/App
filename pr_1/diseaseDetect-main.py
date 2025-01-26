import streamlit as st
import os
import pickle
import streamlit as st  
from streamlit_option_menu import option_menu  
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import streamlit.runtime.scriptrunner as scriptrunner

working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))
cancer_model = pickle.load(open(f'{working_dir}/saved_models/model.pkl', 'rb'))

def main():
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
                "image": "pr_1/static/img/diabetes.png"
            },
            "2. Heart Disease": {
                "description": "Heart disease describes conditions affecting your heart.",
                "symptoms": [
                    "Chest pain or discomfort", "Shortness of breath", "Pain in the legs or arms", 
                    "Pain in the neck, jaw, or back"
                ],
                "image": "pr_1/static/img/heart.png"
            },
            "3. Parkinson's Disease": {
                "description": "Parkinson's disease is a progressive disorder of the nervous system.",
                "symptoms": [
                    "Loss of balance", "Loss of speech", "Loss of movement", 
                    "Fatigue", "Difficulty concentrating"
                ],
                "image": "pr_1/static/img/brain-anatomy.webp"
            },
            "4. Breast Cancer": {
                "description": "Breast cancer is cancer that develops from breast tissue.",
                "symptoms": [
                    "A breast lump or thickening", "Change in the size, shape, or appearance of a breast", 
                    "Changes to the skin over the breast", "A newly inverted nipple", 
                    "Peeling, scaling, crusting or flaking of the skin around the nipple", 
                    "Redness or pitting of the skin over your breast"
                ],
                "image": "pr_1/static/img/breastcancer.jpg"
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
            "image": "pr_1/static/img/avatar.png"
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

if __name__ == "__main__":
    main()