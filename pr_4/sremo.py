import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore
from streamlit_option_menu import option_menu  

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Check if the model file exists
model_path = 'pr_4/model.h5'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure it is in the correct directory.")
else:
    model.load_weights(model_path)  # Load the trained weights

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def main():
    st.title("Live Emotion Detection System")

    with st.sidebar:
        option = option_menu(
                        "Emotion Detection System",
                        ["Home", 
                         "Real-Time Detection", 
                         "About Us"],
                        default_index=0)

    if option == "Home":
        # Overview Page
        st.title("🎭 Emotion Detection Overview")
        st.image(
            r"pr_4/emotion_detection_banner.jpg",
            caption="Real-Time Emotion Detection",
            use_container_width=True,
        )
        st.header("📋 Project Overview")
        st.markdown(
            """
            This project detects **human emotions** in real-time using a webcam feed.
            
            **Supported Emotions:**
            - Angry 😡
            - Disgusted 🤢
            - Fearful 😨
            - Happy 😊
            - Neutral 😐
            - Sad 😢
            - Surprised 😲

            **Highlights:**
            - Uses a pre-trained Convolutional Neural Network (CNN).
            - Processes live webcam feeds for emotion prediction.
            - Highly optimized for real-time performance.
            """
        )
        st.header("✨ Key Features")
        features = [
            "Real-time emotion detection using your webcam.",
            "Accurate predictions based on facial expressions.",
            "Applicable for virtual assistants, customer service, and mental health monitoring.",
        ]
        for feature in features:
            st.markdown(f"- {feature}")

    elif option == "Real-Time Detection":
        # Real-Time Emotion Detection Page
        st.title("🎥 Real-Time Emotion Detection")
        run_app = st.checkbox("Start Webcam Detection")

        if run_app:
            st.warning("Click on 'Stop' in the top right corner of the webcam feed to terminate.")
            cap = cv2.VideoCapture(0)  # Open the webcam
            st_frame = st.empty()  # Streamlit placeholder

            no_face_detected = False  # Track if no face detected state is active

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Unable to access the webcam.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                if len(faces) == 0:
                    if not no_face_detected:
                        st.warning("No faces detected. The app will resume automatically when a face is detected.")
                        no_face_detected = True
                else:
                    no_face_detected = False  # Reset when a face is detected

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)  # Draw rectangle around face
                    roi_gray = gray[y:y + h, x:x + w]  # Region of interest
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

                    try:
                        prediction = model.predict(cropped_img, verbose=0)
                        maxindex = int(np.argmax(prediction))  # Emotion with highest probability
                        emotion = emotion_dict[maxindex]
                        cv2.putText(frame, emotion, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

                resized_frame = cv2.resize(frame, (640, 480))
                st_frame.image(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), channels="RGB")

                if not run_app:
                    break

            cap.release()
            cv2.destroyAllWindows()

    elif option == "About Us":
        # About Us Page
        st.title("About Us")
        st.write("This is an Emotion Detection System developed by a team of students.")
        st.write("Our team consists of:")

        team_info = {
            "Harshil Ameta": {"role": "Lead Developer", "image": r"pr_4/avatar.png"},
        }

        for name, info in team_info.items():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(info["image"], width=150)
            with col1:
                with st.expander(name):
                    st.write(info["role"])


if __name__ == "__main__":
    main()
