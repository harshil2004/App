import json
import streamlit as st
import pickle
import streamlit as st  
from streamlit_option_menu import option_menu  
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import streamlit.runtime.scriptrunner as scriptrunner

# Load location data and ML model
@st.cache_resource
def load_resources():
    with open(r"pr_3/columns.json", "r") as f:  
        data_columns = json.load(f)["data_columns"]
        locations = data_columns[3:]  

    with open(r"pr_3/banglore_home_prices_model.pickle", "rb") as f:
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

def main():
    st.title("House Price Prediction")

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

if __name__ == "__main__":
    main()