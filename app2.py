import streamlit as st
import requests

def main():
    st.title("Parkinson's Disease Detection")

    # Create a form for user input
    st.subheader("Enter Patient Information")
    MDVP_Fo_Hz = st.number_input("MDVP_Fo _Hz")
    MDVP_Fhi_Hz = st.number_input("MDVP_Fhi_Hz)")
    MDVP_Flo_Hz = st.number_input("MDVP_Flo_Hz)")
    MDVP_Jitter = st.number_input("MDVP_Jitter")
    MDVP_Jitter_Abs = st.number_input("MDVP_Jitter_Abs")
    MDVP_RAP = st.number_input("MDVP_RAP")
    MDVP_PPQ = st.number_input("MDVP_PPQ")
    Jitter_DDP = st.number_input("Jitter_DDP")
    MDVP_Shimmer = st.number_input("MDVP_Shimmer")
    MDVP_Shimmer_dB = st.number_input("MDVP_Shimmer_dB")
    Shimmer_APQ3 = st.number_input("Shimmer_APQ3")
    Shimmer_APQ5 = st.number_input("Shimmer_APQ5")
    MDVP_APQ = st.number_input("MDVP_APQ")
    Shimmer_DDA = st.number_input("Shimmer_DDA")
    NHR = st.number_input("NHR")
    HNR = st.number_input("HNR")
    RPDE = st.number_input("RPDE")
    DFA = st.number_input("DFA")
    spread1 = st.number_input("spread1")
    spread2 = st.number_input("spread2")
    D2 = st.number_input("D2")
    PPE = st.number_input("PPE")

    # When the user submits the form, make a POST request to the Flask endpoint
    if st.button("Predict"):
        url = "http://localhost:5000/predict"  # Change this URL if your Flask app is running on a different port or domain
        data = {
            "MDVP_Fo_Hz": MDVP_Fo_Hz,
            "MDVP_Fhi_Hz": MDVP_Fhi_Hz,
            "MDVP_Flo_Hz": MDVP_Flo_Hz,
            "MDVP_Jitter": MDVP_Jitter,
            "MDVP_Jitter_Abs": MDVP_Jitter_Abs,
            "MDVP_RAP": MDVP_RAP,
            "MDVP_PPQ": MDVP_PPQ,
            "Jitter_DDP": Jitter_DDP,
            "MDVP_Shimmer": MDVP_Shimmer,
            "MDVP_Shimmer_dB": MDVP_Shimmer_dB,
            "Shimmer_APQ3": Shimmer_APQ3,
            "Shimmer_APQ5": Shimmer_APQ5,
            "MDVP_APQ": MDVP_APQ,
            "Shimmer_DDA": Shimmer_DDA,
            "NHR": NHR,
            "HNR": HNR,
            "RPDE": RPDE,
            "DFA": DFA,
            "spread1": spread1,
            "spread2": spread2,
            "D2": D2,
            "PPE": PPE
        }
        response = requests.post(url, data=data)

        # Display prediction result
        if response.status_code == 200:
            result = response.json()['prediction_text']
            st.success(result)
        else:
            st.error("Error occurred while making prediction")

if __name__ == "__main__":
    main()
