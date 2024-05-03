from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from tensorflow import keras
import pickle

app1 = Flask(__name__)

# Load the trained model
try:
    model = keras.models.load_model("parkinsondiseasedetectionusingneuralnetworks.h5")
except Exception as e:
    print("Error:", e)
    # Handle the error as needed (e.g., re-train the model or display an error message)

@app1.route("/")
def home():
    return render_template("index1.html")

@app1.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the form
    float_features = [float(x) for x in request.form.values()]
    # features = [np.array(float_features)]
    MDVP_Fo_Hz= float(request.form["MDVP_Fo_Hz"])
    MDVP_Fhi_Hz= float(request.form["MDVP_Fhi_Hz"])
    MDVP_Flo_Hz=float(request.form["MDVP_Flo_Hz"])
    MDVP_Jitter=float(request.form["MDVP_Jitter"])
    MDVP_Jitter_Abs=float(request.form["MDVP_Jitter_Abs"])
    MDVP_RAP= float(request.form["MDVP_RAP"])
    MDVP_PPQ=float(request.form["MDVP_PPQ"])
    Jitter_DDP=float(request.form["Jitter_DDP"])
    MDVP_Shimmer=float(request.form["MDVP_Shimmer"])
    MDVP_Shimmer_dB=float(request.form["MDVP_Shimmer_dB"])
    Shimmer_APQ3=float(request.form["Shimmer_APQ3"])
    Shimmer_APQ5=float(request.form["Shimmer_APQ5"])
    MDVP_APQ=float(request.form["MDVP_APQ"])
    Shimmer_DDA=float(request.form["Shimmer_DDA"])
    NHR=float(request.form["NHR"])
    HNR=float(request.form["HNR"])
    RPDE=float(request.form["RPDE"])
    DFA=float(request.form["DFA"])
    spread1=float(request.form["spread1"])
    spread2=float(request.form["spread2"])
    D2=float(request.form["D2"])
    PPE=float(request.form["PPE"])
                     
    input_data=np.array([[MDVP_Fo_Hz,MDVP_Fhi_Hz,MDVP_Flo_Hz,MDVP_Jitter,MDVP_Jitter_Abs,MDVP_RAP,MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
    features = np.array(float_features)
    input_data=np.array([features])
     # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    # standardize the input data
    std_data = scaler.transform(input_data_reshaped)
    print(std_data)

    prediction =model.predict(std_data)
    print(prediction)

    # Make prediction using the loaded model
    prediction = model.predict(features)

    # Convert prediction to human-readable text   
    if prediction[0] == 0:
        result = 'Person does not suffer from Parkinson\'s Disease'
    else:
        result = 'Person suffers from Parkinson\'s Disease'
 

    pickle.dump(model,open("parkinsondiseasedetectionusingneuralnetworks.pkl","wb"))
    # Return the prediction result as JSON
    return jsonify({'prediction_text': result})

if __name__ == "__main__":
    app1.run(debug=True)
