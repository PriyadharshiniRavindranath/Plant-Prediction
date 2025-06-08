from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your deep learning model
model = tf.keras.models.load_model(r'C:\Users\HP\Plant.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from Flutter app
    data = request.json
    
    # Preprocess data (if needed)
    # Example: Convert data to numpy array
    input_data = np.array(data[
    'aloevera'],  # Class index 0
    data['banana'],  # Class index 1
    data['bilimbi'],
    data['cantaloupe'], 
    data['cassava'],
    data['coconut'],
    data['corn'],
    data['cucumber'],
    data['curcuma'],
    data['eggplant'],
    data['galangal'],
    data['ginger'],
    data['guava'],
    data['kale'],
    data['longbeans'],
    data['mango'],
    data['melon'],
    data['orange'],
    data['paddy'],
    data['papaya'],
    data['peperchili'],
    data['pineapple'],
    data['pomelo'],
    data['shallot'],
    data['soybeans'],
    data['spinach'],
    data['sweetpotatoes'],
    data['tobacco'],
    data['waterapple'],
    data['watermelon'
       # Class index 2
    # Add more plant names as needed...

])
    
    # Perform inference
    prediction = model.predict(input_data)
    
    # Post-process prediction (if needed)
    # Example: Convert prediction to JSON
    output = {'prediction': prediction.tolist()}
    
    # Return prediction to Flutter app
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)