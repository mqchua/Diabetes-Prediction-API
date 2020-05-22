import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict_proba(final_features)

    output = round(prediction[0][1], 2)

    return render_template('home.html', prediction_text='Diabetics probability {}'.format(output))


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')


if __name__ == "__main__":
    app.run(debug=True)
