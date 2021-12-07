from flask import Flask ,render_template,request
import joblib

app = Flask(__name__)
model = joblib.load('Breast Cancer Classification.save')
scaler = joblib.load('scaler.save')

@app.route('/' , methods = ['GET'])
def home():
    return render_template('index.html')

@app.route('/predict' , methods = ['GET'])
def predict():
    inp_data = [
        request.args.get('radius_mean'),
        request.args.get('texture_mean'),
        request.args.get('perimeter_mean'),
        request.args.get('area_mean'),
        request.args.get('perimeter_se'), 
        request.args.get('area_se'),
        request.args.get('radius_worst'),
        request.args.get('texture_worst'),
        request.args.get('perimeter_worst'),
        request.args.get('area_worst')
        ]

    inp_data = [int(n) for n in inp_data]


    prediction = model.predict(scaler.transform([inp_data]))[0]

    return render_template('index.html', prediction='Predicted Class: {}'.format(prediction)) # rendering the predicted result


if __name__ == '__main__':
    app.run(debug=True , host='127.0.0.1')
    
    