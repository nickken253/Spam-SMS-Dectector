from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    message = request.form['message']
    payload = {"message": message}
    url = "http://localhost:3434/predict"
    
    try:
        response = requests.post(url, json=payload)
        prediction = response.json()["prediction"][0]
        
        if prediction == 1:
            result = "THIS IS SPAM SMS"
        else:
            result = "THIS IS NOT SPAM SMS"
            
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='localhost', port=3435)
