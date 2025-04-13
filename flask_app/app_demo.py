from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    #return "hello world"
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    text = request.form['text']
    return text

app.run(debug=True,port=8000)