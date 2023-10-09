from flask import Flask, render_template, request
from utils import model_predict

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    context = request.form.get('context')
    question = request.form.get('question')
    prediction = model_predict(context, question)
    return render_template("index.html", prediction=prediction, 
                           context=context, question = question)


if __name__ == "__main__":
    app.run(debug=True)