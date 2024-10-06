from flask import Flask, request, jsonify, render_template, redirect
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)

# Initialize PropelAuth with your credentials
PROPEL_AUTH_DOMAIN = "https://0669239.propelauthtest.com"  # Replace with your domain


# Load model and tokenizer
model_name = "./model"  # Path to your saved model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/login')
def login():
    # Redirect the user to the PropelAuth login page
    return redirect(f"{PROPEL_AUTH_DOMAIN}/login")
@app.route('/logout')
def logout():
    # Redirect the user to the PropelAuth login page
    return redirect(f"{PROPEL_AUTH_DOMAIN}/logout")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/flirtgen')
def flirtgen():
    return render_template('flirtgen.html')

# New route for LetterLove
@app.route('/letterlove')
def letterlove():
    return render_template('letterlove.html')

# Secure route that requires the user to be authenticated
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({"error": "Text input cannot be empty."}), 400
    if len(text) > 150:
        return jsonify({"error": "Please provide a text input with 150 characters or fewer."}), 400
    
    # Tokenize and prepare input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=128
    )

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    # Convert prediction to "Flirty" or "Not Flirty"
    result = "Flirty" if prediction == 1 else "Not Flirty"
    
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
