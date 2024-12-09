#!/usr/bin/env python
# coding: utf-8

# pip install transformers torch

# import pandas as pd
# fin_data = pd.read_csv("//content/data.csv")
# newfin = fin_data.head(100)

# In[1]:


# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="ahmedrachid/FinancialBERT-Sentiment-Analysis")


# # Assuming your text column is named 'tweet_text'
# newfin['classification'] = newfin['Sentence'].apply(lambda x: pipe(x)[0]['label'])
# 
# # Display the classified data
# print(newfin[['Sentence', 'classification']])

# newfin

# In[4]:


from transformers import pipeline

pipe = pipeline("text-classification", model="ahmedrachid/FinancialBERT-Sentiment-Analysis")

pipe.save_pretrained("saved_model")


# pip install  flask-ngrok

# pip install flask pyngrok joblib

# In[ ]:


from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline
from pyngrok import ngrok

app = Flask(__name__)

# Load the pretrained model pipeline
pipe = pipeline("text-classification", model="ahmedrachid/FinancialBERT-Sentiment-Analysis")

# HTML template embedded in Python
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Sentiment Analysis</title>
</head>
<body>
    <h1>Financial News Sentiment Analysis</h1>
    <form method="POST" action="/predict">
        <label for="news_text">Enter Financial News:</label><br><br>
        <textarea id="news_text" name="news_text" rows="4" cols="50" placeholder="Type your financial news here..."></textarea><br><br>
        <button type="submit">Predict</button>
    </form>
    {% if prediction %}
        <h2>Prediction: {{ prediction }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        user_input = request.form.get("news_text")

        # Get the prediction from the model
        result = pipe(user_input)[0]

        # Render the result with the template
        return render_template_string(html_template, prediction=result['label'])
    except Exception as e:
        return render_template_string(html_template, prediction=f"Error: {str(e)}")

if __name__ == '__main__':

    #ngrok.set_auth_token("2k1WY06GxYVMqdkmmKhLDU7nU05_7mxAbPxA3puA3ZncyC4Yy")

    # Start ngrok tunnel
    #public_url = ngrok.connect(5000)
    #print(' * Tunnel URL:', public_url)

    if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000, debug=True)


# In[ ]:




