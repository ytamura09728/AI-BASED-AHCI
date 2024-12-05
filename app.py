from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import logging

# Initialize Flask app
app = Flask(__name__)

# Initialize the sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Adaptive response based on sentiment
def generate_response(user_input):
    try:
        # Analyze sentiment
        sentiment_result = sentiment_analyzer(user_input)[0]
        sentiment = sentiment_result['label']
        score = sentiment_result['score']

        # Log sentiment analysis
        logging.info(f"Sentiment: {sentiment}, Score: {score}")

        # Adaptive chatbot responses
        if sentiment == "POSITIVE":
            return "I'm so glad to hear that! 😊 How can I assist you further?"
        elif sentiment == "NEGATIVE":
            return "I'm sorry you're feeling this way. Is there something specific I can help you with? 💬"
        else:
            return "Got it. Let me know how I can help! 🖥️"
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}")
        return "I'm having trouble understanding you. Can you please clarify?"

# Route for the chatbot interface
@app.route("/")
def home():
    return render_template("index.html")

# API endpoint for user interaction
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "")
        if not user_input:
            return jsonify({"response": "Please provide a message."}), 400

        # Generate adaptive response
        response = generate_response(user_input)
        return jsonify({"response": response}), 200
    except Exception as e:
        logging.error(f"Error in chat route: {e}")
        return jsonify({"error": "Something went wrong."}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
