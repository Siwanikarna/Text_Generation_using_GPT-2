from flask import Flask, render_template, request, jsonify, session
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask_session import Session
from flask_caching import Cache
from datetime import timedelta
import os

app = Flask(__name__)

# Set up Flask session configuration
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'  # Use filesystem session to avoid needing a database
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=10)  # Set session lifetime to 30 minutes
Session(app)

# Configure Flask-Caching
app.config['CACHE_TYPE'] = 'simple'  # Use simple caching for demonstration
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # Cache timeout in seconds
cache = Cache(app)

# Load the model and tokenizer
try:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

@app.route('/')
def home():
    session.permanent = True  # Make the session permanent
    return render_template('index.html')

@cache.memoize()
def generate_text(input_text):
    try:
        numeric_ids = tokenizer.encode(input_text, return_tensors='pt')
        result = model.generate(numeric_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        generated_text = tokenizer.decode(result[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        return f"Error generating text: {e}"

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        input_text = data.get('prompt')
        if not input_text:
            return jsonify({'error': 'No input text provided'}), 400
        
        # Use the cached generate_text function
        generated_text = generate_text(input_text)

        # Store conversation in session
        if 'conversation' not in session:
            session['conversation'] = []
        session['conversation'].append({'user': input_text, 'bot': generated_text})

        return jsonify({'text': generated_text})
    except Exception as e:
        return jsonify({'error': f"Error in generating response: {e}"}), 500

@app.route('/conversation', methods=['GET'])
def get_conversation():
    try:
        conversation = session.get('conversation', [])
        return jsonify(conversation)
    except Exception as e:
        return jsonify({'error': f"Error retrieving conversation: {e}"}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
