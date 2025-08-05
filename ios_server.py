#!/usr/bin/env python3
"""
TextTwin iOS Server
Runs TextTwin as an API server for iOS app access
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from texttwin import TextTwin

app = Flask(__name__)
CORS(app)

# Initialize TextTwin for your contact
twin = TextTwin("6094620213")

@app.route('/api/generate', methods=['POST'])
def generate_response():
    """Generate a response in your style"""
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    result = twin.generate_response(message)
    return jsonify(result)

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Ask about conversation history"""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    result = twin.ask_question(question)
    return jsonify(result)

@app.route('/api/insights', methods=['GET'])
def get_insights():
    """Get relationship insights"""
    result = twin.get_relationship_insights()
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get conversation statistics"""
    result = twin.get_conversation_stats()
    return jsonify(result)

@app.route('/api/search', methods=['POST'])
def search_conversations():
    """Search conversation history"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    result = twin.search_conversation(query)
    return jsonify(result)

@app.route('/')
def serve_app():
    """Serve the mobile web app"""
    with open('ios_app.html', 'r') as f:
        return f.read()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_available': twin._check_fine_tuned_model(),
        'rag_available': twin.rag is not None,
        'message_count': len(twin.messages)
    })

if __name__ == '__main__':
    print("🚀 Starting TextTwin iOS Server...")
    print("📱 Access from iPhone at: http://[YOUR-MAC-IP]:3000")
    print("💡 Find your IP with: ifconfig | grep 'inet ' | grep -v 127.0.0.1")
    
    # Run server accessible from network
    app.run(host='0.0.0.0', port=3000, debug=False)