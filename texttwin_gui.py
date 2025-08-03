#!/usr/bin/env python3
"""
TextTwin GUI - Simple Web Interface
==================================

A simple web-based GUI for TextTwin that allows:
- Enter a phone number/contact
- View recent message history  
- Generate responses based on natural language input
"""

import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from contact_specific_generator import ContactSpecificGenerator

# Initialize Flask app\app = Flask(__name__)
app = Flask(__name__)
app.secret_key = os.environ.get('TEXTTWIN_SECRET_KEY', 'texttwin_secret_key_2024')

# Global generator instance
generator = None


def get_generator():
    """Initialize the contact generator once and reuse it."""
    global generator
    if generator is None:
        gen = ContactSpecificGenerator()
        try:
            gen.analyze_imessages()            # reads the latest DB copy
        except Exception:
            gen.analyze_sample_messages()
        gen.analyze_contact_conversations()    # re-queries with newest messages
        generator = gen
    return generator


@app.before_first_request
def load_generator_once():
    """Load the iMessage database before the first request."""
    get_generator()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/contacts')
def get_contacts():
    try:
        gen = get_generator()
        contacts = []
        for cid, msgs in gen.contact_conversations.items():
            your = sum(1 for m in msgs if m['is_from_me'])
            contacts.append({
                'id': cid,
                'display_name': cid,
                'total_messages': len(msgs),
                'your_messages': your,
                'last_message': (msgs[-1]['text'][:50] + '...') if msgs else ''
            })
        return jsonify(success=True, contacts=contacts)
    except Exception as e:
        return jsonify(success=False, error=str(e))

@app.route('/api/conversation/<path:contact_id>')
def get_conversation(contact_id):
    try:
        gen = get_generator()
        if contact_id not in gen.contact_conversations:
            return jsonify(success=False, error='Contact not found')
        msgs = gen.contact_conversations[contact_id]
        formatted = []
        for m in msgs[:100][::-1]:
            # original m['timestamp'] is nanoseconds since Jan 1 2001 UTC
            iso_ts = ""
            try:
                iso_ts = datetime.fromtimestamp(
                    m['timestamp'] / 1e9 + 978307200
                ).strftime("%Y-%m-%d %H:%M")
            except:
                pass

            formatted.append({
                "text":       m["text"],
                "is_from_me": m["is_from_me"],
                "timestamp":  iso_ts,
                "sender":     "You" if m["is_from_me"] else "Them",
            })

        if contact_id not in gen.contact_styles:
            gen.analyze_contact_specific_style(contact_id)
        style = gen.contact_styles.get(contact_id, {})
        return jsonify(success=True,
                       contact_id=contact_id,
                       messages=formatted,
                       style_info={
                           'formality_level': style.get('formality_level', 'unknown'),
                           'avg_message_length': style.get('avg_message_length', 0),
                           'message_count': style.get('message_count', 0)
                       })
    except Exception as e:
        return jsonify(success=False, error=str(e))

@app.route('/api/generate', methods=['POST'])
def generate_message():
    data = request.json or {}
    cid = data.get('contact_id')
    raw = data.get('user_input', '')
    if not cid or not raw:
        return jsonify(success=False, error='Missing contact_id or user_input')
    gen = get_generator()
    # Determine type
    txt = raw.lower()
    if any(w in txt for w in ['start', 'initiate', 'begin']):
        mtype = 'initiate'
    elif any(w in txt for w in ['follow up', 'continue', 'check in']):
        mtype = 'follow_up'
    else:
        mtype = 'response'
    # extract content
    def extract_content(inp):
        low = inp.lower()
        for pat in ['respond to "', 'reply to "', 'answer "', 'they said "', 'message: "', 'text: "']:
            if pat in low:
                i = low.find(pat) + len(pat)
                j = low.find('"', i)
                if j > i:
                    return inp[i:j]
        for pre in ['respond to ', 'reply to ', 'answer ', 'they said ', 'message: ', 'text: ']:
            if low.startswith(pre):
                return inp[len(pre):].strip()
        return inp
    ctx = extract_content(raw)
    out = gen.generate_contact_specific_response(
        contact_id=cid,
        context_message=ctx,
        message_type=mtype
    )
    return jsonify(success=True,
                   generated_message=out,
                   message_type=mtype,
                   context_used=ctx)

@app.route('/api/test_connection')
def test_connection():
    try:
        connected = get_generator().test_ollama_connection()
        return jsonify(success=True, connected=connected)
    except Exception as e:
        return jsonify(success=False, error=str(e))

if __name__ == '__main__':
    # Ensure templates directory
    os.makedirs('templates', exist_ok=True)
    # Write updated template
    html = '''<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>TextTwin - Personal Message Generator</title>
    <style>
        /* ——— paste all of your CSS from before here ——— */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }

        .main-content {
            display: flex;
            min-height: 600px;
        }

        .sidebar {
            width: 300px;
            background: #f8f9fa;
            border-right: 1px solid #e9ecef;
            padding: 20px;
        }

        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
        }

        .contact-search {
            margin-bottom: 20px;
        }

        .contact-search input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 16px;
        }

        .contact-search input:focus {
            outline: none;
            border-color: #667eea;
        }

        .contacts-list {
            max-height: 400px;
            overflow-y: auto;
        }

        .contact-item {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.2s;
            border: 2px solid transparent;
        }

        .contact-item:hover {
            background: #e9ecef;
        }

        .contact-item.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }

        .contact-name {
            font-weight: bold;
            margin-bottom: 5px;
        }

        .contact-preview {
            font-size: 0.9em;
            opacity: 0.7;
        }

        .chat-header {
            padding: 20px;
            border-bottom: 1px solid #e9ecef;
            background: #f8f9fa;
        }

        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            max-height: 400px;
        }

        .message {
            margin-bottom: 15px;
            display: flex;
        }

        .message.me {
            justify-content: flex-end;
        }

        .message-bubble {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
        }

        .message.me .message-bubble {
            background: #007AFF;
            color: white;
        }

        .message.them .message-bubble {
            background: #E5E5EA;
            color: black;
        }

        .message-sender {
            font-size: 0.8em;
            opacity: 0.7;
            margin-bottom: 3px;
        }

        .input-area {
            padding: 20px;
            border-top: 1px solid #e9ecef;
            background: #f8f9fa;
        }

        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }

        .input-group input {
            flex: 1;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 16px;
        }

        .input-group input:focus {
            outline: none;
            border-color: #667eea;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-primary {
            background: #667eea;
            color: white;
        }

        .btn-primary:hover {
            background: #5a6fd8;
            transform: translateY(-1px);
        }

        .generated-response {
            margin-top: 15px;
            padding: 15px;
            background: #e3f2fd;
            border-left: 4px solid #007AFF;
            border-radius: 10px;
        }

        .response-label {
            font-weight: bold;
            color: #007AFF;
            margin-bottom: 8px;
        }

        .response-text {
            font-size: 1.1em;
            line-height: 1.4;
        }

        .loading {
            text-align: center;
            padding: 20px;
            opacity: 0.7;
        }

        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }

        .style-info {
            background: #f3e5f5;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }

        .style-info h4 {
            color: #7b1fa2;
            margin-bottom: 10px;
        }

        .style-stat {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }

        @media (max-width: 768px) {
            .main-content {
                flex-direction: column;
            }

            .sidebar {
                width: 100%;
            }
        }
    </style>
</head>

<body>
    <div class="container">
        <div class="header">
            <h1>🤖 TextTwin</h1>
            <p>Generate messages that sound exactly like you</p>
        </div>
        <div class="main-content">
            <div class="sidebar">
                <div class="contact-search">
                    <input id="contactSearch" placeholder="Enter phone number or contact ID" />
                    <button class="btn btn-primary" onclick="loadContact()" style="margin-top:10px;width:100%;">Load
                        Contact</button>
                </div>
                <h3>Recent Contacts</h3>
                <div id="contactsList" class="contacts-list">
                    <div class="loading">Loading contacts...</div>
                </div>
            </div>
            <div class="chat-area">
                <div id="chatHeader" class="chat-header">
                    <h3>Select a contact to view conversation</h3>
                </div>
                <div id="chatMessages" class="chat-messages">
                    <div class="loading">Select a contact to view message history</div>
                </div>
                <div class="input-area">
                    <div class="input-group">
                        <input id="userInput" placeholder='Enter: "respond to hey what'" />
            <button class=" btn btn-primary" onclick="generateMessage()">Generate</button>
                    </div>
                    <div id="generatedResponse"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentContact = null;
        document.addEventListener('DOMContentLoaded', () => {
            loadContacts();
            testConnection();
        });

        async function loadContacts() {
            try {
                const res = await fetch('/api/contacts');
                const data = await res.json();
                if (data.success) {
                    displayContacts(data.contacts);
                } else {
                    document.getElementById('contactsList').innerHTML =
                        `<div class="error">${data.error}</div>`;
                }
            } catch (e) {
                document.getElementById('contactsList').innerHTML =
                    `<div class="error">${e.message}</div>`;
            }
        }

        function displayContacts(contacts) {
            const list = document.getElementById('contactsList');
            if (!contacts.length) {
                list.innerHTML = '<div class="loading">No contacts</div>';
                return;
            }
            list.innerHTML = contacts.map(c => `
        <div class="contact-item"
             data-contact-id="${c.id}"
             onclick="selectContact('${c.id}')">
          <div class="contact-name">${c.display_name}</div>
          <div class="contact-preview">
            ${c.total_messages} msgs • ${c.last_message}
          </div>
        </div>
      `).join('');
        }

        async function selectContact(contactId) {
            currentContact = contactId;
            document.querySelectorAll('.contact-item.active')
                .forEach(el => el.classList.remove('active'));
            const el = document.querySelector(
                `.contact-item[data-contact-id="${contactId}"]`);
            if (el) el.classList.add('active');

            try {
                const res = await fetch(`/api/conversation/${encodeURIComponent(contactId)}`);
                const data = await res.json();
                if (data.success) {
                    displayConversation(data);
                } else {
                    document.getElementById('chatMessages').innerHTML =
                        `<div class="error">${data.error}</div>`;
                }
            } catch (e) {
                document.getElementById('chatMessages').innerHTML =
                    `<div class="error">${e.message}</div>`;
            }
        }

        function loadContact() {
            const id = document.getElementById('contactSearch').value.trim();
            if (id) selectContact(id);
        }

        function displayConversation(data) {
            // Update the header with style info
            const chatHeader = document.getElementById('chatHeader');
            chatHeader.innerHTML = `
            <h3>${data.contact_id}</h3>
            <div class="style-info">
            <h4>Your texting style with this contact:</h4>
            <div class="style-stat">
                <span>Formality:</span>
                <span>${data.style_info.formality_level}</span>
            </div>
            <div class="style-stat">
                <span>Avg message length:</span>
                <span>${data.style_info.avg_message_length.toFixed(1)} chars</span>
            </div>
            <div class="style-stat">
                <span>Total messages:</span>
                <span>${data.style_info.message_count}</span>
            </div>
            </div>
        `;

            // Render the messages
            const chatMessages = document.getElementById('chatMessages');
            if (!data.messages.length) {
                chatMessages.innerHTML = '<div class="loading">No messages found</div>';
            } else {
                chatMessages.innerHTML = data.messages.map(msg => `
            <div class="message ${msg.is_from_me ? 'me' : 'them'}">
                <div class="message-bubble">
                <div class="message-sender">${msg.sender}</div>
                ${msg.text}
                </div>
            </div>
            `).join('');
                // scroll to bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }


        async function generateMessage() {
            if (!currentContact) {
                alert('Please select a contact first');
                return;
            }
            const userInput = document.getElementById('userInput').value.trim();
            if (!userInput) {
                alert('Please enter a request');
                return;
            }

            const responseDiv = document.getElementById('generatedResponse');
            responseDiv.innerHTML = '<div class="loading">Generating response...</div>';

            try {
                const res = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        contact_id: currentContact,
                        user_input: userInput
                    })
                });
                const data = await res.json();
                if (data.success) {
                    responseDiv.innerHTML = `
        <div class="generated-response">
          <div class="response-label">
            Generated Response (${data.message_type}):
          </div>
          <div class="response-text">${data.generated_message}</div>
        </div>
      `;
                } else {
                    responseDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                }
            } catch (err) {
                responseDiv.innerHTML = `<div class="error">Error: ${err.message}</div>`;
            }
        }

        async function testConnection() {
            try {
                const res = await fetch('/api/test_connection');
                const data = await res.json();
                if (!data.success || !data.connected) {
                    document.getElementById('generatedResponse').innerHTML =
                        `<div class="error">
           ⚠️ Ollama not connected. Make sure it's running.
         </div>`;
                }
            } catch (err) {
                console.error('Connection test failed:', err);
            }
        }

    </script>
</body>

</html>'''
    with open('templates/index.html', 'w') as f:
        f.write(html)
    print("🚀 GUI ready at http://localhost:5000")
    app.run(host='0.0.0.0', port=3000)
