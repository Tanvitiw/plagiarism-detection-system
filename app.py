from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from PyPDF2 import PdfReader
import docx
import sqlite3
import os
from langdetect import detect
import torch

app = Flask(__name__)

# Load pre-trained models with CPU optimization
plagiarism_model = SentenceTransformer('paraphrase-mpnet-base-v2', device='cpu')
ai_detector = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)  # -1 for CPU

# Set torch to use less memory
torch.set_num_threads(4)  # Limit to 4 threads to reduce CPU load
torch.set_num_interop_threads(1)

# Dummy reference corpus for plagiarism
reference_texts = [
    "This is a sample text with plagiarism.",
    "Another example of copied content.",
    "Original text without plagiarism."
]

# Initialize SQLite database
if not os.path.exists('plagiarism.db'):
    conn = sqlite3.connect('plagiarism.db')
    conn.execute('''CREATE TABLE reports
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     text TEXT,
                     plagiarism_score REAL,
                     ai_confidence REAL,
                     language TEXT,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/check', methods=['POST'])
def do_check():
    text = request.form.get('text')
    file = request.files.get('file')
    
    if not text and not file:
        return jsonify({'error': 'Text ya file to daal!'}), 400
    
    if text or file:
        # Extract text if file is uploaded
        if file:
            if file.filename.endswith('.pdf'):
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            elif file.filename.endswith('.docx'):
                doc = docx.Document(file)
                text = "\n".join([para.text for para in doc.paragraphs])
            elif file.filename.endswith('.txt'):
                text = file.read().decode('utf-8')
            else:
                return jsonify({'error': 'Sirf PDF, DOCX, ya TXT files allowed hain!'}), 400
        
        # Language detection using langdetect
        try:
            language = detect(text)
        except:
            language = "unknown"  # Agar detection fail ho to unknown set karo

        # Plagiarism check with batch size 1
        user_embedding = plagiarism_model.encode(text, convert_to_tensor=True, batch_size=1)
        reference_embeddings = plagiarism_model.encode(reference_texts, convert_to_tensor=True, batch_size=1)
        cosine_scores = util.pytorch_cos_sim(user_embedding, reference_embeddings)[0]
        plagiarism_score = cosine_scores.max().item() * 100

        # AI content detection
        ai_result = ai_detector(text)[0]
        is_ai_generated = ai_result['label'] == 'POSITIVE' and ai_result['score'] > 0.7
        ai_confidence = ai_result['score'] * 100

        # Save to database
        conn = sqlite3.connect('plagiarism.db')
        conn.execute("INSERT INTO reports (text, plagiarism_score, ai_confidence, language) VALUES (?, ?, ?, ?)",
                     (text[:500], plagiarism_score, ai_confidence, language))
        conn.commit()
        conn.close()

        message = 'Possible plagiarism detected!' if plagiarism_score > 70 else 'No plagiarism detected!'
        ai_message = 'AI-generated content detected!' if is_ai_generated else 'No AI-generated content detected!'

        return jsonify({
            'message': f"{message} (Score: {round(plagiarism_score, 2)})",
            'ai_message': ai_message,
            'ai_confidence': round(ai_confidence, 2),
            'language': language
        })

@app.route('/reports')
def get_reports():
    try:
        conn = sqlite3.connect('plagiarism.db')
        cursor = conn.execute("SELECT * FROM reports ORDER BY timestamp DESC LIMIT 10")
        reports = [{'id': row[0], 'text': row[1], 'plagiarism_score': row[2], 'ai_confidence': row[3], 'language': row[4], 'timestamp': row[5]} for row in cursor.fetchall()]
        conn.close()
        return jsonify(reports)
    except Exception as e:
        print(f"Error in /reports: {e}")  # Debug log
        return jsonify({'error': 'Failed to load reports'}), 500

@app.route('/contact', methods=['POST'])
def contact():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')
    
    if not name or not email or not message:
        return jsonify({'error': 'Name, email, aur message daal!'}), 400
    
    return jsonify({'message': f'Thank you, {name}! Your message has been received. We will contact you at {email}.'})

if __name__ == '__main__':
    app.run(debug=True)