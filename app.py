from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader
import docx
import sqlite3
import os
from langdetect import detect, DetectorFactory
from ddgs import DDGS
import hashlib
import torch

DetectorFactory.seed = 0  # reproducible language detection

app = Flask(__name__)

# ===================== Cache for scraped results =====================
scrape_cache = {}  # key: text hash, value: scraped results

# ===================== Load Models =====================
try:
    plagiarism_model = SentenceTransformer('paraphrase-mpnet-base-v2')
    ai_detector = pipeline("text-classification", model="roberta-large-openai-detector")
    rephrase_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    rephrase_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    rephrase_model.to('cpu')  # change to 'cuda' if GPU available
except Exception as e:
    print(f"Model loading error: {e}")
    plagiarism_model = None
    ai_detector = None
    rephrase_model = None

# ===================== Reference Corpus =====================
reference_texts = [
    "This is a sample text with plagiarism.",
    "Another example of copied content.",
    "Educational content about programming.",
    "A guide to writing original articles."
]

# ===================== SQLite Database =====================
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

# ===================== Helper Functions =====================
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def scrape_web_for_similar(text, max_results=5):
    query_text = text[:400]
    query_hash = hashlib.md5(query_text.encode()).hexdigest()
    if query_hash in scrape_cache:
        return scrape_cache[query_hash]

    try:
        results = []
        with DDGS() as ddgs:
            search_results = ddgs.text(query_text, max_results=max_results)
            for r in search_results:
                snippet = r.get("body", "")
                link = r.get("href", "")
                if snippet and len(snippet) > 50:
                    try:
                        if detect(snippet) == 'en':
                            results.append(f"{snippet} - <a href='{link}' target='_blank'>source</a>")
                    except:
                        continue
        final_results = results if results else ["No relevant content found."]
        scrape_cache[query_hash] = final_results
        return final_results
    except Exception as e:
        print(f"Web scraping error: {e}")
        return ["Error fetching web content."]

def rephrase_content(text):
    input_text = "paraphrase: " + text + " </s>"
    inputs = rephrase_tokenizer(
        [input_text],
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )
    outputs = rephrase_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=256,
        min_length=50,
        num_beams=5,
        num_return_sequences=1,
        early_stopping=True,
        no_repeat_ngram_size=3,
        temperature=0.7,
        top_p=0.9
    )
    return rephrase_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ===================== Flask Routes =====================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/check', methods=['POST'])
def do_check():
    text = request.form.get('text', '').strip()
    file = request.files.get('file')

    if not text and not file:
        return jsonify({'error': 'Please provide text or upload a file!'}), 400

    # ----------- Extract text from file -----------
    if file:
        try:
            if file.filename.endswith('.pdf'):
                pdf_reader = PdfReader(file)
                text = "".join([page.extract_text() or '' for page in pdf_reader.pages])
            elif file.filename.endswith('.docx'):
                doc = docx.Document(file)
                text = "\n".join([para.text for para in doc.paragraphs])
            elif file.filename.endswith('.txt'):
                text = file.read().decode('utf-8')
            else:
                return jsonify({'error': 'Only PDF, DOCX, or TXT files allowed!'}), 400
        except Exception as e:
            return jsonify({'error': f'Error reading file: {e}'}), 400

    if not text.strip():
        return jsonify({'error': 'The provided text is empty!'}), 400

    # ----------- Language detection -----------
    try:
        language = detect(text)
    except Exception as e:
        print(f"Language detection error: {e}")
        language = "unknown"

    # ----------- Scrape web for similar content -----------
    scraped_content = scrape_web_for_similar(text)

    # ----------- Chunk text for long documents -----------
    text_chunks = chunk_text(text)

    # ----------- Plagiarism check -----------
    if plagiarism_model is None:
        return jsonify({'error': 'Plagiarism model not loaded'}), 500

    try:
        user_embeddings = [plagiarism_model.encode(chunk, convert_to_tensor=True) for chunk in text_chunks]
        reference_embeddings = plagiarism_model.encode(reference_texts + scraped_content, convert_to_tensor=True)
        cosine_scores = [util.pytorch_cos_sim(ue, reference_embeddings)[0] for ue in user_embeddings]
        plagiarism_score = max([cs.max().item() for cs in cosine_scores]) * 100
    except Exception as e:
        print(f"Plagiarism check error: {e}")
        return jsonify({'error': 'Failed to calculate plagiarism score'}), 500

    # ----------- AI content detection (first chunk) -----------
    if ai_detector is None:
        return jsonify({'error': 'AI detector not loaded'}), 500

    try:
        truncated_text = text_chunks[0][:512]
        ai_result = ai_detector(truncated_text)[0]
        is_ai_generated = ai_result['label'].lower() == 'ai-generated' and ai_result['score'] > 0.7
        ai_confidence = ai_result['score'] * 100
    except Exception as e:
        print(f"AI detection error: {e}")
        is_ai_generated = False
        ai_confidence = 0.0

    # ----------- Rephrase if plagiarism >70% -----------
    rephrased = None
    if plagiarism_score > 70 and rephrase_model is not None:
        rephrased = rephrase_content(text)

    # ----------- Save report to database -----------
    try:
        conn = sqlite3.connect('plagiarism.db')
        conn.execute(
            "INSERT INTO reports (text, plagiarism_score, ai_confidence, language) VALUES (?, ?, ?, ?)",
            (text[:500], plagiarism_score, ai_confidence, language)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

    # ----------- Messages -----------
    message = 'Possible plagiarism detected!' if plagiarism_score > 70 else 'No plagiarism detected!'
    ai_message = 'AI-generated content detected!' if is_ai_generated else 'No AI-generated content detected!'

    return jsonify({
        'message': f"{message} (Score: {round(plagiarism_score, 2)})",
        'ai_message': ai_message,
        'ai_confidence': round(ai_confidence, 2),
        'language': language,
        'rephrased': rephrased,
        'scraped_sources': scraped_content
    })


# ===================== Reports Endpoint =====================
@app.route('/reports')
def get_reports():
    try:
        conn = sqlite3.connect('plagiarism.db')
        cursor = conn.execute("SELECT * FROM reports ORDER BY timestamp DESC LIMIT 10")
        reports = [
            {'id': row[0], 'text': row[1], 'plagiarism_score': row[2],
             'ai_confidence': row[3], 'language': row[4], 'timestamp': row[5]}
            for row in cursor.fetchall()
        ]
        conn.close()
        return jsonify(reports)
    except Exception as e:
        print(f"Error in /reports: {e}")
        return jsonify({'error': 'Failed to load reports'}), 500


# ===================== Contact Form =====================
@app.route('/contact', methods=['POST'])
def contact():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')

    if not name or not email or not message:
        return jsonify({'error': 'Name, email, and message required!'}), 400

    return jsonify({'message': f'Thank you, {name}! Your message has been received. We will contact you at {email}.'})


# ===================== Run Flask =====================
if __name__ == '__main__':
    app.run(debug=True)
