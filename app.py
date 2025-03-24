import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from flask import Flask, request, jsonify, render_template, send_from_directory
import os

nltk.download('punkt')
nltk.download('stopwords')


# Download NLTK resources silently
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


# Calculate word frequencies
def calculate_word_frequencies(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    word_frequencies = defaultdict(int)
    for word in words:
        if word not in stop_words and word.isalnum():
            word_frequencies[word] += 1
    return word_frequencies


# TextRank algorithm for sentence scoring
def calculate_sentence_scores(sentences, word_frequencies):
    word_graph = defaultdict(lambda: defaultdict(int))
    for sentence in sentences:
        words = [word for word in word_tokenize(sentence.lower()) if word.isalnum()]
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                word_graph[words[i]][words[j]] += 1
                word_graph[words[j]][words[i]] += 1

    scores = defaultdict(int)
    for word1 in word_graph:
        for word2 in word_graph[word1]:
            if word_frequencies[word1] > 0:
                scores[word2] += word_graph[word1][word2] / word_frequencies[word1]

    sentence_scores = defaultdict(int)
    for sentence in sentences:
        words = [word for word in word_tokenize(sentence.lower()) if word.isalnum()]
        for word in words:
            sentence_scores[sentence] += scores[word]

    return sentence_scores


# Generate summary points based on TextRank scores
def generate_summary(text, ratio=0.3):
    sentences = sent_tokenize(text)

    # Handle empty or very short text
    if len(sentences) <= 2:
        return sentences

    word_frequencies = calculate_word_frequencies(text)
    sentence_scores = calculate_sentence_scores(sentences, word_frequencies)
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

    # Determine number of sentences to include (minimum 1, maximum ratio of total)
    num_sentences = max(1, min(len(sentences) // 3, int(len(sentences) * ratio)))
    top_sentences = [sentence for sentence, _ in sorted_sentences[:num_sentences]]

    # Sort by original position to maintain logical flow
    ordered_summary = [s for s in sentences if s in top_sentences]
    return ordered_summary


# Answer questions with keyword matching
def answer_question(text, question):
    # Tokenize the question and remove stopwords
    stop_words = set(stopwords.words('english'))
    question_tokens = set(word for word in word_tokenize(question.lower())
                          if word.isalnum() and word not in stop_words)

    # If no significant words in question, return empty
    if not question_tokens:
        return "Please ask a more specific question."

    # Score sentences based on question token matches
    sentences = sent_tokenize(text)
    matches = []

    for sentence in sentences:
        sentence_tokens = set(word for word in word_tokenize(sentence.lower())
                              if word.isalnum())
        common_words = question_tokens.intersection(sentence_tokens)

        if common_words:
            score = len(common_words) / len(question_tokens)
            matches.append((sentence, score))

    # Sort by score and return top matches
    matches.sort(key=lambda x: x[1], reverse=True)

    if not matches:
        return "No relevant information found in the text."

    # Return top match if it has significant score
    if matches and matches[0][1] > 0.2:
        return matches[0][0]
    else:
        return "No relevant information found in the text."


# Initialize Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/summarize', methods=['POST'])
def summarize():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    summary = generate_summary(text)

    return jsonify({
        'summary_points': summary
    })


@app.route('/api/answer', methods=['POST'])
def answer():
    data = request.get_json()

    if not data or 'question' not in data or 'text' not in data:
        return jsonify({'error': 'Missing question or text'}), 400

    question = data['question']
    text = data['text']

    answer = answer_question(text, question)

    return jsonify({
        'answer': answer
    })


if __name__ == '__main__':
    # Make sure the template and static folders exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists('static/css'):
        os.makedirs('static/css')
    if not os.path.exists('static/js'):
        os.makedirs('static/js')

    app.run(debug=True)
