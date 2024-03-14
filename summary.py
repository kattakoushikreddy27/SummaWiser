import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
import re
from collections import defaultdict

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to calculate word frequency in text
def calculate_word_frequencies(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    word_frequencies = defaultdict(int)
    for word in words:
        if word not in stop_words and word.isalnum():
            word_frequencies[word] += 1
    return word_frequencies

# Function to calculate sentence scores using TextRank
def calculate_sentence_scores_with_textrank(sentences, word_frequencies):
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
            if word_frequencies[word1] != 0:  # Check if word frequency is not zero
                scores[word2] += word_graph[word1][word2] / word_frequencies[word1]

    sentence_scores = defaultdict(int)
    for i, sentence in enumerate(sentences):
        words = [word for word in word_tokenize(sentence.lower()) if word.isalnum()]
        for word in words:
            sentence_scores[sentence] += scores[word]

    return sentence_scores

# Function to generate summary points dynamically based on TextRank scores
def generate_summary_points_with_textrank(text):
    sentences = sent_tokenize(text)
    word_frequencies = calculate_word_frequencies(text)
    sentence_scores = calculate_sentence_scores_with_textrank(sentences, word_frequencies)
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate number of points based on word count
    num_words = len(word_tokenize(text))
    num_sentences = max(num_words // 50, 1)  # Adjust the divisor as needed
    top_sentences = [sentence for sentence, score in sorted_sentences[:num_sentences]]
    return top_sentences



# Welcome message
print("\n" + "*" * 150)
print("\nWelcome to the Text Summarization and Question Answering Tool!\n")

# Ask user to input long text
print("Please enter your long text:")
long_text = input()

# Generate summary points using TextRank
print("\nSummary Points:")
summary_points = generate_summary_points_with_textrank(long_text)
for i, point in enumerate(summary_points, 1):
    print(f"{i}. {point}")

# Separation line after summary
print("\n" + "-" * 150)

# Print ending line
print("\n" + "*" * 150)
