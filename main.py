import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import spacy

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to calculate word frequency in text
def calculate_word_frequencies(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    word_frequencies = {}
    for word in words:
        if word not in stop_words and word.isalnum():
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    return word_frequencies

# Function to calculate sentence scores based on word frequencies
def calculate_sentence_scores(sentences, word_frequencies):
    sentence_scores = {}
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        score = 0
        for word in words:
            if word in word_frequencies:
                score += word_frequencies[word]
        sentence_scores[sentence] = score
    return sentence_scores

# Function to generate summary points dynamically based on word count
def generate_summary_points(text, num_words_per_point=50):
    sentences = sent_tokenize(text)
    word_frequencies = calculate_word_frequencies(text)
    sentence_scores = calculate_sentence_scores(sentences, word_frequencies)
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate number of points based on word count
    num_words = len(word_tokenize(text))
    num_points = max(num_words // num_words_per_point, 1)
    
    # Choose top sentences dynamically based on content
    top_sentences = [sentence for sentence, score in sorted_sentences[:num_points]]
    return top_sentences

# Function to answer questions
def answer_question(text, question):
    # Tokenize the question
    question_tokens = set(word_tokenize(question.lower()))
    doc = nlp(text)
    best_match = ""

    # Find sentences containing any keyword from the question
    for sentence in doc.sents:
        sentence_tokens = set(word_tokenize(sentence.text.lower()))
        if question_tokens.intersection(sentence_tokens):
            best_match = sentence.text
            break

    return best_match

# Welcome message
print("\n" + "*" * 150)
print("\nWelcome to the Text Summarization and Question Answering Tool!\n")

# Ask user to input long text
print("Please enter your long text:")
long_text = input()

# Generate summary points
print("\nSummary Points:")
summary_points = generate_summary_points(long_text)
for i, point in enumerate(summary_points, 1):
    print(f"{i}. {point}")

# Separation line after summary
print("\n" + "-" * 150)

# Ask if the user has any questions
while True:
    have_questions = input("\nDo you have any questions? (yes/no): ").strip().lower()
    if have_questions == "yes":
        question = input("What is your question?\n ")
        answer = answer_question(long_text, question)
        print("\nAnswer:", answer)
        # Separation line after each question and answer
        print("\n" + "-" * 150)
    elif have_questions == "no":
        print("\nThank you for using our summarization tool. Goodbye!\n\n" + "*" * 150)
        break
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")

# Print ending line
print("\n" + "*" * 150)
