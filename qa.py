import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
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

# Function to answer questions and extract sentences containing any keyword from the question
def answer_question_with_keywords(text, question):
    # Tokenize the question
    question_tokens = set(word_tokenize(question.lower()))
    doc = nlp(text)
    relevant_sentences = []

    # Find sentences containing any keyword from the question
    for sentence in doc.sents:
        sentence_tokens = set(word_tokenize(sentence.text.lower()))
        if question_tokens.intersection(sentence_tokens):
            relevant_sentences.append(sentence.text)
    
    # If no relevant information found, return an empty list
    return relevant_sentences


# Function to answer questions and extract sentences containing entities from the question
def answer_question_with_ner(text, question):
    doc = nlp(text)
    question_entities = [ent.text.lower() for ent in doc.ents]
    
    best_match = ""
    for sentence in doc.sents:
        sentence_entities = [ent.text.lower() for ent in sentence.ents]
        if any(entity in sentence_entities for entity in question_entities):
            best_match = sentence.text
            break
    
    return best_match

# Function to answer questions and extract sentences containing any word from the question
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
    
    # If no relevant information found, return a sentence containing any word from the question
    if not best_match:
        for sentence in doc.sents:
            sentence_tokens = set(word_tokenize(sentence.text.lower()))
            if sentence_tokens:
                best_match = sentence.text
                break

    return best_match

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

# Ask if the user has any questions
while True:
    have_questions = input("\nDo you have any questions? (yes/no): ").strip().lower()
    if have_questions == "yes":
        question = input("What is your question?\n ")
        answer = answer_question_with_keywords(long_text, question)
        if not answer:
            answer = answer_question_with_ner(long_text, question)
        if not answer:
            print("No relevant information found. Using worst-case answer:")
            answer = answer_question(long_text, question)
        print("\nSolution:", answer)
        # Separation line after each question and answer
        print("\n" + "-" * 150)
    elif have_questions == "no":
        print("\nThank you for using our summarization tool. Goodbye!\n\n" + "*" * 150)
        break
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")

# Print ending line
print("\n" + "*" * 150)
