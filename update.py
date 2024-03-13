import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

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
    sentences = sent_tokenize(text)
    keyword_matches = []
    for sentence in sentences:
        if question.lower() in sentence.lower():
            keyword_matches.append(sentence)
    if keyword_matches:
        return max(keyword_matches, key=len)
    else:
        return "Unknown"

# Ask user to input long text
long_text = input("Please enter your long text: ")

# Generate summary points
summary_points = generate_summary_points(long_text)
print("Summary Points:")
for i, point in enumerate(summary_points, 1):
    print(f"{i}. {point}")

# Ask if the user has any questions
while True:
    have_questions = input("Do you have any questions? (yes/no): ").strip().lower()
    if have_questions == "yes":
        question = input("What is your question? ")
        answer = answer_question(long_text, question)
        print("Answer:", answer)
    elif have_questions == "no":
        print("Thank you for using our summarization tool. Goodbye!")
        break
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")
