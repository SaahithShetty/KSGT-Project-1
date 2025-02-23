import pdfplumber
import os
import re
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_folder):
    extracted_texts = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            with pdfplumber.open(file_path) as pdf:
                text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                extracted_texts.append(text)
    return extracted_texts

# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# Function to extract important keywords using TF-IDF
def extract_keywords_tfidf(texts, num_keywords=20):
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).tolist()[0]
    keyword_scores = dict(zip(feature_names, tfidf_scores))
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return [keyword for keyword, _ in sorted_keywords[:num_keywords]]

# Function to extract named entities (NER)
def extract_named_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in {"ORG", "PERSON", "GPE", "MONEY", "PRODUCT", "EVENT"}]
    return list(set(entities))

# Main function to process PDFs and extract concepts
def extract_concepts_from_pdfs(pdf_folder):
    raw_texts = extract_text_from_pdfs(pdf_folder)
    processed_texts = [preprocess_text(text) for text in raw_texts]
    
    # Extract keywords using TF-IDF
    tfidf_keywords = extract_keywords_tfidf(processed_texts)
    
    # Extract named entities (organizations, AI models, datasets, etc.)
    named_entities = []
    for text in raw_texts:
        named_entities.extend(extract_named_entities(text))
    
    return {
        "TF-IDF Keywords": tfidf_keywords,
        "Named Entities": list(set(named_entities))
    }

# Provide the path to your folder containing PDFs
pdf_folder_path = "pdfs"

# Run the extraction
extracted_concepts = extract_concepts_from_pdfs(pdf_folder_path)

# Print results
print("🔹 Extracted Key Concepts:")
print("📌 Top Keywords:", extracted_concepts["TF-IDF Keywords"])
print("📌 Named Entities:", extracted_concepts["Named Entities"])