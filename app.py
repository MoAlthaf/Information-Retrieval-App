import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    embeddings=np.load("embeddings.npy")  

    with open("documents.txt",encoding="utf-8") as f:
        documents = f.readlines()
except Exception as e:
    st.error(f"Error loading data: {e}")
    embeddings = np.array([])  
    documents = []  

def retrieve_top_k(query_embedding, embeddings,k=10):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]



st.title("Information Retrieval using Document Embeddings")

query= st.text_input("Enter your query:")

# Load or compute query embedding (Placeholder: Replace with actual embeddingmodel)
def get_query_embedding(query):
    return np.random.rand(embeddings.shape[1]) # Replace with actual embedding function
if st.button("Search"):
    query_embedding = get_query_embedding(query)
    results = retrieve_top_k(query_embedding, embeddings)
    # Display results
    st.write("### Top 10 Relevant Documents:")
    for doc, score in results:
        st.write(f"- **{doc.strip()}** (Score: {score:.4f})")
