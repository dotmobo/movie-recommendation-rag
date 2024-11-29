import json
import ollama
from qdrant_client import QdrantClient
import streamlit as st
import uuid

def generate_uuid(imdb_id):
    """Generates a UUID from a string (e.g., imdbID)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, imdb_id))

def load_movies(file_path="data/data.json"):
    """Loads movie data from the JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def initialize_qdrant():
    """Initializes and returns a Qdrant client."""
    client = QdrantClient(path="./qdrant")
    return client

def index_movies(movies, client, collection_name="movies"):
    """Indexes movies in Qdrant."""
    docs, metadata, ids = [], [], []

    for movie in movies:
        doc_id = generate_uuid(movie["imdbID"])  # Convert imdbID to UUID
        title = movie["Title"]
        plot = movie["Plot"]
        content = f"{title}: {plot}"

        # Append data for batch indexing
        docs.append(content)
        metadata.append({"title": title, "year": movie["Year"]})
        ids.append(doc_id)

    # Add documents to the Qdrant collection
    client.add(
        collection_name=collection_name,
        documents=docs,
        metadata=metadata,
        ids=ids,
    )
    print("Data indexed in Qdrant.")

def perform_query(client, query, collection_name="movies"):
    """Performs a search in Qdrant and returns the results."""
    results = client.query(
        collection_name=collection_name,
        query_text=query,
        limit=3,
    )
    return results

def generate_prompt(context, query):
    """Generates a prompt for Ollama where the model acts as a DVD salesperson."""
    return f"""
You are a knowledgeable DVD salesperson with expertise in movies. Your task is to recommend movies to customers, but you can only suggest films that are available in the store's inventory. Make sure your recommendations are based solely on the list of movies provided.

Context: Below is a list of movies currently available in the store:
{context}

Customer's Question: Ask for a movie about {query}

Your Movie Recommendations (only from the available list):
"""

def query_ollama(prompt, model_name="myllama"):
    """Queries Ollama with the given prompt."""
    client = ollama.Client()
    response = client.chat(model_name, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def main():
    # Load the movies
    movies = load_movies()

    # Initialize Qdrant
    client = initialize_qdrant()

    # Check if the collection exists
    collections = client.get_collections().collections
    collection_name = "movies"
    collection_exists = any(col.name == collection_name for col in collections)

    # Index the movies only if the collection doesn't exist
    if not collection_exists:
        print("Creating collection and indexing movies...")
        index_movies(movies, client)
    else:
        print("The movie index already exists. It was not regenerated.")

    # Streamlit user interface
    st.title("Movie Recommendations")
    query = st.text_input("Ask for a movie about...", "a wormhole in space")

    if st.button("Ask"):
        if query:
            # Perform the search
            results = perform_query(client, query)
            client.close()

            # Prepare the prompt for Ollama as the DVD salesperson
            documents = [result.document for result in results]

            # Display the results in the UI
            st.subheader("Movies Found")
            for doc in documents:
                st.write(doc)

            context = "\n".join([f"- {doc}" for doc in documents])
            prompt = generate_prompt(context, query)

            # Display the prompt in the UI
            st.subheader("Prompt Sent to Ollama")
            st.text_area("Here is the prompt sent to Ollama", prompt, height=200)

            # Show the loading spinner while the request is being processed
            with st.spinner("Searching..."):
                # Query Ollama
                response = query_ollama(prompt, model_name="llama3.2")

            st.subheader("Ollama's Recommendation")
            st.write(response)

if __name__ == "__main__":
    main()
