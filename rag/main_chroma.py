import json
import ollama
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import streamlit as st


def load_movies(file_path="data/data.json"):
    """Loads movie data from the JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def initialize_chromadb():
    """Initializes and returns a ChromaDB client with a movie collection."""
    client = chromadb.Client(Settings(is_persistent=True, anonymized_telemetry=False))
    return client


def generate_embeddings(model, content):
    """Generates embeddings for a given content."""
    return model.encode(content)


def index_movies(movies, model, collection):
    """Indexes movies in ChromaDB only if the index doesn't exist already."""
    for movie in movies:
        doc_id = movie["imdbID"]
        title = movie["Title"]
        plot = movie["Plot"]
        content = f"{title}: {plot}"

        # Check if the document already exists
        existing_document = collection.get(ids=[doc_id])

        # If the document exists, skip it
        if existing_document["documents"]:
            print(
                f"The document with ID {doc_id} already exists. It will not be added."
            )
            continue

        # Generate the embedding
        embedding = generate_embeddings(model, content)

        # Add to the collection
        collection.add(
            ids=[doc_id],
            metadatas=[{"title": title, "year": movie["Year"]}],
            documents=[content],
            embeddings=[embedding],
        )
    print("Data indexed in ChromaDB.")


def perform_query(collection, model, query):
    """Performs a search in ChromaDB and returns the results."""
    query_embedding = generate_embeddings(model, query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,  # Return 3 results
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

    # Initialize SentenceTransformer to generate embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize ChromaDB
    client = initialize_chromadb()

    # Check if the collection exists already
    collection = client.get_or_create_collection(name="movies")

    # If the collection is empty (does not exist), index the movies
    if not collection.count():
        print("Indexing movies in ChromaDB...")
        index_movies(movies, model, collection)
    else:
        print("The movie index already exists. It was not regenerated.")

    # Streamlit user interface
    st.title("Movie Recommendations")
    query = st.text_input("Ask for a movie about...", "a wormhole in space")

    if st.button("Ask"):
        if query:
            # Perform the search
            results = perform_query(collection, model, query)

            # Prepare the prompt for Ollama as the DVD salesperson
            documents = results.get("documents", [])
            if documents is None:
                documents = []

            # Display the results in the UI
            st.subheader("Movies Found")
            for doc in documents:
                st.write(doc)

            context = "\n".join(
                [f"- {doc}" for result in documents if result for doc in result]
            )
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
