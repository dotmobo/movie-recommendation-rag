import unittest
from unittest.mock import patch, MagicMock
from rag.main import (
    load_movies,
    initialize_chromadb,
    generate_embeddings,
    index_movies,
    perform_query,
    generate_prompt,
    query_ollama,
)

class TestMovieRecommendation(unittest.TestCase):

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data='[{"imdbID": "tt123", "Title": "Test Movie", "Plot": "A test plot", "Year": "2022"}]')
    def test_load_movies(self, mock_open):
        movies = load_movies("dummy_path.json")
        self.assertEqual(len(movies), 1)
        self.assertEqual(movies[0]["Title"], "Test Movie")

    @patch("chromadb.Client")
    def test_initialize_chromadb(self, mock_client):
        client = initialize_chromadb()
        mock_client.assert_called_once()
        self.assertIsNotNone(client)

    @patch("sentence_transformers.SentenceTransformer")
    def test_generate_embeddings(self, MockSentenceTransformer):
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        MockSentenceTransformer.return_value = mock_model
        content = "Test content"
        embedding = generate_embeddings(mock_model, content)
        mock_model.encode.assert_called_once_with(content)
        self.assertEqual(embedding, [0.1, 0.2, 0.3])

    @patch("sentence_transformers.SentenceTransformer")
    @patch("chromadb.Client")
    def test_index_movies(self, MockClient, MockSentenceTransformer):
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        MockSentenceTransformer.return_value = mock_model

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"documents": []}
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        MockClient.return_value = mock_client

        movies = [{"imdbID": "tt123", "Title": "Test Movie", "Plot": "A test plot", "Year": "2022"}]

        index_movies(movies, mock_model, mock_collection)

        mock_collection.get.assert_called_once_with(ids=["tt123"])
        mock_collection.add.assert_called_once()
        self.assertEqual(
            mock_collection.add.call_args[1]["ids"], ["tt123"]
        )

    @patch("sentence_transformers.SentenceTransformer")
    @patch("chromadb.Client")
    def test_perform_query(self, MockClient, MockSentenceTransformer):
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        MockSentenceTransformer.return_value = mock_model

        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": ["Result 1", "Result 2", "Result 3"]}

        results = perform_query(mock_collection, mock_model, "Test query")

        mock_collection.query.assert_called_once()
        self.assertEqual(len(results["documents"]), 3)
        self.assertEqual(results["documents"], ["Result 1", "Result 2", "Result 3"])

    def test_generate_prompt(self):
        context = "- Test Movie: A test plot"
        query = "a test"
        expected_prompt = """
You are a knowledgeable DVD salesperson with expertise in movies. Your task is to recommend movies to customers, but you can only suggest films that are available in the store's inventory. Make sure your recommendations are based solely on the list of movies provided.

Context: Below is a list of movies currently available in the store:
- Test Movie: A test plot

Customer's Question: Ask for a movie about a test

Your Movie Recommendations (only from the available list):
"""
        prompt = generate_prompt(context, query)
        self.assertEqual(prompt.strip(), expected_prompt.strip())

    @patch("ollama.Client.chat")
    def test_query_ollama(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "Recommended movie"}}
        response = query_ollama("Test prompt", model_name="test_model")
        self.assertEqual(response, "Recommended movie")
        mock_chat.assert_called_once_with("test_model", messages=[{"role": "user", "content": "Test prompt"}])

if __name__ == "__main__":
    unittest.main()
