import React, { useState, useEffect } from "react";
import "@tensorflow/tfjs";
import * as use from "@tensorflow-models/universal-sentence-encoder";
import "./SearchComponent.css"; // Import the CSS file

const SearchComponent = () => {
  const [model, setModel] = useState(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedModel = await use.load();
        setModel(loadedModel);
        console.log("Model loaded!");
      } catch (error) {
        console.error("Error loading model:", error);
        setError("Failed to load the model. Please try again later.");
      }
    };

    loadModel();
  }, []);

  const search = async () => {
    if (!query) return;
    setResults([]);
    setLoading(true);
    setError("");

    try {
      // Send the query to the backend for combined search
      const response = await fetch(
        `${process.env.REACT_APP_BACKEND_URL}/combined-search`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ query }),
        }
      );

      const data = await response.json();
      if (response.ok) {
        setResults(data.results);
      } else {
        setError(data.detail || "An error occurred during the search.");
      }
    } catch (error) {
      console.error("Search error:", error);
      setError("An error occurred during the search. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="search-container">
      <h1 className="search-title">Translation Search</h1>
      <div className="search-input-container">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter a word or phrase..."
          disabled={loading}
          className="search-input"
        />
        <button
          onClick={search}
          disabled={loading || !model}
          className="search-button"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {error && <p className="error-message">{error}</p>}

      <ul className="results-list">
        {results.map((result, index) => (
          <li key={index} className="result-item">
            <strong className="result-text">{result.english}</strong> →{" "}
            <span className="result-translation">{result.bj_translation}</span>
            <br />
            <small className="result-type">
              {result.type === "text_match"
                ? "🔍 Contains the word"
                : `✨ Semantically similar (Score: ${result.similarity.toFixed(
                    2
                  )})`}
            </small>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default SearchComponent;
