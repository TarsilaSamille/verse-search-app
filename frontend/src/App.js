import { useState } from "react";
import axios from "axios";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);

  const search = async () => {
    if (!query) return;
    const response = await axios.post("http://127.0.0.1:8000/search/", {
      text: query,
      top_k: 5,
    });
    setResults(response.data.results);
  };

  return (
    <div className="App" style={{ padding: "20px", fontFamily: "Arial" }}>
      <h2>Search for Similar Verses</h2>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter a verse..."
        style={{ padding: "10px", width: "300px" }}
      />
      <button onClick={search} style={{ marginLeft: "10px", padding: "10px" }}>
        Search
      </button>

      <div style={{ marginTop: "20px" }}>
        {results.map((result, index) => (
          <div
            key={index}
            style={{
              marginBottom: "15px",
              padding: "10px",
              border: "1px solid #ccc",
            }}
          >
            <p>
              <strong>Similarity:</strong> {result.similarity * 100}%
            </p>
            <p>
              <strong>English:</strong> {result.english}
            </p>
            <p>
              <strong>BJ Translation:</strong> {result.bj_translation}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
