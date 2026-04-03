import { useState, ChangeEvent } from "react";
import axios from "axios";
import "./App.css";

interface PredictionResult {
  prediction: string;
  confidence: number;
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    // Get API URL from environment variables or fallback to localhost
    const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

    try {
      const response = await axios.post<PredictionResult>(
        `${API_URL}/predict`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        },
      );
      setResult(response.data);
    } catch (err) {
      console.error("Error during prediction:", err);
      setError(
        `Failed to get prediction from the server. Make sure the backend (${API_URL}) is running.`,
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>MotoLens</h1>
        <p>Identify cars with AI</p>
      </header>

      <main>
        <div className="upload-section">
          <label htmlFor="file-upload" className="custom-file-upload">
            {selectedFile ? "Change Image" : "Choose Image"}
          </label>
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
          />

          {preview && (
            <div className="preview-container">
              <img src={preview} alt="Preview" className="image-preview" />
            </div>
          )}

          <button
            onClick={handlePredict}
            disabled={!selectedFile || isLoading}
            className="predict-button"
          >
            {isLoading ? "Processing..." : "Identify Car"}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}

        {result && (
          <div className="result-section">
            <h2>Results</h2>
            <div className="result-card">
              <div className="result-item">
                <span className="label">Car Model:</span>
                <span className="value">{result.prediction}</span>
              </div>
              <div className="result-item">
                <span className="label">Confidence:</span>
                <span className="value">
                  {(result.confidence * 100).toFixed(2)}%
                </span>
              </div>
              <div className="confidence-bar-container">
                <div
                  className="confidence-bar"
                  style={{ width: `${result.confidence * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
