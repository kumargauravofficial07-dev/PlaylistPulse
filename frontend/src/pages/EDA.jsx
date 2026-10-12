import { useEffect, useState } from "react";
import api from "../services/api";

function EDA() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.get("/analytics")
      .then((response) => {
        console.log("Backend response:", response.data);
        setData(response.data);
      })
      .catch((error) => {
        console.error(error);
        setError("Unable to connect to PlaylistPulse backend.");
      });
  }, []);

  return (
    <div className="page">
      <h1>Exploratory Data Analysis</h1>

      <p>
        Explore the Spotify track dataset and audio feature distributions.
      </p>

      {error && (
        <div className="card">
          <h3>Connection Error</h3>
          <p>{error}</p>
        </div>
      )}

      {!data && !error && (
        <div className="card">
          <h3>Loading...</h3>
          <p>Connecting to FastAPI...</p>
        </div>
      )}

      {data && (
        <div className="card">
          <h2>Backend Connected ✓</h2>

          <pre>
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

export default EDA;