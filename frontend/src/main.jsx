import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

import "./style.css";


function App() {

  const [health, setHealth] = useState(null);
  const [status, setStatus] = useState(null);

  useEffect(() => {

    fetch("http://127.0.0.1:8000/api/health")
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth(null));

    fetch("http://127.0.0.1:8000/api/output-status")
      .then((response) => response.json())
      .then(setStatus)
      .catch(() => setStatus(null));

  }, []);


  return (
    <div className="container">

      <h1>
        Business Entity Resolution
      </h1>

      <p>
        ML-based matching across multiple business data sources.
      </p>

      <section className="card">

        <h2>Backend Status</h2>

        {health ? (
          <p className="success">
            Backend: {health.status}
          </p>
        ) : (
          <p className="error">
            Backend unavailable
          </p>
        )}

      </section>


      <section className="card">

        <h2>Submission Status</h2>

        {status ? (

          <>
            <p>
              Matching results:{" "}
              {status.matching_results
                ? "Ready"
                : "Not generated"}
            </p>

            <p>
              Candidate pairs:{" "}
              {status.candidate_pairs
                ? "Ready"
                : "Not generated"}
            </p>
          </>

        ) : (

          <p>
            Waiting for backend...
          </p>

        )}

      </section>


      <section className="card">

        <h2>Pipeline</h2>

        <ol>
          <li>Load TSV data</li>
          <li>Normalize business records</li>
          <li>Generate candidate pairs</li>
          <li>Calculate similarity features</li>
          <li>Train ML model</li>
          <li>Optimize F0.5 threshold</li>
          <li>Generate final matches</li>
          <li>Validate submission</li>
        </ol>

      </section>

    </div>
  );
}


createRoot(
  document.getElementById("root")
).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);