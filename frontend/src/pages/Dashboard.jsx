import { useState } from "react";
import { ML_BASE, apiFetch } from "../api/client";

const DEFAULT_FEATURES = {
  tenure: 3,
  monthly_charges: 210,
  request_count_today: 2,
  login_attempts: 1,
};

const DECISION_COLORS = {
  INTERVENE: "#1b7a3c",
  DO_NOTHING: "#616161",
  FLAG: "#b3261e",
};

export default function Dashboard() {
  const [userId, setUserId] = useState("user_001");
  const [features, setFeatures] = useState(DEFAULT_FEATURES);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  function updateFeature(key) {
    return (e) => setFeatures({ ...features, [key]: Number(e.target.value) });
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setResult(null);
    setBusy(true);
    try {
      const res = await apiFetch(`${ML_BASE}/api/v1/decide`, {
        method: "POST",
        body: JSON.stringify({ user_id: userId, features }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Decision request failed");
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <h2>Make a decision</h2>
      <p className="subtitle">
        Submits user features to the ML decision engine (FastAPI) and logs the result to
        audit history (MongoDB) in the background.
      </p>

      <form className="decision-form" onSubmit={handleSubmit}>
        <label>User ID</label>
        <input value={userId} onChange={(e) => setUserId(e.target.value)} />

        <div className="grid">
          <div>
            <label>Tenure (months)</label>
            <input type="number" value={features.tenure} onChange={updateFeature("tenure")} />
          </div>
          <div>
            <label>Monthly charges ($)</label>
            <input
              type="number"
              value={features.monthly_charges}
              onChange={updateFeature("monthly_charges")}
            />
          </div>
          <div>
            <label>Requests today</label>
            <input
              type="number"
              value={features.request_count_today}
              onChange={updateFeature("request_count_today")}
            />
          </div>
          <div>
            <label>Login attempts</label>
            <input
              type="number"
              value={features.login_attempts}
              onChange={updateFeature("login_attempts")}
            />
          </div>
        </div>

        {error && <p className="error">{error}</p>}
        <button type="submit" disabled={busy}>
          {busy ? "Deciding..." : "Get decision"}
        </button>
      </form>

      {result && (
        <div className="result-card" style={{ borderColor: DECISION_COLORS[result.decision] }}>
          <div className="result-header">
            <span
              className="decision-badge"
              style={{ background: DECISION_COLORS[result.decision] }}
            >
              {result.decision}
            </span>
            <span className="latency">{result.latency_ms} ms</span>
          </div>
          <p className="reason">{result.reason}</p>
          <div className="metrics-row">
            <div>
              <span className="label">Expected value</span>
              <span className="value">${result.expected_value.toFixed(2)}</span>
            </div>
            <div>
              <span className="label">Churn probability</span>
              <span className="value">{(result.churn_probability * 100).toFixed(1)}%</span>
            </div>
            <div>
              <span className="label">Anomaly score</span>
              <span className="value">{result.anomaly_score.toFixed(3)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
