import { useEffect, useState } from "react";
import { AUDIT_BASE, apiFetch } from "../api/client";

export default function History() {
  const [logs, setLogs] = useState([]);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    setError("");
    try {
      const [logsRes, statsRes] = await Promise.all([
        apiFetch(`${AUDIT_BASE}/logs?limit=25`),
        apiFetch(`${AUDIT_BASE}/logs/stats`),
      ]);
      if (!logsRes.ok || !statsRes.ok) throw new Error("Failed to load audit history");
      const logsData = await logsRes.json();
      setLogs(logsData.items || []);
      setStats(await statsRes.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="page">
      <h2>Decision history</h2>
      <p className="subtitle">Pulled from audit-service (Node/Express + MongoDB).</p>

      {error && <p className="error">{error}</p>}
      {loading && <p>Loading…</p>}

      {stats && (
        <div className="stats-row">
          <div className="stat-box">
            <span className="label">Total decisions</span>
            <span className="value">{stats.total}</span>
          </div>
          {stats.by_decision.map((s) => (
            <div className="stat-box" key={s._id}>
              <span className="label">{s._id}</span>
              <span className="value">{s.count}</span>
            </div>
          ))}
        </div>
      )}

      <table className="log-table">
        <thead>
          <tr>
            <th>User</th>
            <th>Decision</th>
            <th>Expected value</th>
            <th>Reason</th>
            <th>When</th>
          </tr>
        </thead>
        <tbody>
          {logs.map((log) => (
            <tr key={log._id}>
              <td>{log.user_id || "—"}</td>
              <td>
                <span className={`badge badge-${log.decision}`}>{log.decision}</span>
              </td>
              <td>${Number(log.expected_value).toFixed(2)}</td>
              <td className="reason-cell">{log.reason}</td>
              <td>{new Date(log.created_at).toLocaleString()}</td>
            </tr>
          ))}
          {!loading && logs.length === 0 && (
            <tr>
              <td colSpan={5} className="empty">
                No decisions logged yet. Make one from the Dashboard.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
