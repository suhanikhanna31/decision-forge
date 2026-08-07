import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function Register() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({
    username: "",
    email: "",
    password: "",
    role: "analyst",
    organization: "",
  });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  function update(field) {
    return (e) => setForm({ ...form, [field]: e.target.value });
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await register(form);
      navigate("/");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-card">
      <h1>Create account</h1>
      <p className="subtitle">Join DecisionForge</p>
      <form onSubmit={handleSubmit}>
        <label>Username</label>
        <input value={form.username} onChange={update("username")} required />
        <label>Email</label>
        <input type="email" value={form.email} onChange={update("email")} required />
        <label>Password</label>
        <input type="password" value={form.password} onChange={update("password")} minLength={8} required />
        <label>Role</label>
        <select value={form.role} onChange={update("role")}>
          <option value="viewer">Viewer</option>
          <option value="analyst">Analyst</option>
          <option value="admin">Admin</option>
        </select>
        <label>Organization</label>
        <input value={form.organization} onChange={update("organization")} placeholder="Optional" />
        {error && <p className="error">{error}</p>}
        <button type="submit" disabled={busy}>
          {busy ? "Creating..." : "Create account"}
        </button>
      </form>
      <p className="switch">
        Already have an account? <Link to="/login">Sign in</Link>
      </p>
    </div>
  );
}
