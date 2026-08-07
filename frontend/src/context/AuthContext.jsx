import { createContext, useContext, useEffect, useState } from "react";
import { AUTH_BASE, apiFetch, clearTokens, getTokens, setTokens } from "../api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  async function loadMe() {
    const { access } = getTokens();
    if (!access) {
      setLoading(false);
      return;
    }
    try {
      const res = await apiFetch(`${AUTH_BASE}/auth/me/`);
      if (res.ok) setUser(await res.json());
      else clearTokens();
    } catch {
      // apiFetch already redirects to /login on unrecoverable auth failure
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadMe();
  }, []);

  async function login(username, password) {
    const res = await fetch(`${AUTH_BASE}/auth/login/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Invalid username or password");
    }
    const data = await res.json();
    setTokens({ access: data.access, refresh: data.refresh });
    await loadMe();
  }

  async function register(payload) {
    const res = await fetch(`${AUTH_BASE}/auth/register/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(Object.values(err).flat().join(" ") || "Registration failed");
    }
    await login(payload.username, payload.password);
  }

  function logout() {
    clearTokens();
    setUser(null);
    window.location.href = "/login";
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
