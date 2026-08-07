// Base URLs: dev mode uses the Vite proxy paths (see vite.config.js) so no
// CORS headaches locally; production build points straight at each service
// via env vars baked in at build time (see docker-compose.yml / .env).
const isDev = import.meta.env.DEV;

export const AUTH_BASE = isDev ? "/auth-api" : import.meta.env.VITE_AUTH_SERVICE_URL + "/api";
export const ML_BASE = isDev ? "/ml-api" : import.meta.env.VITE_ML_SERVICE_URL;
export const AUDIT_BASE = isDev ? "/audit-api" : import.meta.env.VITE_AUDIT_SERVICE_URL + "/api";

const ACCESS_KEY = "df_access_token";
const REFRESH_KEY = "df_refresh_token";

export function getTokens() {
  return {
    access: localStorage.getItem(ACCESS_KEY),
    refresh: localStorage.getItem(REFRESH_KEY),
  };
}

export function setTokens({ access, refresh }) {
  if (access) localStorage.setItem(ACCESS_KEY, access);
  if (refresh) localStorage.setItem(REFRESH_KEY, refresh);
}

export function clearTokens() {
  localStorage.removeItem(ACCESS_KEY);
  localStorage.removeItem(REFRESH_KEY);
}

async function refreshAccessToken() {
  const { refresh } = getTokens();
  if (!refresh) throw new Error("No refresh token available");

  const res = await fetch(`${AUTH_BASE}/auth/refresh/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh }),
  });
  if (!res.ok) throw new Error("Session expired, please log in again");

  const data = await res.json();
  setTokens({ access: data.access, refresh: data.refresh || refresh });
  return data.access;
}

/**
 * Authenticated fetch: attaches the access token, and on a single 401
 * transparently refreshes and retries once before giving up.
 */
export async function apiFetch(url, options = {}, retry = true) {
  const { access } = getTokens();
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
    ...(access ? { Authorization: `Bearer ${access}` } : {}),
  };

  const res = await fetch(url, { ...options, headers });

  if (res.status === 401 && retry) {
    try {
      await refreshAccessToken();
      return apiFetch(url, options, false);
    } catch {
      clearTokens();
      window.location.href = "/login";
      throw new Error("Session expired");
    }
  }

  return res;
}
