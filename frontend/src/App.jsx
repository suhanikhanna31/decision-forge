import { NavLink, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import ProtectedRoute from "./context/ProtectedRoute";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import Login from "./pages/Login";
import Register from "./pages/Register";

function Shell({ children }) {
  const { user, logout } = useAuth();
  return (
    <div className="shell">
      <header className="topbar">
        <span className="brand">DecisionForge</span>
        {user && (
          <nav>
            <NavLink to="/" end>
              Dashboard
            </NavLink>
            <NavLink to="/history">History</NavLink>
          </nav>
        )}
        {user && (
          <div className="user-menu">
            <span>
              {user.username} · {user.role}
            </span>
            <button onClick={logout}>Logout</button>
          </div>
        )}
      </header>
      <main>{children}</main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Shell>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/history"
            element={
              <ProtectedRoute>
                <History />
              </ProtectedRoute>
            }
          />
        </Routes>
      </Shell>
    </AuthProvider>
  );
}
