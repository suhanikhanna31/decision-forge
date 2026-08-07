const jwt = require("jsonwebtoken");

// SimpleJWT (Django) signs access tokens with HS256 using DJANGO_SECRET_KEY.
// Sharing that secret with this service lets us verify tokens without an
// extra network round-trip to auth-service on every request.
const SECRET = process.env.DJANGO_SECRET_KEY || "dev-insecure-secret-key-change-me";

function requireAuth(req, res, next) {
  const header = req.headers.authorization || "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : null;

  if (!token) {
    return res.status(401).json({ error: "Missing Authorization: Bearer <token> header" });
  }

  try {
    const payload = jwt.verify(token, SECRET, { algorithms: ["HS256"] });
    if (payload.token_type && payload.token_type !== "access") {
      return res.status(401).json({ error: "Refresh tokens cannot be used to authenticate requests" });
    }
    req.user = {
      id: payload.user_id,
      username: payload.username,
      role: payload.role,
      organization: payload.organization,
    };
    next();
  } catch (err) {
    return res.status(401).json({ error: "Invalid or expired token", detail: err.message });
  }
}

// Some routes (e.g. the internal log-write endpoint hit by ml-service) use
// a shared service key instead of a user JWT.
function requireServiceKey(req, res, next) {
  const key = req.headers["x-service-key"];
  if (key && key === (process.env.INTERNAL_SERVICE_KEY || "dev-internal-key")) {
    return next();
  }
  return requireAuth(req, res, next);
}

module.exports = { requireAuth, requireServiceKey };
