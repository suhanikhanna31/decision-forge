# DecisionForge — Full-Stack Architecture

DecisionForge started as a single FastAPI service (`ml-service/`) implementing
an ROI-aware ML decision engine. This layer adds the rest of a real
deployment around it:

```
┌────────────┐      JWT login       ┌──────────────────┐
│  frontend  │ ───────────────────► │   auth-service    │  Django + DRF
│  (React)   │ ◄─────────────────── │  users / roles /  │  + SimpleJWT
│  :3000     │      access token     │   admin panel      │  :8001
└─────┬──────┘                       └──────────────────┘
      │  Bearer <JWT>
      ▼
┌──────────────────┐   background POST   ┌──────────────────────┐
│    ml-service      │ ───────────────────► │    audit-service       │  Node/Express
│  FastAPI (existing) │                     │  decision history/      │  + MongoDB
│  ROI / churn / ...  │                     │  stats                  │  :5000
│  :8000              │                     └──────────────────────┘
└──────────────────┘   Bearer <JWT> ◄───────────────┘
      ▲                (frontend reads history directly)
      │
   same JWT, verified locally by both services
   (shared DJANGO_SECRET_KEY, HS256, no callback to Django)
```

## Why this split

| Service | Owns | Why here |
|---|---|---|
| **auth-service** (Django) | Users, roles, JWT issuance, admin panel | Django's batteries — admin UI, auth, migrations — are the fastest way to get a solid accounts system, and it's a natural home for staff/ops tooling later. |
| **ml-service** (FastAPI, unchanged core logic) | ML decisions (churn, anomaly, ROI) | Left as-is; only added JWT verification, CORS, and a fire-and-forget log to audit-service. |
| **audit-service** (Node/Express + MongoDB) | Decision history, analytics/aggregates | Document-shaped, high-write, read-light log data — a good fit for Mongo, and keeps this off Django's relational DB. |
| **frontend** (React) | Single UI | Talks to all three: logs in via Django, submits decisions to FastAPI, reads history from Express. |

**Trust model:** auth-service is the only place a password ever goes. It
mints a JWT (HS256) with `role`/`organization`/`username` claims. ml-service
and audit-service both hold the same signing secret (`DJANGO_SECRET_KEY`) and
verify tokens locally — no network call back to Django per request.

## Quick start (Docker)

```bash
cp .env.example .env   # edit DJANGO_SECRET_KEY / INTERNAL_SERVICE_KEY
docker compose up --build
```

- Frontend: http://localhost:3000
- Django admin: http://localhost:8001/admin (create a superuser first: `docker compose exec auth-service python manage.py createsuperuser`)
- FastAPI docs: http://localhost:8000/docs
- Audit API health: http://localhost:5000/health

## Quick start (local, no Docker)

Each service has its own README-equivalent in its `.env.example`. In four terminals:

```bash
# 1. auth-service
cd auth-service && pip install -r requirements.txt
python manage.py migrate && python manage.py runserver 8001

# 2. audit-service (needs a local MongoDB on :27017, or set MONGO_URI)
cd audit-service && npm install && npm run dev

# 3. ml-service
cd ml-service && pip install -r requirements-ai.txt
export DJANGO_SECRET_KEY=dev-insecure-secret-key-change-me
uvicorn app.ai.api:app --reload --port 8000

# 4. frontend
cd frontend && npm install && npm run dev
```

Visit http://localhost:5173.

## Request flow for a decision

1. User logs in on the React app → POST `auth-service` `/api/auth/login/` → JWT.
2. React calls `ml-service` `POST /api/v1/decide` with `Authorization: Bearer <JWT>`.
3. `ml-service` verifies the JWT locally, runs the existing churn/anomaly/ROI
   pipeline (`app/core/decision_engine.py`, untouched), returns the decision,
   and in a `BackgroundTask` POSTs the result to `audit-service`
   `/api/logs` (using an internal service key, not the user's JWT).
4. React's History page calls `audit-service` `GET /api/logs` /
   `GET /api/logs/stats` (with the user's JWT) to render history and
   aggregate counts.

## What each service still needs before production

- **auth-service**: swap SQLite for Postgres, set a real `DJANGO_SECRET_KEY`, put behind HTTPS.
- **ml-service**: unchanged ROI/churn logic is still trained on inline demo
  data at startup (see `README.md` in that folder) — swap for real training data.
- **audit-service**: point `MONGO_URI` at a managed Mongo instance (Atlas, etc.), rotate `INTERNAL_SERVICE_KEY`.
- **frontend**: set the three `VITE_*_URL` build args to real hostnames.

See each subfolder for its own `.env.example`.
