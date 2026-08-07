require("dotenv").config();

const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const morgan = require("morgan");

const connectDB = require("./config/db");
const logsRouter = require("./routes/logs");

const app = express();
const PORT = process.env.PORT || 5000;

app.use(helmet());
app.use(
  cors({
    origin: (process.env.CORS_ALLOWED_ORIGINS || "http://localhost:5173,http://localhost:3000").split(","),
  })
);
app.use(express.json());
app.use(morgan("dev"));

app.get("/health", (req, res) => {
  res.json({ status: "healthy", service: "audit-service" });
});

app.use("/api/logs", logsRouter);

app.use((req, res) => {
  res.status(404).json({ error: "Not found" });
});

// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ error: "Internal server error" });
});

async function start() {
  await connectDB();
  app.listen(PORT, () => {
    console.log(`[audit-service] Listening on port ${PORT}`);
  });
}

start().catch((err) => {
  console.error("[audit-service] Failed to start:", err);
  process.exit(1);
});
