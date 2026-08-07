const express = require("express");
const DecisionLog = require("../models/DecisionLog");
const { requireAuth, requireServiceKey } = require("../middleware/auth");

const router = express.Router();

/**
 * POST /api/logs
 * Called by ml-service right after it makes a decision (service key auth),
 * or by the frontend if it wants to log client-side (user JWT auth).
 */
router.post("/", requireServiceKey, async (req, res) => {
  try {
    const {
      user_id,
      decision,
      reason,
      expected_value,
      churn_probability,
      anomaly_score,
      latency_ms,
      features,
    } = req.body;

    if (!decision) {
      return res.status(400).json({ error: "decision is required" });
    }

    const log = await DecisionLog.create({
      user_id,
      decision,
      reason,
      expected_value,
      churn_probability,
      anomaly_score,
      latency_ms,
      features,
      requested_by: req.user ? req.user.username : "ml-service",
    });

    res.status(201).json(log);
  } catch (err) {
    res.status(500).json({ error: "Failed to save decision log", detail: err.message });
  }
});

/**
 * GET /api/logs?user_id=&decision=&page=&limit=
 * Paginated, filterable decision history. Requires a logged-in user.
 */
router.get("/", requireAuth, async (req, res) => {
  try {
    const { user_id, decision } = req.query;
    const page = Math.max(parseInt(req.query.page) || 1, 1);
    const limit = Math.min(parseInt(req.query.limit) || 25, 100);

    const filter = {};
    if (user_id) filter.user_id = user_id;
    if (decision) filter.decision = decision;

    const [items, total] = await Promise.all([
      DecisionLog.find(filter)
        .sort({ created_at: -1 })
        .skip((page - 1) * limit)
        .limit(limit),
      DecisionLog.countDocuments(filter),
    ]);

    res.json({ items, total, page, limit, pages: Math.ceil(total / limit) });
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch logs", detail: err.message });
  }
});

/**
 * GET /api/logs/stats
 * Aggregate counts per decision type + average expected value, for a
 * dashboard summary widget.
 */
router.get("/stats", requireAuth, async (req, res) => {
  try {
    const stats = await DecisionLog.aggregate([
      {
        $group: {
          _id: "$decision",
          count: { $sum: 1 },
          avg_expected_value: { $avg: "$expected_value" },
          avg_latency_ms: { $avg: "$latency_ms" },
        },
      },
    ]);

    const total = await DecisionLog.countDocuments();

    res.json({ total, by_decision: stats });
  } catch (err) {
    res.status(500).json({ error: "Failed to compute stats", detail: err.message });
  }
});

module.exports = router;
