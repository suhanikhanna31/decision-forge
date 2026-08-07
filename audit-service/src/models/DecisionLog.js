const mongoose = require("mongoose");

/**
 * One document per decision returned by ml-service (FastAPI).
 * ml-service POSTs here right after it computes a decision; the React
 * frontend reads history/analytics back out via GET endpoints.
 */
const DecisionLogSchema = new mongoose.Schema(
  {
    user_id: { type: String, index: true, default: null },
    decision: {
      type: String,
      enum: ["INTERVENE", "DO_NOTHING", "FLAG"],
      required: true,
      index: true,
    },
    reason: { type: String, default: "" },
    expected_value: { type: Number, default: 0 },
    churn_probability: { type: Number, default: 0 },
    anomaly_score: { type: Number, default: 0 },
    latency_ms: { type: Number, default: 0 },
    features: { type: mongoose.Schema.Types.Mixed, default: {} },
    // Who *requested* the decision (from the Django-issued JWT), not who
    // it's about. Useful for auditing which analyst triggered what.
    requested_by: { type: String, default: null },
  },
  { timestamps: { createdAt: "created_at", updatedAt: false } }
);

module.exports = mongoose.model("DecisionLog", DecisionLogSchema);
