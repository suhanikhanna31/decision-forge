const mongoose = require("mongoose");

async function connectDB() {
  const uri = process.env.MONGO_URI || "mongodb://localhost:27017/decisionforge";

  mongoose.set("strictQuery", true);

  await mongoose.connect(uri);
  console.log(`[audit-service] Connected to MongoDB at ${uri}`);

  mongoose.connection.on("error", (err) => {
    console.error("[audit-service] MongoDB connection error:", err);
  });
}

module.exports = connectDB;
