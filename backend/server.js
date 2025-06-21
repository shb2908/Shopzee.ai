const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const app = express();
const port = 8000;

app.use(cors());
app.use(express.json());

// Updated MongoDB connection string with database name
mongoose
  .connect(
    "mongodb+srv://user_amazon:Sohamshb%4091@cluster0.swa3k.mongodb.net/amazon?retryWrites=true&w=majority",
  )
  .then(() => console.log("MongoDB connection successful"))
  .catch((err) => console.error("MongoDB connection error:", err));

// Schema definition
const productSchema = new mongoose.Schema({
  Product_link: String,
  Photo_url: String,
  Description: String,
  Unique_product_id: String,
  Price: Number,
  Product_score: Number,
  review_bold: String,
  ratings: Number,
  review: String,
  verified: Boolean,
  date: String,
  by: String,
  helpful: Number,
  FINAL_SCORE: Number,
});

// Explicitly specifying the collection name
const Product = mongoose.model("Product", productSchema, "amazon");

// Route to get products
app.get("/products", async (req, res) => {
  try {
    const products = await Product.find();
    res.json(products);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

app.get("/products/:Unique_product_id", async (req, res) => {
  const Unique_product_id = req.params.Unique_product_id;

  const products = await Product.find({ Unique_product_id });

  res.json(products);
});

// Start the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
