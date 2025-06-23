const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const cookieParser = require("cookie-parser"); // Import cookie-parser for handling cookies
const userRoutes = require("./routes/userRoutes"); // Import user routes

const app = express();
const port = 8000;

app.use(cors());
app.use(express.json());
app.use(cookieParser()); // Use cookie-parser middleware

// Updated MongoDB connection string with database name
mongoose
  .connect(
    "mongodb+srv://user_amazon:Sohamshb%4091@cluster0.swa3k.mongodb.net/amazon?retryWrites=true&w=majority"
  )
  .then(() => console.log("MongoDB connection successful"))
  .catch((err) => console.error("MongoDB connection error:", err));

app.use("/api/users", userRoutes); 

// Schema definition for products
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

// weights = {
//   'verification_check' : 0.25,   
//   'Sentiment_Score' : 0.15,      # VADER sentiment analysis
//   'Helpful_Score' : 0.15,
//   'vlm_score' : 0.20,
//   'llama_score' : 0.25           # LLaMA-2 text analysis
// }
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

// Route to get a product by Unique_product_id
app.get("/products/:Unique_product_id", async (req, res) => {
  const Unique_product_id = req.params.Unique_product_id;

  try {
    const products = await Product.find({ Unique_product_id });
    res.json(products);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Use user routes for user management
app.use("/api/users", userRoutes); // Mount user routes under /api/users

// Start the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});