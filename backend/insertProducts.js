// const mongoose = require("mongoose");

// // Connect to MongoDB
// mongoose.connect(
//   "mongodb+srv://user_amazon:Sohamshb%4091@cluster0.swa3k.mongodb.net/amazon?retryWrites=true&w=majority",
//   { useNewUrlParser: true, useUnifiedTopology: true }
// );

// // Define the schema
// const productSchema = new mongoose.Schema({
//   Product_link: String,
//   Photo_url: String,
//   Description: String,
//   Unique_product_id: String,
//   Price: Number,
//   Product_score: Number,
//   review_bold: String,
//   ratings: Number,
//   review: String,
//   verified: Boolean,
//   date: String,
//   by: String,
//   helpful: Number,
//   FINAL_SCORE: Number,
// });

// // Create the model
// const Product = mongoose.model("Product", productSchema, "amazon");

// // Example products to insert
// // const products = [
// //   {
// //     Product_link: "http://localhost:8000/products/1",
// //     Photo_url: "https://example.com/photo1.jpg",
// //     Description: "Sample product 1",
// //     Unique_product_id: "0",
// //     Price: 49.99,
// //     Product_score: 80,
// //     review_bold: "Great product!",
// //     ratings: 5,
// //     review: "I loved this product.",
// //     verified: true,
// //     date: "2025-06-21",
// //     by: "Alice",
// //     helpful: 10,
// //     FINAL_SCORE: 20,
// //   },
// //   {
// //     Product_link: "http://localhost:8000/products/1",
// //     Photo_url: "https://example.com/photo1.jpg",
// //     Description: "Sample product 1",
// //     Unique_product_id: "0",
// //     Price: 49.99,
// //     Product_score: 80,
// //     review_bold: "Great product!",
// //     ratings: 5,
// //     review: "I loved this product.",
// //     verified: true,
// //     date: "2025-06-21",
// //     by: "Alice",
// //     helpful: 10,
// //     FINAL_SCORE: 20,
// //   },
// //   {
// //     Product_link: "http://localhost:8000/products/2",
// //     Photo_url: "https://example.com/photo2.jpg",
// //     Description: "Sample product 2",
// //     Unique_product_id: "1",
// //     Price: 4.99,
// //     Product_score: 60,
// //     review_bold: "Good value.",
// //     ratings: 4,
// //     review: "Worth the price.",
// //     verified: false,
// //     date: "2025-06-20",
// //     by: "Bob",
// //     helpful: 5,
// //     FINAL_SCORE: 40,
// //   },
// //   {
// //     Product_link: "http://localhost:8000/products/3",
// //     Photo_url: "https://example.com/photo3.jpg",
// //     Description: "Sample product 3",
// //     Unique_product_id: "2",
// //     Price: 70,
// //     Product_score: 90,
// //     review_bold: "Excellent!",
// //     ratings: 5,
// //     review: "Highly recommend.",
// //     verified: true,
// //     date: "2025-06-19",
// //     by: "Carol",
// //     helpful: 8,
// //     FINAL_SCORE: 10,
// //   },
// //   {
// //     Product_link: "http://localhost:8000/products/3",
// //     Photo_url: "https://example.com/photo4.jpg",
// //     Description: "Sample product 4",
// //     Unique_product_id: "3",
// //     Price: 24.99,
// //     Product_score: 50,
// //     review_bold: "Decent product.",
// //     ratings: 3,
// //     review: "It's okay.",
// //     verified: false,
// //     date: "2025-06-18",
// //     by: "Dave",
// //     helpful: 2,
// //     FINAL_SCORE: 50,
// //   },
// //   {
// //     Product_link: "http://localhost:8000/products/4",
// //     Photo_url: "https://example.com/photo5.jpg",
// //     Description: "Sample product 5",
// //     Unique_product_id: "4",
// //     Price: 49.99,
// //     Product_score: 70,
// //     review_bold: "Not bad.",
// //     ratings: 4,
// //     review: "Met my expectations.",
// //     verified: true,
// //     date: "2025-06-17",
// //     by: "Eve",
// //     helpful: 3,
// //     FINAL_SCORE: 30,
// //   },
// // ];

// const products = [
//   {
//     Product_link: "http://localhost:8000/products/1",
//     Photo_url: "https://m.media-amazon.com/images/I/615SYkkPyDL._AC_SL1500_.jpg",
//     Description: "Superior Sound Quality: Enjoy crystal clear audio with deep bass and crisp highs. These earphones are engineered to deliver an immersive sound experience, making your music, podcasts, and calls sound better than ever.",
//     Unique_product_id: "0",
//     Price: 99.99,
//     Product_score: 80,
//     review_bold: "Awesome product!",
//     ratings: 5,
//     review: "I love this product very much.",
//     verified: true,
//     date: "2025-06-28",
//     by: "Brown",
//     helpful: 30,
//     FINAL_SCORE: 80,
//   },];

// // Insert products
// Product.insertMany(products)
//   .then(() => {
//     console.log("Products inserted successfully!");
//     mongoose.connection.close();
//   })
//   .catch((err) => {
//     console.error("Error inserting products:", err);
//     mongoose.connection.close();
//   });

const mongoose = require("mongoose");
const fs = require("fs");
const csv = require("csv-parser");

// Connect to MongoDB
mongoose.connect(
  "mongodb+srv://user_amazon:Sohamshb%4091@cluster0.swa3k.mongodb.net/amazon?retryWrites=true&w=majority",
  { useNewUrlParser: true, useUnifiedTopology: true }
);

// Define the schema
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

// Create the model
const Product = mongoose.model("Product", productSchema, "amazon");

// Read CSV and update database
const results = [];

fs.createReadStream("/Users/sohambose/Amazon_Adam-1/backend/FinalFrontenddb_updated.csv") // Ensure path is correct
  .pipe(csv())
  .on("data", (data) => {
    // Optional: convert verified to Boolean
    data.verified = data.verified === "true" || data.verified === "1";

    // Optional: convert numerical fields
    data.Price = parseFloat(data.Price);
    data.Product_score = parseFloat(data.Product_score);
    data.ratings = parseFloat(data.ratings);
    data.helpful = parseInt(data.helpful);
    data.FINAL_SCORE = parseFloat(data.FINAL_SCORE);

    results.push(data);
  })
  .on("end", async () => {
    try {
      // Upsert (update if exists, otherwise insert)
      for (const item of results) {
        await Product.updateOne(
          { Unique_product_id: item.Unique_product_id },
          { $set: item },
          { upsert: true }
        );
      }
      console.log("✅ Database updated successfully.");
      mongoose.connection.close();
    } catch (error) {
      console.error("❌ Error updating database:", error);
    }
  });
