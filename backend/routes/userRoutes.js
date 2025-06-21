// backend/routes/userRoutes.js
const express = require("express");
const router = express.Router();
const User = require("../models/userModel"); // Adjust the path as necessary

// Route: Register a new user
router.post("/register", async (req, res) => {
  try {
    const { username, usermail, password } = req.body;

    // Validate required fields
    if (!username || !usermail || !password) {
      return res.status(400).json({ error: "All fields are required" });
    }

    // Create new user
    const newUser = new User({ username, usermail, password });
    await newUser.save();
    return res.status(201).json({ message: "User registered successfully", user: newUser });
  } catch (error) {
    console.error("Error registering user:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// Route: Login a user using username and password
router.post("/login", async (req, res) => {
  const { username, password } = req.body;
  try {
    const user = await User.findOne({ username });
    if (!user || user.password !== password) {
      return res.status(401).json({ error: "Invalid credentials" });
    }
    // Implement session or token logic here
    return res.status(200).json({ message: "Login successful", user });
  } catch (error) {
    console.error("Error logging in user:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// Route: Update user info (email, password, and optionally about)
router.put("/update", async (req, res) => {
  try {
    const { email, password, about } = req.body;
    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }

    // Retrieve user info from cookie
    const rawUserInfo = req.cookies.userInfo;
    if (!rawUserInfo) {
      return res.status(401).json({ error: "User not authenticated" });
    }
    const user = JSON.parse(rawUserInfo);
    const userId = user.userId;

    // Update user document
    const updatedUser = await User.findByIdAndUpdate(userId, { usermail: email, password, about }, { new: true });
    return res.status(200).json({ message: "User updated successfully", user: updatedUser });
  } catch (error) {
    console.error("Error updating user:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// Route: Add balance to the user's account
router.post("/add_balance", async (req, res) => {
  try {
    const { amount } = req.body;
    if (amount == null) {
      return res.status(400).json({ error: "Amount is required" });
    }

    // Retrieve user info from cookie
    const rawUserInfo = req.cookies.userInfo;
    if (!rawUserInfo) {
      return res.status(401).json({ error: "User not authenticated" });
    }
    const user = JSON.parse(rawUserInfo);
    const userId = user.userId;

    // Find the user and update balance
    const updatedUser = await User.findById(userId);
    if (!updatedUser) {
      return res.status(404).json({ error: "User not found" });
    }
    updatedUser.balance = (updatedUser.balance || 0) + Number(amount);
    await updatedUser.save();
    return res.status(200).json({ message: "Balance updated successfully", balance: updatedUser.balance });
  } catch (error) {
    console.error("Error updating balance:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// Route: Get balance for a given user ID
router.get("/get_bal", async (req, res) => {
  try {
    const { userId } = req.query;
    if (!userId) {
      return res.status(400).json({ error: "User ID is required" });
    }

    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }
    return res.status(200).json({ balance: user.balance });
  } catch (error) {
    console.error("Error fetching balance:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// Route: Donation request
router.post("/donation", async (req, res) => {
  try {
    const { userId } = req.query;
    if (!userId) {
      return res.status(400).json({ error: "User ID is required" });
    }
    // Implement your logic to set donation status
    await User.findByIdAndUpdate(userId, { donationEligible: true }); // Example field
    return res.status(200).json({ message: "User is eligible for donation request" });
  } catch (error) {
    console.error("Error setting donation status", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

module.exports = router;