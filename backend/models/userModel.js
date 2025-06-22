// backend/models/userModel.js
const mongoose = require("mongoose");

const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  usermail: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  balance: { type: Number, default: 0 },
  // Add any other fields you need
});

const User = mongoose.model("User", userSchema);

const addUser = async (username, usermail, password) => {
  const user = new User({ username, usermail, password });
  await user.save();
  return user;
};

const loginUserByUsername = async (req, res) => {
  const { username, password } = req.body;
  const user = await User.findOne({ username });
  if (!user || user.password !== password) {
    return res.status(401).json({ error: "Invalid credentials" });
  }
  // Implement session or token logic here
  return res.status(200).json({ message: "Login successful", user });
};

const updateUser = async (userId, email, password, about) => {
  const user = await User.findByIdAndUpdate(userId, { usermail: email, password, about }, { new: true });
  return user;
};

const updateUserFlag = async (userId) => {
  // Implement your logic to update user flag
};

module.exports = { addUser, loginUserByUsername, updateUser, updateUserFlag };