import "./App.css";
import Header from "./Header";
import Home from "./Home";
import Productlisting from "./Productlisting";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import React, { useState, useEffect } from "react";
import SerialVerification from "./SerialVerification.js";
import Container from "@mui/material/Container";
import Box from "@mui/material/Box";
import CssBaseline from "@mui/material/CssBaseline";
import Dashboard from "./component/Dashboard";

function App() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:8080/")
      .then((res) => res.json())
      .then((data) => {
        setData(data);
        console.log(data);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
      });
  }, []);

  return (
    <Router>
      <CssBaseline />
      <Header />
      <Container maxWidth="md">
        <Box className="app" sx={{ mt: 4 }}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route
              path="/serial-verification/:Unique_product_id"
              element={<SerialVerification />}
            />
            <Route
              path="/product/:Unique_product_id"
              element={<Productlisting Data={data} />}
            />
          </Routes>
        </Box>
      </Container>
    </Router>
  );
}

export default App;