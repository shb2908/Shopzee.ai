import React, { useState } from "react"; // Import useState
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import InputBase from "@mui/material/InputBase";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import Badge from "@mui/material/Badge";
import SearchIcon from "@mui/icons-material/Search";
import ShoppingBasketIcon from "@mui/icons-material/ShoppingBasket";
import Button from "@mui/material/Button";
import { styled, alpha } from "@mui/material/styles";
import { Link } from "react-router-dom";

const Search = styled("div")(({ theme }) => ({
  position: "relative",
  borderRadius: theme.shape.borderRadius,
  backgroundColor: alpha(theme.palette.common.white, 0.15),
  "&:hover": {
    backgroundColor: alpha(theme.palette.common.white, 0.25),
  },
  marginLeft: 0,
  width: "100%",
  [theme.breakpoints.up("sm")]: {
    marginLeft: theme.spacing(1),
    width: "auto",
  },
}));

const SearchIconWrapper = styled("div")(({ theme }) => ({
  padding: theme.spacing(0, 2),
  height: "100%",
  position: "absolute",
  pointerEvents: "none",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
}));

const StyledInputBase = styled(InputBase)(({ theme }) => ({
  color: "inherit",
  "& .MuiInputBase-input": {
    padding: theme.spacing(1, 1, 1, 0),
    paddingLeft: `calc(1em + ${theme.spacing(4)})`,
    transition: theme.transitions.create("width"),
    width: "100%",
    [theme.breakpoints.up("md")]: {
      width: "30ch",
    },
  },
}));

function Header() {
  const [searchQuery, setSearchQuery] = useState(""); // State for search input

  const handleSearch = (event) => {
    event.preventDefault(); // Prevent default form submission
    // Implement your search logic here
    console.log("Searching for:", searchQuery);
    // You can call a search function or update the state in a parent component
  };

  return (
    <AppBar position="static" color="default" elevation={2} sx={{ background: "linear-gradient(90deg, #232526 0%, #414345 100%)" }}>
      <Toolbar>
        <Box sx={{ display: "flex", alignItems: "center", mr: 3 }}>
          <img
            src="https://pngimg.com/uploads/amazon/small/amazon_PNG11.png"
            alt="Amazon Logo"
            style={{ height: 40, marginRight: 12 }}
          />
          <Typography variant="h6" noWrap sx={{ fontWeight: 700, letterSpacing: 2, color: "#ff9900" }}>
            Amazon Adam
          </Typography>
        </Box>
        <form onSubmit={handleSearch}> {/* Add form for search */}
          <Search>
            <SearchIconWrapper>
              <SearchIcon />
            </SearchIconWrapper>
            <StyledInputBase
              placeholder="Search products…"
              inputProps={{ "aria-label": "search" }}
              value={searchQuery} // Bind input value to state
              onChange={(e) => setSearchQuery(e.target.value)} // Update state on input change
            />
          </Search>
        </form>
        <Box sx={{ flexGrow: 1 }} />
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <Button color="inherit" sx={{ color: "#fff", textTransform: "none", fontWeight: 500 }}>
            Hello, Guest
          </Button>
          <Button color="inherit" sx={{ color: "#fff", textTransform: "none", fontWeight: 500 }}>
            Returns & Orders
          </Button>
          <Button color="inherit" sx={{ color: "#fff", textTransform: "none", fontWeight: 500 }}>
            Your Prime
          </Button>
          <IconButton size="large" aria-label="show basket items" color="inherit">
            <Badge badgeContent={0} color="warning">
              <ShoppingBasketIcon />
            </Badge>
          </IconButton>
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <Link to="/dashboard" style={{ color: "#fff", textDecoration: "none", fontWeight: 500 }}>
            Dashboard
          </Link>
          <Link to="/orders" style={{ color: "#fff", textDecoration: "none", fontWeight: 500 }}>
            Orders
          </Link>
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Header;