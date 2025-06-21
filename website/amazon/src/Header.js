import React from "react";
import "./Header.css";
import SearchIcon from "@mui/icons-material/Search";
import ShoppingBasketIcon from "@mui/icons-material/ShoppingBasket";

function Header() {
  return (
    <div className="Header">
      <img
        className="header_logo"
        src="https://pngimg.com/uploads/amazon/small/amazon_PNG11.png"
      />

      <div className="header_search">
        <input className="header_searchInput" type="text" />
        <SearchIcon className="header_searchIcon" />
        {/* logo */}
      </div>
      <div className="header_nav">
        <div className="header_option">
          <span className="header_optionLineOne">Hello Guest</span>
          <span className="header_optionLineTwo">Sign In</span>
        </div>
        <div className="header_option">
          <span className="header_optionLineOne">Returns</span>
          <span className="header_optionLineTwo">& orders</span>
        </div>
        <div className="header_option">
          <span className="header_optionLineOne">Your</span>
          <span className="header_optionLineTwo">prime</span>
        </div>
        <div className="header_optionbasket">
          <ShoppingBasketIcon />
          <span
            className="header_optionLineTwo 
          header_basketcount"
          >
            0
          </span>
        </div>
      </div>
    </div>
  );
}

export default Header;
