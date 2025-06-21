import React from "react";
import "./Home.css";
import Product from "./Product";

function Home() {
  return (
    <div className="Home">
      <div className="home-container">
        {/* Premium Hero Section */}
        <section className="hero-section">
          <img
            className="home-image"
            src="https://assets.aboutamazon.com/1d/ef/4c54af0d4f00921dbfc3bb11d1cc/aa-nov2024-profitero-deals-announcement-hero-v1-2000x1125.jpg"
            alt="Premium Shopping Banner"
          />
          <div className="hero-overlay">
            <h1 className="hero-title">Discover Premium Products</h1>
            <p className="hero-subtitle">
              Curated selection. Trusted brands. Unmatched quality.
            </p>
            <a href="#shop-now" className="hero-cta">
              Shop Now
            </a>
          </div>
        </section>

        {/* Product Rows */}
        <div className="home_row" id="shop-now">
          <Product
            title="In Ear Black-Red Earphones"
            price={49.99}
            image="https://m.media-amazon.com/images/I/615SYkkPyDL._AC_SL1500_.jpg"
            rating={3}
            Unique_product_id={0}
          />
          <Product
            title="Amazon Fire-Tv Stick"
            price={100}
            image="https://m.media-amazon.com/images/I/7120GaDFhxL._AC_SL1000_.jpg"
            rating={5}
            Unique_product_id={5}
          />
        </div>
        <div className="home_row">
          <Product
            title="Portable Sub-Woofer"
            price={4.99}
            image="https://m.media-amazon.com/images/I/71FER1UJhcL._AC_SL1500_.jpg"
            rating={1}
            className="fake-prod"
            Unique_product_id={4}
          />
          <Product
            title="Amazon Basics E-300 Headphones"
            price={70}
            image="https://m.media-amazon.com/images/I/71VHRNgvpqL._AC_SL1500_.jpg"
            rating={5}
            Unique_product_id={2}
          />
          <Product
            title="Xu Direct In Line Headphones"
            price={24.99}
            image="https://m.media-amazon.com/images/I/71dtAOC-bLL._AC_SL1500_.jpg"
            rating={4}
            Unique_product_id={3}
          />
        </div>
        <div className="home_row">
          <Product
            className="fake-prod"
            title="In Ear Earphones Black-Green"
            price={49.99}
            image="https://m.media-amazon.com/images/I/618zves-P8L._AC_SL1500_.jpg"
            rating={4}
            Unique_product_id={1}
          />
        </div>
      </div>
    </div>
  );
}

export default Home;