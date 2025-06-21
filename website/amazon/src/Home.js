import React, { useState, useEffect } from "react";
import "./Home.css";
import Product from "./Product";

function Home() {
  const [products, setProducts] = useState([]);

  return (
    <div className="Home">
      <div className="home-container">
        <img
          className="home-image"
          src="https://images-eu.ssl-images-amazon.com/images/G/02/digital/video/merch2016/Hero/Covid19/Generic/GWBleedingHero_ENG_COVIDUPDATE__XSite_1500x600_PV_en-GB._CB428684220_.jpg"
        />
        <div className="home_row">
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
            image="https://m.media-amazon.com/images/I/71FER1UJhcL._AC_SL1500_.jpg "
            rating={1}
            className="fake-prod"
            Unique_product_id={4}
          />

          <Product
            title="Amazon Basics E-300 Headphones"
            price={70}
            image="https://m.media-amazon.com/images/I/71VHRNgvpqL._AC_SL1500_.jpg "
            rating={5}
            Unique_product_id={2}
          />

          <Product
            title="Xu Direct In Line Headphones"
            price={24.99}
            image="https://m.media-amazon.com/images/I/71dtAOC-bLL._AC_SL1500_.jpg "
            rating={4}
            Unique_product_id={3}
          />
        </div>
        <div className="home_row">
          <Product
            className="fake-prod"
            title="In Ear Earphones Black-Green"
            price={49.99}
            image="https://m.media-amazon.com/images/I/618zves-P8L._AC_SL1500_.jpg "
            rating={4}
            Unique_product_id={1}
          />
        </div>
      </div>
    </div>
  );
}

export default Home;
