import React from "react";
import "./Product.css";
import { Link } from "react-router-dom";

function Product({
  className,
  title,
  image,
  price,
  rating,
  Unique_product_id,
}) {
  const productClass = className ? `${className} product` : "product";
  return (
    <div className={productClass}>
      <div className="product_info">
        <p> {title}</p>
        <p className="product_price">
          <small>$</small>
          <strong>{price}</strong>
        </p>
        <div className="product_rating">
          {Array(rating)
            .fill()
            .map((_, i) => (
              <p>⭐️</p>
            ))}
        </div>
      </div>
      <img className="product_img" src={image} />

      <Link to={`/product/${Unique_product_id}`}>
        <button className="Product_button">View Listing</button>
      </Link>
    </div>
  );
}

export default Product;
