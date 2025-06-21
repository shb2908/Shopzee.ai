import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import "./Productlisting.css";
import SortIcon from "@mui/icons-material/Sort";
import FilterListIcon from "@mui/icons-material/FilterList";
import VerifiedIcon from "@mui/icons-material/Verified";
import StarIcon from "@mui/icons-material/Star";
import StarBorderIcon from "@mui/icons-material/StarBorder";

function Productlisting() {
  const { Unique_product_id } = useParams();
  const navigate = useNavigate();
  const [products, setProducts] = useState(null);
  const [sortedReviews, setSortedReviews] = useState([]);
  const [isSorted, setIsSorted] = useState(false);
  const [isFiltered, setIsFiltered] = useState(false);

  const productList = [
    { id: "0", name: "In Ear Black-Red Earphones" },
    { id: "1", name: "In Ear Earphones Black-Green" },
    { id: "2", name: "Amazon Basics E-300 Headphones" },
    { id: "3", name: "Xu Direct In Line Headphones" },
    { id: "4", name: "Portable Sub-Woofer" },
    { id: "5", name: "Amazon Fire TV Stick" },
  ];

  useEffect(() => {
    fetch(`http://localhost:8000/products/${Unique_product_id}`)
      .then((response) => response.json())
      .then((data) => {
        setProducts(data);
        setSortedReviews(data);
      })
      .catch((error) => console.error("Error fetching data:", error));
  }, [Unique_product_id]);

  const sortReviewsByTrustIndex = () => {
    if (isSorted) {
      setSortedReviews(products);
      setIsSorted(false);
    } else {
      const sorted = [...sortedReviews].sort(
        (a, b) => 100 - b.FINAL_SCORE - (100 - a.FINAL_SCORE)
      );
      setSortedReviews(sorted);
      setIsSorted(true);
    }
  };

  const removeLowTrustIndexReviews = () => {
    if (isFiltered) {
      setSortedReviews(products);
      setIsFiltered(false);
    } else {
      const filtered = sortedReviews.filter(
        (review) => 100 - review.FINAL_SCORE >= 25
      );
      setSortedReviews(filtered);
      setIsFiltered(true);
    }
  };

  if (!products) {
    return (
      <div className="product-listing loading-premium">
        <div className="premium-loader"></div>
        <span>Loading premium experience...</span>
      </div>
    );
  }

  const isFake = products[0].Product_score > 70;
  const product = productList.find(
    (product) => product.id === Unique_product_id
  );

  const handleSerialNumberVerification = () => {
    navigate(`/serial-verification/${Unique_product_id}`);
  };

  // Helper for premium rating stars
  const renderStars = (count) => {
    return (
      <span className="premium-stars">
        {[...Array(5)].map((_, i) =>
          i < count ? (
            <StarIcon key={i} style={{ color: "#FFD700", fontSize: 22 }} />
          ) : (
            <StarBorderIcon key={i} style={{ color: "#FFD700", fontSize: 22 }} />
          )
        )}
      </span>
    );
  };

  return (
    <div className="product-listing">
      <div className="product-container">
        <div className="product-image-container">
          <img
            src={products[0].Photo_url}
            alt="Product"
            className="product-image"
          />
          {isFake ? (
            <span className="premium-badge fake">
              <VerifiedIcon style={{ color: "#e53935", marginRight: 6 }} />
              Suspected Fake
            </span>
          ) : (
            <span className="premium-badge genuine">
              <VerifiedIcon style={{ color: "#43a047", marginRight: 6 }} />
              Verified Genuine
            </span>
          )}
        </div>
        <div className="product-details">
          <h1 className="premium-title">
            {product ? product.name : "Product name not found"}
          </h1>
          <p className="desc">
            <strong>Description:</strong> {products[0].Description}
          </p>
          <p className="premium-price">
            <strong>Price:</strong> ${products[0].Price}
          </p>
          <p className={`prod-score ${isFake ? "fake" : ""}`}>
            <strong>Product Trust Index:</strong>{" "}
            {100 - products[0].Product_score}
          </p>
          <div className="progress-bar-prod">
            <div
              className="progress"
              style={{
                width: `${100 - products[0].Product_score}%`,
                background: (() => {
                  if (
                    100 - products[0].Product_score >= 0 &&
                    100 - products[0].Product_score <= 30
                  ) {
                    return "linear-gradient(90deg, #e53935 0%, #ffb347 100%)";
                  } else if (
                    100 - products[0].Product_score >= 31 &&
                    100 - products[0].Product_score <= 69
                  ) {
                    return "linear-gradient(90deg, #ffb347 0%, #ffd700 100%)";
                  } else if (
                    100 - products[0].Product_score >= 70 &&
                    100 - products[0].Product_score <= 100
                  ) {
                    return "linear-gradient(90deg, #43a047 0%, #ffd700 100%)";
                  }
                })(),
              }}
            ></div>
          </div>
          <button
            className="button serial"
            onClick={handleSerialNumberVerification}
          >
            Report Fake Product
          </button>
        </div>
      </div>
      <div className="button-container">
        <button
          className={`button ${isSorted ? "active" : ""}`}
          onClick={sortReviewsByTrustIndex}
        >
          Sort by Trust <SortIcon className="sorticon" />
        </button>
        <button
          className={`button ${isFiltered ? "active" : ""}`}
          onClick={removeLowTrustIndexReviews}
        >
          Trusted Reviews only <FilterListIcon className="filtericon" />
        </button>
      </div>
      <div className="reviews">
        {sortedReviews.map((product) => (
          <div key={product._id} className="product-card">
            <div className="product-info">
              <p>
                <strong>Review:</strong> {product.review_bold}
              </p>
              <p>{product.review}</p>
              <p>
                <strong>Rating:</strong> {renderStars(product.ratings)}
              </p>
              <p>
                <strong>Review by:</strong> {product.by} on {product.date}
              </p>
              <p>
                <strong>Helpful Votes:</strong> {product.helpful}
              </p>
              <p className="rev-score">
                <strong>Review Trust Index:</strong> {100 - product.FINAL_SCORE}
              </p>
              <div className="progress-bar">
                <div
                  className="progress"
                  style={{
                    width: `${100 - product.FINAL_SCORE}%`,
                    background: (() => {
                      if (
                        100 - product.FINAL_SCORE >= 0 &&
                        100 - product.FINAL_SCORE <= 25
                      ) {
                        return "linear-gradient(90deg, #e53935 0%, #ffb347 100%)";
                      } else if (
                        100 - product.FINAL_SCORE >= 26 &&
                        100 - product.FINAL_SCORE <= 74
                      ) {
                        return "linear-gradient(90deg, #ffb347 0%, #ffd700 100%)";
                      } else if (
                        100 - product.FINAL_SCORE >= 75 &&
                        100 - product.FINAL_SCORE <= 100
                      ) {
                        return "linear-gradient(90deg, #43a047 0%, #ffd700 100%)";
                      }
                    })(),
                  }}
                ></div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Productlisting;