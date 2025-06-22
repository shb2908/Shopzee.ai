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

  /* ---------------- state ---------------- */
  const [productDocs, setProductDocs] = useState(null); // API payload (array)
  const [reviews, setReviews]         = useState([]);   // original order
  const [view, setView]               = useState([]);   // list shown
  const [isSorted, setIsSorted]       = useState(false);
  const [isFiltered, setIsFiltered]   = useState(false);

  /* hard-coded names for the demo header */
  const productList = [
    { id: "0", name: "In Ear Black-Red Earphones" },
    { id: "1", name: "In Ear Earphones Black-Green" },
    { id: "2", name: "Amazon Basics E-300 Headphones" },
    { id: "3", name: "Xu Direct In-Line Headphones" },
    { id: "4", name: "Portable Sub-Woofer" },
    { id: "5", name: "Amazon Fire TV Stick" },
  ];

  /* ---------------- fetch ---------------- */
  useEffect(() => {
    fetch(`http://localhost:8000/products/${Unique_product_id}`)
      .then((res) => res.json())
      .then((data) => {
        setProductDocs(data);  // first element = product summary
        setReviews(data);      // keep as canonical order
        setView(data);         // initial list on screen
      })
      .catch((err) => console.error("Error fetching data:", err));
  }, [Unique_product_id]);

  /* ---------------- helpers ---------------- */
  const trustIndex = (doc) => 100 - doc.FINAL_SCORE; // higher = better

  /** toggle sort by trust-index (desc) */
  const toggleSort = () => {
    if (isSorted) {
      const base = isFiltered ? reviews.filter((r) => trustIndex(r) >= 25) : reviews;
      setView(base);
    } else {
      setView([...view].sort((a, b) => trustIndex(b) - trustIndex(a)));
    }
    setIsSorted(!isSorted);
  };

  /** toggle "trusted reviews only" (TI ≥ 25) */
  const toggleFilter = () => {
    if (isFiltered) {
      const base = [...reviews];
      setView(isSorted ? base.sort((a, b) => trustIndex(b) - trustIndex(a)) : base);
    } else {
      setView(view.filter((r) => trustIndex(r) >= 25));
    }
    setIsFiltered(!isFiltered);
  };

  const handleSerialVerification = () =>
    navigate(`/serial-verification/${Unique_product_id}`);

  const renderStars = (n) => (
    <span className="premium-stars">
      {[...Array(5)].map((_, i) =>
        i < n ? (
          <StarIcon key={i} style={{ color: "#FFD700", fontSize: 22 }} />
        ) : (
          <StarBorderIcon key={i} style={{ color: "#FFD700", fontSize: 22 }} />
        )
      )}
    </span>
  );

  /* ---------------- loading ---------------- */
  if (!productDocs) {
    return (
      <div className="product-listing loading-premium">
        <div className="premium-loader" />
        <span>Loading premium experience…</span>
      </div>
    );
  }

  /* ---------------- main product header ---------------- */
  const mainDoc = productDocs[0];
  const isFake  = mainDoc.Product_score > 70;
  const productName =
    (productList.find((p) => p.id === Unique_product_id) || {}).name ||
    "Product";

  return (
    <div className="product-listing">
      {/* ---------- product info ---------- */}
      <div className="product-container">
        <div className="product-image-container">
          <img src={mainDoc.Photo_url} alt="Product" className="product-image" />
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
          <h1 className="premium-title">{productName}</h1>

          <p className="desc">
            <strong>Description:</strong> {mainDoc.Description}
          </p>
          <p className="premium-price">
            <strong>Price:</strong> ${mainDoc.Price}
          </p>

          <p className={`prod-score ${isFake ? "fake" : ""}`}>
            <strong>Product Trust Index:</strong> {trustIndex(mainDoc)}
          </p>

          <div className="progress-bar-prod">
            <div
              className="progress"
              style={{
                width: `${trustIndex(mainDoc)}%`,
                background:
                  trustIndex(mainDoc) <= 30
                    ? "linear-gradient(90deg,#e53935 0%,#ffb347 100%)"
                    : trustIndex(mainDoc) <= 69
                    ? "linear-gradient(90deg,#ffb347 0%,#ffd700 100%)"
                    : "linear-gradient(90deg,#43a047 0%,#ffd700 100%)",
              }}
            />
          </div>

          <button className="button serial" onClick={handleSerialVerification}>
            Report Fake Product
          </button>
        </div>
      </div>

      {/* ---------- actions ---------- */}
      <div className="button-container">
        <button className={`button ${isSorted ? "active" : ""}`} onClick={toggleSort}>
          Sort by Trust <SortIcon className="sorticon" />
        </button>
        <button
          className={`button ${isFiltered ? "active" : ""}`}
          onClick={toggleFilter}
        >
          Trusted Reviews only <FilterListIcon className="filtericon" />
        </button>
      </div>

      {/* ---------- reviews ---------- */}
      <div className="reviews">
        {view.map((rev) => (
          <div key={rev._id} className="product-card">
            <div className="product-info">
              <p>
                <strong>Review:</strong> {rev.review_bold}
              </p>
              <p>{rev.review}</p>
              <p>
                <strong>Rating:</strong> {renderStars(rev.ratings)}
              </p>
              <p>
                <strong>Review by:</strong> {rev.by} on {rev.date}
              </p>
              <p>
                <strong>Helpful Votes:</strong> {rev.helpful}
              </p>
              <p className="rev-score">
                <strong>Review Trust Index:</strong> {trustIndex(rev)}
              </p>

              <div className="progress-bar">
                <div
                  className="progress"
                  style={{
                    width: `${trustIndex(rev)}%`,
                    background:
                      trustIndex(rev) <= 25
                        ? "linear-gradient(90deg,#e53935 0%,#ffb347 100%)"
                        : trustIndex(rev) <= 74
                        ? "linear-gradient(90deg,#ffb347 0%,#ffd700 100%)"
                        : "linear-gradient(90deg,#43a047 0%,#ffd700 100%)",
                  }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Productlisting;