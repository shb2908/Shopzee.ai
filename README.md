## Amazon_Hackon (Team Adam)
To foster lasting trust across the marketplace, our solution places both buyers and sellers at the core
of our product verification architecture. From the moment a product is listed to the time it is reviewed,
every stage of our pipeline is designed to detect, flag, and filter out counterfeit or misrepresented
products, ensuring that only verified, authentic listings are promoted.

# Problems :


# 1: Fake Sellers Upload Fake Items & Self‑Ratings  
**Solution:** Seller Trust Scoring  
An account trust system assesses sellers using a weighted score: **business verification (30%)**, **financial history (25%)**, and **compliance history (10%)**. This tiered approach ensures that sellers with low trust scores are automatically flagged for suspicious activity, helping maintain marketplace integrity.

#  2: Serial/ID Authenticity Checks  
**Solution:** Serial Number Verification  
The `SerialVerification.js` component enables real-time validation of serial numbers against official product databases. This process effectively prevents duplicate or invalid serial numbers from being listed, significantly reducing instances of counterfeit listings.

#  3: Low Ratings from Fake Reviews or Damaged Products  
**Solution:** VADER Sentiment Analysis  
Combining the VADER sentiment analysis tool with a **compound score threshold (|compound| > 0.85)** and LLaMA‑2 chain-of-thought reasoning, this system distinguishes genuine negative feedback from fake or product-damage–induced reviews. It ensures authentic user concerns are recognized while filtering out malicious patterns.

#  4: Fake Sellers Posting Poor Reviews on Competitors  
**Solution:** Review Pattern Detection  
By analyzing IP addresses and using clustering algorithms, this feature identifies coordinated negative reviews from the same seller accounts or IP locations. Suspicious review behavior triggers alerts, helping to protect honest competitors from sabotage.

#  5: Missing Parts/Features Not Detectable by Images  
**Solution:** LLaMA‑2 Text Analysis  
Advanced NLP powered by LLaMA‑2 scans user reviews for phrases indicating missing parts. These flags are weighted (25%) against overall review authenticity, enabling proactive detection of incomplete or misleading product listings.

# 6: Multiple Accounts After Ban  
**Solution:** Device/IP Fingerprinting  
This module tracks IP changes, device fingerprints, and behavioral usage patterns to detect users who create new accounts after being banned. It links related profiles and prevents repeated rule violations.

#  7: Multiple Accounts Posting Reviews for Fake Products  
**Solution:** Review Clustering Detection  
Detects fake product promotion by identifying identical reviews across multiple user profiles. It also spots review bursts and spam patterns, ensuring fraudulent review farms are brought to light.

#  8: Damaged Products, Variant Mismatch, Expired Warranty  
**Solution:** Multi‑Modal Verification  
A three-pronged approach:  
1. **CLIP (vision-language model)** scans for image-text inconsistencies,  
2. **Metadata analysis** verifies brand and variant details,  
3. **Warranty checks** flag expired coverage.  
This ensures product authenticity and accuracy before listing.

# 9: Bot Flooding with Fake Reviews  
**Solution:** Review Burst Detection  
Real-time monitoring identifies sudden spikes in reviews from new accounts. When detected, it activates rate limiting and triggers anomaly alerts, effectively mitigating bot-driven review attacks.

#  10: Fake Images or Misleading Description Videos  
**Solution:** CLIP Vision‑Language Analysis  
Using a vision-language model to compare product images to textual descriptions, this component detects media inconsistency. Listings with **low CLIP similarity scores (20% weight in final authenticity rating)** are flagged for review.

#  11: Price Manipulation & Stock Fraud  
**Solution:** Price History Monitoring  
By correlating price trends with stock levels, this module alerts when stock is removed soon after a price drop. Any suspicious patterns negatively impact seller trust scores and product authenticity ratings.


## Project Files 

Key Features  

•⁠  ⁠Multi-Modal Analysis: Combines text, image, and metadata analysis   
•⁠  ⁠Advanced AI Models: Integrates CLIP (VLM) and LLaMA-2 for sophisticated analysis  
•⁠  ⁠Real-time API: Flask-based REST API for instant product analysis  
•⁠  ⁠Modern Web Interface: React-based frontend with Material-UI  
•⁠  ⁠Comprehensive Scoring: Weighted combination of multiple authenticity indicators  
•⁠  ⁠Scalable Architecture: Modular design for easy extension and maintenance  
•⁠  ⁠Account Trust System: Seller and customer trust scoring mechanisms  
•⁠  ⁠Verification Workflows: Multi-level account verification processes


1. Main.py: Contains the actual machine learning models and all the data frame used in the models.
2. Backend : Our Backend server follows a modular Node.js/Express architecture with MongoDB as the primary database,designed for an Amazon product authenticity detection platform with user management and product review analysis capabilities.

A modular Node.js + Express backend with MongoDB:

```bash
backend/
├── server.js               # Entry point
├── db.js                   # MongoDB connector (currently unused)
├── insertProducts.js       # Seeds database
├── routes/
│   └── userRoutes.js       # API endpoints
├── models/
│   └── userModel.js        # MongoDB schemas
├── middleware/
│   └── authware.js         # Auth middleware
├── package.json            # Dependencies

```
Two main collections:
users - User accounts with balance tracking
amazon - Product reviews with authenticity scores

```bash
{
  Product_link: String,
  Photo_url: String,
  Description: String,
  Unique_product_id: String,    // Product identifier
  Price: Number,
  Product_score: Number,        // Authenticity score
  review_bold: String,          // Review headline
  ratings: Number,              // Star rating (1-5)
  review: String,               // Full review text
  verified: Boolean,            // Verified purchase
  date: String,                 // Review date
  by: String,                   // Reviewer name
  helpful: Number,              // Helpfulness votes
  FINAL_SCORE: Number          // Computed authenticity score
}
```
Product Authenticity Detection - Uses ML-generated scores
Review Analysis - Trust index calculation based on multiple factors
User Balance System - Virtual currency for platform interactions

3. Frontend : an Amazon-inspired e-commerce interface with advanced product authenticity detection capabilities. The application follows a component-based architecture with Material-UI for modern styling.

```bash
src/
├── App.js                 # Main router & layout orchestrator
├── Header.js             # Navigation bar with search
├── Home.js               # Landing page with product grid
├── Product.js            # Product card component
├── Productlisting.js     # Detailed product view with reviews
├── SerialVerification.js # Fake product reporting system
├── ItemSearchBar.js      # Advanced search functionality
└── [Component].css       # Component-specific styling
```

Real-time authenticity scoring with visual indicators  
-> color-coded trust bars (red/yellow/green gradient)  
->Fake product flagging with prominent warnings  
-> ML-powered scoring integration from backend  
Trust-based review sorting (most to least trustworthy)  
->Trusted reviews filtering (≥25% trust threshold)  
->Interactive toggle controls with active states  
->Star rating visualization with Material-UI icons  







