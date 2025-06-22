### Implementation Details

1.
2.
3. #Backend : Our Backend server follows a modular Node.js/Express architecture with MongoDB as the primary database,designed for an Amazon product authenticity detection platform with user management and product review analysis capabilities.

backend/
├── server.js           # Main application entry point
├── db.js              # Database connection utility (unused in current setup)
├── package.json       # Dependencies and project metadata
├── insertProducts.js  # Data seeding script
├── routes/            # API endpoint definitions
│   └── userRoutes.js  # User management endpoints
├── models/            # Database schemas
│   └── userModel.js   # User data model
└── middleware/        # Custom middleware functions
    └── authware.js    # Authentication middleware

Two main collections:
users - User accounts with balance tracking
amazon - Product reviews with authenticity scores

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

Product Authenticity Detection - Uses ML-generated scores
Review Analysis - Trust index calculation based on multiple factors
User Balance System - Virtual currency for platform interactions

4. Frontend : an Amazon-inspired e-commerce interface with advanced product authenticity detection capabilities. The application follows a component-based architecture with Material-UI for modern styling.

src/
├── App.js                 # Main router & layout orchestrator
├── Header.js             # Navigation bar with search
├── Home.js               # Landing page with product grid
├── Product.js            # Product card component
├── Productlisting.js     # Detailed product view with reviews
├── SerialVerification.js # Fake product reporting system
├── ItemSearchBar.js      # Advanced search functionality
└── [Component].css       # Component-specific styling

Real-time authenticity scoring with visual indicators
-> color-coded trust bars (red/yellow/green gradient)
->Fake product flagging with prominent warnings
-> ML-powered scoring integration from backend
Trust-based review sorting (most to least trustworthy)
->Trusted reviews filtering (≥25% trust threshold)
->Interactive toggle controls with active states
->Star rating visualization with Material-UI icons

5. Dataset files 


