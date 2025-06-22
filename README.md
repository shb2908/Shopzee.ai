# Amazon_Adam
# Amazon Product Authenticity Detection System

## 🚀 Project Overview

This is a comprehensive *Amazon Product Authenticity Detection System* that leverages advanced AI/ML techniques including *Vision-Language Models (VLM)* and *Large Language Models (LLaMA-2)* to detect fake products and reviews on Amazon. The system provides a multi-modal approach to authenticity verification by analyzing product reviews, images, and metadata.

## 🎯 Key Features

- *Multi-Modal Analysis*: Combines text, image, and metadata analysis
- *Advanced AI Models*: Integrates CLIP (VLM) and LLaMA-2 for sophisticated analysis
- *Real-time API*: Flask-based REST API for instant product analysis
- *Modern Web Interface*: React-based frontend with Material-UI
- *Comprehensive Scoring*: Weighted combination of multiple authenticity indicators
- *Scalable Architecture*: Modular design for easy extension and maintenance
- *Account Trust System*: Seller and customer trust scoring mechanisms
- *Verification Workflows*: Multi-level account verification processes

## 🏗 System Architecture


┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   ML Pipeline   │
│   (React)       │◄──►│   (Flask)       │◄──►│   (Python)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Database      │
                       │   (CSV/JSON)    │
                       └─────────────────┘


## 🔧 Technology Stack

### Backend & ML
- *Python 3.8+*: Core programming language
- *Flask*: Web framework for API
- *TensorFlow*: Deep learning framework
- *PyTorch*: Neural network framework
- *Transformers*: Hugging Face library for LLMs
- *scikit-learn*: Machine learning utilities
- *NLTK*: Natural language processing
- *Pandas & NumPy*: Data manipulation
- *MongoDB*: NoSQL database for product storage
- *Node.js/Express*: Additional backend server

### Frontend
- *React 18*: Frontend framework
- *Material-UI*: UI component library
- *React Router*: Navigation
- *Axios*: HTTP client

### AI/ML Models
- *CLIP (Vision-Language Model)*: Image-text understanding
- *LLaMA-2 (Large Language Model)*: Advanced text analysis
- *VADER Sentiment Analyzer*: Sentiment analysis
- *Custom Classifiers*: Ensemble scoring system

### Database & Storage
- *MongoDB Atlas*: Cloud database for product data
- *CSV Files*: Local data storage for ML training
- *Model Checkpoints*: Saved fine-tuned models

## 📊 Workflow & Algorithm

### 1. Data Processing Pipeline

Raw Data → Preprocessing → Feature Extraction → Model Analysis → Scoring → Results


### 2. Multi-Parameter Authenticity Scoring

The system uses a *weighted ensemble approach* with 5 key parameters:

| Parameter | Weight | Description | Model Used |
|-----------|--------|-------------|------------|
| *Verification Check* | 25% | Purchase verification status | Rule-based |
| *Sentiment Analysis* | 15% | Review sentiment scoring | VADER |
| *Helpful Score* | 15% | User helpfulness metrics | MinMax Scaler |
| *VLM Score* | 20% | Image authenticity analysis | CLIP |
| *LLaMA-2 Score* | 25% | Advanced text understanding | LLaMA-2 |

### 3. Final Score Calculation
python
FINAL_SCORE = (
    verification_score × 0.25 +
    sentiment_score × 0.15 +
    helpful_score × 0.15 +
    vlm_score × 0.20 +
    llama_score × 0.25
)


## 🤖 AI/ML Training Pipelines

### 1. VLM Training (vlm_training.py)
- *Model*: CLIP (Vision-Language Model)
- *Purpose*: Image authenticity classification
- *Dataset*: Real/Fake product image dataset
- *Features*:
  - Custom ProductDataset class
  - CLIPClassifier wrapper with classification head
  - Multi-brand support (Nike, Adidas, Fila, Puma)
  - GPU acceleration support
  - Learning rate: 5e-5
  - Batch size: 16
  - Epochs: 100

### 2. LLaMA-2 Training (llama_training.py)
- *Model*: LLaMA-2-7b-hf (Large Language Model)
- *Purpose*: Advanced text classification for authenticity detection
- *Training Data*: Custom dataset with labeled authentic/fake products
- *Features*:
  - Custom tokenization with max length 512
  - Train/test split (80/20)
  - Learning rate: 2e-5
  - Batch size: 2 (optimized for large model)
  - Epochs: 3
  - Weight decay: 0.01
  - GPU memory optimization
  - Model checkpointing and saving

## 🔍 Advanced Features

### 1. Serial Number Verification System
- *Purpose*: Verify product authenticity via serial numbers
- *Features*:
  - Real-time serial number validation
  - Duplicate detection (prevents multiple verifications)
  - Genuine/fake classification
  - User-friendly verification interface
  - Integration with product listing pages

### 2. MongoDB Integration
- *Database*: MongoDB Atlas cloud database
- *Schema*: Comprehensive product schema including:
  - Product metadata (links, photos, descriptions)
  - Review data (text, ratings, helpfulness)
  - Authenticity scores (FINAL_SCORE, Product_score)
  - Verification status and timestamps

### 3. Advanced Frontend Features

#### Product Trust Index Visualization
- *Dynamic Progress Bars*: Color-coded trust indicators
- *Real-time Scoring*: Live authenticity score updates
- *Visual Badges*: Genuine/Fake product indicators
- *Trust Level Categories*:
  - 0-30: High Risk (Red)
  - 31-69: Medium Risk (Yellow)
  - 70-100: Low Risk (Green)

#### Review Management System
- *Sort by Trust*: Arrange reviews by authenticity score
- *Filter Trusted Reviews*: Show only high-trust reviews (≥25 score)
- *Review Trust Index*: Individual review authenticity scoring
- *Star Rating System*: Visual rating display with Material-UI icons

#### Premium User Experience
- *Loading Animations*: Premium-style loading indicators
- *Responsive Design*: Mobile-friendly interface
- *Interactive Elements*: Hover effects and transitions
- *Real-time Updates*: Dynamic content without page refresh

### 4. Dual Backend Architecture

#### Flask API (Port 5000)
- *ML Pipeline Integration*: Direct connection to Python ML models
- *Real-time Analysis*: Instant product authenticity scoring
- *CORS Support*: Cross-origin resource sharing
- *RESTful Endpoints*: Standard API design

#### Node.js Server (Port 8000)
- *MongoDB Integration*: Database operations and queries
- *Product Management*: CRUD operations for products
- *Data Persistence*: Long-term data storage
- *Scalable Architecture*: Microservices approach

### 5. Model Optimization Features

#### Quantization & Pruning
- *Model Quantization*: Reduced model size for faster inference
- *Pruning*: Removal of unnecessary model parameters
- *Memory Optimization*: Efficient resource utilization

#### GPU Acceleration
- *CUDA Support*: GPU-accelerated training and inference
- *Device Detection*: Automatic CPU/GPU selection
- *Batch Processing*: Optimized for parallel processing

### 6. Data Pipeline Features

#### Multi-Format Support
- *CSV Processing*: Large dataset handling
- *JSON Integration*: API data exchange
- *Image Processing*: PIL-based image manipulation
- *Text Preprocessing*: NLTK-based text cleaning

#### Real-time Processing
- *Streaming Data*: Continuous data ingestion
- *Batch Updates*: Periodic model retraining
- *Incremental Learning*: Model updates without full retraining

### 7. Real-time Monitoring & Alerts

#### System Health Monitoring
- *Model Performance Tracking*: Real-time accuracy monitoring
- *API Response Time*: Performance metrics tracking
- *Database Health*: Connection and query performance
- *Resource Utilization*: CPU, GPU, and memory monitoring

#### Alert System
- *Fraud Detection Alerts*: Immediate notification of suspicious activity
- *Model Drift Detection*: Alerts when model performance degrades
- *System Failures*: Automatic notification of service disruptions
- *Threshold Violations*: Alerts when trust scores exceed dangerous levels

### 8. Security & Compliance Features

#### Data Security
- *Encryption*: End-to-end data encryption
- *Access Control*: Role-based permissions
- *Audit Logging*: Comprehensive activity tracking
- *GDPR Compliance*: Data privacy and protection measures

#### API Security
- *Rate Limiting*: Prevent API abuse
- *Authentication*: JWT-based authentication
- *Input Validation*: Sanitize all user inputs
- *CORS Protection*: Secure cross-origin requests

### 9. Deployment & Scalability

#### Cloud Deployment Options
- *AWS Deployment*: EC2, Lambda, and S3 integration
- *Google Cloud*: GCP services for ML pipeline
- *Azure Integration*: Microsoft cloud services
- *Docker Containerization*: Portable deployment

#### Scalability Features
- *Load Balancing*: Distribute traffic across multiple servers
- *Auto-scaling*: Automatic resource allocation
- *Microservices*: Modular architecture for easy scaling
- *Caching*: Redis-based caching for improved performance

### 10. Analytics & Reporting

#### Business Intelligence
- *Dashboard Analytics*: Real-time business metrics
- *Trend Analysis*: Historical data analysis
- *Predictive Analytics*: Future fraud prediction
- *Custom Reports*: Tailored reporting solutions

#### Performance Metrics
- *Model Accuracy*: Continuous performance tracking
- *User Engagement*: Frontend usage analytics
- *API Usage*: Backend service utilization
- *Cost Optimization*: Resource usage analysis

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git
- MongoDB Atlas account (for cloud database)

### 1. Clone the Repository
bash
git clone https://github.com/shb2908/Amazon_Adam.git
cd Amazon_Adam


### 2. Backend Setup

#### Install Python Dependencies
bash
pip install -r requirements.txt


#### Additional Dependencies (if needed)
bash
pip install torch torchvision
pip install transformers[torch]
pip install accelerate
pip install pymongo


#### Install Node.js Dependencies
bash
cd backend
npm install


### 3. Frontend Setup

#### Install Node.js Dependencies
bash
cd website/amazon
npm install


### 4. Database Setup

#### MongoDB Atlas Configuration
1. Create MongoDB Atlas account
2. Create new cluster
3. Get connection string
4. Update connection string in backend/server.js

#### Local Data Preparation
Ensure you have the required dataset files:
- final_dataset.csv: Main dataset with product reviews and metadata
- authenticity_score.csv: Pre-computed authenticity scores
- Classifier_dataset.csv: Training dataset for classifiers

## 🏃‍♂ Running the Application

### 1. Start the ML Pipeline
bash
python main3.py

This will:
- Load and process the dataset
- Run VLM (CLIP) analysis on product images
- Execute LLaMA-2 text analysis
- Calculate final authenticity scores
- Start Flask API server on port 5000

### 2. Start the Node.js Backend
bash
cd backend
node server.js

The Node.js server will start on port 8000

### 3. Start the Frontend
bash
cd website/amazon
npm start

The React app will start on http://localhost:3000

### 4. API Endpoints

#### Flask API (Port 5000)
- GET /: API information
- GET /analyze/<product_id>: Get authenticity analysis for a specific product

#### Node.js API (Port 8000)
- GET /products: Get all products
- GET /products/:Unique_product_id: Get specific product by ID

#### Example API Response
json
{
  "product_id": 123,
  "review": "Great product, highly recommended!",
  "final_score": 0.85,
  "vlm_score": 0.92,
  "llama_score": 0.88,
  "verification_score": 0.75,
  "sentiment_score": 0.90,
  "helpful_score": 0.82
}


## 📁 Project Structure


Amazon_Adam/
├── main3.py                 # Main ML pipeline with VLM & LLaMA-2
├── vlm_training.py          # CLIP model training pipeline
├── llama_training.py        # LLaMA-2 model training pipeline
├── requirements.txt         # Python dependencies
├── final_dataset.csv        # Main dataset
├── authenticity_score.csv   # Pre-computed scores
├── Classifier_dataset.csv   # Training data
├── fine-tuned-llama/        # Fine-tuned LLaMA-2 model
├── backend/
│   ├── server.js           # Node.js server with MongoDB
│   ├── insertProducts.js   # Database operations
│   └── package.json        # Node.js dependencies
└── website/
    └── amazon/
        ├── src/            # React components
        │   ├── SerialVerification.js  # Serial number verification
        │   ├── Productlisting.js      # Product listing with trust scores
        │   └── ...                    # Other components
        ├── public/         # Static assets
        └── package.json    # Frontend dependencies


## 🤖 AI Models Explained

### 1. CLIP (Vision-Language Model)
- *Purpose*: Analyze product images for authenticity
- *Model*: openai/clip-vit-base-patch32
- *Function*: Compares product images with text descriptions
- *Output*: Score between 0-1 (1 = authentic)

### 2. LLaMA-2 (Large Language Model)
- *Purpose*: Advanced text analysis and understanding
- *Model*: meta-llama/Llama-2-7b-hf
- *Function*: Deep semantic analysis of product reviews
- *Output*: Score between 0-1 (1 = authentic)

### 3. VADER Sentiment Analyzer
- *Purpose*: Sentiment analysis of reviews
- *Function*: Analyzes emotional tone and polarity
- *Threshold*: Reviews with |compound| > 0.85 flagged

### 4. Custom Ensemble Classifier
- *Purpose*: Combine all scores into final authenticity rating
- *Method*: Weighted average of all parameters
- *Threshold*: Score > 0.7 indicates potential fake product

## 📈 Performance Metrics

The system provides comprehensive analysis with:
- *Accuracy*: Based on ensemble model performance
- *Precision*: High precision in detecting fake products
- *Recall*: Comprehensive coverage of suspicious items
- *F1-Score*: Balanced precision and recall

## 🔍 Usage Examples

### 1. Analyze a Product via API
bash
curl http://localhost:5000/analyze/123


### 2. Batch Processing
python
import pandas as pd
from main3 import analyze_product

# Load dataset
df = pd.read_csv('final_dataset.csv')

# Analyze all products
for idx, row in df.iterrows():
    result = analyze_product(idx)
    print(f"Product {idx}: Score = {result['final_score']}")


### 3. Custom Thresholds
python
# Modify weights in main3.py
weights = {
    'verification_check': 0.30,   # Increase verification weight
    'Sentiment_Score': 0.10,      # Decrease sentiment weight
    'Helpful_Score': 0.15,
    'vlm_score': 0.25,           # Increase VLM weight
    'llama_score': 0.20
}


## 🛠 Customization

### Adding New Models
1. Create model class in main3.py
2. Add scoring function
3. Update weights dictionary
4. Integrate into final score calculation

### Modifying Weights
Edit the weights dictionary in main3.py to adjust the importance of each parameter.

### Extending API
Add new routes in the Flask app section of main3.py.

## 🐛 Troubleshooting

### Common Issues

1. *Model Loading Errors*
   - Ensure sufficient RAM (LLaMA-2 requires 16GB+)
   - Check internet connection for model downloads
   - Verify CUDA installation for GPU acceleration

2. *Memory Issues*
   - Reduce batch size in model processing
   - Use smaller model variants
   - Process data in chunks

3. *API Connection Issues*
   - Verify Flask server is running on port 5000
   - Check CORS configuration
   - Ensure frontend proxy settings are correct

### Performance Optimization
- Use GPU acceleration for model inference
- Implement caching for repeated analyses
- Optimize data preprocessing pipeline

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

## 🔮 Future Enhancements

- [ ] Real-time image processing
- [ ] Mobile app development
- [ ] Advanced fraud detection algorithms
- [ ] Integration with Amazon API
- [ ] Multi-language support
- [ ] Cloud deployment options

---

*Note*: This system is designed for educational and research purposes. Always comply with Amazon's terms of service and applicable laws when using this tool.
