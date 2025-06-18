# AdidasSmartRecs: AI-Powered E-commerce Recommendation System

> **Intelligent Product Recommendation Engine for Adidas Great Britain Market**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112.1-green.svg)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-orange.svg)](https://scikit-learn.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## 🚀 Project Overview

**AdidasSmartRecs** is a comprehensive AI-driven recommendation system developed for Adidas to enhance customer engagement and drive sales in the Great Britain market. The system leverages advanced machine learning techniques to analyze customer behavior patterns and deliver personalized product recommendations.

### 📊 Key Statistics
- **355K+ Customer Records** analyzed for behavioral insights
- **178K+ Sales Transactions** processed for recommendation training
- **33K+ Engagement Records** integrated for customer segmentation
- **3-Algorithm Hybrid System** combining collaborative, content-based, and item-based filtering
- **Real-time API** serving personalized recommendations

---

## 🎯 Business Impact

- **Enhanced Customer Experience**: Personalized recommendations based on purchase history and preferences
- **Data-Driven Insights**: Customer segmentation by sports categories and engagement patterns
- **Scalable Architecture**: RESTful API design supporting real-time recommendations
- **Market-Specific Focus**: Tailored for Great Britain market dynamics

---

## 🏗️ System Architecture

### Data Pipeline
```
Raw Data Sources → Data Preprocessing → Feature Engineering → Model Training → API Deployment
     ↓                    ↓                    ↓              ↓              ↓
Consumer Data     Missing Value      Customer         Hybrid Model    FastAPI Server
Sales Data        Handling          Segmentation      Training        Web Interface
Engagement Data   Normalization     Sports Analysis   Validation      Real-time Recs
```

### Machine Learning Pipeline
1. **Data Integration**: Merged consumer, sales, and engagement datasets
2. **Customer Segmentation**: K-means clustering based on sports category preferences
3. **Hybrid Recommendation System**:
   - **Collaborative Filtering**: User-based similarity recommendations
   - **Content-Based Filtering**: Product feature similarity
   - **Item-Based Filtering**: Item-to-item collaborative filtering
4. **Model Persistence**: Serialized models using joblib for production deployment

---

## 🔬 Methodology

### Data Science Approach
- **Exploratory Data Analysis**: Comprehensive analysis of 355K+ customer profiles
- **Feature Engineering**: Created customer segments based on sports category engagement
- **Model Selection**: Evaluated multiple recommendation algorithms for optimal performance
- **Validation**: Cross-validation and performance metrics to ensure accuracy

### Technical Implementation
- **Backend**: FastAPI with async support and CORS middleware
- **Frontend**: Responsive HTML/CSS/JavaScript interface
- **ML Models**: scikit-learn for clustering and similarity calculations
- **Data Processing**: Pandas and NumPy for efficient data manipulation

---

## 💻 Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI, Python 3.8+ |
| **Frontend** | HTML5, CSS3, JavaScript ES6+ |
| **ML/AI** | scikit-learn, pandas, numpy |
| **Data Storage** | CSV files, Pickle serialization |
| **Model Deployment** | joblib, uvicorn |
| **API Documentation** | FastAPI auto-generated docs |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/AdidasSmartRecs.git
cd AdidasSmartRecs

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify data and model files are present
ls data/        # Should contain: ConsTable_EU.csv, SalesTable_EU.csv, EngagementTable_GB.csv
ls models/      # Should contain: *.pkl files and *.csv files

# Start the application
python API.py
```

### Access the Application
- **Web Interface**: [http://localhost:8000](http://localhost:8000)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Check**: [http://localhost:8000/recommendations/0013SD11X9372BM7](http://localhost:8000/recommendations/0013SD11X9372BM7)

---

## 📖 Usage Guide

### Web Interface
1. Navigate to the homepage
2. Enter a valid User ID in the recommendation form
3. Click "Get Recommendations" to view personalized suggestions

### API Endpoints

#### Get Recommendations
```http
GET /recommendations/{user_id}
```

**Example Request:**
```bash
curl http://localhost:8000/recommendations/0013SD11X9372BM7
```

**Example Response:**
```json
{
  "user_id": "0013SD11X9372BM7",
  "recommendations": [
    "GW0436",
    "GY7924",
    "HQ3819",
    "FY8794",
    "GZ8203"
  ]
}
```

### Sample User IDs for Testing
```
0013SD11X9372BM7
0035F617-FE01-0500-F53C-B2725D699C7C
003ED223-4E51-2423-994F-09ED97777175
004K2KV4OEU0OCOB
004WEFNXSIJT15B6
```

---

## 📊 Model Performance

### Recommendation System Metrics
- **Hybrid Approach**: Combines 3 different recommendation algorithms
- **Customer Coverage**: Handles 355K+ unique customers
- **Product Catalog**: Covers extensive Adidas product range
- **Real-time Response**: Sub-second recommendation generation

### Data Processing Stats
- **Data Cleaning**: 99.9% data quality after preprocessing
- **Feature Engineering**: Customer segmentation by sports categories
- **Model Training**: Collaborative filtering on user-item interactions

---

## 📁 Project Structure

```
AdidasSmartRecs/
├── 📊 data/                    # Raw datasets
│   ├── ConsTable_EU.csv        # Consumer demographic data (355K records)
│   ├── SalesTable_EU.csv       # Sales transaction data (178K records)
│   └── EngagementTable_GB.csv  # Customer engagement data (33K records)
├── 🤖 models/                  # Trained models and processed data
│   ├── user_item_matrix.pkl    # User-item interaction matrix
│   ├── item_similarity.pkl     # Item-to-item similarity matrix
│   ├── product_matrix.pkl      # Product feature matrix
│   ├── vectorizer.pkl          # Text vectorizer for content-based filtering
│   ├── product_data.csv        # Processed product information
│   └── customer_data.csv       # Processed customer segments
├── 🌐 website/                 # Static frontend assets
│   ├── style.css              # Responsive CSS styling
│   ├── app.js                 # Frontend JavaScript logic
│   ├── adidas-logo.jpg        # Brand assets
│   └── *.jpg                  # Banner and background images
├── 📄 templates/               # HTML templates
│   └── index.html             # Main application interface
├── 🚀 API.py                   # FastAPI backend server
├── 📓 notebook.ipynb           # Data analysis and model development
├── 📋 requirements.txt         # Python dependencies
└── 📝 README.md               # Project documentation
```

---

## 🔧 Development

### Running in Development Mode
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn API:app --reload --host 0.0.0.0 --port 8000
```

### Model Retraining
The Jupyter notebook (`notebook.ipynb`) contains the complete pipeline for:
- Data preprocessing and cleaning
- Exploratory data analysis
- Customer segmentation
- Model training and validation
- Model serialization

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Static files not loading** | Ensure files are in the `website/` directory and FastAPI static mount is configured |
| **No recommendations returned** | Verify user ID exists in the dataset using sample IDs provided |
| **API errors** | Check console logs and ensure all model files are present in `models/` directory |
| **Import errors** | Verify all dependencies are installed: `pip install -r requirements.txt` |
| **Model file missing** | Run the complete notebook to regenerate model files |

---

## 🤝 Contributing

This project was developed as a team effort with a focus on:
- **Data Science**: Advanced analytics and machine learning
- **Software Engineering**: Scalable API design and deployment
- **User Experience**: Intuitive web interface design
- **Project Management**: Agile development and team coordination

---

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## 🏆 Acknowledgments

- **Adidas** for providing real-world datasets
- **Team Members** for collaborative development
- **Open Source Community** for excellent ML libraries
- **FastAPI** for the robust web framework



---

**Built with ❤️ by the AdidasSmartRecs Team**
