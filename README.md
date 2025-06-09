# AdidasSmartRecs

> **Personalized Product Recommendations for Adidas (Great Britain)**

---

## Overview
AdidasSmartRecs is a data-driven recommendation system designed to enhance customer engagement and drive sales for Adidas in Great Britain. It leverages customer segmentation and hybrid recommendation algorithms to provide personalized product suggestions based on user behavior and preferences.

---

## Features
- 🏷️ **Hybrid Recommendation System** (Collaborative, Item-based, Content-based)
- 👥 **Customer Segmentation** using machine learning
- 🌐 **Modern Web Interface** for user interaction
- ⚡ **RESTful API** for real-time recommendations
- 📊 **Data-driven insights** from real Adidas datasets

---

## Screenshots
> _Add your own screenshots here!_

![Web UI Screenshot](website/screenshot.png)

---

## Tech Stack
- **Backend:** FastAPI (Python)
- **Frontend:** HTML, CSS, JavaScript
- **ML/Data:** scikit-learn, pandas, numpy
- **Model Storage:** joblib

---

## Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/AdidasSmartRecs.git
   cd AdidasSmartRecs
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure data and models are present:**
   - `data/ConsTable_EU.csv`, `data/SalesTable_EU.csv`, `data/EngagementTable_GB.csv`
   - `models/user_item_matrix.pkl`, `models/item_similarity.pkl`, `models/product_matrix.pkl`, `models/product_data.csv`, `models/customer_data.csv`
4. **Run the API server:**
   ```bash
   python API.py
   ```
5. **Access the web interface:**
   - Open [http://localhost:8000](http://localhost:8000) in your browser

---

## Usage
### Web Interface
- Enter a valid User ID (see below) and click **Get Recommendations** to view personalized product suggestions.

### Example User IDs
- `0013SD11X9372BM7`
- `0035F617-FE01-0500-F53C-B2725D699C7C`
- `003ED223-4E51-2423-994F-09ED97777175`
- `004K2KV4OEU0OCOB`
- `004WEFNXSIJT15B6`

### API Usage
- Get recommendations for a user:
  ```bash
  curl http://localhost:8000/recommendations/0013SD11X9372BM7
  ```
- Response:
  ```json
  {
    "user_id": "0013SD11X9372BM7",
    "recommendations": ["product_id_1", "product_id_2", ...]
  }
  ```

---

## Project Structure
```
├── data/           # Raw datasets
├── models/         # Trained models and processed data
├── website/        # Static files (CSS, JS, images)
├── templates/      # HTML templates
├── API.py          # FastAPI backend
├── requirements.txt
├── README.md
```

---

## Troubleshooting
- **Static files not loading?** Ensure CSS/JS/images are in the `website/` directory.
- **No recommendations?** Make sure you use a valid User ID from the dataset.
- **API errors?** Check the terminal for error messages and ensure all model files are present.

---

## License
This project is licensed under the MIT License.

---

## Contact
For questions or feedback, please open an issue or contact the maintainer.
