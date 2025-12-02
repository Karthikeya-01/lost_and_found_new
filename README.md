# 🔍 Campus Lost and Found with AutoMatch

An intelligent lost and found matching system for university campuses that uses classical machine learning techniques to automatically match lost items with found items based on image and text features.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

---

## 🎯 Overview

Losing personal belongings on campus is a common problem, and connecting those who have lost items with those who have found them is often inefficient. This project implements an **automated matching system** that uses machine learning to identify potential matches between lost and found items without relying on deep learning methods.

### Key Highlights

- ✅ **Classical ML approach** - No deep learning, focuses on feature engineering
- ✅ **Multi-modal matching** - Combines image and text features
- ✅ **Real-time matching** - Instant results for new uploads
- ✅ **Interactive web app** - User-friendly Streamlit interface
- ✅ **Explainable results** - Shows why items match

---

## ✨ Features

### Core Functionality

- **Report Lost Items**: Upload images and descriptions of lost items
- **Report Found Items**: Upload images and descriptions of found items
- **Automatic Matching**: AI-powered similarity matching between lost and found items
- **Top-K Retrieval**: Returns the most similar items ranked by confidence
- **Category Filtering**: Matches items only within the same category
- **Temporal Validation**: Ensures found date ≥ lost date

### User Interface

- 🏠 **Home Dashboard**: Overview and statistics
- 😢 **Lost Item Reporting**: Upload and get instant matches
- 🎉 **Found Item Reporting**: Help find potential owners
- 🔎 **Browse Matches**: Explore existing matched items
- 📊 **Statistics**: Visual analytics and insights

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/campus-lost-found.git
cd campus-lost-found
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python test_setup.py
```

---

## 📖 Usage

### Quick Start

1. **Prepare Your Dataset**
   ```bash
   # Place your images in: dataset/images/{category}/
   # Place metadata in: dataset/metadata_labelled.json
   ```

2. **Extract Features**
   ```bash
   cd scripts
   python extract_features.py
   ```
   This will create feature vectors for all items and save them to `features/`

3. **Run Matching Algorithm**
   ```bash
   python run_matching.py
   ```
   This computes similarities and generates matches, saving results to `features/matches/`

4. **Launch Web Application**
   ```bash
   cd ../app
   streamlit run streamlit_app.py
   ```
   The app will open at `http://localhost:8501`

### Step-by-Step Guide

#### 1. Data Exploration

Analyze your dataset:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

#### 2. Feature Extraction

Extract image and text features:
```bash
cd scripts
python extract_features.py
```

**What it does:**
- Loads all images and metadata
- Extracts color, texture, and shape features from images (184 dimensions)
- Extracts TF-IDF and keyword features from text (126 dimensions)
- Saves features to `features/` directory

**Output:**
- `features/all_features.pkl` - Complete feature data
- `features/feature_matrices.pkl` - Numpy arrays for fast computation
- `features/tfidf_vectorizer.pkl` - Trained TF-IDF model
- `features/feature_summary.csv` - Statistics

#### 3. Run Matching

Compute similarities and generate matches:
```bash
python run_matching.py
```

**What it does:**
- Loads extracted features
- Computes similarity between all lost and found items
- Ranks matches by similarity score
- Applies category and temporal filters

**Output:**
- `features/matches/matching_results.pkl` - Complete results
- `features/matches/matches.json` - Matches in JSON format
- `features/matches/matching_summary.csv` - Summary statistics
- `features/matches/matching_report.txt` - Detailed report

#### 4. Evaluate Performance

Analyze matching quality:
```bash
python evaluate_matching.py
```

**What it does:**
- Analyzes similarity distributions
- Shows performance by category
- Suggests optimal thresholds
- Generates visualizations

#### 5. Launch Web App

Start the interactive interface:
```bash
cd ../app
streamlit run streamlit_app.py
```

**Features:**
- Upload new lost/found items
- Get instant AI matches
- Browse existing matches
- View system statistics

---

## 📁 Project Structure

```
campus-lost-found/
│
├── dataset/                          # Dataset directory
│   ├── images/                      # Item images
│   │   ├── bags/
│   │   ├── books/
│   │   ├── calculators/
│   │   ├── computer_mouse/
│   │   ├── earphones/
│   │   ├── glasses/
│   │   ├── id_cards/
│   │   ├── keys/
│   │   ├── wallets/
│   │   └── water_bottles/
│   └── metadata_labelled.json       # Item metadata
│
├── src/                             # Source code
│   ├── feature_extraction/
│   │   ├── __init__.py
│   │   ├── image_features.py       # Image feature extraction
│   │   └── text_features.py        # Text feature extraction
│   └── matching/
│       ├── __init__.py
│       ├── similarity.py            # Similarity computation
│       └── ranking.py               # Match ranking
│
├── scripts/                         # Executable scripts
│   ├── extract_features.py         # Feature extraction pipeline
│   ├── run_matching.py             # Matching pipeline
│   ├── evaluate_matching.py        # Evaluation tools
│   └── view_matches.py             # View sample matches
│
├── app/                            # Web application
│   └── streamlit_app.py           # Streamlit interface
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # Dataset analysis
│   └── 02_feature_extraction.ipynb # Feature engineering
│
├── features/                       # Extracted features (generated)
│   ├── all_features.pkl
│   ├── feature_matrices.pkl
│   ├── tfidf_vectorizer.pkl
│   └── matches/
│       ├── matching_results.pkl
│       └── matches.json
│
├── docs/                          # Documentation
│   ├── screenshots/
│   └── demo.mp4
│
├── tests/                         # Unit tests
│   └── test_setup.py
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LICENSE                        # License file
└── .gitignore                    # Git ignore rules
```

---

## 🧠 Methodology

### 1. Feature Extraction

#### Image Features (184 dimensions)

**Color Features:**
- **Color Histograms (96 dims)**: HSV color space histograms
- **Dominant Colors (20 dims)**: K-means clustering (5 colors)
- **Color Moments (9 dims)**: Mean, std, skewness for each RGB channel

**Texture Features:**
- **LBP - Local Binary Patterns (26 dims)**: Texture descriptor
- **GLCM - Gray-Level Co-occurrence Matrix (10 dims)**: Texture properties

**Shape Features:**
- **Edge Features (17 dims)**: Canny edge detection + distribution
- **Shape Features (6 dims)**: Contours, compactness, circularity

#### Text Features (126 dimensions)

**TF-IDF Features (100 dims):**
- Word and bigram frequencies
- Captures semantic content

**Keyword Features (18 dims):**
- Colors (red, blue, black, white, green)
- Materials (leather, plastic, metal)
- Conditions (new, worn, damaged)
- Sizes (small, medium, large)
- Brands (Apple, Nike, etc.)

**Text Statistics (8 dims):**
- Character count, word count
- Average word length
- Unique word ratio

### 2. Similarity Computation

**Combined Score:**
```
similarity = α × image_similarity + β × text_similarity
```

Where:
- α = 0.6 (image weight)
- β = 0.4 (text weight)

**Similarity Metrics:**
- **Cosine Similarity**: For both image and text features
- **Euclidean Distance**: Alternative metric (configurable)

**Filtering:**
- **Category Match**: Only matches items in same category
- **Temporal Validation**: Found date must be ≥ lost date

### 3. Ranking Algorithm

**Top-K Retrieval:**
1. Compute similarity with all candidate items
2. Filter by minimum threshold (default: 0.3)
3. Sort by similarity score (descending)
4. Return top-K matches (default: K=5)

**Confidence Levels:**
- **High (≥0.7)**: Very likely match
- **Medium (0.5-0.7)**: Possible match
- **Low (0.3-0.5)**: Weak match

---

## 📊 Results

### Example Matches

**Lost Item: Blue Nike Backpack**
```
Rank 1: Found Item F042 - Similarity: 87.3%
  ├── Image Similarity: 85.2%
  ├── Text Similarity: 90.4%
  └── Category: bags ✓

Rank 2: Found Item F015 - Similarity: 72.1%
  ├── Image Similarity: 68.9%
  ├── Text Similarity: 76.8%
  └── Category: bags ✓
```


## 🎓 Technical Details

### Technologies Used

- **Python 3.8+**: Core programming language
- **OpenCV**: Image processing and feature extraction
- **scikit-learn**: Machine learning utilities (TF-IDF, similarity metrics)
- **scikit-image**: Advanced image processing (LBP, GLCM)
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Streamlit**: Web application framework
- **Pillow**: Image handling

### Machine Learning Techniques

- **Feature Engineering**: Hand-crafted features from images and text
- **TF-IDF Vectorization**: Text representation
- **Cosine Similarity**: Distance metric
- **K-Means Clustering**: Dominant color extraction
- **Information Retrieval**: Top-K ranking

### Why Classical ML?

This project deliberately avoids deep learning to:
- ✅ Focus on feature engineering skills
- ✅ Maintain interpretability
- ✅ Work with limited data
- ✅ Reduce computational requirements
- ✅ Enable faster experimentation

---

## 🔮 Future Work

### Short-term Improvements

- [ ] **Database Integration**: Store items persistently (SQLite/PostgreSQL)
- [ ] **User Authentication**: Track who reported items
- [ ] **Email Notifications**: Alert users of potential matches
- [ ] **Feedback System**: "Is this your item?" confirmation
- [ ] **Admin Panel**: Verify and manage matches
- [ ] **Ground Truth Evaluation**: Create labeled test set

### Long-term Enhancements

- [ ] **Deep Learning**: Use CNNs for image features (ResNet, EfficientNet)
- [ ] **Active Learning**: Learn from user feedback
- [ ] **Mobile Application**: iOS and Android apps
- [ ] **Multi-language Support**: Internationalization
- [ ] **Location-based Filtering**: GPS-aware matching
- [ ] **QR Code System**: Quick reporting via QR codes
- [ ] **Integration with Campus Systems**: Connect to student ID database

### Alternative Approaches

- [ ] **Supervised Classification**: Train SVM or Random Forest on labeled pairs
- [ ] **Siamese Networks**: Learn similarity metric end-to-end
- [ ] **Graph-based Matching**: Model as bipartite graph matching
- [ ] **Ensemble Methods**: Combine multiple similarity metrics

---

## 📚 References

1. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*.
2. Ojala, T., Pietikäinen, M., & Harwood, D. (1996). A comparative study of texture measures with classification based on featured distributions. *Pattern Recognition*.
3. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*.


<div align="center">
  
**Made with ❤️ for helping people find their lost belongings**

[⬆ Back to Top](#-campus-lost-and-found-with-automatch)

</div>
