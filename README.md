# Movie Genre Classification using Machine Learning

A comprehensive multi-label classification system that automatically predicts movie genres based on textual features using machine learning techniques.

## Overview

This project implements a **multi-label classification system** to predict movie genres using natural language processing and machine learning techniques. The system can classify movies into multiple genres simultaneously, making it suitable for real-world applications where movies often belong to several categories.

### Key Highlights
- **Multi-label Classification**: Handles movies with multiple genres
- **TF-IDF Vectorization**: Advanced text feature extraction
- **One-vs-Rest Classifier**: Scalable approach for multi-label problems
- **Comprehensive Evaluation**: Multiple metrics for thorough assessment
- **Production Ready**: Includes prediction function for new movies

## Dataset

**Source**: [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata)

**Dataset Statistics**:
- **Movies**: 4,803 movies
- **Features**: 20+ attributes per movie
- **Genres**: 20 different genres
- **Time Period**: Various years of movie releases

**Key Columns Used**:
- `overview`: Movie plot summary
- `tagline`: Movie tagline
- `keywords`: Associated keywords
- `genres`: Target variable (multiple genres per movie)
- `cast` & `crew`: Additional metadata

## Features

### Core Functionality
- **Automated Genre Prediction**: Predict genres for new movies
- **Multi-label Support**: Handle movies with multiple genres
- **Performance Analytics**: Comprehensive model evaluation
- **Data Visualization**: Genre distribution analysis
- **Results Export**: Save predictions to CSV format

### Technical Features
- **Text Preprocessing**: Advanced cleaning and preparation
- **Feature Engineering**: TF-IDF vectorization with n-grams
- **Model Training**: Logistic Regression with One-vs-Rest approach
- **Cross-validation**: Robust model evaluation
- **Scalable Architecture**: Easy to extend and modify

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/movie-genre-classification.git
cd movie-genre-classification
```

2. **Install required packages**
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

3. **Download the dataset**
- Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` from [Kaggle](https://www.kaggle.com/tmdb/tmdb-movie-metadata) or this repository
- Place the CSV files in the project root directory

4. **Run the notebook**
```bash
jupyter notebook movie_genre_classification.ipynb
```

## Usage

### Quick Start

1. **Basic Usage**: Run all cells in the Jupyter notebook to train the model and see results

2. **Predict New Movie Genres**:
```python
# Example prediction
new_movie = {
    'title': 'The Last Starship',
    'overview': 'A daring space crew embarks on a mission to save humanity from a rogue black hole.',
    'tagline': 'Their last hope is an ancient ship.',
    'keywords': ['space', 'sci-fi', 'mission', 'black hole']
}

predicted_genres = predict_genre(new_movie)
print(f"Predicted genres: {predicted_genres}")
# Output: Predicted genres: ('Action', 'Adventure')
```

3. **Access Results**: Check the generated `classified_movies_results.csv` for all predictions

### Advanced Usage

- **Custom Features**: Modify the `combined_features` creation to include additional text fields
- **Model Tuning**: Adjust TF-IDF parameters or try different classifiers
- **Evaluation**: Use the built-in evaluation functions to assess performance

## Model Performance

### Performance Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **Subset Accuracy** | 12.90% | Exact match accuracy (all genres correct) |
| **F1-Score (Macro)** | 0.23 | Average F1-score across all genres |
| **F1-Score (Micro)** | 0.47 | Global F1-score considering all predictions |

### Performance Analysis

- **Subset Accuracy**: While 12.90% may seem low, it's typical for multi-label classification where ALL genres must be predicted correctly
- **Micro F1-Score**: The 0.47 score indicates good performance at individual genre level
- **Macro F1-Score**: Shows variation across genres due to class imbalance

### Model Strengths
Handles class imbalance well  
Good performance on common genres  
Fast training and prediction  
Interpretable results  

## Results & Visualizations

### Genre Distribution
![Genre Distribution](genre_distribution.png)

**Key Insights**:
- **Drama** is the most common genre (2000+ movies)
- **Comedy** and **Thriller** are also well-represented
- **TV Movie** and **Foreign** are less common
- Class imbalance affects model performance across genres

### Sample Predictions

| Movie Title | Original Genres | Predicted Genres | Match |
|-------------|----------------|------------------|-------|
| Avatar | Action, Adventure, Fantasy, Science Fiction | Action, Adventure | ✅ Partial |
| The Dark Knight Rises | Action, Crime, Drama, Thriller | Action, Crime | ✅ Partial |

## 🔧 Technical Implementation

### Architecture Overview

```
Input Text → Preprocessing → TF-IDF Vectorization → One-vs-Rest Classifier → Genre Predictions
```

### Key Components

1. **Text Preprocessing**:
   - JSON parsing for structured data
   - Text cleaning and normalization
   - Feature combination

2. **Feature Engineering**:
   - TF-IDF vectorization (5000 features)
   - N-gram analysis (1-2 grams)
   - Stop word removal

3. **Model Architecture**:
   - One-vs-Rest Classifier
   - Logistic Regression base estimator
   - Multi-label binarization

4. **Evaluation Framework**:
   - Train-test split (80-20)
   - Multiple evaluation metrics
   - Cross-validation ready

### Algorithm Choice Rationale

- **TF-IDF**: Effective for text classification, handles vocabulary size well
- **Logistic Regression**: Fast, interpretable, works well with TF-IDF
- **One-vs-Rest**: Standard approach for multi-label classification
- **Multi-label Binarizer**: Efficient handling of multiple genres

## Project Structure

```
movie-genre-classification/
│
├── movie_genre_classification.ipynb    # Main notebook with complete analysis
├── classified_movies_results.csv       # Model predictions output
├── genre_distribution.png             # Visualization of genre distribution
├── README.md                          # Project documentation
├── tmdb_5000_movies.csv                # Movies dataset
└── tmdb_5000_credits.csv               # Credits dataset
```

## Acknowledgments

- **TMDB**: For providing the comprehensive movie dataset

---

*NOTE: This is an educational/portfolio project using publicly available TMDB movie data from Kaggle. The results demonstrate technical competency in machine learning pipeline development, multi-label classification, and natural language processing. This project is intended for learning purposes and to showcase data science skills*
