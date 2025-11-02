# Video Category Classification System

## Overview
This project implements a machine learning system for classifying videos into different categories based on their engagement metrics and characteristics. It uses Principal Component Analysis (PCA) for dimensionality reduction and a Neural Network for classification.

## Features
- Multi-class video classification
- PCA-based feature reduction
- Neural Network classifier using scikit-learn
- Comprehensive metrics analysis
- Support for various engagement features

## Requirements
- Python 3.x
- Required packages:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - scipy

## Installation
1. Clone the repository
2. Create a virtual environment:
```powershell
python -m venv venv
```
3. Activate the virtual environment:
```powershell
.\venv\Scripts\activate
```
4. Install required packages:
```powershell
pip install numpy pandas scikit-learn matplotlib scipy
```

## Data Features
The system uses the following features for classification:
- Video views
- Likes and dislikes
- Comment count
- Video duration
- Days since publication
- Engagement metrics (derived)
  - Likes per view
  - Comments per view
  - Dislikes per view
  - Overall engagement rate

## Model Architecture
### Feature Processing
1. **Log Transformation** for highly skewed features:
   - Views
   - Likes
   - Dislikes
   - Comments

2. **PCA Transformation**
   - Tested with components k=6 through k=10
   - Best performance with k=8 (99.24% explained variance)

3. **Neural Network**
   - MLPClassifier from scikit-learn
   - Adaptive learning rate
   - ReLU activation function
   - Adam optimizer

## Performance Metrics
### Best Configuration (k=8 components)
- Overall Accuracy: 29.26%
- Per-Category Performance:
  ```
  Category    Precision    Recall    F1-Score
  ---------------------------------------------
  A           0.10        0.62      0.18
  B           0.27        0.45      0.34
  C           0.18        0.42      0.26
  D           0.77        0.19      0.30
  E           0.33        0.29      0.31
  F           0.23        0.47      0.31
  G           0.37        0.39      0.38
  H           0.09        0.44      0.15
  ```

## Component Analysis
### Top PCA Components
1. Component 1: Engagement volume (likes, dislikes, comments)
2. Component 2: Engagement rates per view
3. Component 3: Video duration and timing metrics
4. Component 4: Dislike patterns
5. Component 5: Publication timing
6. Component 6: Comment engagement
7. Component 7: Comment-dislike relationships
8. Component 8: View-like relationships

## Usage
1. Activate the virtual environment:
```powershell
.\venv\Scripts\activate
```

2. Run the classification system:
```powershell
python video_classification_sklearn.py
```

## Model Output
The system provides:
- Classification accuracy for each k value
- Detailed classification reports
- PCA component analysis
- Training progress monitoring
- Validation scores during training

## Limitations and Considerations
1. Class Imbalance
   - Some categories have significantly more samples
   - Category D: 4,419 samples
   - Category A: 181 samples

2. Performance Variations
   - Higher precision for high-volume categories
   - Lower precision for minority classes
   - Trade-off between precision and recall

3. Feature Importance
   - Engagement metrics are most influential
   - Time-based features provide secondary patterns
   - View-normalized metrics create distinct patterns

## Future Improvements
1. Class Imbalance Solutions:
   - Implement SMOTE or other sampling techniques
   - Class weights adjustment
   - Stratified sampling

2. Model Enhancements:
   - Ensemble methods
   - Different neural network architectures
   - Feature engineering optimization

3. Performance Optimization:
   - Hyperparameter tuning
   - Cross-validation strategies
   - Alternative scaling methods

## Contributing
Contributions to improve the classification system are welcome. Please ensure to:
1. Follow the existing code style
2. Add tests for new features
3. Update documentation as needed
4. Submit pull requests with detailed descriptions

## License
This project is available under the MIT License.
