# 🧠 Face Recognition Using PCA and ANN

This project implements a face recognition system using **Principal Component Analysis (PCA)** for feature extraction and an **Artificial Neural Network (ANN)** for classification. It uses Python libraries such as NumPy, SciPy, and OpenCV.

## 📁 Dataset

Download the dataset from [this GitHub link](https://github.com/robaita/introduction_to_machine_learning/blob/main/dataset.zip).

## 🛠 Libraries Used

- `NumPy`, `SciPy`: Matrix operations, SVD, eigen decomposition
- `OpenCV`: Image reading and preprocessing
- `Python`: PCA and ANN implementation

---

## 🚀 Project Workflow

### 🔧 Training Phase

1. **Generate Face Database**
   - Convert each image to a column vector.
   - Stack vectors to form a matrix of size `mn × p`.

2. **Mean Calculation**
   - Compute the mean face vector.

3. **Mean Normalization**
   - Subtract the mean face from each image.

4. **Covariance Matrix**
   - Use surrogate covariance method (Turk & Pentland) to compute a `p × p` matrix.

5. **Eigen Decomposition**
   - Extract eigenvalues and eigenvectors.
   - Sort eigenvalues and select top `k` eigenvectors.

6. **Feature Vector Generation**
   - Project mean-normalized faces onto selected eigenvectors.

7. **Eigenfaces**
   - Generate eigenfaces from feature vectors.

8. **Signature Generation**
   - Project each face onto eigenfaces to get its signature.

9. **ANN Training**
   - Train a backpropagation neural network using the signature vectors.

---

### 🧪 Testing Phase

1. **Preprocess Test Image**
   - Convert to column vector and subtract mean face.

2. **Feature Projection**
   - Project onto eigenfaces to get test signature.

3. **Classification**
   - Use trained ANN to classify the test image.

---

## 📊 Evaluation

- **Accuracy vs. k Value**
  - Vary `k` and plot accuracy to analyze performance.

- **Imposter Detection**
  - Add unknown faces to test set and evaluate rejection capability.

---

## 📈 Output

- Accuracy graph vs. number of eigenfaces (`k`)
- Classification results from ANN
- Imposter recognition performance

---

## 🧪 Notes

- Use 60% of data for training and 40% for testing.
- Experiment with different values of `k` to optimize accuracy.

---

## 👨‍💻 Author

This project was implemented as part of a machine learning module focused on dimensionality reduction and neural networks.
