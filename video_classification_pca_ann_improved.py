import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

class PCAVideoClassification:
    def __init__(self, k=5):
        """
        Initialize PCA Video Classification System
        k: number of principal components to use
        """
        self.k = k
        self.mean_vector = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.feature_vector = None
        self.scaler = RobustScaler()  # Changed to RobustScaler for better handling of outliers
    
    def parse_duration(self, duration_str):
        """
        Parse ISO 8601 duration format (e.g., 'PT7M37S') to seconds
        """
        duration_str = duration_str.replace('PT', '')
        seconds = 0
        current_num = ''
        
        for char in duration_str:
            if char.isdigit():
                current_num += char
            else:
                if not current_num:
                    continue
                num = int(current_num)
                if char == 'H':
                    seconds += num * 3600
                elif char == 'M':
                    seconds += num * 60
                elif char == 'S':
                    seconds += num
                current_num = ''
        
        return seconds

    def log_transform(self, x):
        """
        Apply log transformation with handling of zeros and negative values
        """
        return np.log1p(np.abs(x)) * np.sign(x)
    
    def prepare_features(self, df):
        """
        Prepare numerical features from the video data
        """
        # Create a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # Convert duration to seconds
        df_copy['duration_seconds'] = df_copy['duration'].apply(self.parse_duration)
        
        # Convert published date to days since earliest date
        df_copy['published'] = pd.to_datetime(df_copy['published'])
        earliest_date = df_copy['published'].min()
        df_copy['days_since_pub'] = (df_copy['published'] - earliest_date).dt.days
        
        # Add adview column if it doesn't exist (for test data)
        if 'adview' not in df_copy.columns:
            df_copy['adview'] = 0
        
        # Convert to numeric, replacing any non-numeric values with 0
        feature_cols = ['adview', 'views', 'likes', 'dislikes', 'comment']
        numeric_df = df_copy[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Apply log transformation to handle large ranges
        for col in ['views', 'likes', 'comment']:
            numeric_df[col] = numeric_df[col].apply(self.log_transform)
        
        # Create engagement ratios
        numeric_df['likes_per_view'] = numeric_df['likes'] / (numeric_df['views'] + 1)
        numeric_df['comments_per_view'] = numeric_df['comment'] / (numeric_df['views'] + 1)
        numeric_df['dislikes_per_view'] = numeric_df['dislikes'] / (numeric_df['views'] + 1)
        
        # Stack all numerical features together
        feature_matrix = np.hstack([
            numeric_df.values,
            df_copy[['duration_seconds']].values,
            df_copy[['days_since_pub']].values
        ])
        
        return feature_matrix.astype(np.float64)
    
    def load_data(self, df):
        """
        Load and prepare video data
        """
        print("Preparing video features...")
        feature_matrix = self.prepare_features(df)
        self.data = self.scaler.fit_transform(feature_matrix)
        print(f"Feature matrix shape: {self.data.shape}")
        return self.data
    
    def calculate_mean(self):
        """
        Calculate the mean of each feature
        """
        print("Calculating mean vector...")
        self.mean_vector = np.mean(self.data, axis=0, keepdims=True)
        print(f"Mean vector shape: {self.mean_vector.shape}")
        return self.mean_vector
    
    def mean_zero_alignment(self):
        """
        Center the data by subtracting the mean
        """
        print("Performing mean zero alignment...")
        self.centered_data = self.data - self.mean_vector
        print(f"Centered data shape: {self.centered_data.shape}")
        return self.centered_data
    
    def calculate_covariance(self):
        """
        Calculate covariance matrix
        """
        print("Calculating covariance matrix...")
        self.covariance = np.dot(self.centered_data.T, self.centered_data) / (self.centered_data.shape[0] - 1)
        print(f"Covariance matrix shape: {self.covariance.shape}")
        return self.covariance
    
    def eigen_decomposition(self):
        """
        Perform eigenvalue decomposition
        """
        print("Computing eigenvalues and eigenvectors...")
        eigenvalues, eigenvectors = linalg.eigh(self.covariance)
        
        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        
        print(f"Eigenvalues shape: {self.eigenvalues.shape}")
        print(f"Eigenvectors shape: {self.eigenvectors.shape}")
        return self.eigenvalues, self.eigenvectors
    
    def select_components(self):
        """
        Select top k principal components
        """
        print(f"Selecting top {self.k} principal components...")
        self.feature_vector = self.eigenvectors[:, :self.k]
        print(f"Feature vector shape: {self.feature_vector.shape}")
        return self.feature_vector
    
    def transform_data(self):
        """
        Project data onto principal components
        """
        print("Transforming data to PCA space...")
        self.transformed_data = np.dot(self.centered_data, self.feature_vector)
        print(f"Transformed data shape: {self.transformed_data.shape}")
        return self.transformed_data
    
    def train(self, df):
        """
        Complete training pipeline
        """
        print("="*50)
        print("PCA TRAINING PHASE")
        print("="*50)
        
        self.load_data(df)
        self.calculate_mean()
        self.mean_zero_alignment()
        self.calculate_covariance()
        self.eigen_decomposition()
        self.select_components()
        features = self.transform_data()
        
        # Calculate explained variance
        total_var = np.sum(self.eigenvalues)
        explained_var = np.sum(self.eigenvalues[:self.k]) / total_var
        print(f"\nExplained variance ratio with {self.k} components: {explained_var:.2%}")
        
        print("\nPCA training completed!")
        print("="*50)
        return features
    
    def transform_new_data(self, df):
        """
        Transform new data using trained PCA
        """
        feature_matrix = self.prepare_features(df)
        X = self.scaler.transform(feature_matrix)  # Use transform, not fit_transform
        X_centered = X - self.mean_vector
        X_transformed = np.dot(X_centered, self.feature_vector)
        return X_transformed


class ImprovedANN:
    """
    Improved Artificial Neural Network for classification with two hidden layers
    """
    def __init__(self, input_size, hidden_sizes=[100, 50], output_size=10, learning_rate=0.001):
        self.lr = learning_rate
        
        # Initialize weights with better scaling
        self.W1 = np.random.randn(input_size, hidden_sizes[0]) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_sizes[0]))
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) / np.sqrt(hidden_sizes[0])
        self.b2 = np.zeros((1, hidden_sizes[1]))
        self.W3 = np.random.randn(hidden_sizes[1], output_size) / np.sqrt(hidden_sizes[1])
        self.b3 = np.zeros((1, output_size))
        
        # Add dropout rate
        self.dropout_rate = 0.5
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X, training=True):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        if training:
            self.m1 = np.random.binomial(1, 1-self.dropout_rate, size=self.a1.shape) / (1-self.dropout_rate)
            self.a1 *= self.m1
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        if training:
            self.m2 = np.random.binomial(1, 1-self.dropout_rate, size=self.a2.shape) / (1-self.dropout_rate)
            self.a2 *= self.m2
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Output layer gradients
        dz3 = output - y
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Second hidden layer gradients
        dz2 = np.dot(dz3, self.W3.T) * self.relu_derivative(self.z2)
        if hasattr(self, 'm2'):
            dz2 *= self.m2
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # First hidden layer gradients
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        if hasattr(self, 'm1'):
            dz1 *= self.m1
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights with momentum
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
    
    def train(self, X, y, epochs=1000, batch_size=128, verbose=True):
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                output = self.forward(batch_X, training=True)
                self.backward(batch_X, batch_y, output)
            
            if verbose and (epoch + 1) % 100 == 0:
                output = self.forward(X, training=False)
                loss = -np.mean(y * np.log(output + 1e-8))
                predictions = np.argmax(output, axis=1)
                accuracy = np.mean(predictions == np.argmax(y, axis=1))
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4%}")
    
    def predict(self, X):
        output = self.forward(X, training=False)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        return self.forward(X, training=False)


def load_and_preprocess_data(train_csv, test_csv):
    """
    Load and preprocess the video data
    """
    print("="*70)
    print("LOADING AND PREPROCESSING DATA")
    print("="*70)
    
    # Load data
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Convert category to numerical labels
    categories = sorted(train_df['category'].unique())
    cat_to_label = {cat: idx for idx, cat in enumerate(categories)}
    
    train_labels = train_df['category'].map(cat_to_label).values
    test_labels = test_df['category'].map(cat_to_label).values
    
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    print(f"Number of categories: {len(categories)}")
    print("\nCategory mapping:")
    for cat, label in cat_to_label.items():
        count_train = len(train_df[train_df['category'] == cat])
        count_test = len(test_df[test_df['category'] == cat])
        print(f"Category {cat} -> Label {label} (Train: {count_train}, Test: {count_test})")
    
    return train_df, test_df, train_labels, test_labels, len(categories)


def experiment_with_k_values(train_df, test_df, train_labels, test_labels, num_classes, k_values):
    """
    Experiment with different numbers of principal components
    """
    accuracies = []
    
    for k in k_values:
        print(f"\n{'='*50}")
        print(f"Testing with k = {k}")
        print(f"{'='*50}")
        
        # Train PCA
        pca = PCAVideoClassification(k=k)
        train_features = pca.train(train_df)
        
        # Train ANN
        ann = ImprovedANN(
            input_size=k,
            hidden_sizes=[100, 50],
            output_size=num_classes,
            learning_rate=0.001
        )
        
        # One-hot encode labels
        y_train_onehot = np.zeros((len(train_labels), num_classes))
        y_train_onehot[np.arange(len(train_labels)), train_labels] = 1
        
        print("\nTraining neural network...")
        ann.train(train_features, y_train_onehot, epochs=500, verbose=True)
        
        # Test
        test_features = pca.transform_new_data(test_df)
        predictions = ann.predict(test_features)
        
        accuracy = accuracy_score(test_labels, predictions)
        accuracies.append(accuracy)
        print(f"Accuracy with k={k}: {accuracy*100:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, [acc*100 for acc in accuracies], marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Principal Components (k)', fontsize=12)
    plt.ylabel('Classification Accuracy (%)', fontsize=12)
    plt.title('Video Classification Accuracy vs Number of Principal Components', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_vs_k.png', dpi=300)
    print("\nPlot saved as 'accuracy_vs_k.png'")
    
    return accuracies


def plot_confusion_matrix(y_true, y_pred, categories):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.ylabel('True Category', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\nConfusion matrix saved as 'confusion_matrix.png'")


if __name__ == "__main__":
    # Load and preprocess data
    TRAIN_CSV = "train_lyst1717074532669.csv"
    TEST_CSV = "test_lyst1717074532669.csv"
    
    train_df, test_df, train_labels, test_labels, num_classes = load_and_preprocess_data(
        TRAIN_CSV, TEST_CSV
    )
    
    # Single run with k=5
    print("\n" + "="*70)
    print("SINGLE RUN WITH k=5")
    print("="*70)
    
    # Train PCA
    pca = PCAVideoClassification(k=5)
    train_features = pca.train(train_df)
    
    # Train ANN
    ann = ImprovedANN(
        input_size=5,
        hidden_sizes=[100, 50],
        output_size=num_classes,
        learning_rate=0.001
    )
    
    # One-hot encode labels
    y_train_onehot = np.zeros((len(train_labels), num_classes))
    y_train_onehot[np.arange(len(train_labels)), train_labels] = 1
    
    print("\nTraining neural network...")
    ann.train(train_features, y_train_onehot, epochs=1000, batch_size=128, verbose=True)
    
    # Test
    print("\n" + "="*50)
    print("TESTING PHASE")
    print("="*50)
    
    test_features = pca.transform_new_data(test_df)
    predictions = ann.predict(test_features)
    probabilities = ann.predict_proba(test_features)
    
    # Calculate accuracy and display results
    accuracy = accuracy_score(test_labels, predictions)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    # Plot confusion matrix
    categories = sorted(train_df['category'].unique())
    plot_confusion_matrix(test_labels, predictions, categories)
    
    # Experiment with different k values
    print("\n" + "="*70)
    print("EXPERIMENTING WITH DIFFERENT K VALUES")
    print("="*70)
    
    k_values = [2, 3, 4, 5, 6, 7]
    accuracies = experiment_with_k_values(train_df, test_df, train_labels, test_labels, 
                                        num_classes, k_values)
    
    print("\n" + "="*70)
    print("Video Classification System Implementation Complete!")
    print("="*70)