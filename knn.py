import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Encode Gender (Male/Female) to numeric
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    
    X = df[['Age', 'Gender', 'Height', 'Weight']] 
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

class KNNClassifier:
    def __init__(self, k, distance_metric='e'):
        self.k = k
        self.distance_metric = distance_metric
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate distances to all training points
            distances = np.sum(np.abs(self.X_train - x), axis=1) if self.distance_metric == 'm' else np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train.iloc[k_indices]
            
            # Majority vote
            prediction = k_nearest_labels.mode()[0]
            predictions.append(prediction)
            
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)

def visualize_results(y_test, predictions, title, k, dist):
    # Create confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(y_test)),
                yticklabels=sorted(set(y_test)))
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'results/at_k{k}_with_{dist}.png')
    plt.show()


# Load the data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('Obesity Classification.csv')

# Initialization
k=int(input("Enter the value of k(√110 = 10): "))
dist=input("euclidean(e) or manhattan(m): ")
knn = KNNClassifier(k, dist)
knn.fit(pd.DataFrame(X_train), y_train)

# Make predictions
predictions = knn.predict(X_test)

accuracy = knn.score(X_test, y_test)
print(f"\nAccuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
report = classification_report(y_test, predictions)
print(report)

report_df = pd.DataFrame([classification_report(y_test, predictions, output_dict=True)]).transpose()
report_df.to_csv(f'results/classification_report_k{k}_with_{dist}.csv')


visualize_results(y_test, predictions, "KNN Classification Results", k, dist)
