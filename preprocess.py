import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

# Load dataset (comma separated)
df = pd.read_csv("student-mat.csv", sep=',')

# Select relevant columns
df = df[['studytime', 'absences', 'G1', 'G2', 'health', 'G3']]

# Convert final grade into performance category
def categorize(g3):
    if g3 >= 85:
        return 3  # Excellent
    elif g3 >= 70:
        return 2  # Good
    elif g3 >= 50:
        return 1  # Average
    else:
        return 0  # At Risk

df['performance'] = df['G3'].apply(categorize)
df.drop('G3', axis=1, inplace=True)

# Split features and target
X = df.drop('performance', axis=1).values
y = df['performance'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

print("Preprocessing complete")


