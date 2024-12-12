import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load Data
df = pd.read_excel('PURINEDATABASEANDDATASOURCES2023.XLSX', sheet_name='Table1_Food data_NAm sources')
df = df.iloc[:-8]

# Rename Columns
df.rename(columns={
    'Adenine': 'Adenine_Mean',
    'Unnamed: 4': 'Adenine_SE',
    'Unnamed: 5': 'Adenine_Min',
    'Unnamed: 6': 'Adenine_Max',
    'Guanine': 'Guanine_Mean',
    'Unnamed: 8': 'Guanine_SE',
    'Unnamed: 9': 'Guanine_Min',
    'Unnamed: 10': 'Guanine_Max',
    'Hypoxanthine': 'Hypoxanthine_Mean',
    'Unnamed: 12': 'Hypoxanthine_SE',
    'Unnamed: 13': 'Hypoxanthine_Min',
    'Unnamed: 14': 'Hypoxanthine_Max',
    'Xanthine': 'Xanthine_Mean',
    'Unnamed: 16': 'Xanthine_SE',
    'Unnamed: 17': 'Xanthine_Min',
    'Unnamed: 18': 'Xanthine_Max'
}, inplace=True)

# Drop irrelevant rows and strip column names
df = df.drop(index=0)
df.columns = df.columns.str.strip()

# Step 2: Handle Missing Data
numerical_cols = ['Adenine_Mean', 'Guanine_Mean', 'Hypoxanthine_Mean', 'Xanthine_Mean']
categorical_cols = ['Category', 'Country of Origin for Sample']

# Convert invalid entries to NaN
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Use KNN Imputer for numerical columns
imputer = KNNImputer(n_neighbors=5)
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Impute categorical columns with mode
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
df['Category'] = LabelEncoder().fit_transform(df['Category'])

# Step 3: Correlation Analysis
numeric_columns = df.select_dtypes(include=['number']).columns  # Select numeric columns
correlation_matrix = df[numeric_columns].corr()  # Compute correlation only for numeric data

# Step 4: Prepare Data for Model
X = df.drop(['Total'], axis=1)
X = X.select_dtypes(include=['number'])
y = pd.to_numeric(df['Total'], errors='coerce')

# Remove rows with NaN in target
y = y.dropna()
X = X.loc[y.index]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Build and Train MLP Model
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])

# Initialize PDF
with PdfPages('visualizations.pdf') as pdf:
    # Step 6: Visualize Loss Curve
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Function')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    pdf.savefig()  # Save to PDF
    plt.close()

    # Step 7: Predictions and Metrics
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")
    print(f"Accuracy (R^2 as %): {r2 * 100:.2f}%")

    # Visualization: Predictions vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid()
    pdf.savefig()  # Save to PDF
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    pdf.savefig()  # Save to PDF
    plt.close()

# Step 8: Add KNN Model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_knn_pred = knn.predict(X_test)

knn_rmse = np.sqrt(mean_squared_error(y_test, y_knn_pred))
knn_mae = mean_absolute_error(y_test, y_knn_pred)
knn_r2 = r2_score(y_test, y_knn_pred)

print(f"KNN Root Mean Squared Error (RMSE): {knn_rmse}")
print(f"KNN Mean Absolute Error (MAE): {knn_mae}")
print(f"KNN R^2 Score: {knn_r2}")
print(f"KNN Accuracy (R^2 as %): {knn_r2 * 100:.2f}%")
