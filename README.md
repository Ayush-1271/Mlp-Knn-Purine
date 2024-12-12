# Mlp-Knn-Purine
# Purine Database Analysis and Prediction

This repository contains code for analyzing and predicting data from the Purine Database using an MLP (Multi-Layer Perceptron) model and KNN (K-Nearest Neighbors) regression. The project focuses on imputing missing values, preprocessing data, training machine learning models, and visualizing results.

## Features

1. **Data Imputation**: Missing values in numerical columns are handled using KNN Imputation, and categorical columns are imputed with the mode.
2. **Data Encoding**: Categorical features are encoded using label encoding.
3. **Data Scaling**: Features are standardized for optimal model performance.
4. **MLP Model**:
   - Multi-layered perceptron with dropout layers for regularization.
   - Early stopping to avoid overfitting.
   - Visualization of loss function.
5. **KNN Regression**: Comparison of performance metrics with KNN regression.
6. **Visualization**:
   - Feature correlation heatmap.
   - Loss function graph.
   - Actual vs. Predicted values.

## Installation and Usage

### Prerequisites
Ensure the following libraries are installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `matplotlib`
- `seaborn`

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn openpyxl
```

### Steps to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/purine-analysis.git
   cd purine-analysis
   ```
2. Place the data file `PURINEDATABASEANDDATASOURCES2023.XLSX` in the root directory.
3. Run the script:
   ```bash
   python analysis.py
   ```
4. Check the `visualizations.pdf` file for the generated plots.

## Results

- **MLP Metrics**:
  - Root Mean Squared Error (RMSE): Computed on test data.
  - Mean Absolute Error (MAE): Computed on test data.
  - R-squared (R²): Indicates the accuracy of the model.

- **KNN Metrics**:
  - Comparison of RMSE, MAE, and R² values with MLP.

## Visualization
All plots, including the loss function graph, actual vs. predicted values scatter plot, and correlation heatmap, are saved in a single PDF (`visualizations.pdf`).

## File Structure

```
.
├── analysis.py         # Main script containing the code
├── PURINEDATABASEANDDATASOURCES2023.XLSX # Input data file
├── visualizations.pdf  # Generated visualizations
├── README.md           # Documentation file
```

## Contact
For questions or contributions, feel free to open an issue or submit a pull request!
