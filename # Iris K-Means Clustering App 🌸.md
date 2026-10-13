# Iris K-Means Clustering App 🌸

An interactive Machine Learning web application that uses the K-Means Clustering algorithm to group Iris flowers based on their measurements.

## Features

* Predicts flower clusters using four measurements.
* Uses a trained K-Means model saved as a `.pkl` file.
* Applies a saved scaler when available.
* Interactive Streamlit interface.
* Displays the predicted cluster and input measurements.

## Technologies Used

* Python
* Machine Learning
* Scikit-learn
* Pandas
* Streamlit
* Pickle

## Project Structure

```text
Iris_KMeans_App/
├── app.py
├── iris_kmeans_model.pkl
└── README.md
```

## Installation

Install the required libraries:

```bash
pip install streamlit pandas scikit-learn
```

## Run the Application

1. Place `app.py` and `iris_kmeans_model.pkl` in the same folder.
2. Open the folder in VS Code.
3. Open the terminal.
4. Run:

```bash
streamlit run app.py
```

5. Open the local URL shown in the terminal.

## How to Use

1. Enter the sepal length and width.
2. Enter the petal length and width.
3. Click **Predict Cluster**.
4. View the predicted cluster.

## Machine Learning Algorithm

K-Means Clustering is an unsupervised learning algorithm that groups similar data points into clusters.

**Note:** Cluster numbers do not automatically correspond to Iris species names. A separate mapping is required to display species names.
