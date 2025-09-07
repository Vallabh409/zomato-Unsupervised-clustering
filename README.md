# Zomato – Restaurant Clustering (Unsupervised Learning)

Cluster restaurants using **metadata features** such as cost, cuisines, location, ratings, online ordering, and table booking availability to uncover natural groupings.

> **Repository:** `Vallabh409/zomato-Unsupervised-clustering`
> **Main asset:** `Zomato_Restaurant_clustering.ipynb`

---

## 📦 Project Structure

```
zomato-Unsupervised-clustering/
├── Zomato_Restaurant_clustering.ipynb
├── Zomato Restaurant names and Metadata.csv
└── Zomato Restaurant reviews.csv
```

## 🎯 Objective

The goal of this project is to apply **unsupervised clustering (K-Means)** on restaurant data to:

* Discover natural clusters of restaurants based on metadata (cuisines, cost, ratings, etc.)
* Enable **targeted marketing** for specific segments
* Provide **personalized recommendations**
* Support **market gap analysis** for business decisions

## 🛠️ Requirements

Create a fresh Python environment (≥ 3.9). Required libraries:

```bash
pip install -U pandas numpy scikit-learn matplotlib seaborn jupyter
```

Optional:

```bash
pip install -U plotly
```

> Notebook was originally built on **Google Colab**. If running locally, adjust file paths accordingly.

## 🗂️ Data

* **`Zomato Restaurant names and Metadata.csv`**: Restaurant name, cuisines, location, cost, rating, and service availability (online order/book table).
* **`Zomato Restaurant reviews.csv`**: Contains raw reviews (not the main focus of this clustering project).

> Ensure both files are in the same folder as the notebook or update paths inside the notebook.

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Vallabh409/zomato-Unsupervised-clustering.git
   cd zomato-Unsupervised-clustering
   ```
2. Launch Jupyter Notebook:

   ```bash
   jupyter notebook Zomato_Restaurant_clustering.ipynb
   ```
3. Run all cells in order. Adjust dataset paths if necessary.

## 📚 Methodology

1. **Data Cleaning & Preparation**

   * Remove irrelevant columns
   * Handle missing values in cuisines, location, restaurant type, ratings
   * Convert `cost` and `rating` to numeric
   * Encode categorical features (`online_order`, `book_table`)

2. **Exploratory Data Analysis (EDA)**

   * Visualize cost distributions, cuisines, and ratings
   * Analyze correlations between features

3. **Clustering**

   * Apply **K-Means** clustering on selected features
   * Determine optimal number of clusters using the **Elbow Method**

4. **Insights & Interpretation**

   * Analyze each cluster’s characteristics (e.g., high-cost fine dining vs. budget-friendly eateries)
   * Provide business insights for marketing, recommendations, and expansion

## 📝 Example Code Snippets

**K-Means clustering with elbow method:**

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Scale numeric features
df_scaled = StandardScaler().fit_transform(df_selected)

# Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init="auto")
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Fit model
kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
df["Cluster"] = kmeans.fit_predict(df_scaled)
```

## 📊 Outputs

* Cluster labels for each restaurant
* Distribution plots for cost, ratings, and cuisines
* Elbow plot to determine optimal clusters
* Cluster-wise insights and summaries

## 🔧 Troubleshooting

* **Incorrect paths:** Update CSV file paths in the notebook if not using Google Colab.
* **Imbalanced clusters:** Try different values of `k` or include/exclude certain features.

## 📄 License

Currently no license file included. Consider adding one (e.g., MIT, Apache-2.0) for clarity.

## 🙌 Acknowledgements

* Zomato dataset provider(s)
* Open-source libraries: pandas, numpy, scikit-learn, seaborn, matplotlib

## 👤 Author

**Vallabh Bashyakarla**
Project developed and maintained by Vallabh Bashyakarla.

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests to suggest improvements or add features.

## ✅ Checklist

* [ ] CSV files placed in the correct folder
* [ ] Paths updated in notebook if needed
* [ ] All dependencies installed
* [ ] Notebook runs successfully
* [ ] Clusters interpreted and results saved
