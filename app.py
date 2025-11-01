import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Zomato Restaurant Clustering",
    page_icon="🍽️",
    layout="wide"
)

# Title and description
st.title("🍽️ Zomato Restaurant Clustering Analysis")
st.markdown("""
    This application performs unsupervised clustering analysis on Zomato restaurant data.
    Explore restaurant patterns and discover insights through interactive visualizations.
""")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('zomato.csv', encoding='latin-1')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'zomato.csv' is in the repository.")
        return None

df = load_data()

if df is not None:
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Show data overview
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("📊 Raw Dataset")
        st.dataframe(df.head(100))
        st.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Data preprocessing
    st.subheader("🔧 Data Preprocessing & Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Restaurants", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        if 'Country Code' in df.columns:
            st.metric("Countries", df['Country Code'].nunique())
    
    # Feature selection for clustering
    st.subheader("🎯 Clustering Configuration")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 0:
        selected_features = st.multiselect(
            "Select features for clustering:",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))]
        )
        
        n_clusters = st.slider("Number of clusters:", min_value=2, max_value=10, value=3)
        
        if st.button("Run Clustering Analysis") and len(selected_features) >= 2:
            with st.spinner("Performing clustering analysis..."):
                # Prepare data
                df_cluster = df[selected_features].dropna()
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_cluster)
                
                # Perform K-Means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                
                # Add cluster labels to dataframe
                df_cluster['Cluster'] = clusters
                
                # Display clustering results
                st.success(f"✅ Successfully created {n_clusters} clusters!")
                
                # Cluster distribution
                st.subheader("📈 Cluster Distribution")
                cluster_counts = pd.DataFrame(df_cluster['Cluster'].value_counts()).reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                
                fig_bar = px.bar(cluster_counts, x='Cluster', y='Count', 
                                title='Number of Restaurants per Cluster',
                                color='Cluster')
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # 2D/3D Visualization
                if len(selected_features) >= 2:
                    st.subheader("🎨 Cluster Visualization")
                    
                    if len(selected_features) >= 3:
                        # 3D scatter plot
                        fig_3d = px.scatter_3d(
                            df_cluster, 
                            x=selected_features[0], 
                            y=selected_features[1], 
                            z=selected_features[2],
                            color='Cluster',
                            title='3D Cluster Visualization',
                            labels={'Cluster': 'Cluster'},
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        # 2D scatter plot
                        fig_2d = px.scatter(
                            df_cluster,
                            x=selected_features[0],
                            y=selected_features[1],
                            color='Cluster',
                            title='2D Cluster Visualization',
                            labels={'Cluster': 'Cluster'},
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig_2d, use_container_width=True)
                
                # Cluster statistics
                st.subheader("📊 Cluster Statistics")
                cluster_stats = df_cluster.groupby('Cluster')[selected_features].mean()
                st.dataframe(cluster_stats)
                
                # Download clustered data
                st.subheader("💾 Download Results")
                csv = df_cluster.to_csv(index=False)
                st.download_button(
                    label="Download Clustered Data as CSV",
                    data=csv,
                    file_name="zomato_clustered.csv",
                    mime="text/csv"
                )
        
        elif len(selected_features) < 2:
            st.warning("⚠️ Please select at least 2 features for clustering.")
    
    else:
        st.error("No numeric columns found in the dataset for clustering.")
    
    # Additional insights
    st.subheader("📌 Dataset Information")
    with st.expander("View Column Information"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(col_info)

else:
    st.error("Unable to load data. Please check if 'zomato.csv' exists in the repository.")

# Footer
st.markdown("---")
st.markdown("### 🚀 Ready to Deploy on Streamlit Cloud")
st.markdown("""
    **Deployment Instructions:**
    1. Ensure all data files are in the repository
    2. Create a `requirements.txt` with: streamlit, pandas, numpy, plotly, scikit-learn
    3. Deploy directly from GitHub via Streamlit Cloud
""")
