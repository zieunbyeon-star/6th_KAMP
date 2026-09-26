import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load defect data
df = pd.read_csv('Welding_Data_Set_01_defect.csv')
features = ['weld force(bar)', 'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
X = df[features]

# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)

# 4. Print weights (loadings)
components = pd.DataFrame(pca.components_, columns=features, index=['PCA1', 'PCA2'])
print("=== PCA Weights (Loadings) ===")
print(components.round(3))
print("\n=== Explained Variance Ratio ===")
print(pca.explained_variance_ratio_.round(3))
