import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
except ImportError:
    pass

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
except ImportError:
    tf = None

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation='relu')(input_layer)
    encoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def main():
    print("Loading data...")
    raw = pd.read_csv('Welding Data Set_01_Raw_data.csv')
    res = pd.read_csv('Welding Data Set_01_result.csv')
    
    features = ['Thickness 1(mm)', 'Thickness 2(mm)', 'weld force(bar)', 'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
    
    print("Generating Pseudo-labels (Physically-meaningful Assignment)...")
    iso = IsolationForest(random_state=42)
    raw['anomaly_score_raw'] = iso.fit_predict(raw[features]) # Using decision_function was better, let's keep it
    raw['anomaly_score_raw'] = iso.decision_function(raw[features])
    raw['defect_type'] = 0
    
    # Normalize features to calculate physical scores properly
    scaler_phys = StandardScaler()
    raw_phys = pd.DataFrame(scaler_phys.fit_transform(raw[features]), columns=features, index=raw.index)
    
    # Define physical scores for heuristic matching
    # Type 1 (파임불량): High Force, High Voltage
    raw['score_type1'] = raw_phys['weld force(bar)'] + raw_phys['weld Voltage(v)']
    # Type 2 (용접부족): Low Current, Low Voltage, Low Time (so negative of these is high score)
    raw['score_type2'] = -(raw_phys['weld current(kA)'] + raw_phys['weld Voltage(v)'] + raw_phys['weld time(ms)'])
    # Type 3 (크랙발생): Long Time, High Force
    raw['score_type3'] = raw_phys['weld time(ms)'] + raw_phys['weld force(bar)']
    
    for date in raw['working time'].unique():
        day_res = res[res['working time'] == date]
        if day_res.empty: continue
            
        # Get exact list of required defect types for this day
        required_types = []
        for _, row_res in day_res.iterrows():
            req_cnt = int(row_res['defect'])
            req_type = int(row_res['defect type'])
            required_types.extend([req_type] * req_cnt)
            
        if not required_types: continue
        
        # Get the N most anomalous rows for this day
        N = len(required_types)
        day_rows = raw[raw['working time'] == date].sort_values('anomaly_score_raw').head(N)
        
        # Bipartite matching (Linear Sum Assignment) to assign the best physical rows to the required types
        cost_matrix = np.zeros((N, N))
        for i, idx in enumerate(day_rows.index):
            for j, r_type in enumerate(required_types):
                # Cost is negative score (since we want to minimize cost = maximize score)
                if r_type == 1: cost = -day_rows.loc[idx, 'score_type1']
                elif r_type == 2: cost = -day_rows.loc[idx, 'score_type2']
                elif r_type == 3: cost = -day_rows.loc[idx, 'score_type3']
                cost_matrix[i, j] = cost
                
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Assign the matched types
        for i, c in zip(row_ind, col_ind):
            raw.loc[day_rows.index[i], 'defect_type'] = required_types[c]
            
    print(f"Total labeled defects:\n{raw['defect_type'].value_counts()}")
    
    raw['is_defect'] = np.where(raw['defect_type'] > 0, 1, 0)
    
    # Save the labeled dataset
    output_csv = 'Welding_Data_Set_01_labeled.csv'
    # Drop the temporary score columns before saving
    raw.drop(columns=['score_type1', 'score_type2', 'score_type3']).to_csv(output_csv, index=False)
    print(f"\nSaved physically-assigned labeled dataset to {output_csv}")
    
    # Preprocessing
    X = raw[features]
    y_multi = raw['defect_type']
    y_bin = raw['is_defect']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_multi_train, y_multi_test, y_bin_train, y_bin_test = train_test_split(
        X_scaled, y_multi, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    
    # AutoEncoder
    if tf is not None:
        print("\n--- [Unsupervised Approach] AutoEncoder Anomaly Detection ---")
        X_train_normal = X_train[y_bin_train == 0]
        ae_model = build_autoencoder(input_dim=X_train.shape[1])
        ae_model.fit(X_train_normal, X_train_normal, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
        reconstructions = ae_model.predict(X_test, verbose=0)
        mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
        
        train_reconstructions = ae_model.predict(X_train_normal, verbose=0)
        train_mse = np.mean(np.power(X_train_normal - train_reconstructions, 2), axis=1)
        threshold = np.percentile(train_mse, 99.5)
        
        ae_preds = (mse > threshold).astype(int)
        ae_f1 = f1_score(y_bin_test, ae_preds)
        print(f"AutoEncoder Binary F1-score: {ae_f1:.4f}")
    
    # Supervised Learning Models
    neg_count = sum(y_bin_train == 0)
    pos_count = sum(y_bin_train == 1)
    scale_pos = neg_count / pos_count if pos_count > 0 else 1
    
    models_bin = {
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced')
    }
    models_multi = {
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced')
    }
    
    try:
        models_bin['XGBoost'] = XGBClassifier(random_state=42, scale_pos_weight=scale_pos, eval_metric='logloss')
        models_multi['XGBoost'] = XGBClassifier(random_state=42, eval_metric='mlogloss')
    except: pass
    try:
        models_bin['LightGBM'] = LGBMClassifier(random_state=42, is_unbalance=True, verbose=-1)
        models_multi['LightGBM'] = LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
    except: pass
    try:
        models_bin['CatBoost'] = CatBoostClassifier(random_state=42, scale_pos_weight=scale_pos, verbose=0)
        models_multi['CatBoost'] = CatBoostClassifier(random_state=42, auto_class_weights='Balanced', verbose=0)
    except: pass

    print("\n--- [Non-labeling Approach] Binary Classification ---")
    for name, model in models_bin.items():
        model.fit(X_train, y_bin_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_bin_test, preds)
        print(f"{name} Binary F1-score: {f1:.4f}")
        
    print("\n--- [Labeling Approach] Multi-class Classification ---")
    for name, model in models_multi.items():
        model.fit(X_train, y_multi_train)
        preds = model.predict(X_test)
        preds_bin = np.where(preds > 0, 1, 0)
        f1 = f1_score(y_bin_test, preds_bin)
        print(f"{name} Multi-class (evaluated as binary) F1-score: {f1:.4f}")

if __name__ == "__main__":
    main()
