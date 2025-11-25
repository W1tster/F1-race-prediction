"""
Test F1 Model with São Paulo Grand Prix 2025 Data
Fetches data from OpenF1 API and makes predictions
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model definition (same as training)
class F1DeepPredictor(nn.Module):
    def __init__(self, input_features=21):
        super(F1DeepPredictor, self).__init__()
        self.fc1 = nn.Linear(input_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

print("=" * 80)
print("SÃO PAULO GRAND PRIX 2025 - MODEL PREDICTION TEST")
print("=" * 80)

# Step 1: Find São Paulo GP session
print("\n[1/6] Finding São Paulo Grand Prix session...")
try:
    # URL encode São Paulo
    sessions_url = "https://api.openf1.org/v1/sessions?year=2025&location=S%C3%A3o%20Paulo&csv=true"
    df_sessions = pd.read_csv(sessions_url)
    
    if df_sessions.empty:
        print("ERROR: No São Paulo GP found for 2025")
        exit(1)
    
    print(f"  Found {len(df_sessions)} sessions")
    
    # Get session keys
    practice_sessions = df_sessions[df_sessions['session_name'].str.contains('Practice', na=False)]
    qualifying_session = df_sessions[df_sessions['session_name'] == 'Qualifying']
    race_session = df_sessions[df_sessions['session_name'] == 'Race']
    
    if qualifying_session.empty or race_session.empty:
        print("ERROR: Missing qualifying or race session")
        exit(1)
    
    practice_keys = practice_sessions['session_key'].tolist()
    quali_key = qualifying_session.iloc[0]['session_key']
    race_key = race_session.iloc[0]['session_key']
    
    print(f"  Practice sessions: {len(practice_keys)}")
    print(f"  Qualifying key: {quali_key}")
    print(f"  Race key: {race_key}")
    
except Exception as e:
    print(f"ERROR fetching session data: {e}")
    exit(1)

# Step 2: Fetch lap data for all sessions
print("\n[2/6] Fetching lap data from OpenF1 API...")

all_practice_laps = []
for session_key in practice_keys:
    try:
        laps_url = f"https://api.openf1.org/v1/laps?session_key={session_key}&csv=true"
        df_laps = pd.read_csv(laps_url)
        all_practice_laps.append(df_laps)
    except:
        pass

practice_laps = pd.concat(all_practice_laps, ignore_index=True) if all_practice_laps else pd.DataFrame()

quali_laps_url = f"https://api.openf1.org/v1/laps?session_key={quali_key}&csv=true"
quali_laps = pd.read_csv(quali_laps_url)

race_laps_url = f"https://api.openf1.org/v1/laps?session_key={race_key}&csv=true"
race_laps = pd.read_csv(race_laps_url)

print(f"  Practice laps: {len(practice_laps)}")
print(f"  Qualifying laps: {len(quali_laps)}")
print(f"  Race laps: {len(race_laps)}")

# Step 3: Get driver names and race results
print("\n[3/6] Fetching driver names and race results...")

# Fetch driver names
try:
    drivers_url = f"https://api.openf1.org/v1/drivers?session_key={race_key}&csv=true"
    df_drivers = pd.read_csv(drivers_url)
    
    # Get driver names
    driver_names = df_drivers[['driver_number', 'name_acronym']].drop_duplicates()
    driver_names.columns = ['driver_number', 'driver_name']
    
    print(f"  Got driver names for {len(driver_names)} drivers")
except Exception as e:
    print(f"  Warning: Could not fetch driver names: {e}")
    driver_names = pd.DataFrame()

# Try to get race results
try:
    # Get final positions from last lap of race
    final_lap = race_laps.groupby('driver_number')['lap_number'].max().reset_index()
    final_lap.columns = ['driver_number', 'final_lap_number']
    
    # Merge to get final positions - use lap count as proxy for finishing position
    # (more laps = better finish, typically)
    lap_counts = race_laps.groupby('driver_number')['lap_number'].max().reset_index()
    lap_counts.columns = ['driver_number', 'total_laps']
    lap_counts['finishing_position'] = lap_counts['total_laps'].rank(ascending=False, method='min').astype(int)
    
    final_positions = lap_counts[['driver_number', 'finishing_position']]
    
    print(f"  Got race results for {len(final_positions)} drivers")
    
except Exception as e:
    print(f"  Warning: Could not fetch race results: {e}")
    final_positions = pd.DataFrame()

# Step 4: Process data into model features (same as feature engineering)
print("\n[4/6] Processing data into model features...")

# Get driver list from race
drivers = race_laps['driver_number'].unique()

# Aggregate practice data per driver
practice_agg = practice_laps.groupby('driver_number').agg({
    'lap_duration': ['min', 'mean', 'count'],
    'duration_sector_1': 'min',
    'duration_sector_2': 'min',
    'duration_sector_3': 'min',
    'i1_speed': 'mean',
    'i2_speed': 'mean',
    'st_speed': 'mean',
}).reset_index()

practice_agg.columns = ['driver_number', 'practice_best_lap', 'practice_avg_best_lap', 'practice_total_laps',
                        'practice_best_s1', 'practice_best_s2', 'practice_best_s3',
                        'practice_avg_i1_speed', 'practice_avg_i2_speed', 'practice_avg_st_speed']

# Add dummy purple/green sectors (not available in API)
practice_agg['practice_purple_sectors'] = 0
practice_agg['practice_green_sectors'] = 0

# Add dummy tire data
practice_agg['practice_soft_laps'] = 0
practice_agg['practice_medium_laps'] = 0
practice_agg['practice_hard_laps'] = 0

# Aggregate qualifying data
quali_agg = quali_laps.groupby('driver_number').agg({
    'lap_duration': 'min',
    'duration_sector_1': 'min',
    'duration_sector_2': 'min',
    'duration_sector_3': 'min',
}).reset_index()

quali_agg.columns = ['driver_number', 'quali_best_lap', 'quali_best_s1', 'quali_best_s2', 'quali_best_s3']

# Get grid positions from qualifying results
grid_positions = quali_laps.groupby('driver_number')['lap_duration'].min().rank().astype(int).reset_index()
grid_positions.columns = ['driver_number', 'grid_position']

# Merge all data
dataset = practice_agg.merge(quali_agg, on='driver_number', how='inner')
dataset = dataset.merge(grid_positions, on='driver_number', how='inner')

# Add driver names
if not driver_names.empty:
    dataset = dataset.merge(driver_names, on='driver_number', how='left')
else:
    dataset['driver_name'] = 'Unknown'

# Create derived features
dataset['practice_total_sectors'] = dataset['practice_best_s1'] + dataset['practice_best_s2'] + dataset['practice_best_s3']
dataset['practice_vs_quali_improvement'] = dataset['practice_best_lap'] - dataset['quali_best_lap']
dataset['practice_total_tire_laps'] = dataset['practice_soft_laps'] + dataset['practice_medium_laps'] + dataset['practice_hard_laps']

# Merge with actual results
if not final_positions.empty:
    dataset = dataset.merge(final_positions, on='driver_number', how='left')

print(f"  Processed data for {len(dataset)} drivers")

# Step 5: Load model and make predictions
print("\n[5/6] Loading trained model...")
checkpoint = torch.load('f1_model.pth', weights_only=False)
scaler = checkpoint['scaler']
feature_cols = checkpoint['features']

model = F1DeepPredictor(input_features=len(feature_cols))
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Prepare features
X = dataset[feature_cols].values

# Handle NaN
nan_mask = np.isnan(X)
if nan_mask.any():
    col_means = np.nanmean(X, axis=0)
    for i in range(X.shape[1]):
        X[nan_mask[:, i], i] = col_means[i]

# Scale and predict
X_scaled = scaler.transform(X)
X_scaled = np.clip(X_scaled, -5, 5)

with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    predictions_raw = model(X_tensor).squeeze().numpy()
    
    # Rank drivers by raw prediction scores (lower score = better position)
    # This ensures each position 1-20 is assigned exactly once
    predicted_ranks = predictions_raw.argsort().argsort() + 1  # Convert to 1-indexed positions
    predictions = predicted_ranks.astype(int)

# Add predictions to dataset
dataset['predicted_position'] = predictions
dataset['predicted_raw'] = predictions_raw

# Step 6: Show results
print("\n[6/6] Results:")
print("\n" + "=" * 80)
print("SÃO PAULO GRAND PRIX 2025 - PREDICTIONS")
print("=" * 80)

# Just show predictions
results = dataset[['driver_number', 'driver_name', 'grid_position', 'predicted_position']].sort_values('predicted_position')

print(f"\n{'Driver':<25} {'Grid':<8} {'Predicted':<12}")
print("-" * 50)
for _, row in results.iterrows():
    driver = f"#{int(row['driver_number'])} {row['driver_name']}"
    grid = f"P{int(row['grid_position'])}"
    pred = f"P{int(row['predicted_position'])}"
    print(f"{driver:<25} {grid:<8} {pred:<12}")

print("\n" + "=" * 80)
