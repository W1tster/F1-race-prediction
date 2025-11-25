import pandas as pd
import numpy as np

print("=" * 80)
print("F1 FEATURE ENGINEERING - CREATING TRAINING DATASET")
print("=" * 80)

# Load master dataset
df = pd.read_csv('master_dataset.csv')
print(f"\nLoaded {len(df):,} rows from master_dataset.csv")

# Split by session type
practice = df[df['session_type'] == 'Practice'].copy()
qualifying = df[df['session_type'] == 'Qualifying'].copy()
race = df[df['session_type'] == 'Race'].copy()

print(f"\nSession breakdown:")
print(f"  Practice: {len(practice):,} rows")
print(f"  Qualifying: {len(qualifying):,} rows")  
print(f"  Race: {len(race):,} rows")

# Step 1: Aggregate practice sessions (P1, P2, P3 combined per driver/race)
print("\n" + "=" * 80)
print("Step 1: Aggregating Practice Sessions")
print("=" * 80)

practice_agg = practice.groupby(['meeting_key', 'driver_name']).agg({
    'best_lap_time': ['min', 'mean'],
    'total_laps': 'sum',
    'best_sector_1': 'min',
    'best_sector_2': 'min',
    'best_sector_3': 'min',
    'avg_i1_speed': 'mean',
    'avg_i2_speed': 'mean',
    'avg_st_speed': 'mean',
    'purple_sectors_count': 'sum',
    'green_sectors_count': 'sum',
    'total_soft_laps': 'sum',
    'total_medium_laps': 'sum',
    'total_hard_laps': 'sum',
}).reset_index()

# Flatten multi-level column names
practice_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in practice_agg.columns.values]

# Rename to meaningful names
practice_agg = practice_agg.rename(columns={
    'best_lap_time_min': 'practice_best_lap',
    'best_lap_time_mean': 'practice_avg_best_lap',
    'total_laps_sum': 'practice_total_laps',
    'best_sector_1_min': 'practice_best_s1',
    'best_sector_2_min': 'practice_best_s2',
    'best_sector_3_min': 'practice_best_s3',
    'avg_i1_speed_mean': 'practice_avg_i1_speed',
    'avg_i2_speed_mean': 'practice_avg_i2_speed',
    'avg_st_speed_mean': 'practice_avg_st_speed',
    'purple_sectors_count_sum': 'practice_purple_sectors',
    'green_sectors_count_sum': 'practice_green_sectors',
    'total_soft_laps_sum': 'practice_soft_laps',
    'total_medium_laps_sum': 'practice_medium_laps',
    'total_hard_laps_sum': 'practice_hard_laps',
})

print(f"  Aggregated practice data: {len(practice_agg):,} driver-race combinations")

# Step 2: Add qualifying data
print("\n" + "=" * 80)
print("Step 2: Adding Qualifying Data")
print("=" * 80)

quali_agg = qualifying.groupby(['meeting_key', 'driver_name']).agg({
    'best_lap_time': 'min',
    'best_sector_1': 'min',
    'best_sector_2': 'min',
    'best_sector_3': 'min',
}).reset_index().rename(columns={
    'best_lap_time': 'quali_best_lap',
    'best_sector_1': 'quali_best_s1',
    'best_sector_2': 'quali_best_s2',
    'best_sector_3': 'quali_best_s3',
})

print(f"  Qualifying data: {len(quali_agg):,} driver-race combinations")

# Step 3: Add race results (target variable) and grid position
print("\n" + "=" * 80)
print("Step 3: Adding Race Results and Grid Position")
print("=" * 80)

race_agg = race.groupby(['meeting_key', 'driver_name']).agg({
    'starting_position': 'first',
    'finishing_position': 'first',
    'points': 'first',
}).reset_index().rename(columns={
    'starting_position': 'grid_position'
})

print(f"  Race results: {len(race_agg):,} driver-race combinations")

# Step 4: Merge all data (practice + qualifying + race)
print("\n" + "=" * 80)
print("Step 4: Merging Everything")
print("=" * 80)

dataset = practice_agg.merge(quali_agg, on=['meeting_key', 'driver_name'], how='inner')
print(f"  After merging practice + qualifying: {len(dataset):,} rows")

dataset = dataset.merge(race_agg, on=['meeting_key', 'driver_name'], how='inner')
print(f"  After adding race results: {len(dataset):,} rows")

# Check for missing critical data
print("\nMissing values check:")
print(f"  practice_best_lap: {dataset['practice_best_lap'].isnull().sum()}")
print(f"  grid_position: {dataset['grid_position'].isnull().sum()}")
print(f"  finishing_position: {dataset['finishing_position'].isnull().sum()}")

# Remove incomplete records
dataset_before = len(dataset)
dataset = dataset[
    dataset['practice_best_lap'].notna() &
    dataset['grid_position'].notna() &
    dataset['finishing_position'].notna()
].copy()

print(f"\nRows removed due to missing data: {dataset_before - len(dataset)}")
print(f"Final complete records: {len(dataset):,}")

# Step 5: Create derived features
print("\n" + "=" * 80)
print("Step 5: Creating Derived Features")
print("=" * 80)

dataset['practice_total_sectors'] = dataset['practice_best_s1'] + dataset['practice_best_s2'] + dataset['practice_best_s3']
dataset['practice_vs_quali_improvement'] = dataset['practice_best_lap'] - dataset['quali_best_lap']
dataset['practice_total_tire_laps'] = dataset['practice_soft_laps'] + dataset['practice_medium_laps'] + dataset['practice_hard_laps']

print(f"  Added 3 derived features")

# Save training dataset
output_file = 'training_dataset.csv'
dataset.to_csv(output_file, index=False)

feature_cols = [col for col in dataset.columns if col not in ['meeting_key', 'driver_name', 'finishing_position', 'points']]

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 80)
print(f"\nFinal dataset saved to: {output_file}")
print(f"  Shape: {dataset.shape}")
print(f"  Training examples: {len(dataset):,}")
print(f"  Input features: {len(feature_cols)}")
print(f"  Target: finishing_position (1-20)")

print("\nSample (first 3 rows):")
print(dataset[['driver_name', 'practice_best_lap', 'grid_position', 'finishing_position']].head(3))

print("\nReady for neural network training!")
