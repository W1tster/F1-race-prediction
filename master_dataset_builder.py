import pandas as pd
import numpy as np
import os
import ast
import time
from glob import glob

# Configuration
DATA_DIR = "all_session_data"
SESSIONS_DIR = "sessions"
OUTPUT_FILE = "master_dataset.csv"

driver_cache = {}  # Cache to avoid redundant API calls

# Normalize location names for file matching
def normalize_location_name(location):
    normalized = location.replace(' ', '_')
    normalized = normalized.replace('-', '_')
    normalized = normalized.replace('ã', '_')
    normalized = normalized.replace('á', '_')
    normalized = normalized.replace('é', '_')
    normalized = normalized.replace('ó', '_')
    return normalized

# Fetch driver info from OpenF1 API
def get_driver_info(session_key, driver_number):
    if session_key not in driver_cache:
        print(f"    -> Fetching driver list for Session {session_key}...")
        try:
            url = f"https://api.openf1.org/v1/drivers?session_key={session_key}&csv=true"
            df_d = pd.read_csv(url)
            
            session_lookup = {}
            if not df_d.empty:
                for _, row in df_d.iterrows():
                    d_num = row['driver_number']
                    session_lookup[d_num] = {
                        'name': row.get('name_acronym', row.get('broadcast_name', 'UNK')),
                        'team': row.get('team_name', 'Unknown')
                    }
            driver_cache[session_key] = session_lookup
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            print(f"    [!] Error fetching drivers: {e}")
            driver_cache[session_key] = {}
    
    sess_data = driver_cache.get(session_key, {})
    return sess_data.get(driver_number, {'name': 'UNK', 'team': 'Unknown'})

# Parse segment values and count purple/green sectors
def parse_segment_counts(segments_str):
    if pd.isna(segments_str):
        return 0, 0
    
    try:
        segments = ast.literal_eval(segments_str) if isinstance(segments_str, str) else segments_str
        purple_count = sum(1 for s in segments if s == 2051)  # Overall fastest
        green_count = sum(1 for s in segments if s == 2049)   # Personal best
        return purple_count, green_count
    except:
        return 0, 0

# Aggregate lap performance data for a driver
def aggregate_lap_data(session_key, driver_number, year, location, session_name):
    normalized_location = normalize_location_name(location)
    laps_file = f"{DATA_DIR}/laps/{year}_{normalized_location}_{session_name}_laps.csv"
    
    if not os.path.exists(laps_file):
        return {}
    
    try:
        df_laps = pd.read_csv(laps_file)
        df_driver = df_laps[df_laps['driver_number'] == driver_number].copy()
        
        if df_driver.empty:
            return {}
        
        # Filter out pit laps for clean performance metrics
        df_clean = df_driver[df_driver['is_pit_out_lap'] == False].copy()
        
        if df_clean.empty:
            df_clean = df_driver.copy()
        
        # Calculate lap statistics
        lap_stats = {
            'total_laps': len(df_driver),
            'best_lap_time': df_clean['lap_duration'].min() if 'lap_duration' in df_clean.columns else np.nan,
            'median_lap_time': df_clean['lap_duration'].median() if 'lap_duration' in df_clean.columns else np.nan,
            'avg_lap_time': df_clean['lap_duration'].mean() if 'lap_duration' in df_clean.columns else np.nan,
        }
        
        # Sector statistics
        for sector_num in [1, 2, 3]:
            sector_col = f'duration_sector_{sector_num}'
            if sector_col in df_clean.columns:
                lap_stats[f'best_sector_{sector_num}'] = df_clean[sector_col].min()
                lap_stats[f'avg_sector_{sector_num}'] = df_clean[sector_col].mean()
            else:
                lap_stats[f'best_sector_{sector_num}'] = np.nan
                lap_stats[f'avg_sector_{sector_num}'] = np.nan
        
        # Speed statistics
        for speed_col in ['i1_speed', 'i2_speed', 'st_speed']:
            if speed_col in df_clean.columns:
                lap_stats[f'avg_{speed_col}'] = df_clean[speed_col].mean()
            else:
                lap_stats[f'avg_{speed_col}'] = np.nan
        
        # Count purple/green sectors
        purple_total, green_total = 0, 0
        for sector_num in [1, 2, 3]:
            segment_col = f'segments_sector_{sector_num}'
            if segment_col in df_driver.columns:
                for seg_str in df_driver[segment_col]:
                    p, g = parse_segment_counts(seg_str)
                    purple_total += p
                    green_total += g
        
        lap_stats['purple_sectors_count'] = purple_total
        lap_stats['green_sectors_count'] = green_total
        
        return lap_stats
        
    except Exception as e:
        print(f"    [!] Error processing laps for driver {driver_number}: {e}")
        return {}

# Aggregate tire stint data for a driver
def aggregate_stint_data(session_key, driver_number, year, location, session_name):
    normalized_location = normalize_location_name(location)
    stints_file = f"{DATA_DIR}/stints/{year}_{normalized_location}_{session_name}_stints.csv"
    
    if not os.path.exists(stints_file):
        return {}
    
    try:
        df_stints = pd.read_csv(stints_file)
        df_driver = df_stints[df_stints['driver_number'] == driver_number].copy()
        
        if df_driver.empty:
            return {}
        
        # Calculate stint lengths
        df_driver['stint_length'] = df_driver['lap_end'] - df_driver['lap_start'] + 1
        
        stint_stats = {
            'num_stints': len(df_driver),
            'compounds_used': ','.join(df_driver['compound'].unique()),
            'avg_stint_length': df_driver['stint_length'].mean(),
        }
        
        # Count laps per tire compound
        for compound in ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']:
            compound_laps = df_driver[df_driver['compound'] == compound]['stint_length'].sum()
            stint_stats[f'total_{compound.lower()}_laps'] = compound_laps if compound_laps > 0 else 0
        
        return stint_stats
        
    except Exception as e:
        print(f"    [!] Error processing stints for driver {driver_number}: {e}")
        return {}

# Aggregate weather data for a session
def aggregate_weather_data(session_key, year, location, session_name):
    normalized_location = normalize_location_name(location)
    weather_file = f"{DATA_DIR}/weather/{year}_{normalized_location}_{session_name}_weather.csv"
    
    if not os.path.exists(weather_file):
        return {}
    
    try:
        df_weather = pd.read_csv(weather_file)
        
        if df_weather.empty:
            return {}
        
        weather_stats = {
            'avg_air_temp': df_weather['air_temperature'].mean() if 'air_temperature' in df_weather.columns else np.nan,
            'avg_track_temp': df_weather['track_temperature'].mean() if 'track_temperature' in df_weather.columns else np.nan,
            'avg_humidity': df_weather['humidity'].mean() if 'humidity' in df_weather.columns else np.nan,
            'total_rainfall': df_weather['rainfall'].sum() if 'rainfall' in df_weather.columns else 0,
            'avg_wind_speed': df_weather['wind_speed'].mean() if 'wind_speed' in df_weather.columns else np.nan,
        }
        
        return weather_stats
        
    except Exception as e:
        print(f"    [!] Error processing weather: {e}")
        return {}

# Get race results (starting/finishing positions, points)
def get_race_results(session_key, driver_number, year, location):
    results = {}
    normalized_location = normalize_location_name(location)
    
    # Starting grid position
    grid_file = f"{DATA_DIR}/starting_grid/{year}_{normalized_location}_Race_starting_grid.csv"
    if os.path.exists(grid_file):
        try:
            df_grid = pd.read_csv(grid_file)
            grid_row = df_grid[df_grid['driver_number'] == driver_number]
            if not grid_row.empty:
                results['starting_position'] = grid_row.iloc[0]['position']
            else:
                results['starting_position'] = np.nan
        except:
            results['starting_position'] = np.nan
    else:
        results['starting_position'] = np.nan
    
    # Race results (finishing position, points, DNF status)
    result_file = f"{DATA_DIR}/session_result/{year}_{normalized_location}_Race_session_result.csv"
    if os.path.exists(result_file):
        try:
            df_results = pd.read_csv(result_file)
            result_row = df_results[df_results['driver_number'] == driver_number]
            if not result_row.empty:
                results['finishing_position'] = result_row.iloc[0]['position']
                results['points'] = result_row.iloc[0]['points']
                results['dnf'] = result_row.iloc[0].get('dnf', False)
                results['dns'] = result_row.iloc[0].get('dns', False)
                results['dsq'] = result_row.iloc[0].get('dsq', False)
            else:
                results['finishing_position'] = np.nan
                results['points'] = 0
                results['dnf'] = False
                results['dns'] = False
                results['dsq'] = False
        except:
            results['finishing_position'] = np.nan
            results['points'] = 0
            results['dnf'] = False
            results['dns'] = False
            results['dsq'] = False
    else:
        results['finishing_position'] = np.nan
        results['points'] = 0
        results['dnf'] = False
        results['dns'] = False
        results['dsq'] = False
    
    return results

# Main function to build the master dataset
def build_master_dataset():
    print("=== BUILDING F1 MASTER DATASET ===\n")
    
    master_data = []
    
    # Load all session metadata files
    session_files = glob(f"{SESSIONS_DIR}/session_*.csv")
    
    if not session_files:
        print(f"ERROR: No session files found in {SESSIONS_DIR}/")
        return
    
    print(f"Found {len(session_files)} session files")
    
    # Process each session file
    for session_file in sorted(session_files):
        print(f"\nProcessing {os.path.basename(session_file)}...")
        
        try:
            df_sessions = pd.read_csv(session_file)
        except Exception as e:
            print(f"  [!] Error reading {session_file}: {e}")
            continue
        
        # Process each session
        for idx, session_row in df_sessions.iterrows():
            session_key = session_row['session_key']
            meeting_key = session_row['meeting_key']
            year = session_row['year']
            location = session_row['location']
            circuit_short_name = session_row['circuit_short_name']
            session_name = session_row['session_name'].replace(' ', '_')
            session_type = session_row['session_type']
            
            print(f"  {year} {location} - {session_name} (Session {session_key})")
            
            # Get list of drivers from lap data
            normalized_location = normalize_location_name(location)
            laps_file = f"{DATA_DIR}/laps/{year}_{normalized_location}_{session_name}_laps.csv"
            
            if not os.path.exists(laps_file):
                print(f"    [!] No lap data found, skipping session")
                continue
            
            try:
                df_laps = pd.read_csv(laps_file)
                driver_numbers = df_laps['driver_number'].unique()
            except Exception as e:
                print(f"    [!] Error reading laps: {e}")
                continue
            
            # Build row for each driver in this session
            for driver_number in driver_numbers:
                driver_info = get_driver_info(session_key, driver_number)
                
                row_data = {
                    'year': year,
                    'meeting_key': meeting_key,
                    'session_key': session_key,
                    'location': location,
                    'circuit_short_name': circuit_short_name,
                    'session_name': session_name,
                    'session_type': session_type,
                    'driver_number': driver_number,
                    'driver_name': driver_info['name'],
                    'team_name': driver_info['team'],
                }
                
                # Aggregate all data sources
                lap_data = aggregate_lap_data(session_key, driver_number, year, location, session_name)
                row_data.update(lap_data)
                
                stint_data = aggregate_stint_data(session_key, driver_number, year, location, session_name)
                row_data.update(stint_data)
                
                weather_data = aggregate_weather_data(session_key, year, location, session_name)
                row_data.update(weather_data)
                
                # Add race results (only for race sessions)
                if 'Race' in session_name:
                    race_data = get_race_results(session_key, driver_number, year, location)
                    row_data.update(race_data)
                
                master_data.append(row_data)
    
    # Create and save DataFrame
    df_master = pd.DataFrame(master_data)
    df_master.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n=== SUCCESS ===")
    print(f"Created master dataset with {len(df_master)} rows")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"\nDataset shape: {df_master.shape}")
    print(f"\nColumns: {df_master.columns.tolist()}")
    print(f"\nSample data:")
    print(df_master.head())
    print(f"\nData info:")
    print(df_master.info())


if __name__ == "__main__":
    build_master_dataset()
