import os
import pandas as pd
import geopandas as gpd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
import json
import warnings
from datetime import datetime
from pathlib import Path
import logging
from functools import lru_cache
import gc

# Import the SpatialInterpolator
try:
    from interpolation import SpatialInterpolator
except ImportError:
    logging.error("Failed to import SpatialInterpolator. Ensure interpolation.py is in the correct path.")
    SpatialInterpolator = None

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='geopandas')
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- Constants for file paths ---
BASE_DIR = Path(__file__).resolve().parent
SHAPEFILE_PATH = BASE_DIR / 'DISTRICTs_Corrected' / 'DISTRICTs_Corrected.shp'
STATE_AVERAGES_PATH = BASE_DIR / 'aggregated_data' / 'state_averages.csv'
DISTRICT_DETAILS_DIR = BASE_DIR / 'aggregated_data'
POINT_DATA_PARQUET_DIR = BASE_DIR / 'data_parquet'
ANALYTICS_DATA_DIR = BASE_DIR / 'analytics_data'

# --- Global data cache with lazy loading ---
app_data_cache = {
    'state_gdf': None,
    'district_gdf_master': None,
    'state_avg_data': None,
    'processed_state_timeseries_avg': None,
    'available_years': [],
    'min_year': None,
    'max_year': None,
    'all_states_list': [],
    'district_map': {},
    'initialization_complete': False,
    'analytics_cache': {}
}

# --- Helper Functions ---
def convert_numpy_types(data):
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(i) for i in data]
    elif isinstance(data, (np.integer, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8,
                           np.uint16, np.uint32, np.uint64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float16, np.float32, np.float64)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, (bool, np.bool_)):
        return bool(data)
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif isinstance(data, (int, float, str, bool)) or data is None:
        return data
    return data

def get_valid_range(values):
    if isinstance(values, pd.Series):
        valid_values = values[(values > 0) & values.notna() & ~np.isinf(values)]
    else: # assuming list-like
        valid_values_list = [v for v in values if v is not None and isinstance(v, (int, float)) and v > 0 and not np.isnan(v) and not np.isinf(v)]
        valid_values = pd.Series(valid_values_list) if valid_values_list else pd.Series(dtype=float) # ensure dtype for empty

    if valid_values.empty:
        return 1700.0, 2200.0 # Default range as float

    min_val = float(valid_values.min())
    max_val = float(valid_values.max())

    # Ensure max_val is greater than min_val if they are too close or equal, for legend stability
    if max_val <= min_val:
        max_val = min_val + 1.0 # Add a small delta

    return min_val, max_val

@lru_cache(maxsize=528)
def load_shapefiles():
    try:
        logger.info(f"Loading shapefile from: {SHAPEFILE_PATH}")
        if not SHAPEFILE_PATH.exists():
            logger.error(f"Shapefile not found at {SHAPEFILE_PATH}")
            return None, None

        district_gdf = gpd.read_file(SHAPEFILE_PATH)

        if district_gdf.crs is None:
            district_gdf.crs = 'EPSG:4326'
        elif district_gdf.crs != 'EPSG:4326':
            district_gdf = district_gdf.to_crs('EPSG:4326')

        rename_map = {}
        if 'STATE' not in district_gdf.columns:
            state_col = next((col for col in district_gdf.columns if col.upper() in ['STATE', 'ST_NM', 'STATE_NAME']), None)
            if state_col: rename_map[state_col] = 'STATE'

        if 'District_1' not in district_gdf.columns:
            district_col = next((col for col in district_gdf.columns if col.upper() in ['DISTRICT_1', 'DISTRICT', 'DT_NM', 'DIST_NAME']), None)
            if district_col: rename_map[district_col] = 'District_1'

        if rename_map: district_gdf.rename(columns=rename_map, inplace=True)

        if 'STATE' not in district_gdf.columns or 'District_1' not in district_gdf.columns:
            logger.error(f"Required columns 'STATE' or 'District_1' not found in shapefile. Found: {district_gdf.columns.tolist()}")
            return None, None

        district_gdf['STATE'] = district_gdf['STATE'].astype(str).str.upper().str.strip()
        district_gdf['District_1'] = district_gdf['District_1'].astype(str).str.upper().str.strip()
        
        #district_gdf = district_gdf[district_gdf.geometry.is_valid & ~district_gdf.geometry.is_empty & district_gdf.geometry.notna()]

        if district_gdf.empty:
            logger.error("No valid geometries found after cleaning shapefile.")
            return None, None

        state_gdf = district_gdf.dissolve(by='STATE', aggfunc='first').reset_index()

        state_gdf['geometry'] = state_gdf.geometry.simplify(0.01, preserve_topology=True)
        district_gdf['geometry'] = district_gdf.geometry.simplify(0.001, preserve_topology=True)
        print(district_gdf[district_gdf['STATE'] == 'RAJASTHAN']['District_1'].unique())
        logger.info(f"Loaded {len(state_gdf)} states and {len(district_gdf)} districts from shapefile.")
        return state_gdf, district_gdf

    except Exception as e:
        logger.error(f"Error loading shapefiles: {e}", exc_info=True)
        return None, None

def process_state_averages_csv():
    try:
        logger.info(f"Processing state averages from: {STATE_AVERAGES_PATH}")
        if not STATE_AVERAGES_PATH.exists():
            logger.error(f"State averages file not found at {STATE_AVERAGES_PATH}")
            return None, None, [], None, None, []

        df = pd.read_csv(STATE_AVERAGES_PATH, engine='c')

        state_col = next((col for col in df.columns if col.lower() in ['state', 'st_nm', 'state_name']), None)
        if not state_col:
            logger.error(f"State column not found in {STATE_AVERAGES_PATH}. Columns: {df.columns.tolist()}")
            return None, None, [], None, None, []
        if state_col != 'state': df.rename(columns={state_col: 'state'}, inplace=True)

        df['state'] = df['state'].astype(str).str.upper().str.strip()
        all_states_list = sorted(df['state'].unique().tolist())

        id_vars = ['state']
        date_cols = [col for col in df.columns if col not in id_vars and '_' in str(col) and str(col).split('_')[0].isdigit()]

        if not date_cols:
            logger.warning(f"No date-like columns (e.g., YYYY_MM_DD) found in {STATE_AVERAGES_PATH} for melting.")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                 logger.error("No numeric columns to calculate overall average state methane.")
                 return None, None, [], None, None, all_states_list

            df['avg_methane'] = df[numeric_cols].mean(axis=1, skipna=True)
            overall_avg = df[['state', 'avg_methane']].copy()
            overall_avg = overall_avg[overall_avg['avg_methane'].notna() & (overall_avg['avg_methane'] > 0)]
            return overall_avg, pd.DataFrame(), [], None, None, all_states_list


        melted_df = df.melt(id_vars=id_vars, value_vars=date_cols, var_name='date_str', value_name='methane_value')
        melted_df['methane_value'] = pd.to_numeric(melted_df['methane_value'], errors='coerce')
        melted_df = melted_df[melted_df['methane_value'].notna() & (melted_df['methane_value'] > 0)]

        if melted_df.empty:
            logger.warning(f"No valid methane data after melting from {STATE_AVERAGES_PATH}.")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols :
                df['avg_methane'] = df[numeric_cols].mean(axis=1, skipna=True)
                overall_avg = df[['state', 'avg_methane']].copy()
                overall_avg = overall_avg[overall_avg['avg_methane'].notna() & (overall_avg['avg_methane'] > 0)]
                return overall_avg, pd.DataFrame(), [], None, None, all_states_list
            return None, None, [], None, None, all_states_list


        def parse_date_str(date_str):
            try:
                parts = date_str.split('_')
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    return pd.Timestamp(int(parts[0]), int(parts[1]), 1) # Year, Month, Day=1
            except ValueError: pass # Handle cases like non-integer parts
            return pd.NaT

        melted_df['datetime'] = melted_df['date_str'].apply(parse_date_str)
        melted_df = melted_df[melted_df['datetime'].notna()]

        if melted_df.empty:
            logger.warning(f"No valid dates found after parsing 'date_str' in {STATE_AVERAGES_PATH}.")
            return None, None, [], None, None, all_states_list

        melted_df['year'] = melted_df['datetime'].dt.year
        melted_df['month'] = melted_df['datetime'].dt.month

        overall_avg = melted_df.groupby('state')['methane_value'].mean().reset_index()
        overall_avg.rename(columns={'methane_value': 'avg_methane'}, inplace=True)

        available_years = sorted(melted_df['year'].unique().tolist())
        min_year = min(available_years) if available_years else None
        max_year = max(available_years) if available_years else None

        logger.info(f"Processed state averages: {len(overall_avg)} states, years {min_year}-{max_year}.")
        gc.collect()
        return (overall_avg, melted_df[['state', 'year', 'month', 'methane_value']],
                available_years, min_year, max_year, all_states_list)

    except Exception as e:
        logger.error(f"Error processing state averages CSV: {e}", exc_info=True)
        return None, None, [], None, None, []

@lru_cache(maxsize=64) # Cache results for state details
def get_district_details_for_state(state_name_upper, year=None, month=None):
    cache_key = f"district_details_{state_name_upper}_{year}_{month}"
    if cache_key in app_data_cache.get('analytics_cache', {}):
        return app_data_cache['analytics_cache'][cache_key]

    state_dir = DISTRICT_DETAILS_DIR / state_name_upper
    district_details_path = state_dir / 'district_details.csv'

    if not district_details_path.exists():
        logger.warning(f"District details file not found for {state_name_upper} at {district_details_path}")
        app_data_cache.setdefault('analytics_cache', {})[cache_key] = (None, None)
        return None, None

    try:
        df = pd.read_csv(district_details_path, engine='c')
        df['state'] = state_name_upper # Add state column
        df.columns = [str(col).strip().lower() for col in df.columns]

        district_col = next((col for col in df.columns if 'district' in col), None)
        if not district_col:
            logger.error(f"No 'district' column in {district_details_path}. Columns: {df.columns.tolist()}")
            app_data_cache.setdefault('analytics_cache', {})[cache_key] = (None, None)
            return None, None
        if district_col != 'district': df.rename(columns={district_col: 'district'}, inplace=True)

        df['district'] = df['district'].astype(str).str.upper().str.strip()

        id_vars = ['district', 'state'] # 'state' is now in df
        date_cols = [col for col in df.columns if col not in id_vars and '_' in str(col) and str(col).split('_')[0].isdigit()]

        avg_district_methane = pd.DataFrame() # To store the results

        if year and month:
            month_str = str(month).zfill(2)
            target_cols_in_order = [f"{year}_{month_str}_01", f"{year}_{month_str}"]
            selected_col = next((col for col in target_cols_in_order if col in df.columns), None)

            if selected_col:
                logger.info(f"Using specific column '{selected_col}' for {state_name_upper} districts ({year}-{month}).")
                avg_district_methane = df[['state', 'district', selected_col]].copy()
                avg_district_methane.rename(columns={selected_col: 'avg_methane'}, inplace=True)
                avg_district_methane['avg_methane'] = pd.to_numeric(avg_district_methane['avg_methane'], errors='coerce')
            else:
                 logger.info(f"Specific column for {year}-{month} not found in {district_details_path}. Will calculate overall average.")

        if avg_district_methane.empty and date_cols:
            logger.info(f"Calculating overall average district methane for {state_name_upper} from {len(date_cols)} date columns.")
            numeric_cols_for_avg = []
            for col in date_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_cols_for_avg.append(col)

            if numeric_cols_for_avg:
                 df['avg_methane'] = df[numeric_cols_for_avg].mean(axis=1, skipna=True)
                 avg_district_methane = df[['state', 'district', 'avg_methane']].copy()
            else:
                 logger.warning(f"No numeric date columns to calculate average for {state_name_upper} in {district_details_path}")

        if avg_district_methane.empty:
            logger.warning(f"No valid district methane data derived for {state_name_upper} from {district_details_path}.")
            app_data_cache.setdefault('analytics_cache', {})[cache_key] = (None, None)
            return None, None

        avg_district_methane = avg_district_methane[avg_district_methane['avg_methane'].notna() & (avg_district_methane['avg_methane'] > 0)]

        if avg_district_methane.empty:
            logger.warning(f"All district methane values are invalid or zero for {state_name_upper}.")
            app_data_cache.setdefault('analytics_cache', {})[cache_key] = (None, None)
            return None, None

        valid_values = avg_district_methane['avg_methane']
        data_min, data_max = get_valid_range(valid_values)
        stats = {
            'mean': float(valid_values.mean()), 'min': float(valid_values.min()),
            'max': float(valid_values.max()), 'median': float(valid_values.median()),
            'std': float(valid_values.std()), 'count': int(len(valid_values)),
            'data_min': data_min, 'data_max': data_max
        }

        result = (avg_district_methane, stats)
        app_data_cache.setdefault('analytics_cache', {})[cache_key] = result
        return result

    except Exception as e:
        logger.error(f"Error loading district details for {state_name_upper}: {e}", exc_info=True)
        app_data_cache.setdefault('analytics_cache', {})[cache_key] = (None, None)
        return None, None

@lru_cache(maxsize=256) # Cache results for point data
def get_point_data_for_district(state_name_upper, district_name_upper, year=None, month=None):
    """Load point data from Parquet files, no sampling."""
    cache_key = f"point_data_{state_name_upper}_{district_name_upper}_{year}_{month}"
    if cache_key in app_data_cache.get('analytics_cache', {}):
        cached_data = app_data_cache['analytics_cache'][cache_key]
        if cached_data and cached_data != (None, None):
             return cached_data

    state_dir_path = POINT_DATA_PARQUET_DIR / state_name_upper
    if not state_dir_path.exists():
        found_dir = next((d for d in POINT_DATA_PARQUET_DIR.iterdir() if d.is_dir() and d.name.upper() == state_name_upper), None)
        if not found_dir:
            logger.warning(f"State Parquet directory not found for '{state_name_upper}' in {POINT_DATA_PARQUET_DIR}")
            app_data_cache.setdefault('analytics_cache', {})[cache_key] = (None, None)
            return None, None
        state_dir_path = found_dir

    parquet_path = None
    district_name_cleaned = district_name_upper.replace(' ', '_')
    possible_stems = [district_name_upper, district_name_cleaned, district_name_upper.replace(' ', '')]

    for stem_candidate in possible_stems:
        for file_in_dir in state_dir_path.glob('*.parquet'):
            if file_in_dir.stem.upper() == stem_candidate.upper():
                parquet_path = file_in_dir
                break
        if parquet_path: break

    if not parquet_path:
        logger.warning(f"Parquet file not found for district '{district_name_upper}' (tried stems: {possible_stems}) in {state_dir_path}")
        app_data_cache.setdefault('analytics_cache', {})[cache_key] = (None, None)
        return None, None

    try:
        df = pd.read_parquet(parquet_path)
        df.columns = [str(col).strip().lower() for col in df.columns]

        lat_col = next((col for col in df.columns if 'lat' in col or 'latitude' in col), None)
        lon_col = next((col for col in df.columns if 'lon' in col or 'long' in col or 'longitude' in col), None)

        if not lat_col or not lon_col:
            logger.error(f"Latitude/Longitude columns ('lat*', 'lon*') not found in {parquet_path}. Columns: {df.columns.tolist()}")
            app_data_cache.setdefault('analytics_cache', {})[cache_key] = (None, None)
            return None, None

        df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'}, inplace=True)

        id_vars = ['latitude', 'longitude']
        date_cols = [
            col for col in df.columns
            if col not in id_vars and
            isinstance(col, str) and '_' in col and
            col.split('_')[0].isdigit() and
            (len(col.split('_')) > 1 and col.split('_')[1].isdigit())
        ]

        df_filtered = pd.DataFrame()

        if year and month:
            month_str = str(month).zfill(2)
            target_col_patterns = [f"{year}_{month_str}_01", f"{year}_{month_str}"]
            selected_col = next((col for col in target_col_patterns if col in df.columns), None)

            if selected_col:
                logger.info(f"Using specific column '{selected_col}' for point data in {district_name_upper} ({year}-{month}).")
                df_filtered = df[['latitude', 'longitude', selected_col]].copy()
                df_filtered.rename(columns={selected_col: 'avg_methane'}, inplace=True)
                df_filtered['avg_methane'] = pd.to_numeric(df_filtered['avg_methane'], errors='coerce')

        if df_filtered.empty and date_cols:
            logger.info(f"Calculating overall average point methane for {district_name_upper} from {len(date_cols)} date columns.")
            numeric_cols_for_avg = []
            for col in date_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_cols_for_avg.append(col)

            if numeric_cols_for_avg:
                df['avg_methane'] = df[numeric_cols_for_avg].mean(axis=1, skipna=True)
                df_filtered = df[['latitude', 'longitude', 'avg_methane']].copy()
            else:
                logger.warning(f"No numeric date columns to calculate average for {district_name_upper} in {parquet_path}")

        if df_filtered.empty:
             logger.warning(f"No methane data (df_filtered empty) for {district_name_upper} from {parquet_path}")
             app_data_cache.setdefault('analytics_cache', {})[cache_key] = (None, None)
             return None, None


        df_filtered = df_filtered[
            (df_filtered['avg_methane'].notna()) & (df_filtered['avg_methane'] > 0) &
            (df_filtered['latitude'].notna()) & (df_filtered['latitude'].between(-90, 90)) &
            (df_filtered['longitude'].notna()) & (df_filtered['longitude'].between(-180, 180))
        ]

        if df_filtered.empty:
            logger.warning(f"No valid point data after filtering (lat/lon/value checks) for {district_name_upper}.")
            app_data_cache.setdefault('analytics_cache', {})[cache_key] = (None, None)
            return None, None

        points_list = df_filtered[['latitude', 'longitude', 'avg_methane']].values.tolist()

        min_lat, max_lat = (df_filtered['latitude'].min(), df_filtered['latitude'].max())
        min_lon, max_lon = (df_filtered['longitude'].min(), df_filtered['longitude'].max())

        bounds_dict = {
            'min_lat': float(min_lat), 'max_lat': float(max_lat),
            'min_lon': float(min_lon), 'max_lon': float(max_lon)
        }

        valid_values = df_filtered['avg_methane']
        data_min, data_max = get_valid_range(valid_values)

        point_stats = {
            'mean': float(valid_values.mean()), 'min': float(valid_values.min()),
            'max': float(valid_values.max()), 'median': float(valid_values.median()),
            'std': float(valid_values.std()), 'count': int(len(valid_values)),
            'data_min': data_min, 'data_max': data_max
        }

        result = ({'points': points_list, 'bounds': bounds_dict}, point_stats)
        app_data_cache.setdefault('analytics_cache', {})[cache_key] = result
        return result

    except Exception as e:
        logger.error(f"Error loading point data for {district_name_upper} from {parquet_path}: {e}", exc_info=True)
        app_data_cache.setdefault('analytics_cache', {})[cache_key] = (None, None)
        return None, None


def initialize_global_data():
    if app_data_cache['initialization_complete']: return
    logger.info("Initializing global data...")

    state_gdf, district_gdf = load_shapefiles()
    app_data_cache['state_gdf'] = state_gdf
    app_data_cache['district_gdf_master'] = district_gdf

    if district_gdf is not None:
        for state in district_gdf['STATE'].unique():
            districts = sorted(district_gdf[district_gdf['STATE'] == state]['District_1'].unique().tolist())
           
            app_data_cache['district_map'][state] = districts
            
    
    results = process_state_averages_csv()
    if results:
        (overall_avg, processed_ts, years, min_y, max_y, states_list) = results
        app_data_cache['state_avg_data'] = overall_avg
        app_data_cache['processed_state_timeseries_avg'] = processed_ts
        app_data_cache['available_years'] = years
        app_data_cache['min_year'] = min_y
        app_data_cache['max_year'] = max_y
        app_data_cache['all_states_list'] = states_list

        if (app_data_cache['state_gdf'] is not None and
            app_data_cache['state_avg_data'] is not None and
            not app_data_cache['state_avg_data'].empty):
            state_gdf_copy = app_data_cache['state_gdf'].copy()
            state_avg_data_copy = app_data_cache['state_avg_data'].copy()

            merged_gdf = state_gdf_copy.merge(
                state_avg_data_copy,
                left_on='STATE',
                right_on='state',
                how='left'
            )
            if 'state' in merged_gdf.columns:
                merged_gdf.drop(columns=['state'], inplace=True)

            if 'avg_methane' not in merged_gdf.columns:
                merged_gdf['avg_methane'] = 0.0
            else:
                merged_gdf['avg_methane'] = merged_gdf['avg_methane'].fillna(0.0)

            app_data_cache['state_gdf'] = merged_gdf
        elif app_data_cache['state_gdf'] is not None and 'avg_methane' not in app_data_cache['state_gdf'].columns:
            app_data_cache['state_gdf']['avg_methane'] = 0.0


    app_data_cache['initialization_complete'] = True
    logger.info("Global data initialization completed.")

# --- API Endpoints ---
@app.route('/')
def index_route():
    return render_template('index.html')

@app.route('/api/metadata')
def metadata():
    if not app_data_cache.get('initialization_complete', False):
        logger.warning("Metadata requested before full initialization.")

    total_states = 0
    current_state_gdf = app_data_cache.get('state_gdf')
    if current_state_gdf is not None and not current_state_gdf.empty:
        total_states = len(current_state_gdf['STATE'].unique())
    elif app_data_cache.get('all_states_list'):
        total_states = len(app_data_cache['all_states_list'])

    data = {
        "years": app_data_cache.get('available_years', []),
        "min_year": app_data_cache.get('min_year'),
        "max_year": app_data_cache.get('max_year'),
        "total_states": total_states,
        "all_states_list": app_data_cache.get('all_states_list', []),
        "district_map": app_data_cache.get('district_map', {})
    }
    return jsonify(convert_numpy_types(data))

@app.route('/api/india/<int:year>/<int:month>')
def india_data(year, month):
    if not app_data_cache['initialization_complete']:
        return jsonify({"error": "Data not initialized"}), 500

    viz_type = request.args.get('viz', 'choropleth')

    state_gdf_orig = app_data_cache.get('state_gdf')
    if state_gdf_orig is None or state_gdf_orig.empty:
        return jsonify({"error": "State geometries not loaded"}), 500

    state_gdf = state_gdf_orig.copy()
    processed_ts = app_data_cache.get('processed_state_timeseries_avg')

    if 'methane_ppb' in state_gdf.columns: state_gdf = state_gdf.drop(columns=['methane_ppb'], errors='ignore')
    if 'avg_methane' in state_gdf.columns and 'avg_methane' != 'methane_ppb':
         state_gdf.rename(columns={'avg_methane': 'methane_ppb'}, inplace=True)
    else:
         state_gdf['methane_ppb'] = 0.0


    if processed_ts is not None and not processed_ts.empty:
        monthly_data = processed_ts[(processed_ts['year'] == year) & (processed_ts['month'] == month)]
        if not monthly_data.empty:
            logger.info(f"India: Using monthly data for {year}-{month}.")
            state_gdf = state_gdf.drop(columns=['methane_ppb'], errors='ignore')
            state_gdf = state_gdf.merge(
                monthly_data[['state', 'methane_value']],
                left_on='STATE', right_on='state', how='left'
            )
            state_gdf.rename(columns={'methane_value': 'methane_ppb'}, inplace=True)
            if 'state' in state_gdf.columns: state_gdf.drop(columns=['state'], inplace=True, errors='ignore')
        else:
            logger.info(f"India: Monthly data for {year}-{month} not found. Using overall averages (already in methane_ppb).")
    else:
        logger.info("India: Processed time series not available. Using overall averages (already in methane_ppb).")

    state_gdf['methane_ppb'] = state_gdf['methane_ppb'].fillna(0.0)

    valid_values_series = state_gdf['methane_ppb'][state_gdf['methane_ppb'] > 0]
    data_min_overall, data_max_overall = get_valid_range(valid_values_series)

    stats_overall = {
        'mean': float(valid_values_series.mean()) if not valid_values_series.empty else 0,
        'min': float(valid_values_series.min()) if not valid_values_series.empty else 0,
        'max': float(valid_values_series.max()) if not valid_values_series.empty else 0,
        'median': float(valid_values_series.median()) if not valid_values_series.empty else 0,
        'std': float(valid_values_series.std()) if not valid_values_series.empty else 0,
        'count': len(valid_values_series), 'data_min': data_min_overall, 'data_max': data_max_overall
    }

    top_states_df = state_gdf[state_gdf['methane_ppb'] > 0].nlargest(5, 'methane_ppb')
    top_states_list = [{"name": row['STATE'], "methane_ppb": row['methane_ppb']} for _, row in top_states_df.iterrows()]

    if viz_type == 'heatmap' and SpatialInterpolator is not None:
        logger.info(f"India: Generating heatmap-style data ({year}-{month}).")
        points_for_interp = []
        state_gdf_valid_geom = state_gdf[state_gdf.geometry.is_valid & ~state_gdf.geometry.is_empty]
        for _, row in state_gdf_valid_geom.iterrows():
            if row['methane_ppb'] > 0 and row.geometry.centroid:
                points_for_interp.append([row.geometry.centroid.y, row.geometry.centroid.x, row['methane_ppb']])

        if not points_for_interp or len(points_for_interp) < 3:
            logger.warning("India: Not enough valid state data points for interpolation. Falling back to choropleth.")
            viz_type = 'choropleth'
        else:
            min_lon, min_lat, max_lon, max_lat = state_gdf_orig.total_bounds
            india_bounds = {'min_lat': float(min_lat), 'max_lat': float(max_lat), 'min_lon': float(min_lon), 'max_lon': float(max_lon)}

            interpolator = SpatialInterpolator()
            interpolated_result = interpolator.interpolate_grid(points_for_interp, india_bounds, grid_size=60, method='idw')

            if interpolated_result and interpolated_result.get('grid'):
                display_stats = stats_overall.copy()
                if 'value_range' in interpolated_result and interpolated_result['value_range']:
                    is_vr = interpolated_result['value_range']
                    display_stats['data_min'] = is_vr.get('min', stats_overall['data_min'])
                    display_stats['data_max'] = is_vr.get('max', stats_overall['data_max'])

                return jsonify(convert_numpy_types({
                    "type": "interpolated_contour", "interpolated_grid_bundle": interpolated_result,
                    "stats": display_stats, "top_regions_label": "Top Emitting States",
                    "top_regions_data": top_states_list
                }))
            else:
                logger.warning("India: Interpolation failed. Falling back to choropleth.")
                viz_type = 'choropleth'

    if viz_type == 'choropleth':
        try:
            state_gdf_simplified = state_gdf[['STATE', 'geometry', 'methane_ppb']].copy()
            state_gdf_simplified['geometry'] = state_gdf_simplified.geometry.simplify(0.01, preserve_topology=True)
            geojson_str = state_gdf_simplified.to_json()

            return jsonify(convert_numpy_types({
                "type": "choropleth", "geojson": json.loads(geojson_str),
                "stats": stats_overall, "top_states": top_states_list
            }))
        except Exception as e:
            logger.error(f"Error creating India GeoJSON: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Unknown visualization type or processing error"}), 500


@app.route('/api/state/<state_name>/<int:year>/<int:month>')
def state_data(state_name, year, month):
    if not app_data_cache['initialization_complete']:
        return jsonify({"error": "Data not initialized"}), 500

    viz_type = request.args.get('viz', 'choropleth')
    state_name_upper = state_name.upper().strip()

    district_gdf_master = app_data_cache.get('district_gdf_master')
    if district_gdf_master is None or district_gdf_master.empty:
         return jsonify({"error": "District geometries not loaded"}), 500

    state_districts_orig = district_gdf_master[district_gdf_master['STATE'] == state_name_upper]
    if state_districts_orig.empty:
        return jsonify({"error": f"No districts found for state: {state_name}"}), 404

    state_districts_gdf = state_districts_orig.copy()

    district_methane_df, stats_from_details = get_district_details_for_state(state_name_upper, year, month)

    if 'methane_ppb' in state_districts_gdf.columns: state_districts_gdf = state_districts_gdf.drop(columns=['methane_ppb'], errors='ignore')

    if district_methane_df is not None and not district_methane_df.empty:
        state_districts_gdf = state_districts_gdf.merge(
            district_methane_df[['district', 'avg_methane']],
            left_on='District_1', right_on='district', how='left'
        )
        state_districts_gdf.rename(columns={'avg_methane': 'methane_ppb'}, inplace=True)
        if 'district' in state_districts_gdf.columns: state_districts_gdf.drop(columns=['district'], inplace=True, errors='ignore')
    else:
        state_districts_gdf['methane_ppb'] = 0.0

    state_districts_gdf['methane_ppb'] = state_districts_gdf['methane_ppb'].fillna(0.0)

    current_stats_for_state = stats_from_details
    if current_stats_for_state is None or current_stats_for_state.get('count', 0) == 0:
        valid_values_s = state_districts_gdf['methane_ppb'][state_districts_gdf['methane_ppb'] > 0]
        data_min_s, data_max_s = get_valid_range(valid_values_s)
        current_stats_for_state = {
            'mean': float(valid_values_s.mean()) if not valid_values_s.empty else 0,
            'min': float(valid_values_s.min()) if not valid_values_s.empty else 0,
            'max': float(valid_values_s.max()) if not valid_values_s.empty else 0,
            'median': float(valid_values_s.median()) if not valid_values_s.empty else 0,
            'std': float(valid_values_s.std()) if not valid_values_s.empty else 0,
            'count': len(valid_values_s), 'data_min': data_min_s, 'data_max': data_max_s
        }

    top_districts_df = state_districts_gdf[state_districts_gdf['methane_ppb'] > 0].nlargest(5, 'methane_ppb')
    top_districts_list = [{"name": row['District_1'], "methane_ppb": row['methane_ppb']} for _, row in top_districts_df.iterrows()]

    if viz_type == 'heatmap' and SpatialInterpolator is not None:
        logger.info(f"State ({state_name_upper}): Generating heatmap-style data ({year}-{month}).")
        points_for_interp = []
        state_districts_valid_geom = state_districts_gdf[state_districts_gdf.geometry.is_valid & ~state_districts_gdf.geometry.is_empty]
        for _, row in state_districts_valid_geom.iterrows():
            if row['methane_ppb'] > 0 and row.geometry.centroid:
                points_for_interp.append([row.geometry.centroid.y, row.geometry.centroid.x, row['methane_ppb']])

        if not points_for_interp or len(points_for_interp) < 3:
            logger.warning(f"State ({state_name_upper}): Not enough district data for interpolation. Falling back to choropleth.")
            viz_type = 'choropleth'
        else:
            min_lon, min_lat, max_lon, max_lat = state_districts_orig.total_bounds
            state_bounds = {'min_lat': float(min_lat), 'max_lat': float(max_lat), 'min_lon': float(min_lon), 'max_lon': float(max_lon)}

            interpolator = SpatialInterpolator()
            interpolated_result = interpolator.interpolate_grid(points_for_interp, state_bounds, grid_size=50, method='idw')

            if interpolated_result and interpolated_result.get('grid'):
                display_stats = current_stats_for_state.copy()
                if 'value_range' in interpolated_result and interpolated_result['value_range']:
                    is_vr = interpolated_result['value_range']
                    display_stats['data_min'] = is_vr.get('min', current_stats_for_state['data_min'])
                    display_stats['data_max'] = is_vr.get('max', current_stats_for_state['data_max'])

                return jsonify(convert_numpy_types({
                    "type": "interpolated_contour", "interpolated_grid_bundle": interpolated_result,
                    "stats": display_stats, "top_regions_label": f"Top Districts in {state_name}",
                    "top_regions_data": top_districts_list
                }))
            else:
                logger.warning(f"State ({state_name_upper}): Interpolation failed. Falling back to choropleth.")
                viz_type = 'choropleth'

    if viz_type == 'choropleth':
        try:
            state_districts_simplified = state_districts_gdf[['District_1', 'STATE', 'geometry', 'methane_ppb']].copy()
            state_districts_simplified['geometry'] = state_districts_simplified.geometry.simplify(0.001, preserve_topology=True)
            geojson_str = state_districts_simplified.to_json()

            return jsonify(convert_numpy_types({
                "type": "choropleth", "geojson": json.loads(geojson_str),
                "stats": current_stats_for_state, "top_districts": top_districts_list
            }))
        except Exception as e:
            logger.error(f"Error creating state GeoJSON for {state_name}: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Unknown visualization type or processing error for state"}), 500

# Enhanced district_data endpoint with improved interpolation
@app.route('/api/district/<state_name>/<district_name>/<int:year>/<int:month>')
def district_data(state_name, district_name, year, month):
    state_name_upper = state_name.upper().strip()
    district_name_upper = district_name.upper().strip()

    point_bundle, point_stats = get_point_data_for_district(state_name_upper, district_name_upper, year, month)

    if point_bundle is None or point_stats is None or not point_bundle.get('points'):
        logger.warning(f"No point data found for district {district_name_upper}, {state_name_upper} ({year}-{month})")
        return jsonify({"error": f"No data for {district_name}, {state_name} for the selected period."}), 404

    # Always attempt interpolation for districts (like your GeoTIFF approach)
    if SpatialInterpolator is None:
        logger.warning("SpatialInterpolator not available. Returning points only.")
        return jsonify(convert_numpy_types({
            "type": "contour_points_only", 
            "points": point_bundle['points'],
            "bounds": point_bundle['bounds'], 
            "stats": point_stats
        }))

    try:
        interpolator = SpatialInterpolator()
        
        # Enhanced interpolation with multiple methods
        points_data = point_bundle['points']
        bounds_data = point_bundle['bounds']
        
        # Determine optimal grid size based on data density
        num_points = len(points_data)
        if num_points > 1000:
            grid_size = 80
        elif num_points > 500:
            grid_size = 60
        elif num_points > 100:
            grid_size = 40
        else:
            grid_size = 30
        
        logger.info(f"District ({district_name_upper}): Using grid size {grid_size} for {num_points} points")
        
        # Try different interpolation methods in order of preference
        interpolation_methods = ['idw', 'rbf', 'cubic', 'linear']
        interpolated_result = None
        
        for method in interpolation_methods:
            try:
                logger.info(f"Attempting {method.upper()} interpolation for {district_name_upper}")
                
                interpolated_result = interpolator.interpolate_grid(
                    points_data, bounds_data,
                    grid_size=grid_size, 
                    method=method,
                    power=2 if method == 'idw' else None,
                    smooth=True  # Apply Gaussian smoothing for smoother surface
                )
                
                if interpolated_result and interpolated_result.get('grid'):
                    logger.info(f"District ({district_name_upper}): {method.upper()} interpolation successful. Coverage: {interpolated_result.get('coverage', 0):.1%}")
                    break
                else:
                    logger.warning(f"District ({district_name_upper}): {method.upper()} interpolation failed")
                    
            except Exception as method_error:
                logger.warning(f"District ({district_name_upper}): {method.upper()} interpolation error: {method_error}")
                continue
        
        if interpolated_result and interpolated_result.get('grid'):
            # Enhance the interpolated result
            display_stats = point_stats.copy()
            
            # Update stats with interpolated data range
            if 'value_range' in interpolated_result and interpolated_result['value_range']:
                interp_range = interpolated_result['value_range']
                display_stats.update({
                    'data_min': interp_range.get('min', point_stats['data_min']),
                    'data_max': interp_range.get('max', point_stats['data_max']),
                    'interpolated_mean': interp_range.get('mean', point_stats['mean']),
                    'interpolated_std': interp_range.get('std', point_stats['std']),
                    'grid_coverage': interpolated_result.get('coverage', 0)
                })
            
            # Add interpolation metadata
            interpolated_result['interpolation_info'] = {
                'method_used': interpolated_result.get('method', 'unknown'),
                'original_points': len(points_data),
                'grid_resolution': f"{grid_size}x{grid_size}",
                'smoothing_applied': True
            }
            
            response_data = {
                "type": "interpolated_contour",
                "original_points": points_data,
                "bounds": bounds_data,
                "interpolated_grid_bundle": interpolated_result,
                "stats": display_stats
            }
            
            logger.info(f"District ({district_name_upper}): Returning interpolated surface with {grid_size}x{grid_size} grid")
            
        else:
            logger.warning(f"District ({district_name_upper}): All interpolation methods failed. Returning points only.")
            response_data = {
                "type": "contour_points_only",
                "points": points_data,
                "bounds": bounds_data,
                "stats": point_stats
            }

    except Exception as e:
        logger.error(f"District ({district_name_upper}): Interpolation error: {e}", exc_info=True)
        response_data = {
            "type": "contour_points_only",
            "points": point_bundle['points'],
            "bounds": point_bundle['bounds'],
            "stats": point_stats
        }

    return jsonify(convert_numpy_types(response_data))


# Enhanced helper function for creating raster-like interpolation (similar to your GeoTIFF approach)
def create_raster_like_interpolation(points, bounds, resolution=0.01):
    """
    Create a raster-like interpolation similar to the GeoTIFF approach in paste.txt
    but with smooth interpolation instead of discrete rasterization.
    """
    try:
        if not points or len(points) < 3:
            return None
            
        # Convert points to numpy array
        points_array = np.array(points, dtype=np.float64)
        lats = points_array[:, 0]
        lons = points_array[:, 1]
        values = points_array[:, 2]
        
        # Filter valid points
        valid_mask = (values > 0) & np.isfinite(values) & np.isfinite(lats) & np.isfinite(lons)
        if np.sum(valid_mask) < 3:
            return None
            
        lats = lats[valid_mask]
        lons = lons[valid_mask]
        values = values[valid_mask]
        
        # Create regular grid (similar to your GeoTIFF approach)
        min_lon, max_lon = bounds['min_lon'], bounds['max_lon']
        min_lat, max_lat = bounds['min_lat'], bounds['max_lat']
        
        # Calculate grid dimensions
        width = int(np.ceil((max_lon - min_lon) / resolution))
        height = int(np.ceil((max_lat - min_lat) / resolution))
        
        # Limit grid size for performance
        max_size = 200
        if width > max_size or height > max_size:
            scale_factor = max_size / max(width, height)
            width = int(width * scale_factor)
            height = int(height * scale_factor)
        
        # Create coordinate arrays
        lon_range = np.linspace(min_lon, max_lon, width)
        lat_range = np.linspace(min_lat, max_lat, height)
        lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
        
        # Use IDW interpolation for smooth surface
        interpolator = SpatialInterpolator()
        interpolated_grid = interpolator._idw_interpolation_optimized(
            lats, lons, values, lat_grid, lon_grid, power=2
        )
        
        # Apply Gaussian smoothing for smoother surface
        from scipy.ndimage import gaussian_filter
        interpolated_grid = gaussian_filter(interpolated_grid, sigma=1.0)
        
        # Calculate statistics
        valid_grid_values = interpolated_grid[~np.isnan(interpolated_grid)]
        
        result = {
            'grid': interpolated_grid.tolist(),
            'lat_range': lat_range.tolist(),
            'lon_range': lon_range.tolist(),
            'bounds': bounds,
            'method': 'raster_idw',
            'grid_size': f"{width}x{height}",
            'value_range': {
                'min': float(np.nanmin(valid_grid_values)),
                'max': float(np.nanmax(valid_grid_values)),
                'mean': float(np.nanmean(valid_grid_values)),
                'std': float(np.nanstd(valid_grid_values))
            } if len(valid_grid_values) > 0 else None
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Raster-like interpolation error: {e}")
        return None

# --- Analytics Endpoints ---
def load_analytics_json(filename):
    cache_key = f"analytics_file_{filename}"
    if cache_key in app_data_cache.get('analytics_cache', {}):
        return app_data_cache['analytics_cache'][cache_key], None

    file_path = ANALYTICS_DATA_DIR / filename
    if not file_path.exists():
        return None, {"error": f"Analytics file {filename} not found at {file_path}"}

    try:
        with open(file_path, 'r') as f: data = json.load(f)
        app_data_cache.setdefault('analytics_cache', {})[cache_key] = data
        return data, None
    except Exception as e:
        logger.error(f"Error loading analytics JSON {file_path}: {e}", exc_info=True)
        return None, {"error": f"Could not read or parse {filename}"}

@app.route('/api/analytics/ranking/<int:year>/<int:month>')
def analytics_ranking(year, month):
    all_rankings, error = load_analytics_json('ranking_analytics.json')
    if error: return jsonify(error), 404

    date_key = f"{year}_{month:02d}"
    period_rankings = all_rankings.get(date_key)

    if not period_rankings:
        logger.warning(f"Ranking data for key '{date_key}' not directly found. Searching alternatives.")
        found_alternative = False
        for key_iter, data_iter in all_rankings.items():
            if key_iter.startswith(str(year)):
                period_rankings = data_iter
                logger.info(f"Found alternative ranking for year {year} under key '{key_iter}'.")
                found_alternative = True
                break
        if not found_alternative and all_rankings:
             first_key = next(iter(all_rankings))
             period_rankings = all_rankings[first_key]
             logger.warning(f"No ranking data for {year} found. Using data from key '{first_key}' as fallback.")


    if not period_rankings:
        return jsonify({"error": f"No ranking data available for {year}-{month} or any fallback."}), 404

    return jsonify(convert_numpy_types({
        "state_rankings": period_rankings.get('state_rankings', [])[:50],
        "district_rankings": period_rankings.get('district_rankings', [])[:100],
        "period": period_rankings.get('date_column', date_key),
        "total_states_ranked": period_rankings.get('total_states', 0),
        "total_districts_ranked": period_rankings.get('total_districts', 0)
    }))

def find_analytics_key(data_dict, state_name_query, district_name_query=None):
    state_upper_query = state_name_query.upper().strip()
    district_upper_query = district_name_query.upper().strip() if district_name_query else None

    if district_upper_query:
        key_to_try = f"{state_upper_query}_{district_upper_query}"
        if key_to_try in data_dict: return data_dict[key_to_try]
    elif state_upper_query in data_dict:
        return data_dict[state_upper_query]

    if isinstance(data_dict, dict):
        for _, item_data in data_dict.items():
            if isinstance(item_data, dict):
                item_state = str(item_data.get("state", "")).upper().strip()
                if item_state == state_upper_query:
                    if district_upper_query:
                        item_district = str(item_data.get("district", "")).upper().strip()
                        if item_district == district_upper_query: return item_data
                    else:
                        return item_data
    elif isinstance(data_dict, list):
         for item_data in data_dict:
            if isinstance(item_data, dict):
                item_state = str(item_data.get("state", "")).upper().strip()
                if item_state == state_upper_query:
                    if district_upper_query:
                        item_district = str(item_data.get("district", "")).upper().strip()
                        if item_district == district_upper_query: return item_data
                    else: return item_data
    return None

@app.route('/api/analytics/timeseries_data/<state_name>/<district_name>')
def analytics_timeseries_data(state_name, district_name):
    data, error = load_analytics_json('time_series_analytics.json')
    if error: return jsonify(error), 404
    result = find_analytics_key(data, state_name, district_name)
    if not result: return jsonify({"error": f"Time series data not found for {district_name}, {state_name}"}), 404
    return jsonify(convert_numpy_types(result))

@app.route('/api/analytics/correlation_data/<state_name>')
def analytics_correlation_data(state_name):
    data, error = load_analytics_json('correlation_analytics.json')
    if error: return jsonify(error), 404
    result = find_analytics_key(data, state_name)
    if not result: return jsonify({"error": f"Correlation data not found for {state_name}"}), 404
    return jsonify(convert_numpy_types(result))

@app.route('/api/analytics/clustering_data/<state_name>')
def analytics_clustering_data(state_name):
    data, error = load_analytics_json('clustering_analytics.json')
    if error: return jsonify(error), 404
    result = find_analytics_key(data, state_name)
    if not result: return jsonify({"error": f"Clustering data not found for {state_name}"}), 404
    return jsonify(convert_numpy_types(result))

@app.route('/api/analytics/extreme_events_data/<state_name>/<district_name>')
def analytics_extreme_events_data(state_name, district_name):
    data, error = load_analytics_json('extreme_events_analytics.json')
    if error: return jsonify(error), 404
    result = find_analytics_key(data, state_name, district_name)
    if not result: return jsonify({"error": f"Extreme events data not found for {district_name}, {state_name}"}), 404
    return jsonify(convert_numpy_types(result))

# List endpoints
@app.route('/api/states_list')
def states_list_api():
    states = app_data_cache.get('all_states_list', [])
    if not states: logger.warning("/api/states_list: No states found in cache.")
    return jsonify(convert_numpy_types(states))

@app.route('/api/districts_list/<state_name>')
def districts_list_api(state_name):
    state_upper = state_name.upper().strip()
    districts = app_data_cache.get('district_map', {}).get(state_upper, [])
    if not districts : logger.warning(f"/api/districts_list/{state_name}: No districts found for state '{state_upper}'.")
    return jsonify(convert_numpy_types(districts))

# --- New Clear Cache Endpoint ---
@app.route('/api/clear_cache', methods=['POST'])
def clear_all_caches():
    logger.info("Attempting to clear all caches...")

    # Clear LRU caches
    try:
        load_shapefiles.cache_clear()
        get_district_details_for_state.cache_clear()
        get_point_data_for_district.cache_clear()
        logger.info("LRU caches cleared.")
    except Exception as e:
        logger.error(f"Error clearing LRU caches: {e}")

    # Clear app_data_cache (analytics part and re-initialize global data)
    try:
        if 'analytics_cache' in app_data_cache:
            app_data_cache['analytics_cache'].clear()
            logger.info("Analytics file cache cleared.")

        app_data_cache['initialization_complete'] = False
        logger.info("Re-initializing global data...")
        initialize_global_data()
        logger.info("Global data re-initialized.")

    except Exception as e:
        logger.error(f"Error clearing app_data_cache or re-initializing: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "success", "message": "Caches cleared and data re-initialized."})

# --- Initialize on startup ---
with app.app_context():
    initialize_global_data()

if __name__ == '__main__':
    if not app_data_cache.get('initialization_complete', False):
        logger.critical("CRITICAL: Application initialization failed or did not complete.")
    else:
        logger.info("Flask application starting...")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)