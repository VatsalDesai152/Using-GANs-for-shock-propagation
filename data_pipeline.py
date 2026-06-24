import os
import concurrent.futures
import pandas as pd
import numpy as np
import logging
import requests
from fredapi import Fred
from owid.catalog import Client
import zipfile
import pycountry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

START_YEAR = 1995
END_YEAR = 2024
SECTORS = ["Food", "Energy", "Semiconductors", "Consumer Goods"]

def map_unique_countries_to_iso3(df, col_name):
    """Maps country names to ISO3 codes efficiently using unique values."""
    unique_names = df[col_name].dropna().unique()
    mapping = {}
    for name in unique_names:
        try:
            mapping[name] = pycountry.countries.lookup(name).alpha_3
        except LookupError:
            try:
                # search_fuzzy can be slow, but we only do it once per unique failed name
                mapping[name] = pycountry.countries.search_fuzzy(name)[0].alpha_3
            except Exception:
                mapping[name] = None
    df['country_code_iso3'] = df[col_name].map(mapping)
    return df

def process_baci_trade():
    """Thread 1: BACI Trade Data (Bulk ZIP + Parquet mapping)"""
    logging.info("Starting BACI trade data processing (Thread 1)...")
    baci_path = "data/raw/BACI_HS92_V202601.zip"
    if not os.path.exists(baci_path):
        raise FileNotFoundError(f"BACI dataset missing. Please download to {baci_path}")
    
    edges = []
    
    def map_sector(hs_code):
        try:
            code_str = str(hs_code).zfill(6)
            chapter = int(code_str[:2])
            if 1 <= chapter <= 24: return "Food"
            if chapter == 27: return "Energy"
            if code_str.startswith("8541") or code_str.startswith("8542"): return "Semiconductors"
            return "Consumer Goods"
        except:
            return "Consumer Goods"

    with zipfile.ZipFile(baci_path) as z:
        for year in range(START_YEAR, END_YEAR + 1):
            filename = f"BACI_HS92_Y{year}_V202601.csv"
            if filename in z.namelist():
                logging.info(f"Processing BACI year {year}...")
                with z.open(filename) as f:
                    chunk = pd.read_csv(f)
                    chunk['sector'] = chunk['k'].apply(map_sector)
                    agg_chunk = chunk.groupby(['t', 'i', 'j', 'sector'])['v'].sum().reset_index()
                    edges.append(agg_chunk)
                    
    if not edges:
        return pd.DataFrame(columns=['year', 'source_iso3', 'target_iso3', 'sector', 'log_trade_volume'])
        
    df_edges = pd.concat(edges, ignore_index=True)
    df_edges = df_edges.groupby(['t', 'i', 'j', 'sector'])['v'].sum().reset_index()
    df_edges['log_trade_volume'] = np.log1p(df_edges['v'])
    df_edges = df_edges.drop(columns=['v'])
    df_edges.rename(columns={'t': 'year', 'i': 'source', 'j': 'target'}, inplace=True)
    
    # ISO3 Mapping using parquet
    map_path = "data/raw/country_mapping.parquet"
    if os.path.exists(map_path):
        c_map = pd.read_parquet(map_path)
        code_to_iso = dict(zip(c_map['comtrade_code'], c_map['iso3']))
        df_edges['source_iso3'] = df_edges['source'].map(code_to_iso)
        df_edges['target_iso3'] = df_edges['target'].map(code_to_iso)
    else:
        logging.warning("country_mapping.parquet not found. Assuming source/target are ISO3 strings already.")
        df_edges['source_iso3'] = df_edges['source'].astype(str)
        df_edges['target_iso3'] = df_edges['target'].astype(str)
        
    # Drop rows where we couldn't map
    df_edges = df_edges.dropna(subset=['source_iso3', 'target_iso3'])
    
    # Drop numeric source/target columns and rename ISO3 columns to source/target
    df_edges = df_edges.drop(columns=['source', 'target'])
    df_edges = df_edges.rename(columns={'source_iso3': 'source', 'target_iso3': 'target'})
    
    logging.info("BACI trade data processing complete.")
    return df_edges

def process_fred():
    """Thread 2: FRED API Data"""
    logging.info("Starting FRED data processing (Thread 2)...")
    
    # FRED API key: set via FRED_API_KEY environment variable
    # Register for free at https://fredaccount.stlouisfed.org/apikeys
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY environment variable not set. Register at https://fredaccount.stlouisfed.org/apikeys")
    fred = Fred(api_key=api_key)
    
    fed_funds = fred.get_series('FEDFUNDS')
    cpi = fred.get_series('CPIAUCSL')
    usd_index = fred.get_series('DTWEXBGS')
    unemp = fred.get_series('UNRATE')
    
    # Resample to annual frequency (mean)
    df_fred = pd.DataFrame({
        'fed_funds': fed_funds.resample('YE').mean(),
        'cpi': cpi.resample('YE').mean(),
        'usd_index': usd_index.resample('YE').mean(),
        'unemployment': unemp.resample('YE').mean()
    })
    
    df_fred['year'] = df_fred.index.year
    df_fred = df_fred[(df_fred['year'] >= START_YEAR) & (df_fred['year'] <= END_YEAR)].reset_index(drop=True)
    
    df_fred['is_global_macro'] = True 
    
    logging.info("FRED data processing complete.")
    return df_fred

def process_owid():
    """Thread 3: Our World in Data"""
    logging.info("Starting OWID data processing (Thread 3)...")
    client = Client()
    
    def safe_fetch(search_term, value_name):
        results = client.indicators.search(search_term)
        for r in results:
            try:
                df = pd.DataFrame(r.fetch()).reset_index()
                if 'country' in df.columns and 'year' in df.columns:
                    # Rename the last column (which is the actual value)
                    df = df[['country', 'year', df.columns[-1]]].rename(columns={df.columns[-1]: value_name})
                    return df
            except Exception:
                continue
        logging.error(f"Failed to fetch {search_term} from OWID")
        return pd.DataFrame(columns=['country', 'year', value_name])

    # Fetch datasets safely
    gdp_df = safe_fetch("gross domestic product", "gdp")
    pop_df = safe_fetch("population", "population")
    energy_df = safe_fetch("fossil fuel energy consumption", "fossil_energy_share")
    
    # Merge OWID datasets
    df_owid = pd.merge(gdp_df, pop_df, on=['country', 'year'], how='outer')
    df_owid = pd.merge(df_owid, energy_df, on=['country', 'year'], how='outer')
    
    df_owid = df_owid[(df_owid['year'] >= START_YEAR) & (df_owid['year'] <= END_YEAR)]
    
    # Map country to ISO3
    df_owid = map_unique_countries_to_iso3(df_owid, 'country')
    df_owid = df_owid.dropna(subset=['country_code_iso3']).drop(columns=['country'])
    
    logging.info("OWID data processing complete.")
    return df_owid

def process_faostat():
    """Thread 4: FAOSTAT Data (REST API)"""
    logging.info("Starting FAOSTAT data processing (Thread 4)...")
    
    # FAOSTAT API token: set via FAOSTAT_TOKEN environment variable
    # Register at https://www.fao.org/faostat/en/ to obtain a token
    token = os.environ.get("FAOSTAT_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    all_data = []
    
    # We loop by year to avoid massive payload limits or timeouts
    for year in range(START_YEAR, END_YEAR + 1):
        try:
            # QCL domain, items: 15(Wheat), 27(Rice), 56(Maize), 236(Soybeans), element: 2510(Production)
            url = f"https://faostatservices.fao.org/api/v1/en/data/QCL?item=15,27,56,236&element=2510&year={year}"
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
            if 'data' in data:
                all_data.extend(data['data'])
        except Exception as e:
            logging.error(f"FAOSTAT API Error for year {year}: {e}")
            
    df = pd.DataFrame(all_data)
    if df.empty:
        return pd.DataFrame(columns=['country_code_iso3', 'year', 'food_production_mt'])
        
    df['food_production_mt'] = pd.to_numeric(df['Value'], errors='coerce').fillna(0)
    df = df.rename(columns={"Area": "country", "Year": "year"})
    df['year'] = df['year'].astype(int)
    
    df_fao = df.groupby(['country', 'year'])['food_production_mt'].sum().reset_index()
    
    # Map country to ISO3
    df_fao = map_unique_countries_to_iso3(df_fao, 'country')
    df_fao = df_fao.dropna(subset=['country_code_iso3']).drop(columns=['country'])
    
    logging.info("FAOSTAT data processing complete.")
    return df_fao

def process_ucdp():
    """Thread 5: UCDP Data (Bulk CSV)"""
    logging.info("Starting UCDP data processing (Thread 5)...")
    
    ucdp_path = "data/raw/GEDEvent_v26_0_4.csv"
    if not os.path.exists(ucdp_path):
        raise FileNotFoundError(f"UCDP dataset missing. Please download to {ucdp_path}")
        
    # Read the bulk CSV
    # Usually uses low_memory=False for large GED sets
    df = pd.read_csv(ucdp_path, low_memory=False)
    
    df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)]
    
    df['conflict_fatalities'] = pd.to_numeric(df['best'], errors='coerce').fillna(0)
    
    df_ucdp = df.groupby(['country', 'year'])['conflict_fatalities'].sum().reset_index()
    
    # Map country to ISO3
    df_ucdp = map_unique_countries_to_iso3(df_ucdp, 'country')
    df_ucdp = df_ucdp.dropna(subset=['country_code_iso3']).drop(columns=['country'])
    
    logging.info("UCDP data processing complete.")
    return df_ucdp

def impute_and_merge_features(df_list, countries):
    logging.info("Merging and imputing features...")
    df_fred = df_list[0]
    df_owid = df_list[1]
    df_fao = df_list[2]
    df_ucdp = df_list[3]
    
    # Broadcast FRED (US Macro) to all countries
    fred_records = []
    for c in countries:
        df_c = df_fred.copy()
        df_c['country_code_iso3'] = c
        fred_records.append(df_c)
    df_fred_broad = pd.concat(fred_records, ignore_index=True)
    df_fred_broad.drop(columns=['is_global_macro'], inplace=True, errors='ignore')
    
    # Start with standard Cartesian product of all countries and years to ensure full grid
    years = np.arange(START_YEAR, END_YEAR + 1)
    df_base = pd.DataFrame([(y, c) for y in years for c in countries], columns=["year", "country_code_iso3"])
    
    # Merge everything
    df_features = pd.merge(df_base, df_fred_broad, on=["year", "country_code_iso3"], how="left")
    df_features = pd.merge(df_features, df_owid, on=["year", "country_code_iso3"], how="left")
    df_features = pd.merge(df_features, df_fao, on=["year", "country_code_iso3"], how="left")
    df_features = pd.merge(df_features, df_ucdp, on=["year", "country_code_iso3"], how="left")
        
    # Forward and backward filling along each specific country's temporal sequence
    df_features = df_features.sort_values(["country_code_iso3", "year"])
    df_features = df_features.groupby("country_code_iso3").apply(
        lambda group: group.ffill().bfill()
    ).reset_index(drop=True)
    
    # Fill remaining NaNs with 0 (for conflict fatalities, production, etc.)
    df_features = df_features.fillna(0)
    
    # Rename country_code_iso3 back to country for compatibility with existing dataset loader
    df_features = df_features.rename(columns={"country_code_iso3": "country"})
    
    from sklearn.preprocessing import StandardScaler
    
    # 1. Transform non-stationary variables (log-differences for gdp and population)
    df_features['log_gdp'] = np.log1p(df_features['gdp'])
    df_features['log_pop'] = np.log1p(df_features['population'])
    
    df_features['gdp_growth'] = df_features.groupby('country')['log_gdp'].diff().fillna(0)
    df_features['pop_growth'] = df_features.groupby('country')['log_pop'].diff().fillna(0)
    
    # Drop original non-stationary and intermediate variables
    df_features = df_features.drop(columns=['gdp', 'population', 'log_gdp', 'log_pop'])
    
    # 2. StandardScaler on all features, fit only on training years (not 2008, 2020, 2022)
    feature_cols = [c for c in df_features.columns if c not in ["year", "country"]]
    
    train_mask = ~df_features['year'].isin([2008, 2020, 2022])
    scaler = StandardScaler()
    
    # Fit only on non-shock years
    scaler.fit(df_features.loc[train_mask, feature_cols])
    
    # Transform entire timeline
    df_features[feature_cols] = scaler.transform(df_features[feature_cols])
    
    logging.info("Feature imputation and standardization complete.")
    return df_features

def generate_negative_samples(df_edges, countries, years):
    logging.info("Generating negative samples for sparse graph...")
    neg_edges = []
    
    for y in years:
        for s in SECTORS:
            existing = df_edges[(df_edges["year"] == y) & (df_edges["sector"] == s)]
            existing_set = set(zip(existing["source"], existing["target"]))
            
            num_pos = len(existing)
            num_neg = 0
            attempts = 0 
            max_attempts = num_pos * 10
            
            while num_neg < num_pos and attempts < max_attempts:
                src = np.random.choice(countries)
                dst = np.random.choice(countries)
                if src != dst and (src, dst) not in existing_set:
                    neg_edges.append([y, src, dst, s, 0.0])
                    existing_set.add((src, dst))
                    num_neg += 1
                attempts += 1
                    
    df_neg = pd.DataFrame(neg_edges, columns=["year", "source", "target", "sector", "log_trade_volume"])
    df_all_edges = pd.concat([df_edges, df_neg], ignore_index=True)
    
    logging.info("Negative sampling complete.")
    return df_all_edges

def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_baci = executor.submit(process_baci_trade)
        future_fred = executor.submit(process_fred)
        future_owid = executor.submit(process_owid)
        future_fao = executor.submit(process_faostat)
        future_ucdp = executor.submit(process_ucdp)
        
        df_edges = future_baci.result()
        df_fred = future_fred.result()
        df_owid = future_owid.result()
        df_fao = future_fao.result()
        df_ucdp = future_ucdp.result()
        
    # Extract unique countries from the trade graph to define the node set
    countries = list(set(df_edges["source"].unique()).union(set(df_edges["target"].unique())))
    
    df_features = impute_and_merge_features([df_fred, df_owid, df_fao, df_ucdp], countries)
    
    years = np.arange(START_YEAR, END_YEAR + 1)
    df_all_edges = generate_negative_samples(df_edges, countries, years)
    
    # Save processed data
    df_features.drop_duplicates(subset=["year", "country"]).to_csv("data/processed/node_features.csv", index=False)
    df_all_edges.drop_duplicates(subset=["year", "source", "target", "sector"]).to_csv("data/processed/multiplex_edges.csv", index=False)
    logging.info("Data pipeline complete. Files saved to data/processed/")

if __name__ == "__main__":
    main()
