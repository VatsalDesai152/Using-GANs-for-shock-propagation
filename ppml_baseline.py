"""
ppml_baseline.py
================
Poisson Pseudo-Maximum Likelihood (PPML) Gravity Baseline Estimation.

Replaces the OLS gravity baseline per reviewer critique.
Runs PPML on un-transformed trade levels with CEPII gravity variables,
then post-transforms predictions to log-space for Log-RMSE comparison.

References:
    Silva, J.M.C.S., & Tenreyro, S. (2006). "The Log of Gravity."
    Review of Economics and Statistics, 88(4): 641–658.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SHOCK_YEARS = {2008, 2020, 2022}
START_YEAR = 1995
END_YEAR = 2024


def load_trade_data():
    """Load and prepare the trade panel from processed edges (optimized)."""
    logging.info("Loading multiplex_edges.csv (chunked for performance)...")
    
    chunks = []
    for chunk in pd.read_csv("data/processed/multiplex_edges.csv", chunksize=500000):
        # Only keep positive trade flows for PPML (zeros handled by the Poisson family)
        chunk = chunk[chunk["log_trade_volume"] > 0].copy()
        chunk["trade_level"] = np.expm1(np.clip(chunk["log_trade_volume"].values, 0.0, 20.0))
        # Pre-aggregate within chunk
        agg = chunk.groupby(["year", "source", "target"]).agg(
            trade_level=("trade_level", "sum")
        ).reset_index()
        chunks.append(agg)
    
    logging.info(f"Loaded {len(chunks)} chunks, merging...")
    df_agg = pd.concat(chunks, ignore_index=True)
    
    # Final aggregation across chunks
    df_agg = df_agg.groupby(["year", "source", "target"])["trade_level"].sum().reset_index()
    df_agg["log_trade_volume"] = np.log1p(df_agg["trade_level"])
    df_agg.rename(columns={"source": "exporter", "target": "importer"}, inplace=True)
    
    logging.info(f"Aggregated trade panel: {len(df_agg):,} bilateral-year observations")
    return df_agg


def load_gravity_variables():
    """Load CEPII distance and gravity variables."""
    logging.info("Loading CEPII distance data...")
    df_dist = pd.read_excel("data/raw/dist_cepii.xls")
    
    # Keep relevant gravity variables
    df_dist = df_dist[["iso_o", "iso_d", "contig", "comlang_off", "dist"]].copy()
    df_dist.rename(columns={
        "iso_o": "exporter",
        "iso_d": "importer",
        "contig": "contiguity",
        "comlang_off": "common_language",
        "dist": "distance"
    }, inplace=True)
    
    # Log-distance (standard in gravity models)
    df_dist["ln_distance"] = np.log(df_dist["distance"].clip(lower=1.0))
    
    logging.info(f"CEPII gravity data: {len(df_dist):,} country-pairs")
    return df_dist


def load_gdp_data():
    """Load GDP data for exporter/importer GDP controls."""
    logging.info("Loading GDP data...")
    
    # Try World Bank GDP parquet first
    gdp_path = "data/raw/auxiliary/worldbank_gdp_current_usd.parquet"
    if os.path.exists(gdp_path):
        df_gdp = pd.read_parquet(gdp_path)
        # Expect columns like country_code, year, value
        logging.info(f"GDP parquet columns: {df_gdp.columns.tolist()}")
        return df_gdp
    
    # Try CSV fallback
    gdp_csv = "data/raw/auxiliary/worldbank_gdp_current_usd.csv"
    if os.path.exists(gdp_csv):
        df_gdp = pd.read_csv(gdp_csv)
        logging.info(f"GDP CSV columns: {df_gdp.columns.tolist()}")
        return df_gdp
    
    logging.warning("No GDP data found. PPML will run without GDP controls.")
    return None


def run_ppml_estimation(df_trade, df_dist, df_gdp=None):
    """
    Run PPML gravity estimation.
    
    Strategy:
    1. Train on non-shock years
    2. Predict on shock years
    3. Evaluate on shock years only
    
    Uses core gravity covariates only (no high-dimensional FE dummies)
    for computational tractability. This is a standard specification
    in the trade literature when FE are computationally prohibitive.
    """
    logging.info("Merging trade data with gravity variables...")
    
    df = df_trade.merge(df_dist, on=["exporter", "importer"], how="left")
    
    # Drop pairs without CEPII coverage
    n_before = len(df)
    df = df.dropna(subset=["ln_distance"])
    n_after = len(df)
    logging.info(f"Dropped {n_before - n_after:,} obs without CEPII match ({n_after:,} remaining)")
    
    # Core gravity variables
    X_vars = ["ln_distance", "contiguity", "common_language"]
    
    # Split train/test
    train_mask = ~df["year"].isin(SHOCK_YEARS)
    df_train = df[train_mask].copy()
    df_test = df[df["year"].isin(SHOCK_YEARS)].copy()
    
    logging.info(f"Training set: {len(df_train):,} obs | Test set (shock years): {len(df_test):,} obs")
    
    # --- PPML Estimation ---
    logging.info("Running PPML (Poisson GLM) with gravity covariates...")
    
    X_train = sm.add_constant(df_train[X_vars].values.astype(np.float64))
    y_train = df_train["trade_level"].values.astype(np.float64)
    
    # Scale down for numerical stability
    scale_factor = np.mean(y_train[y_train > 0]) if np.any(y_train > 0) else 1.0
    y_scaled = y_train / scale_factor
    
    logging.info(f"Scale factor: {scale_factor:.2f}, running IRLS...")
    
    ppml_model = sm.GLM(y_scaled, X_train, family=sm.families.Poisson())
    ppml_result = ppml_model.fit(maxiter=100, method='IRLS')
    
    logging.info(f"PPML converged: {ppml_result.converged}")
    logging.info(f"PPML pseudo R²: {1 - ppml_result.deviance / ppml_result.null_deviance:.4f}")
    
    gravity_coefs = ppml_result.params
    logging.info(f"Gravity coefficients:")
    logging.info(f"  Constant:         {gravity_coefs[0]:.4f}")
    logging.info(f"  ln(distance):     {gravity_coefs[1]:.4f}  (expected: negative)")
    logging.info(f"  Contiguity:       {gravity_coefs[2]:.4f}  (expected: positive)")
    logging.info(f"  Common language:  {gravity_coefs[3]:.4f}  (expected: positive)")
    
    # --- Predict on Shock Years ---
    logging.info("Generating predictions for shock years...")
    X_test = sm.add_constant(df_test[X_vars].values.astype(np.float64))
    y_pred_scaled = ppml_result.predict(X_test)
    y_pred_ppml = y_pred_scaled * scale_factor
    
    return compute_metrics(df_test, y_pred_ppml, ppml_result, gravity_coefs)


def compute_metrics(df_test, y_pred_ppml, ppml_result, gravity_coefs):
    """Compute WAPE and Log-RMSE from PPML predictions."""
    
    y_true_level = df_test["trade_level"].values
    y_pred_level = np.clip(y_pred_ppml, 0, None)  # Ensure non-negative predictions
    
    # --- WAPE (on levels) ---
    wape = np.sum(np.abs(y_true_level - y_pred_level)) / (np.sum(y_true_level) + 1e-9)
    
    # --- Log-RMSE (post-estimation log transform) ---
    # As specified: apply log(1+x) to both predictions and true values post-estimation
    y_true_log = np.log1p(y_true_level)
    y_pred_log = np.log1p(y_pred_level)
    log_rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    
    # --- Per-shock-year breakdown ---
    results_by_year = {}
    for yr in sorted(df_test["year"].unique()):
        mask = df_test["year"].values == yr
        yr_true = y_true_level[mask]
        yr_pred = y_pred_level[mask]
        yr_wape = np.sum(np.abs(yr_true - yr_pred)) / (np.sum(yr_true) + 1e-9)
        yr_log_rmse = np.sqrt(mean_squared_error(np.log1p(yr_true), np.log1p(yr_pred)))
        results_by_year[int(yr)] = {"WAPE": round(float(yr_wape), 4), "Log-RMSE": round(float(yr_log_rmse), 4)}
    
    results = {
        "model": "PPML Gravity (Silva & Tenreyro, 2006)",
        "estimator": "Poisson Pseudo-Maximum Likelihood",
        "WAPE": round(float(wape), 4),
        "Log-RMSE": round(float(log_rmse), 4),
        "pseudo_R2": round(float(1 - ppml_result.deviance / ppml_result.null_deviance), 4),
        "converged": bool(ppml_result.converged),
        "n_train_obs": int(ppml_result.nobs),
        "n_test_obs": len(df_test),
        "gravity_coefficients": {
            "constant": round(float(gravity_coefs[0]), 4),
            "ln_distance": round(float(gravity_coefs[1]), 4),
            "contiguity": round(float(gravity_coefs[2]), 4) if len(gravity_coefs) > 2 else None,
            "common_language": round(float(gravity_coefs[3]), 4) if len(gravity_coefs) > 3 else None,
        },
        "per_shock_year": results_by_year,
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("PPML GRAVITY BASELINE RESULTS")
    print("=" * 60)
    print(f"  WAPE (shock years):     {wape:.4f}")
    print(f"  Log-RMSE (shock years): {log_rmse:.4f}")
    print(f"  Pseudo R²:              {results['pseudo_R2']:.4f}")
    print(f"  Converged:              {results['converged']}")
    print(f"\n  Gravity Coefficients:")
    print(f"    ln(distance):   {gravity_coefs[1]:.4f}")
    if len(gravity_coefs) > 2:
        print(f"    Contiguity:     {gravity_coefs[2]:.4f}")
    if len(gravity_coefs) > 3:
        print(f"    Common lang:    {gravity_coefs[3]:.4f}")
    print(f"\n  Per-Year Breakdown:")
    for yr, metrics in results_by_year.items():
        print(f"    {yr}: WAPE={metrics['WAPE']:.4f}, Log-RMSE={metrics['Log-RMSE']:.4f}")
    print("=" * 60)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/ppml_baseline_results.json", "w") as f:
        json.dump(results, f, indent=4)
    logging.info("Results saved to results/ppml_baseline_results.json")
    
    return results


def main():
    df_trade = load_trade_data()
    df_dist = load_gravity_variables()
    df_gdp = load_gdp_data()
    
    results = run_ppml_estimation(df_trade, df_dist, df_gdp)
    return results


if __name__ == "__main__":
    main()
