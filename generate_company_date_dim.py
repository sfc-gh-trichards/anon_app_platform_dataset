from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
INPUT_FACT_PATH = APP_DIR / "anon_company_day_fact.csv"
OUTPUT_DIM_PATH = APP_DIR / "anon_company_date_dim.csv"


@dataclass(frozen=True)
class GenerationConfig:
    random_seed: int = 42
    convert_trialing_ratio: float = 0.10
    # Avoid converting right at the boundaries for realism
    min_conversion_offset_days: int = 3
    max_conversion_offset_days: int | None = None  # if None, computed from range length


INDUSTRIES: List[str] = [
    "Technology",
    "Healthcare",
    "Finance",
    "Retail",
    "Manufacturing",
    "Education",
    "Media",
    "Energy",
    "Telecommunications",
    "Government",
    "Transportation",
    "Hospitality",
]


def assign_industry_deterministically(company_id: str) -> str:
    # Deterministic assignment based on hash for stability across runs
    hashed = hashlib.md5(company_id.encode("utf-8")).hexdigest()
    index = int(hashed, 16) % len(INDUSTRIES)
    return INDUSTRIES[index]


def compute_baseline_status_by_company(fact_df: pd.DataFrame) -> Dict[str, str]:
    # Use mode status per company as baseline. If no non-null status exists, default to "Trialing".
    def _mode_or_trialing(s: pd.Series) -> str:
        s_non_null = s.dropna()
        if s_non_null.empty:
            return "Trialing"
        modes = s_non_null.mode()
        return str(modes.iloc[0]) if not modes.empty else "Trialing"

    baseline = (
        fact_df.groupby("COMPANY_ID", as_index=True)["STATUS"].apply(_mode_or_trialing).to_dict()
    )
    return baseline


def generate_company_date_dim(config: GenerationConfig = GenerationConfig()) -> pd.DataFrame:
    if not INPUT_FACT_PATH.exists():
        raise FileNotFoundError(f"Input fact file not found: {INPUT_FACT_PATH}")

    fact_df = pd.read_csv(INPUT_FACT_PATH, low_memory=False)
    # Normalize headers to uppercase so we can write DB-style logic regardless of CSV casing
    fact_df.columns = [str(c).upper() for c in fact_df.columns]

    if "COMPANY_ID" not in fact_df.columns:
        raise KeyError("Expected column 'COMPANY_ID' in anon_company_day_fact.csv")
    if "VIEW_DATE" not in fact_df.columns:
        raise KeyError("Expected column 'VIEW_DATE' in anon_company_day_fact.csv")

    # Normalize status column name
    if "STATUS" not in fact_df.columns and "COMPANY_STATUS" in fact_df.columns:
        fact_df["STATUS"] = fact_df["COMPANY_STATUS"]

    # Normalize/parse dates
    fact_df["VIEW_DATE"] = pd.to_datetime(fact_df["VIEW_DATE"]).dt.date

    unique_companies = fact_df["COMPANY_ID"].dropna().unique()
    min_date = fact_df["VIEW_DATE"].min()
    max_date = fact_df["VIEW_DATE"].max()

    date_index = pd.date_range(start=min_date, end=max_date, freq="D").date

    # Cartesian product of companies x dates
    dim_df = (
        pd.MultiIndex.from_product([unique_companies, date_index], names=["COMPANY_ID", "VIEW_DATE"])  # type: ignore[arg-type]
        .to_frame(index=False)
    )

    # Baseline status per company
    baseline_status_by_company = compute_baseline_status_by_company(fact_df)
    dim_df["STATUS"] = dim_df["COMPANY_ID"].map(baseline_status_by_company).fillna("Trialing")

    # Deterministic industry per company
    dim_df["INDUSTRY"] = dim_df["COMPANY_ID"].map(assign_industry_deterministically)

    # Convert a random 10% of companies whose baseline is Trialing to Paying after a random date
    rng = np.random.default_rng(config.random_seed)

    trialing_companies = [cid for cid, status in baseline_status_by_company.items() if str(status) == "Trialing"]
    if trialing_companies:
        num_to_convert = int(len(trialing_companies) * config.convert_trialing_ratio)
        # Convert at least one if there are 10+? Keep zero if less than 10% yields 0 to respect spec strictly
        companies_to_convert = (
            rng.choice(trialing_companies, size=num_to_convert, replace=False).tolist()
            if num_to_convert > 0
            else []
        )

        if companies_to_convert:
            total_days = len(date_index)
            min_offset = min(config.min_conversion_offset_days, max(total_days - 1, 0))
            max_offset = (
                (config.max_conversion_offset_days if config.max_conversion_offset_days is not None else total_days - 1)
            )
            max_offset = max(min_offset, min(max_offset, total_days - 1))

            # Assign a conversion date index per company
            conversion_offsets = rng.integers(low=min_offset, high=max_offset + 1, size=len(companies_to_convert))
            company_to_convdate: Dict[str, object] = {
                cid: date_index[offset] for cid, offset in zip(companies_to_convert, conversion_offsets)
            }

            # Apply conversion: if company selected and date >= conversion date, set Paying
            convdate_series = dim_df["COMPANY_ID"].map(company_to_convdate)
            is_selected = dim_df["COMPANY_ID"].isin(companies_to_convert)
            view_dates = pd.to_datetime(dim_df["VIEW_DATE"])  # datetime64[ns]
            conv_dt = pd.to_datetime(convdate_series)
            is_after_or_on = view_dates >= conv_dt
            dim_df.loc[is_selected & is_after_or_on, "STATUS"] = "Paying"

    # Ensure column order and types. Output with lowercase headers to match other CSVs in this repo
    dim_df = dim_df[["VIEW_DATE", "COMPANY_ID", "STATUS", "INDUSTRY"]].copy()
    dim_df["VIEW_DATE"] = pd.to_datetime(dim_df["VIEW_DATE"]).dt.strftime("%Y-%m-%d")

    dim_df.columns = ["view_date", "company_id", "status", "industry"]

    return dim_df


def main() -> None:
    dim_df = generate_company_date_dim()
    dim_df.to_csv(OUTPUT_DIM_PATH, index=False)
    print(f"Wrote {len(dim_df):,} rows to {OUTPUT_DIM_PATH}")


if __name__ == "__main__":
    main()


