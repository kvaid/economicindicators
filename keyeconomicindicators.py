import subprocess
import sys
import threading
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from flask import Response

BASE_DIR = Path(__file__).resolve().parent

SERIES = {
    "1Y": "BC_1YEAR",
    "2Y": "BC_2YEAR",
    "5Y": "BC_5YEAR",
    "10Y": "BC_10YEAR",
    "30Y": "BC_30YEAR",
}
MATURITY_PRESETS = list(SERIES.keys())
CS_BASELINE_PRESETS = ["1Y", "2Y", "5Y", "10Y", "30Y"]

PRESETS = ["1W", "YTD", "1M", "3M", "6M", "1Y", "5Y", "10Y"]
YIELD_COLORS = {
    "1Y": "#B7D7F5",
    "2Y": "#8EBFEA",
    "5Y": "#5B9EDB",
    "10Y": "#2E73B8",
    "30Y": "#124A80",
}
CREDIT_YIELD_COLS = {
    "us_ig_corp": "ice_bofa_us_corporate_effective_yield",
    "aaa_corp": "ice_bofa_aaa_us_corporate_effective_yield",
    "us_hy_corp": "ice_bofa_us_high_yield_effective_yield",
    "ig_muni": "IG_MUNIS:MUB",
    "hy_muni": "HY_MUNIS:HYD",
    "aaa_clo": "AAA_CLO:JAAA",
    "senior_loans": "SENIOR_LOANS:BKLN",
    "agency_mbs": "AGENCY_MBS:MBB",
}
BOND_LINE_COLORS = {
    "us_ig_corp": "#A1D99B",
    "aaa_corp": "#74C476",
    "us_hy_corp": "#41AB5D",
    "ig_muni": "#238B45",
    "hy_muni": "#1E7A3D",
    "aaa_clo": "#146B34",
    "senior_loans": "#0D572A",
    "agency_mbs": "#08441E",
}
CREDIT_SPREAD_BUTTON_COLORS = {
    "us_ig_corp": "#F8B4B4",
    "aaa_corp": "#F28B82",
    "us_hy_corp": "#EF5350",
    "ig_muni": "#E53935",
    "hy_muni": "#D32F2F",
    "aaa_clo": "#C62828",
    "senior_loans": "#B71C1C",
    "agency_mbs": "#8E0000",
}
VOLATILITY_COLS = [
    "vix",
    "vxn",
    "rvx",
    "vxeem",
    "skew",
    "gvz",
    "ovx",
    "stlfsi",
    "hy_oas",
    "ig_oas",
    "move",
    "dxy",
    "cnn_fear_greed",
]
VOLATILITY_DATA_COLS = {col: col for col in VOLATILITY_COLS}
VOLATILITY_BUTTON_COLORS = {
    "vix": "#BE185D",
    "vxn": "#0891B2",
    "rvx": "#0EA5E9",
    "vxeem": "#2563EB",
    "skew": "#7C3AED",
    "gvz": "#D97706",
    "ovx": "#DC2626",
    "stlfsi": "#6D28D9",
    "hy_oas": "#B45309",
    "ig_oas": "#0F766E",
    "move": "#374151",
    "dxy": "#92400E",
    "cnn_fear_greed": "#EA580C",
}
VOLATILITY_HOVER_LABELS = {
    "vix": "VIX (Equity Volatility)",
    "vxn": "VXN (NASDAQ-100 Volatility)",
    "rvx": "VIX of small-caps, measures expected 30-day volatility in the Russell 2000 Index (small-cap U.S. stocks) based on options prices for the RUT (Russell 2000 Index futures/options)",
    "vxeem": "Primary gauge for emerging markets volatility. It estimates expected 30-day volatility of returns on the MSCI Emerging Markets Index (via options on the iShares MSCI Emerging Markets ETF, ticker EEM)",
    "skew": "SKEW measures how expensive far OOM put options are relative to NTM options on S&P 500. SKEW rises when investors fear (or hedge) for black swan events",
    "gvz": "Gold Volatility",
    "ovx": "Oil Volatility",
    "stlfsi": "Fed Financial Stress Index",
    "hy_oas": "High Yield Corporate OAS",
    "ig_oas": "IG Corporate OAS",
    "dxy": "Measures the value of the USD relative to a basket of euro (57.6% weight), Japanese yen (13.6%), British pound (11.9%), Canadian dollar (9.1%), Swedish krona (4.2%), and Swiss franc (3.6%). It serves as a broad gauge of USD strength/weakness in global FX markets, often inversely correlated with equity/commodity assets",
    "cnn_fear_greed": "CNN Fear & Greed Proxy (0-100 composite)",
    "move": "MOVE Index is the bond market’s analog to VIX: it’s a market-implied measure of expected volatility in U.S. Treasury yields over the next ~30 days. It’s built from a yield-curve-weighted basket of at-the-money, 1-month options tied to key points on the Treasury curve",
}
VOLATILITY_BUTTON_LABELS = {
    "vix": "VIX (S&P 500)",
    "vxn": "VXN (NASDAQ-100)",
    "rvx": "RVX (RUSSELL 2000)",
    "vxeem": "VXEEM (MSCI EM)",
    "skew": "SKEW (TAIL PRICING)",
    "gvz": "GVZ (GOLD)",
    "ovx": "OVX (OIL)",
    "stlfsi": "STLFSI4 (FED STRESS INDEX)",
    "hy_oas": "HY OAS (HY BOND SPREADS)",
    "ig_oas": "IG OAS (IG BOND SPREADS)",
    "move": "MOVE (BONDS)",
    "dxy": "DXY (US DOLLAR)",
    "cnn_fear_greed": "CNN FEAR AND GREED",
}
REFRESH_SCRIPTS = [
    "download_scripts/download_fedrate.py",
    "download_scripts/download_inflation.py",
    "download_scripts/download_treasury_data.py",
    "download_scripts/download_bond_yields.py",
    "download_scripts/download_volatility.py",
    "download_scripts/download_unemployment.py",
]

refresh_lock = threading.Lock()
refresh_state = {
    "running": False,
    "done": False,
    "ok": True,
    "message": "",
    "progress": 0,
}
chart_dataset_lock = threading.Lock()
chart_dataset_cache = {"csv": "date\n"}


def load_and_process_csv(path: str, date_col: str = "date") -> pd.DataFrame:
    csv_path = BASE_DIR / path
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, parse_dates=[date_col])
        df = df.rename(columns={date_col: "DATE"}).sort_values("DATE")
        return df
    except Exception:
        return pd.DataFrame()


def align_to_month_end(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["DATE"] = df["DATE"] + pd.offsets.MonthEnd(0)
    return df.sort_values("DATE").reset_index(drop=True)


def align_to_treasury_daily_calendar(
    indicator_df: pd.DataFrame,
    value_cols: list[str],
    treasury_dates: pd.Series,
) -> pd.DataFrame:
    if indicator_df.empty:
        return indicator_df

    base = indicator_df[["DATE"] + value_cols].copy()
    base = base.sort_values("DATE").drop_duplicates(subset=["DATE"]).set_index("DATE")

    daily_index = pd.DatetimeIndex(pd.Series(treasury_dates).dropna().sort_values().unique())
    aligned = base.reindex(daily_index).ffill()
    aligned.index.name = "DATE"
    return aligned.reset_index()


def filter_by_date(df: pd.DataFrame, start: object, end: object) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["DATE"].dt.date >= start) & (df["DATE"].dt.date <= end)
    return df.loc[mask].copy()


def _set_refresh_state(**kwargs) -> None:
    with refresh_lock:
        refresh_state.update(kwargs)


def _refresh_worker() -> None:
    total = len(REFRESH_SCRIPTS)
    ok = True
    _set_refresh_state(running=True, done=False, ok=True, message="Starting refresh...", progress=0)

    for i, script in enumerate(REFRESH_SCRIPTS, start=1):
        _set_refresh_state(message=f"Running {script}...", progress=int(((i - 1) / total) * 100))
        try:
            subprocess.run(
                [sys.executable, str(BASE_DIR / script)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            ok = False
            err_lines = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()
            summary = err_lines[-1] if err_lines else str(exc)
            _set_refresh_state(
                message=f"Error in {script}: {summary}",
                progress=int((i / total) * 100),
                ok=False,
            )
        else:
            _set_refresh_state(
                message=f"Completed {script}",
                progress=int((i / total) * 100),
            )

    final_message = "Refresh complete." if ok else "Refresh complete with errors."
    _set_refresh_state(running=False, done=True, ok=ok, message=final_message, progress=100)


def start_refresh_worker() -> bool:
    with refresh_lock:
        if refresh_state["running"]:
            return False
        refresh_state.update({"running": True, "done": False, "ok": True, "message": "Starting refresh...", "progress": 0})
    worker = threading.Thread(target=_refresh_worker, daemon=True)
    worker.start()
    return True


def run_startup_refresh() -> None:
    for script in REFRESH_SCRIPTS:
        try:
            subprocess.run(
                [sys.executable, str(BASE_DIR / script)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            err_lines = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()
            summary = err_lines[-1] if err_lines else str(exc)
            print(f"[startup refresh] Error in {script}: {summary}", file=sys.stderr)


if __name__ == "__main__" and os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    run_startup_refresh()


def get_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    ust_df = load_and_process_csv("data/ust.csv")
    today = pd.Timestamp.today().normalize()
    if ust_df.empty:
        return today, today
    return ust_df["DATE"].min(), max(ust_df["DATE"].max(), today)


def compute_preset_range(preset: str, min_date: pd.Timestamp, max_date: pd.Timestamp) -> tuple[str, str]:
    now = pd.Timestamp(max_date)
    if preset == "1W":
        start = now - pd.DateOffset(weeks=1)
    elif preset == "1M":
        start = now - pd.DateOffset(months=1)
    elif preset == "YTD":
        start = now.replace(month=1, day=1)
    elif preset == "3M":
        start = now - pd.DateOffset(months=3)
    elif preset == "6M":
        start = now - pd.DateOffset(months=6)
    elif preset == "1Y":
        start = now - pd.DateOffset(years=1)
    elif preset == "5Y":
        start = now - pd.DateOffset(years=5)
    elif preset == "10Y":
        start = now - pd.DateOffset(years=10)
    else:
        start = now

    start = max(pd.Timestamp(min_date), start)
    return start.date().isoformat(), pd.Timestamp(max_date).date().isoformat()


def date_to_timeline_idx(date_value: object, min_date: pd.Timestamp, max_date: pd.Timestamp) -> int:
    dt = pd.Timestamp(date_value)
    dt = max(min_date, min(max_date, dt))
    return int((dt - min_date).days)


def timeline_idx_to_date(idx: int, min_date: pd.Timestamp, max_date: pd.Timestamp) -> str:
    total_days = max((max_date - min_date).days, 1)
    idx = int(max(0, min(total_days, idx)))
    return (min_date + pd.Timedelta(days=idx)).date().isoformat()


def build_timeline_marks(min_date: pd.Timestamp, max_date: pd.Timestamp) -> dict[int, str]:
    total_days = max((max_date - min_date).days, 1)
    marks = {0: min_date.strftime("%Y"), total_days: max_date.strftime("%Y")}
    step = 10
    for year in range(min_date.year + 1, max_date.year):
        if year % step != 0:
            continue
        year_start = pd.Timestamp(year=year, month=1, day=1)
        marks[int((year_start - min_date).days)] = str(year)
    return dict(sorted(marks.items()))


def build_figure(
    plot_ust: pd.DataFrame,
    plot_bond_yields: pd.DataFrame,
    plot_fed: pd.DataFrame,
    plot_infl: pd.DataFrame,
    plot_unrate: pd.DataFrame,
    plot_vol: pd.DataFrame,
    selected_maturities: list[str],
    show_yields: bool,
    show_spread: bool,
    show_us_ig_corp: bool,
    show_aaa_corp: bool,
    show_us_hy_corp: bool,
    show_ig_muni: bool,
    show_hy_muni: bool,
    show_aaa_clo: bool,
    show_senior_loans: bool,
    show_agency_mbs: bool,
    show_cs_us_ig_corp: bool,
    show_cs_aaa_corp: bool,
    show_cs_us_hy_corp: bool,
    show_cs_ig_muni: bool,
    show_cs_hy_muni: bool,
    show_cs_aaa_clo: bool,
    show_cs_senior_loans: bool,
    show_cs_agency_mbs: bool,
    cs_baseline_tenor: str,
    show_fed_rate: bool,
    show_inflation: bool,
    show_unemployment: bool,
    show_u6_unemployment: bool,
    show_unemp_ind: bool,
    show_vol_vix: bool,
    show_vol_vxn: bool,
    show_vol_rvx: bool,
    show_vol_vxeem: bool,
    show_vol_skew: bool,
    show_vol_gvz: bool,
    show_vol_ovx: bool,
    show_vol_stlfsi: bool,
    show_vol_hy_oas: bool,
    show_vol_ig_oas: bool,
    show_vol_move: bool,
    show_vol_dxy: bool,
    show_vol_cnn_fear_greed: bool,
    vol_band_mode: str | None,
    vol_median_mode: str | None,
) -> go.Figure:
    all_vals: list[float] = []
    baseline_tenor = cs_baseline_tenor if cs_baseline_tenor in SERIES else "10Y"
    baseline_col = SERIES[baseline_tenor]

    if show_yields and selected_maturities:
        for maturity in selected_maturities:
            all_vals.extend(plot_ust[SERIES[maturity]].dropna().tolist())
    if show_spread and not plot_ust.empty:
        all_vals.extend(plot_ust["SPREAD_10Y_2Y"].dropna().tolist())
    if (
        show_us_ig_corp
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["us_ig_corp"] in plot_bond_yields.columns
    ):
        all_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["us_ig_corp"]].dropna().tolist())
    if (
        show_aaa_corp
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["aaa_corp"] in plot_bond_yields.columns
    ):
        all_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["aaa_corp"]].dropna().tolist())
    if (
        show_us_hy_corp
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["us_hy_corp"] in plot_bond_yields.columns
    ):
        all_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["us_hy_corp"]].dropna().tolist())
    if (
        show_ig_muni
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["ig_muni"] in plot_bond_yields.columns
    ):
        all_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["ig_muni"]].dropna().tolist())
    if (
        show_hy_muni
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["hy_muni"] in plot_bond_yields.columns
    ):
        all_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["hy_muni"]].dropna().tolist())
    if (
        show_aaa_clo
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["aaa_clo"] in plot_bond_yields.columns
    ):
        all_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["aaa_clo"]].dropna().tolist())
    if (
        show_senior_loans
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["senior_loans"] in plot_bond_yields.columns
    ):
        all_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["senior_loans"]].dropna().tolist())
    if (
        show_agency_mbs
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["agency_mbs"] in plot_bond_yields.columns
    ):
        all_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["agency_mbs"]].dropna().tolist())
    if not plot_bond_yields.empty and not plot_ust.empty and baseline_col in plot_ust.columns:
        cs_flags = {
            "us_ig_corp": show_cs_us_ig_corp,
            "aaa_corp": show_cs_aaa_corp,
            "us_hy_corp": show_cs_us_hy_corp,
            "ig_muni": show_cs_ig_muni,
            "hy_muni": show_cs_hy_muni,
            "aaa_clo": show_cs_aaa_clo,
            "senior_loans": show_cs_senior_loans,
            "agency_mbs": show_cs_agency_mbs,
        }
        base = plot_ust[["DATE", baseline_col]]
        for key, enabled in cs_flags.items():
            col = CREDIT_YIELD_COLS[key]
            if not enabled or col not in plot_bond_yields.columns:
                continue
            s_df = plot_bond_yields[["DATE", col]].merge(base, on="DATE", how="left")
            spread_vals = (
                pd.to_numeric(s_df[col], errors="coerce")
                - pd.to_numeric(s_df[baseline_col], errors="coerce")
            ).dropna()
            all_vals.extend(spread_vals.tolist())
    if show_inflation and not plot_infl.empty:
        all_vals.extend(plot_infl["PCE_YoY"].dropna().tolist())
    if show_unemployment and not plot_unrate.empty:
        all_vals.extend(plot_unrate["UNRATE"].dropna().tolist())
    if show_u6_unemployment and not plot_unrate.empty and "U6RATE" in plot_unrate.columns:
        all_vals.extend(plot_unrate["U6RATE"].dropna().tolist())
    if show_unemp_ind and not plot_unrate.empty:
        all_vals.extend(plot_unrate["UNEMP_INDICATOR"].dropna().tolist())
    if show_fed_rate and not plot_fed.empty:
        all_vals.extend(plot_fed["FED_RATE"].dropna().tolist())
    if not plot_vol.empty:
        selected_band_mode = str(vol_band_mode or "").strip()
        if selected_band_mode not in {"25_75", "10_90"}:
            selected_band_mode = "none"
        selected_median_mode = str(vol_median_mode or "").strip()
        if selected_median_mode not in {"none", "3y_median", "10y_median", "15y_median", "3y_mean", "10y_mean", "15y_mean"}:
            selected_median_mode = "none"
        vol_flags = {
            "vix": show_vol_vix,
            "vxn": show_vol_vxn,
            "rvx": show_vol_rvx,
            "vxeem": show_vol_vxeem,
            "skew": show_vol_skew,
            "gvz": show_vol_gvz,
            "ovx": show_vol_ovx,
            "stlfsi": show_vol_stlfsi,
            "hy_oas": show_vol_hy_oas,
            "ig_oas": show_vol_ig_oas,
            "move": show_vol_move,
            "dxy": show_vol_dxy,
            "cnn_fear_greed": show_vol_cnn_fear_greed,
        }
        for key, enabled in vol_flags.items():
            z_col = VOLATILITY_DATA_COLS[key]
            if enabled and z_col in plot_vol.columns:
                all_vals.extend(pd.to_numeric(plot_vol[z_col], errors="coerce").dropna().tolist())

    if all_vals:
        v_min, v_max = min(all_vals), max(all_vals)
        pad = 0.05 * (v_max - v_min) if v_max != v_min else 1.0
        y_range = [v_min - pad, v_max + pad]
    else:
        y_range = [0, 10]

    fig = go.Figure()

    if show_yields:
        for maturity in selected_maturities:
            fig.add_trace(
                go.Scatter(
                    x=plot_ust["DATE"],
                    y=plot_ust[SERIES[maturity]],
                    name=f"{maturity} Treasury Yield",
                    mode="lines",
                    line={"color": YIELD_COLORS.get(maturity, "#2E73B8"), "width": 2.4},
                    hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
                )
            )

    if show_fed_rate and not plot_fed.empty:
        fig.add_trace(
            go.Scatter(
                x=plot_fed["DATE"],
                y=plot_fed["FED_RATE"],
                name="Fed Rate",
                mode="lines",
                line={"color": "#1B8F3A", "width": 2.4, "dash": "dot"},
                line_shape="hv",
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
        rate_delta = pd.to_numeric(plot_fed["FED_RATE"], errors="coerce").diff()
        hikes = plot_fed[rate_delta > 0.001]
        cuts = plot_fed[rate_delta < -0.001]
        if not hikes.empty:
            fig.add_trace(
                go.Scatter(
                    x=hikes["DATE"],
                    y=hikes["FED_RATE"],
                    mode="markers",
                    marker={"color": "red", "size": 6},
                    showlegend=False,
                )
            )
        if not cuts.empty:
            fig.add_trace(
                go.Scatter(
                    x=cuts["DATE"],
                    y=cuts["FED_RATE"],
                    mode="markers",
                    marker={"color": "green", "size": 6},
                    showlegend=False,
                )
            )

    if show_spread and not plot_ust.empty:
        fig.add_trace(
            go.Scatter(
                x=plot_ust["DATE"],
                y=plot_ust["SPREAD_10Y_2Y"],
                name="10Y-2Y Treasury Yield Spread",
                line={"color": "#0B3A63", "width": 2, "dash": "dot"},
                yaxis="y2",
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
    if (
        show_us_ig_corp
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["us_ig_corp"] in plot_bond_yields.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=plot_bond_yields["DATE"],
                y=plot_bond_yields[CREDIT_YIELD_COLS["us_ig_corp"]],
                name="IG CORP BOND YIELD",
                line={"color": BOND_LINE_COLORS["us_ig_corp"], "width": 1.6},
                yaxis="y2",
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
    if (
        show_aaa_corp
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["aaa_corp"] in plot_bond_yields.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=plot_bond_yields["DATE"],
                y=plot_bond_yields[CREDIT_YIELD_COLS["aaa_corp"]],
                name="AAA CORP BOND YIELD",
                line={"color": BOND_LINE_COLORS["aaa_corp"], "width": 1.6},
                yaxis="y2",
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
    if (
        show_us_hy_corp
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["us_hy_corp"] in plot_bond_yields.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=plot_bond_yields["DATE"],
                y=plot_bond_yields[CREDIT_YIELD_COLS["us_hy_corp"]],
                name="HY CORP BOND YIELD",
                line={"color": BOND_LINE_COLORS["us_hy_corp"], "width": 1.6},
                yaxis="y2",
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
    if (
        show_ig_muni
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["ig_muni"] in plot_bond_yields.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=plot_bond_yields["DATE"],
                y=plot_bond_yields[CREDIT_YIELD_COLS["ig_muni"]],
                name="IG MUNI BOND YIELD",
                line={"color": BOND_LINE_COLORS["ig_muni"], "width": 1.6},
                yaxis="y2",
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
    if (
        show_hy_muni
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["hy_muni"] in plot_bond_yields.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=plot_bond_yields["DATE"],
                y=plot_bond_yields[CREDIT_YIELD_COLS["hy_muni"]],
                name="HY MUNI BOND YIELD",
                line={"color": BOND_LINE_COLORS["hy_muni"], "width": 1.6},
                yaxis="y2",
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
    if (
        show_aaa_clo
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["aaa_clo"] in plot_bond_yields.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=plot_bond_yields["DATE"],
                y=plot_bond_yields[CREDIT_YIELD_COLS["aaa_clo"]],
                name="AAA CLO BOND YIELD",
                line={"color": BOND_LINE_COLORS["aaa_clo"], "width": 1.6},
                yaxis="y2",
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
    if (
        show_senior_loans
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["senior_loans"] in plot_bond_yields.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=plot_bond_yields["DATE"],
                y=plot_bond_yields[CREDIT_YIELD_COLS["senior_loans"]],
                name="SENIOR LOANS BOND YIELD",
                line={"color": BOND_LINE_COLORS["senior_loans"], "width": 1.6},
                yaxis="y2",
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
    if (
        show_agency_mbs
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["agency_mbs"] in plot_bond_yields.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=plot_bond_yields["DATE"],
                y=plot_bond_yields[CREDIT_YIELD_COLS["agency_mbs"]],
                name="AGENCY MBS BOND YIELD",
                line={"color": BOND_LINE_COLORS["agency_mbs"], "width": 1.6},
                yaxis="y2",
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
    if not plot_bond_yields.empty and not plot_ust.empty and baseline_col in plot_ust.columns:
        cs_traces = [
            ("us_ig_corp", show_cs_us_ig_corp, f"IG CORP BOND - {baseline_tenor} SPREAD"),
            ("aaa_corp", show_cs_aaa_corp, f"AAA CORP BOND - {baseline_tenor} SPREAD"),
            ("us_hy_corp", show_cs_us_hy_corp, f"HY CORP BOND - {baseline_tenor} SPREAD"),
            ("ig_muni", show_cs_ig_muni, f"IG MUNI BOND - {baseline_tenor} SPREAD"),
            ("hy_muni", show_cs_hy_muni, f"HY MUNI BOND - {baseline_tenor} SPREAD"),
            ("aaa_clo", show_cs_aaa_clo, f"AAA CLO BOND - {baseline_tenor} SPREAD"),
            ("senior_loans", show_cs_senior_loans, f"SENIOR LOANS BOND - {baseline_tenor} SPREAD"),
            ("agency_mbs", show_cs_agency_mbs, f"AGENCY MBS BOND - {baseline_tenor} SPREAD"),
        ]
        base = plot_ust[["DATE", baseline_col]]
        for key, enabled, label in cs_traces:
            col = CREDIT_YIELD_COLS[key]
            if not enabled or col not in plot_bond_yields.columns:
                continue
            s_df = plot_bond_yields[["DATE", col]].merge(base, on="DATE", how="left")
            spread = pd.to_numeric(s_df[col], errors="coerce") - pd.to_numeric(s_df[baseline_col], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=s_df["DATE"],
                    y=spread,
                    name=label,
                    line={"color": CREDIT_SPREAD_BUTTON_COLORS[key], "width": 1.6, "dash": "dot"},
                    yaxis="y2",
                    hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
                )
            )
    if show_inflation and not plot_infl.empty:
        fig.add_trace(
            go.Scatter(
                x=plot_infl["DATE"],
                y=plot_infl["PCE_YoY"],
                name="Core PCE Inflation",
                line={"color": "#FF1F1F", "width": 2.4, "dash": "dot"},
                yaxis="y2",
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )

    if show_unemployment and not plot_unrate.empty:
        fig.add_trace(
            go.Scatter(
                x=plot_unrate["DATE"],
                y=plot_unrate["UNRATE"],
                name="Unemployment Rate",
                line={"color": "#F28C28", "width": 2, "dash": "dot"},
                connectgaps=True,
                yaxis="y2",
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )

    if show_u6_unemployment and not plot_unrate.empty and "U6RATE" in plot_unrate.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_unrate["DATE"],
                y=plot_unrate["U6RATE"],
                name="U-6 Unemployment Rate",
                line={"color": "#8B5A2B", "width": 2, "dash": "dot"},
                connectgaps=True,
                yaxis="y2",
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )

    if show_unemp_ind and not plot_unrate.empty:
        fig.add_trace(
            go.Scatter(
                x=plot_unrate["DATE"],
                y=plot_unrate["UNEMP_INDICATOR"],
                name="Unemp. Indicator (U3-NROU)",
                line={"color": "#8B0000", "width": 2, "dash": "dot"},
                connectgaps=True,
                yaxis="y2",
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
    if not plot_vol.empty:
        vix_plotted = False
        vxn_plotted = False
        rvx_plotted = False
        vxeem_plotted = False
        skew_plotted = False
        gvz_plotted = False
        ovx_plotted = False
        stlfsi_plotted = False
        ig_oas_plotted = False
        hy_oas_plotted = False
        move_plotted = False
        dxy_plotted = False
        cnn_fear_greed_plotted = False
        vol_flags = {
            "vix": show_vol_vix,
            "vxn": show_vol_vxn,
            "rvx": show_vol_rvx,
            "vxeem": show_vol_vxeem,
            "skew": show_vol_skew,
            "gvz": show_vol_gvz,
            "ovx": show_vol_ovx,
            "stlfsi": show_vol_stlfsi,
            "hy_oas": show_vol_hy_oas,
            "ig_oas": show_vol_ig_oas,
            "move": show_vol_move,
            "dxy": show_vol_dxy,
            "cnn_fear_greed": show_vol_cnn_fear_greed,
        }
        for key, enabled in vol_flags.items():
            z_col = VOLATILITY_DATA_COLS[key]
            if not enabled or z_col not in plot_vol.columns:
                continue
            display_label = VOLATILITY_BUTTON_LABELS.get(key, key.upper())
            vol_series = pd.Series(
                pd.to_numeric(plot_vol[z_col], errors="coerce").values,
                index=pd.to_datetime(plot_vol["DATE"], errors="coerce"),
            ).sort_index()
            if key == "vix" and not vol_series.dropna().empty:
                vix_plotted = True
            if key == "vxn" and not vol_series.dropna().empty:
                vxn_plotted = True
            if key == "rvx" and not vol_series.dropna().empty:
                rvx_plotted = True
            if key == "vxeem" and not vol_series.dropna().empty:
                vxeem_plotted = True
            if key == "skew" and not vol_series.dropna().empty:
                skew_plotted = True
            if key == "gvz" and not vol_series.dropna().empty:
                gvz_plotted = True
            if key == "ovx" and not vol_series.dropna().empty:
                ovx_plotted = True
            if key == "stlfsi" and not vol_series.dropna().empty:
                stlfsi_plotted = True
            if key == "ig_oas" and not vol_series.dropna().empty:
                ig_oas_plotted = True
            if key == "hy_oas" and not vol_series.dropna().empty:
                hy_oas_plotted = True
            if key == "move" and not vol_series.dropna().empty:
                move_plotted = True
            if key == "dxy" and not vol_series.dropna().empty:
                dxy_plotted = True
            if key == "cnn_fear_greed" and not vol_series.dropna().empty:
                cnn_fear_greed_plotted = True
            window_map = {
                "3y_median": ("1096D", "3Y"),
                "10y_median": ("3652D", "10Y"),
                "15y_median": ("5479D", "15Y"),
                "3y_mean": ("1096D", "3Y"),
                "10y_mean": ("3652D", "10Y"),
                "15y_mean": ("5479D", "15Y"),
            }
            rolling_window, window_label = window_map.get(selected_median_mode, ("3652D", "10Y"))
            fig.add_trace(
                go.Scatter(
                    x=vol_series.index,
                    y=vol_series.values,
                    name=display_label,
                    mode="lines",
                    line={"color": VOLATILITY_BUTTON_COLORS[key], "width": 2.0},
                    yaxis="y2",
                    hovertemplate=f"{display_label}<br>%{{x|%b %d, %Y}}<br>%{{y:.2f}}<extra></extra>",
                )
            )
            if selected_median_mode.endswith("_median"):
                vol_overlay = vol_series.rolling(rolling_window, min_periods=1).median()
                overlay_name = f"{display_label} {window_label} MEDIAN"
                overlay_hover = f"{display_label} {window_label} Median"
                fig.add_trace(
                    go.Scatter(
                        x=vol_overlay.index,
                        y=vol_overlay.values,
                        name=overlay_name,
                        mode="lines",
                        line={"color": VOLATILITY_BUTTON_COLORS[key], "width": 2.0, "dash": "dot"},
                        yaxis="y2",
                        hovertemplate=f"{overlay_hover}<br>%{{x|%b %d, %Y}}<br>%{{y:.2f}}<extra></extra>",
                    )
                )
            elif selected_median_mode.endswith("_mean"):
                vol_overlay = vol_series.rolling(rolling_window, min_periods=1).mean()
                overlay_name = f"{display_label} {window_label} AVERAGE"
                overlay_hover = f"{display_label} {window_label} Average"
                fig.add_trace(
                    go.Scatter(
                        x=vol_overlay.index,
                        y=vol_overlay.values,
                        name=overlay_name,
                        mode="lines",
                        line={"color": VOLATILITY_BUTTON_COLORS[key], "width": 2.0, "dash": "dot"},
                        yaxis="y2",
                        hovertemplate=f"{overlay_hover}<br>%{{x|%b %d, %Y}}<br>%{{y:.2f}}<extra></extra>",
                    )
                )
            if selected_band_mode == "10_90":
                vol_p10 = vol_series.rolling(rolling_window, min_periods=1).quantile(0.10)
                vol_p90 = vol_series.rolling(rolling_window, min_periods=1).quantile(0.90)
                fig.add_trace(
                    go.Scatter(
                        x=vol_p10.index,
                        y=vol_p10.values,
                        name=f"{display_label} {window_label} P10",
                        mode="lines",
                        line={"color": VOLATILITY_BUTTON_COLORS[key], "width": 1.0, "dash": "dashdot"},
                        opacity=0.45,
                        yaxis="y2",
                        hovertemplate=f"{display_label} {window_label} P10<br>%{{x|%b %d, %Y}}<br>%{{y:.2f}}<extra></extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=vol_p90.index,
                        y=vol_p90.values,
                        name=f"{display_label} {window_label} P90",
                        mode="lines",
                        line={"color": VOLATILITY_BUTTON_COLORS[key], "width": 1.0, "dash": "dashdot"},
                        opacity=0.45,
                        fill="tonexty",
                        fillcolor="rgba(15, 23, 42, 0.05)",
                        yaxis="y2",
                        hovertemplate=f"{display_label} {window_label} P90<br>%{{x|%b %d, %Y}}<br>%{{y:.2f}}<extra></extra>",
                    )
                )
            elif selected_band_mode == "25_75":
                vol_p25 = vol_series.rolling(rolling_window, min_periods=1).quantile(0.25)
                vol_p75 = vol_series.rolling(rolling_window, min_periods=1).quantile(0.75)
                fig.add_trace(
                    go.Scatter(
                        x=vol_p25.index,
                        y=vol_p25.values,
                        name=f"{display_label} {window_label} P25",
                        mode="lines",
                        line={"color": VOLATILITY_BUTTON_COLORS[key], "width": 1.2, "dash": "dash"},
                        opacity=0.55,
                        yaxis="y2",
                        hovertemplate=f"{display_label} {window_label} P25<br>%{{x|%b %d, %Y}}<br>%{{y:.2f}}<extra></extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=vol_p75.index,
                        y=vol_p75.values,
                        name=f"{display_label} {window_label} P75",
                        mode="lines",
                        line={"color": VOLATILITY_BUTTON_COLORS[key], "width": 1.2, "dash": "dash"},
                        opacity=0.55,
                        fill="tonexty",
                        fillcolor="rgba(15, 23, 42, 0.08)",
                        yaxis="y2",
                        hovertemplate=f"{display_label} {window_label} P75<br>%{{x|%b %d, %Y}}<br>%{{y:.2f}}<extra></extra>",
                    )
                )
        if vix_plotted:
            fig.add_hrect(y0=0, y1=15, yref="y2", fillcolor="rgba(22, 163, 74, 0.12)", line_width=0, layer="below")
            fig.add_hrect(y0=25, y1=35, yref="y2", fillcolor="rgba(194, 65, 12, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=35, y1=50, yref="y2", fillcolor="rgba(185, 28, 28, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=50, y1=100, yref="y2", fillcolor="rgba(127, 29, 29, 0.16)", line_width=0, layer="below")
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=7.5,
                text="Low Vol / Complacency",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#166534"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=20,
                text="Normal / Watchful",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#4B5563"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=30,
                text="Elevated Stress",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#9A3412"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=42.5,
                text="High Stress / Fear",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#B91C1C"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=75,
                text="Extreme Stress / Panic",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#7F1D1D"},
                bgcolor="rgba(255,255,255,0.65)",
            )
        if vxn_plotted:
            fig.add_hrect(y0=0, y1=20, yref="y2", fillcolor="rgba(22, 163, 74, 0.12)", line_width=0, layer="below")
            fig.add_hrect(y0=30, y1=40, yref="y2", fillcolor="rgba(194, 65, 12, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=40, y1=55, yref="y2", fillcolor="rgba(185, 28, 28, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=55, y1=100, yref="y2", fillcolor="rgba(127, 29, 29, 0.16)", line_width=0, layer="below")
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=10,
                text="Low Vol / Complacency",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#166534"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=25,
                text="Normal / Watchful",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#4B5563"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=35,
                text="Elevated Stress",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#9A3412"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=47.5,
                text="High Stress / Fear",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#B91C1C"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=77.5,
                text="Extreme Stress / Panic",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#7F1D1D"},
                bgcolor="rgba(255,255,255,0.65)",
            )
        if rvx_plotted:
            fig.add_hrect(y0=0, y1=22, yref="y2", fillcolor="rgba(22, 163, 74, 0.12)", line_width=0, layer="below")
            fig.add_hrect(y0=32, y1=42, yref="y2", fillcolor="rgba(194, 65, 12, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=42, y1=58, yref="y2", fillcolor="rgba(185, 28, 28, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=58, y1=100, yref="y2", fillcolor="rgba(127, 29, 29, 0.16)", line_width=0, layer="below")
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=11,
                text="Low Vol / Complacency",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#166534"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=27,
                text="Normal / Watchful",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#4B5563"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=37,
                text="Elevated Stress",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#9A3412"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=50,
                text="High Stress / Fear",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#B91C1C"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=79,
                text="Extreme Stress / Panic",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#7F1D1D"},
                bgcolor="rgba(255,255,255,0.65)",
            )
        if skew_plotted:
            fig.add_hrect(y0=0, y1=130, yref="y2", fillcolor="rgba(22, 163, 74, 0.12)", line_width=0, layer="below")
            fig.add_hrect(y0=145, y1=160, yref="y2", fillcolor="rgba(194, 65, 12, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=160, y1=175, yref="y2", fillcolor="rgba(185, 28, 28, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=175, y1=260, yref="y2", fillcolor="rgba(127, 29, 29, 0.16)", line_width=0, layer="below")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=65, text="Low Tail-Risk Concern", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#166534"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=137.5, text="Watchful", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#4B5563"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=152.5, text="Elevated Tail-Risk Pricing", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#9A3412"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=167.5, text="High Tail-Risk Concern", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#B91C1C"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=217.5, text="Extreme Tail-Risk Pricing", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#7F1D1D"}, bgcolor="rgba(255,255,255,0.65)")
        if vxeem_plotted:
            fig.add_hrect(y0=0, y1=18, yref="y2", fillcolor="rgba(22, 163, 74, 0.12)", line_width=0, layer="below")
            fig.add_hrect(y0=28, y1=40, yref="y2", fillcolor="rgba(194, 65, 12, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=40, y1=55, yref="y2", fillcolor="rgba(185, 28, 28, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=55, y1=100, yref="y2", fillcolor="rgba(127, 29, 29, 0.16)", line_width=0, layer="below")
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=9,
                text="Low Vol / Complacency",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#166534"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=23,
                text="Normal / Watchful",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#4B5563"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=34,
                text="Elevated Stress",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#9A3412"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=47.5,
                text="High Stress / Fear",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#B91C1C"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=77.5,
                text="Extreme Stress / Panic",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#7F1D1D"},
                bgcolor="rgba(255,255,255,0.65)",
            )
        if gvz_plotted:
            fig.add_hrect(y0=0, y1=20, yref="y2", fillcolor="rgba(22, 163, 74, 0.12)", line_width=0, layer="below")
            fig.add_hrect(y0=30, y1=40, yref="y2", fillcolor="rgba(194, 65, 12, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=40, y1=55, yref="y2", fillcolor="rgba(185, 28, 28, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=55, y1=120, yref="y2", fillcolor="rgba(127, 29, 29, 0.16)", line_width=0, layer="below")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=10, text="Low Vol / Complacency", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#166534"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=25, text="Normal / Watchful", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#4B5563"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=35, text="Elevated Stress", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#9A3412"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=47.5, text="High Stress / Fear", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#B91C1C"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=87.5, text="Extreme Stress / Panic", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#7F1D1D"}, bgcolor="rgba(255,255,255,0.65)")
        if ovx_plotted:
            fig.add_hrect(y0=0, y1=35, yref="y2", fillcolor="rgba(22, 163, 74, 0.12)", line_width=0, layer="below")
            fig.add_hrect(y0=50, y1=70, yref="y2", fillcolor="rgba(194, 65, 12, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=70, y1=90, yref="y2", fillcolor="rgba(185, 28, 28, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=90, y1=160, yref="y2", fillcolor="rgba(127, 29, 29, 0.16)", line_width=0, layer="below")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=17.5, text="Low Vol / Complacency", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#166534"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=42.5, text="Normal / Watchful", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#4B5563"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=60, text="Elevated Stress", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#9A3412"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=80, text="High Stress / Fear", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#B91C1C"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=125, text="Extreme Stress / Panic", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#7F1D1D"}, bgcolor="rgba(255,255,255,0.65)")
        if stlfsi_plotted:
            fig.add_hrect(y0=-5, y1=-0.5, yref="y2", fillcolor="rgba(22, 163, 74, 0.12)", line_width=0, layer="below")
            fig.add_hrect(y0=0.5, y1=2.0, yref="y2", fillcolor="rgba(194, 65, 12, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=2.0, y1=3.0, yref="y2", fillcolor="rgba(185, 28, 28, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=3.0, y1=6.0, yref="y2", fillcolor="rgba(127, 29, 29, 0.16)", line_width=0, layer="below")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=-2.75, text="Easy Conditions", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#166534"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=0.0, text="Normal", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#4B5563"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=1.25, text="Elevated Stress", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#9A3412"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=2.5, text="High Stress", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#B91C1C"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=4.5, text="Extreme Systemic Stress", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#7F1D1D"}, bgcolor="rgba(255,255,255,0.65)")
        if ig_oas_plotted:
            fig.add_hline(
                y=0.8,
                yref="y2",
                line_dash="dash",
                line_color="#16A34A",
                line_width=1.4,
                annotation_text="Below 1.0: Low Credit Risk/Complacency (tight spreads, strong investor demand for corporates, stable economy)",
                annotation_position="bottom left",
                annotation_font={"size": 11, "color": "#166534"},
            )
            fig.add_hline(
                y=1.5,
                yref="y2",
                line_dash="dash",
                line_color="#FF8C00",
                line_width=1.4,
                annotation_text="1.5-2.0: Elevated Stress/Nervousness(widening due to recession fears, inflation surprises, or liquidity issues)",
                annotation_position="top left",
                annotation_font={"size": 11, "color": "#C2410C"},
            )
            fig.add_hline(
                y=3,
                yref="y2",
                line_dash="dash",
                line_color="#FF0000",
                line_width=1.4,
                annotation_text="Above 3.0: Significant Credit Fear, often during crises or sharp economic downturns",
                annotation_position="top left",
                annotation_font={"size": 11, "color": "#B91C1C"},
            )
        if hy_oas_plotted:
            fig.add_hrect(y0=0, y1=3, yref="y2", fillcolor="rgba(22, 163, 74, 0.12)", line_width=0, layer="below")
            fig.add_hrect(y0=6, y1=8, yref="y2", fillcolor="rgba(194, 65, 12, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=8, y1=10, yref="y2", fillcolor="rgba(185, 28, 28, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=10, y1=20, yref="y2", fillcolor="rgba(127, 29, 29, 0.16)", line_width=0, layer="below")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=1.5, text="Low Credit Risk / Complacency", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#166534"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=4.5, text="Normal / Watchful", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#4B5563"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=7, text="Elevated Credit Stress", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#9A3412"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=9, text="High Credit Stress", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#B91C1C"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=15, text="Extreme Credit Stress", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#7F1D1D"}, bgcolor="rgba(255,255,255,0.65)")
        if move_plotted:
            fig.add_hrect(y0=0, y1=80, yref="y2", fillcolor="rgba(22, 163, 74, 0.12)", line_width=0, layer="below")
            fig.add_hrect(y0=100, y1=120, yref="y2", fillcolor="rgba(194, 65, 12, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=120, y1=140, yref="y2", fillcolor="rgba(185, 28, 28, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=140, y1=240, yref="y2", fillcolor="rgba(127, 29, 29, 0.16)", line_width=0, layer="below")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=40, text="Low Rates Vol", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#166534"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=90, text="Watchful", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#4B5563"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=110, text="Elevated Uncertainty", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#9A3412"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=130, text="High Stress", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#B91C1C"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=190, text="Extreme Stress", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#7F1D1D"}, bgcolor="rgba(255,255,255,0.65)")
        if dxy_plotted:
            fig.add_hrect(y0=0, y1=90, yref="y2", fillcolor="rgba(185, 28, 28, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=100, y1=105, yref="y2", fillcolor="rgba(194, 65, 12, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=105, y1=110, yref="y2", fillcolor="rgba(22, 163, 74, 0.12)", line_width=0, layer="below")
            fig.add_hrect(y0=110, y1=150, yref="y2", fillcolor="rgba(21, 128, 61, 0.16)", line_width=0, layer="below")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=45, text="USD Weak Regime", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#B91C1C"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=95, text="Neutral USD", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#4B5563"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=102.5, text="Moderately Strong USD", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#9A3412"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=107.5, text="Strong USD", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#166534"}, bgcolor="rgba(255,255,255,0.65)")
            fig.add_annotation(xref="paper", x=0.005, yref="y2", y=130, text="Extreme USD Strength", showarrow=False, xanchor="left", yanchor="middle", font={"size": 11, "color": "#14532D"}, bgcolor="rgba(255,255,255,0.65)")
        if cnn_fear_greed_plotted:
            fig.add_hrect(y0=0, y1=25, yref="y2", fillcolor="rgba(185, 28, 28, 0.16)", line_width=0, layer="below")
            fig.add_hrect(y0=25, y1=45, yref="y2", fillcolor="rgba(194, 65, 12, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=55, y1=75, yref="y2", fillcolor="rgba(22, 163, 74, 0.12)", line_width=0, layer="below")
            fig.add_hrect(y0=75, y1=100, yref="y2", fillcolor="rgba(21, 128, 61, 0.16)", line_width=0, layer="below")
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=12.5,
                text="Extreme Fear",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#7F1D1D"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=35,
                text="Fear",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#9A3412"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=50,
                text="Neutral",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#4B5563"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=65,
                text="Greed",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#166534"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            fig.add_annotation(
                xref="paper",
                x=0.005,
                yref="y2",
                y=87.5,
                text="Extreme Greed",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font={"size": 11, "color": "#14532D"},
                bgcolor="rgba(255,255,255,0.65)",
            )
            y_range = [min(y_range[0], 0.0), max(y_range[1], 100.0)]

    fig.update_layout(
        template="plotly_white",
        font={"family": "Plus Jakarta Sans, Segoe UI, Arial, sans-serif", "size": 13, "color": "#1f2a37"},
        xaxis_title="Date",
        xaxis={
            "showgrid": True,
            "gridcolor": "rgba(15, 23, 42, 0.08)",
            "linecolor": "rgba(15, 23, 42, 0.25)",
        },
        yaxis={
            "title": "Yield (%)",
            "side": "left",
            "range": y_range,
            "tickformat": ".1f",
            "showgrid": True,
            "gridcolor": "rgba(15, 23, 42, 0.08)",
            "zeroline": True,
            "zerolinecolor": "rgba(15, 23, 42, 0.28)",
            "linecolor": "rgba(15, 23, 42, 0.25)",
        },
        yaxis2={
            "title": "%",
            "side": "right",
            "overlaying": "y",
            "showgrid": False,
            "range": y_range,
            "tickformat": ".1f",
            "linecolor": "rgba(15, 23, 42, 0.25)",
        },
        hovermode="x unified",
        hoverlabel={"bgcolor": "white", "font_size": 12, "font_family": "Plus Jakarta Sans, Segoe UI, Arial"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "rgba(15, 23, 42, 0.15)",
            "borderwidth": 1,
        },
        height=450,
        margin={"t": 36, "b": 40, "l": 52, "r": 52},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(15,23,42,0.45)", opacity=0.7, yref="y2")

    return fig


def latest_non_null_value(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def series_items_to_csv_text(series_items: list[tuple[str, pd.Series]]) -> str:
    if not series_items:
        return "date\n"

    series_list: list[pd.Series] = []
    for name, src in series_items:
        if not name or src is None:
            continue

        s = pd.Series(src).copy()
        if s.empty:
            continue

        s.index = pd.to_datetime(s.index, errors="coerce")
        s = pd.to_numeric(s, errors="coerce")
        s.name = name
        s = s.dropna()
        if s.empty:
            continue
        if isinstance(s.index, pd.DatetimeIndex):
            if s.index.tz is not None:
                s.index = s.index.tz_convert("UTC").tz_localize(None)
            s.index = s.index.normalize()
        s = s[~s.index.isna()]
        if not s.empty:
            series_list.append(s)

    if not series_list:
        return "date\n"

    out = pd.concat(series_list, axis=1, sort=False).sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    out.index = out.index.strftime("%Y-%m-%d")
    out.index.name = "date"
    return out.to_csv(index=True, float_format="%.4f")


def indicator_card(label: str, value: float | None) -> html.Div:
    value_txt = "N/A" if value is None else f"{value:.2f}%"
    return html.Div(
        [
            html.Div(label, className="indicator-label"),
            html.Div(value_txt, className="indicator-value"),
        ],
        className="indicator-card",
    )


min_dt, max_dt = get_date_bounds()
timeline_min_dt = max(min_dt, pd.to_datetime("2000-01-01"))
default_start_str, default_end_str = compute_preset_range("5Y", timeline_min_dt, max_dt)
default_start = pd.to_datetime(default_start_str)
default_end = pd.to_datetime(default_end_str)
timeline_total_days = max((max_dt - timeline_min_dt).days, 1)
default_slider_range = [
    date_to_timeline_idx(default_start, timeline_min_dt, max_dt),
    date_to_timeline_idx(default_end, timeline_min_dt, max_dt),
]
timeline_marks = build_timeline_marks(timeline_min_dt, max_dt)
dataset_range_text = f"Downloaded range: {min_dt.strftime('%b-%Y')} to {max_dt.strftime('%b-%Y')}"

app = Dash(__name__)
server = app.server
app.title = "Key Economic Indicators"


@app.server.route("/chart_dataset.csv")
def chart_dataset_csv():
    with chart_dataset_lock:
        csv_text = chart_dataset_cache.get("csv", "date\n")
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "inline; filename=chart_dataset.csv"},
    )

app.layout = html.Div(
    [
        dcc.Store(id="refresh-token", data=0),
        dcc.Store(id="active-preset", data="5Y"),
        dcc.Store(id="selected-maturities", data=["10Y"]),
        dcc.Store(id="show-spread-state", data=False),
        dcc.Store(id="show-us-ig-corp-state", data=False),
        dcc.Store(id="show-aaa-corp-state", data=False),
        dcc.Store(id="show-us-hy-corp-state", data=False),
        dcc.Store(id="show-ig-muni-state", data=False),
        dcc.Store(id="show-hy-muni-state", data=False),
        dcc.Store(id="show-aaa-clo-state", data=False),
        dcc.Store(id="show-senior-loans-state", data=False),
        dcc.Store(id="show-agency-mbs-state", data=False),
        dcc.Store(id="show-cs-us-ig-corp-state", data=False),
        dcc.Store(id="show-cs-aaa-corp-state", data=False),
        dcc.Store(id="show-cs-us-hy-corp-state", data=False),
        dcc.Store(id="show-cs-ig-muni-state", data=False),
        dcc.Store(id="show-cs-hy-muni-state", data=False),
        dcc.Store(id="show-cs-aaa-clo-state", data=False),
        dcc.Store(id="show-cs-senior-loans-state", data=False),
        dcc.Store(id="show-cs-agency-mbs-state", data=False),
        dcc.Store(id="show-vol-vix-state", data=False),
        dcc.Store(id="show-vol-vxn-state", data=False),
        dcc.Store(id="show-vol-rvx-state", data=False),
        dcc.Store(id="show-vol-vxeem-state", data=False),
        dcc.Store(id="show-vol-skew-state", data=False),
        dcc.Store(id="show-vol-gvz-state", data=False),
        dcc.Store(id="show-vol-ovx-state", data=False),
        dcc.Store(id="show-vol-stlfsi-state", data=False),
        dcc.Store(id="show-vol-hy_oas-state", data=False),
        dcc.Store(id="show-vol-ig_oas-state", data=False),
        dcc.Store(id="show-vol-move-state", data=False),
        dcc.Store(id="show-vol-dxy-state", data=False),
        dcc.Store(id="show-vol-cnn_fear_greed-state", data=False),
        dcc.Interval(id="refresh-progress-interval", interval=600, n_intervals=0, disabled=True),
        html.Div(
            [
                html.H1("Key Economic Indicators", className="page-title"),
                html.Div(id="warning-message", className="warning-message"),
                html.Div(id="latest-indicators", className="latest-indicators"),
                html.Div(
                    [
                        html.Div("Select Date Range", className="chart-panel-date-label"),
                        html.Div(
                            [
                                dcc.DatePickerSingle(
                                    id="start-date",
                                    min_date_allowed=timeline_min_dt.date(),
                                    max_date_allowed=max_dt.date(),
                                    date=default_start.date(),
                                    display_format="YYYY-MM-DD",
                                    className="date-single",
                                ),
                                dcc.DatePickerSingle(
                                    id="end-date",
                                    min_date_allowed=timeline_min_dt.date(),
                                    max_date_allowed=max_dt.date(),
                                    date=default_end.date(),
                                    display_format="YYYY-MM-DD",
                                    className="date-single",
                                ),
                            ],
                            className="date-range chart-panel-date-range",
                        ),
                        html.Div(
                            [html.Button(p, id=f"preset-{p}", n_clicks=0, className="preset-btn") for p in PRESETS],
                            className="chart-panel-presets",
                        ),
                        html.Div(
                            [
                                html.Div("", id="dataset-range-label", className="freq-label", style={"display": "none"}),
                                dcc.RangeSlider(
                                    id="timeline-slider",
                                    min=0,
                                    max=timeline_total_days,
                                    step=1,
                                    value=default_slider_range,
                                    marks=timeline_marks,
                                    allowCross=False,
                                    className="timeline-slider chart-panel-timeline-slider",
                                ),
                            ],
                            className="chart-panel-timeline",
                        ),
                        html.Button("Refresh Data", id="refresh-btn", n_clicks=0, className="primary-btn chart-action-btn"),
                        html.A(
                            "Download Chart",
                            id="download-dataset-btn",
                            href="/chart_dataset.csv",
                            target="_blank",
                            className="primary-btn download-dataset-btn chart-action-btn",
                        ),
                    ],
                    className="chart-panel-actions",
                ),
                html.Div(
                    [
                        html.Div(id="refresh-progress-msg", className="refresh-status"),
                        html.Progress(id="refresh-progress-bar", value="0", max=100, className="refresh-progress-bar"),
                    ],
                    id="refresh-progress-wrap",
                    className="chart-panel-progress",
                    style={"display": "none"},
                ),
                dcc.Graph(id="indicator-graph"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Major Economic Crisis", className="row-tag"),
                                html.Div(
                                    [
                                        dcc.Checklist(
                                            id="show-crisis-dotcom",
                                            options=[{"label": "Dot Com Bubble", "value": "on"}],
                                            value=[],
                                            className="control-group",
                                        ),
                                        dcc.Checklist(
                                            id="show-crisis-gfc",
                                            options=[{"label": "Global Financial Crisis", "value": "on"}],
                                            value=[],
                                            className="control-group",
                                        ),
                                        dcc.Checklist(
                                            id="show-crisis-eu-debt",
                                            options=[{"label": "EU Debt Crisis", "value": "on"}],
                                            value=[],
                                            className="control-group",
                                        ),
                                        dcc.Checklist(
                                            id="show-crisis-covid",
                                            options=[{"label": "COVID-19 Recession", "value": "on"}],
                                            value=[],
                                            className="control-group",
                                        ),
                                        dcc.Checklist(
                                            id="show-crisis-us-banking",
                                            options=[{"label": "US Regional Banking Crisis", "value": "on"}],
                                            value=[],
                                            className="control-group",
                                        ),
                                    ],
                                    className="secondary-controls",
                                ),
                            ],
                            className="spread-row",
                        ),
                    ],
                    className="below-chart-controls crisis-controls-box",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Macro indicators", className="row-tag"),
                                html.Div(
                                    [
                                        dcc.Checklist(
                                            id="show-fed-rate",
                                            options=[{"label": "Federal Reserve Rate", "value": "on"}],
                                            value=[],
                                            className="control-group",
                                        ),
                                        dcc.Checklist(
                                            id="show-inflation",
                                            options=[{"label": "Core PCE Inflation", "value": "on"}],
                                            value=[],
                                            className="control-group",
                                        ),
                                        dcc.Checklist(
                                            id="show-unemployment",
                                            options=[{"label": "U-3 Unemployment Rate", "value": "on"}],
                                            value=[],
                                            className="control-group",
                                        ),
                                        dcc.Checklist(
                                            id="show-u6-unemployment",
                                            options=[{"label": "U-6 Unemployment Rate", "value": "on"}],
                                            value=[],
                                            className="control-group",
                                        ),
                                        dcc.Checklist(
                                            id="show-unemp-ind",
                                            options=[{"label": "U3-NROU Unemployment", "value": "on"}],
                                            value=[],
                                            className="control-group",
                                        ),
                                    ],
                                    className="secondary-controls",
                                ),
                            ],
                            className="spread-row",
                        ),
                    ],
                    className="below-chart-controls macro-controls-box",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("", className="control-label"),
                                html.Div(
                                    [
                                        html.Div("Treasury Yields", className="row-tag"),
                                        html.Div(
                                            [
                                                html.Button("10Y-2Y", id="spread-btn", n_clicks=0, className="maturity-btn"),
                                                *[html.Button(m, id=f"maturity-{m}", n_clicks=0, className="maturity-btn") for m in MATURITY_PRESETS],
                                            ],
                                            className="maturity-grid",
                                        ),
                                    ],
                                    className="yield-row",
                                ),
                                html.Div("", className="row-spacer"),
                                html.Div(
                                    [
                                        html.Div("Bond Yields", className="row-tag"),
                                        html.Div(
                                            [
                                                html.Button("IG CORP", id="ig-corp-spread-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("AAA CORP", id="aaa-corp-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("HY CORP", id="ig-muni-spread-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("IG MUNI", id="ig-muni-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("HY MUNI", id="hy-muni-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("AAA CLO", id="aaa-clo-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("SENIOR LOANS", id="senior-loans-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("AGENCY MBS", id="agency-mbs-yield-btn", n_clicks=0, className="maturity-btn"),
                                            ],
                                            className="spread-grid",
                                        ),
                                    ],
                                    className="spread-row",
                                ),
                            ],
                            className="below-chart-controls",
                        ),
                        html.Div("", className="row-spacer"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(["Credit Spreads", html.Br(), "(vs Treasuries)"], className="row-tag credit-row-tag"),
                                        dcc.Dropdown(
                                            id="cs-baseline-tenor",
                                            options=[{"label": m, "value": m} for m in CS_BASELINE_PRESETS],
                                            value="10Y",
                                            clearable=False,
                                            searchable=False,
                                            className="cs-baseline-dropdown",
                                        ),
                                        html.Div(
                                            [
                                                html.Button("IG CORP", id="cs-ig-corp-spread-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("AAA CORP", id="cs-aaa-corp-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("HY CORP", id="cs-ig-muni-spread-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("IG MUNI", id="cs-ig-muni-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("HY MUNI", id="cs-hy-muni-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("AAA CLO", id="cs-aaa-clo-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("SENIOR LOANS", id="cs-senior-loans-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("AGENCY MBS", id="cs-agency-mbs-yield-btn", n_clicks=0, className="maturity-btn"),
                                            ],
                                            className="spread-grid",
                                        ),
                                    ],
                                    className="spread-row",
                                ),
                            ],
                            className="below-chart-controls credit-spreads-box",
                        ),
                        html.Div("", className="row-spacer"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("Volatility", className="row-tag"),
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id="vol-band-select",
                                                    options=[
                                                        {"label": "Select band", "value": "none"},
                                                        {"label": "25/75 (Elevated)", "value": "25_75"},
                                                        {"label": "10/90 (Stress)", "value": "10_90"},
                                                    ],
                                                    value="none",
                                                    placeholder="Percentile bands",
                                                    clearable=False,
                                                    searchable=False,
                                                    persistence=False,
                                                    className="vol-band-dropdown",
                                                ),
                                                dcc.Dropdown(
                                                    id="vol-median-select",
                                                    options=[
                                                        {"label": "No overlay", "value": "none"},
                                                        {"label": "3Y median overlay", "value": "3y_median"},
                                                        {"label": "10Y median overlay", "value": "10y_median"},
                                                        {"label": "15Y median overlay", "value": "15y_median"},
                                                        {"label": "3Y average overlay", "value": "3y_mean"},
                                                        {"label": "10Y average overlay", "value": "10y_mean"},
                                                        {"label": "15Y average overlay", "value": "15y_mean"},
                                                    ],
                                                    value="none",
                                                    clearable=False,
                                                    searchable=False,
                                                    persistence=False,
                                                    className="vol-median-dropdown",
                                                ),
                                            ],
                                            className="vol-dropdown-stack",
                                        ),
                                        html.Div(
                                            [
                                                *[
                                                    html.Button(
                                                        VOLATILITY_BUTTON_LABELS.get(col, col),
                                                        id=f"vol-{col}-btn",
                                                        n_clicks=0,
                                                        className="maturity-btn",
                                                        title=VOLATILITY_HOVER_LABELS.get(col, col),
                                                    )
                                                    for col in ["vix", "vxn", "rvx", "vxeem"]
                                                ],
                                                *[
                                                    html.Button(
                                                        VOLATILITY_BUTTON_LABELS.get(col, col),
                                                        id=f"vol-{col}-btn",
                                                        n_clicks=0,
                                                        className="maturity-btn",
                                                        title=VOLATILITY_HOVER_LABELS.get(col, col),
                                                    )
                                                    for col in ["skew", "move", "hy_oas", "dxy", "cnn_fear_greed"]
                                                ],
                                                html.Button(
                                                    VOLATILITY_BUTTON_LABELS.get("stlfsi", "stlfsi"),
                                                    id="vol-stlfsi-btn",
                                                    n_clicks=0,
                                                    className="maturity-btn",
                                                    title=VOLATILITY_HOVER_LABELS.get("stlfsi", "stlfsi"),
                                                ),
                                                html.Button(
                                                    VOLATILITY_BUTTON_LABELS.get("gvz", "gvz"),
                                                    id="vol-gvz-btn",
                                                    n_clicks=0,
                                                    className="maturity-btn",
                                                    title=VOLATILITY_HOVER_LABELS.get("gvz", "gvz"),
                                                ),
                                                html.Button(
                                                    VOLATILITY_BUTTON_LABELS.get("ovx", "ovx"),
                                                    id="vol-ovx-btn",
                                                    n_clicks=0,
                                                    className="maturity-btn",
                                                    title=VOLATILITY_HOVER_LABELS.get("ovx", "ovx"),
                                                ),
                                            ],
                                            className="spread-grid",
                                        ),
                                    ],
                                    className="spread-row",
                                ),
                            ],
                            className="below-chart-controls volatility-box",
                        ),
                    ],
                    className="below-chart-controls-wrap",
                ),
            ],
            className="main-content",
        ),
    ],
    className="app-shell",
)


@app.callback(
    Output("start-date", "date"),
    Output("end-date", "date"),
    Output("timeline-slider", "value"),
    Output("active-preset", "data"),
    [Input(f"preset-{p}", "n_clicks") for p in PRESETS],
    Input("show-crisis-dotcom", "value"),
    Input("show-crisis-gfc", "value"),
    Input("show-crisis-eu-debt", "value"),
    Input("show-crisis-covid", "value"),
    Input("show-crisis-us-banking", "value"),
    Input("start-date", "date"),
    Input("end-date", "date"),
    Input("timeline-slider", "value"),
    Input("indicator-graph", "relayoutData"),
    prevent_initial_call=True,
)
def apply_preset(*args):
    trigger = callback_context.triggered_id
    if not trigger:
        return no_update, no_update, no_update, no_update
    rest = args[len(PRESETS):]
    crisis_dotcom_val = rest[0]
    crisis_gfc_val = rest[1]
    crisis_eu_debt_val = rest[2]
    crisis_covid_val = rest[3]
    crisis_us_banking_val = rest[4]
    start_date_str = rest[5]
    end_date_str = rest[6]
    slider_range = rest[7]
    relayout_data = rest[8]

    if str(trigger).startswith("preset-"):
        preset = str(trigger).replace("preset-", "")
        start_date, end_date = compute_preset_range(preset, timeline_min_dt, max_dt)
        slider_value = [
            date_to_timeline_idx(start_date, timeline_min_dt, max_dt),
            date_to_timeline_idx(end_date, timeline_min_dt, max_dt),
        ]
        return start_date, end_date, slider_value, preset

    if trigger == "show-crisis-dotcom":
        if "on" in (crisis_dotcom_val or []):
            start_date = max(timeline_min_dt, pd.Timestamp("2000-01-01")).date().isoformat()
            end_date = pd.Timestamp(max_dt).date().isoformat()
            slider_value = [
                date_to_timeline_idx(start_date, timeline_min_dt, max_dt),
                date_to_timeline_idx(end_date, timeline_min_dt, max_dt),
            ]
            return start_date, end_date, slider_value, None
        return no_update, no_update, no_update, no_update

    if trigger == "show-crisis-gfc":
        if "on" in (crisis_gfc_val or []):
            start_date = max(timeline_min_dt, pd.Timestamp("2006-07-01")).date().isoformat()
            end_date = pd.Timestamp(max_dt).date().isoformat()
            slider_value = [
                date_to_timeline_idx(start_date, timeline_min_dt, max_dt),
                date_to_timeline_idx(end_date, timeline_min_dt, max_dt),
            ]
            return start_date, end_date, slider_value, None
        return no_update, no_update, no_update, no_update

    if trigger == "show-crisis-eu-debt":
        if "on" in (crisis_eu_debt_val or []):
            start_date = max(timeline_min_dt, pd.Timestamp("2009-01-01")).date().isoformat()
            end_date = pd.Timestamp(max_dt).date().isoformat()
            slider_value = [
                date_to_timeline_idx(start_date, timeline_min_dt, max_dt),
                date_to_timeline_idx(end_date, timeline_min_dt, max_dt),
            ]
            return start_date, end_date, slider_value, None
        return no_update, no_update, no_update, no_update

    if trigger == "show-crisis-covid":
        if "on" in (crisis_covid_val or []):
            start_date = max(timeline_min_dt, pd.Timestamp("2019-07-01")).date().isoformat()
            end_date = pd.Timestamp(max_dt).date().isoformat()
            slider_value = [
                date_to_timeline_idx(start_date, timeline_min_dt, max_dt),
                date_to_timeline_idx(end_date, timeline_min_dt, max_dt),
            ]
            return start_date, end_date, slider_value, None
        return no_update, no_update, no_update, no_update

    if trigger == "show-crisis-us-banking":
        if "on" in (crisis_us_banking_val or []):
            start_date = max(timeline_min_dt, pd.Timestamp("2022-07-01")).date().isoformat()
            end_date = pd.Timestamp(max_dt).date().isoformat()
            slider_value = [
                date_to_timeline_idx(start_date, timeline_min_dt, max_dt),
                date_to_timeline_idx(end_date, timeline_min_dt, max_dt),
            ]
            return start_date, end_date, slider_value, None
        return no_update, no_update, no_update, no_update

    if trigger == "timeline-slider":
        if not slider_range or len(slider_range) != 2:
            return no_update, no_update, no_update, no_update
        start_idx, end_idx = sorted([int(slider_range[0]), int(slider_range[1])])
        start_date = timeline_idx_to_date(start_idx, timeline_min_dt, max_dt)
        end_date = timeline_idx_to_date(end_idx, timeline_min_dt, max_dt)
        return start_date, end_date, [start_idx, end_idx], None

    if trigger in ("start-date", "end-date"):
        if not start_date_str or not end_date_str:
            return no_update, no_update, no_update, no_update
        start_dt = max(timeline_min_dt, min(max_dt, pd.Timestamp(start_date_str)))
        end_dt = max(timeline_min_dt, min(max_dt, pd.Timestamp(end_date_str)))
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt
        slider_value = [
            date_to_timeline_idx(start_dt, timeline_min_dt, max_dt),
            date_to_timeline_idx(end_dt, timeline_min_dt, max_dt),
        ]
        return start_dt.date().isoformat(), end_dt.date().isoformat(), slider_value, None

    if trigger == "indicator-graph":
        if not isinstance(relayout_data, dict):
            return no_update, no_update, no_update, no_update
        x0 = None
        x1 = None
        if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
            x0 = relayout_data.get("xaxis.range[0]")
            x1 = relayout_data.get("xaxis.range[1]")
        elif isinstance(relayout_data.get("xaxis.range"), list) and len(relayout_data["xaxis.range"]) == 2:
            x0, x1 = relayout_data["xaxis.range"]
        else:
            return no_update, no_update, no_update, no_update
        try:
            start_dt = max(timeline_min_dt, min(max_dt, pd.Timestamp(x0)))
            end_dt = max(timeline_min_dt, min(max_dt, pd.Timestamp(x1)))
        except Exception:
            return no_update, no_update, no_update, no_update
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt
        slider_value = [
            date_to_timeline_idx(start_dt, timeline_min_dt, max_dt),
            date_to_timeline_idx(end_dt, timeline_min_dt, max_dt),
        ]
        return start_dt.date().isoformat(), end_dt.date().isoformat(), slider_value, None

    return no_update, no_update, no_update, no_update


@app.callback(
    [Output(f"preset-{p}", "className") for p in PRESETS],
    Input("active-preset", "data"),
)
def update_preset_button_styles(active_preset):
    return [
        "preset-btn preset-btn-active" if preset == active_preset else "preset-btn"
        for preset in PRESETS
    ]


@app.callback(
    Output("selected-maturities", "data"),
    [Input(f"maturity-{m}", "n_clicks") for m in MATURITY_PRESETS],
    State("selected-maturities", "data"),
    prevent_initial_call=True,
)
def update_selected_maturities(*args):
    selected = list(args[-1] or [])
    trigger = callback_context.triggered_id
    if not trigger:
        return selected

    maturity = str(trigger).replace("maturity-", "")
    if maturity in selected:
        selected.remove(maturity)
    else:
        selected.append(maturity)

    ordered = [m for m in MATURITY_PRESETS if m in selected]
    return ordered


@app.callback(
    [Output(f"maturity-{m}", "className") for m in MATURITY_PRESETS]
    + [Output(f"maturity-{m}", "style") for m in MATURITY_PRESETS],
    Input("selected-maturities", "data"),
)
def update_maturity_button_styles(selected):
    selected_set = set(selected or [])
    class_names = [
        "maturity-btn maturity-btn-active" if maturity in selected_set else "maturity-btn"
        for maturity in MATURITY_PRESETS
    ]
    styles = []
    for maturity in MATURITY_PRESETS:
        if maturity in selected_set:
            color = YIELD_COLORS.get(maturity, "#2E73B8")
            styles.append(
                {
                    "background": "transparent",
                    "backgroundColor": "transparent",
                    "backgroundImage": "none",
                    "borderColor": color,
                    "borderWidth": "2px",
                    "color": color,
                }
            )
        else:
            styles.append({})
    return class_names + styles


@app.callback(
    Output("show-spread-state", "data"),
    Output("spread-btn", "className"),
    Output("spread-btn", "style"),
    Input("spread-btn", "n_clicks"),
    State("show-spread-state", "data"),
    prevent_initial_call=True,
)
def toggle_spread_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": "#0B3A63",
            "borderWidth": "2px",
            "color": "#0B3A63",
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-us-ig-corp-state", "data"),
    Output("ig-corp-spread-btn", "className"),
    Output("ig-corp-spread-btn", "style"),
    Input("ig-corp-spread-btn", "n_clicks"),
    State("show-us-ig-corp-state", "data"),
    prevent_initial_call=True,
)
def toggle_us_ig_corp_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["us_ig_corp"],
            "borderWidth": "2px",
            "color": BOND_LINE_COLORS["us_ig_corp"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-aaa-corp-state", "data"),
    Output("aaa-corp-yield-btn", "className"),
    Output("aaa-corp-yield-btn", "style"),
    Input("aaa-corp-yield-btn", "n_clicks"),
    State("show-aaa-corp-state", "data"),
    prevent_initial_call=True,
)
def toggle_aaa_corp_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["aaa_corp"],
            "borderWidth": "2px",
            "color": BOND_LINE_COLORS["aaa_corp"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-us-hy-corp-state", "data"),
    Output("ig-muni-spread-btn", "className"),
    Output("ig-muni-spread-btn", "style"),
    Input("ig-muni-spread-btn", "n_clicks"),
    State("show-us-hy-corp-state", "data"),
    prevent_initial_call=True,
)
def toggle_us_hy_corp_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["us_hy_corp"],
            "borderWidth": "2px",
            "color": BOND_LINE_COLORS["us_hy_corp"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-ig-muni-state", "data"),
    Output("ig-muni-yield-btn", "className"),
    Output("ig-muni-yield-btn", "style"),
    Input("ig-muni-yield-btn", "n_clicks"),
    State("show-ig-muni-state", "data"),
    prevent_initial_call=True,
)
def toggle_ig_muni_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["ig_muni"],
            "borderWidth": "2px",
            "color": BOND_LINE_COLORS["ig_muni"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-hy-muni-state", "data"),
    Output("hy-muni-yield-btn", "className"),
    Output("hy-muni-yield-btn", "style"),
    Input("hy-muni-yield-btn", "n_clicks"),
    State("show-hy-muni-state", "data"),
    prevent_initial_call=True,
)
def toggle_hy_muni_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["hy_muni"],
            "borderWidth": "2px",
            "color": BOND_LINE_COLORS["hy_muni"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-aaa-clo-state", "data"),
    Output("aaa-clo-yield-btn", "className"),
    Output("aaa-clo-yield-btn", "style"),
    Input("aaa-clo-yield-btn", "n_clicks"),
    State("show-aaa-clo-state", "data"),
    prevent_initial_call=True,
)
def toggle_aaa_clo_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["aaa_clo"],
            "borderWidth": "2px",
            "color": BOND_LINE_COLORS["aaa_clo"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-senior-loans-state", "data"),
    Output("senior-loans-yield-btn", "className"),
    Output("senior-loans-yield-btn", "style"),
    Input("senior-loans-yield-btn", "n_clicks"),
    State("show-senior-loans-state", "data"),
    prevent_initial_call=True,
)
def toggle_senior_loans_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["senior_loans"],
            "borderWidth": "2px",
            "color": BOND_LINE_COLORS["senior_loans"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-agency-mbs-state", "data"),
    Output("agency-mbs-yield-btn", "className"),
    Output("agency-mbs-yield-btn", "style"),
    Input("agency-mbs-yield-btn", "n_clicks"),
    State("show-agency-mbs-state", "data"),
    prevent_initial_call=True,
)
def toggle_agency_mbs_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["agency_mbs"],
            "borderWidth": "2px",
            "color": BOND_LINE_COLORS["agency_mbs"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-cs-us-ig-corp-state", "data"),
    Output("cs-ig-corp-spread-btn", "className"),
    Output("cs-ig-corp-spread-btn", "style"),
    Input("cs-ig-corp-spread-btn", "n_clicks"),
    State("show-cs-us-ig-corp-state", "data"),
    prevent_initial_call=True,
)
def toggle_cs_us_ig_corp_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["us_ig_corp"],
            "borderWidth": "2px",
            "color": CREDIT_SPREAD_BUTTON_COLORS["us_ig_corp"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-cs-aaa-corp-state", "data"),
    Output("cs-aaa-corp-yield-btn", "className"),
    Output("cs-aaa-corp-yield-btn", "style"),
    Input("cs-aaa-corp-yield-btn", "n_clicks"),
    State("show-cs-aaa-corp-state", "data"),
    prevent_initial_call=True,
)
def toggle_cs_aaa_corp_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["aaa_corp"],
            "borderWidth": "2px",
            "color": CREDIT_SPREAD_BUTTON_COLORS["aaa_corp"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-cs-us-hy-corp-state", "data"),
    Output("cs-ig-muni-spread-btn", "className"),
    Output("cs-ig-muni-spread-btn", "style"),
    Input("cs-ig-muni-spread-btn", "n_clicks"),
    State("show-cs-us-hy-corp-state", "data"),
    prevent_initial_call=True,
)
def toggle_cs_us_hy_corp_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["us_hy_corp"],
            "borderWidth": "2px",
            "color": CREDIT_SPREAD_BUTTON_COLORS["us_hy_corp"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-cs-ig-muni-state", "data"),
    Output("cs-ig-muni-yield-btn", "className"),
    Output("cs-ig-muni-yield-btn", "style"),
    Input("cs-ig-muni-yield-btn", "n_clicks"),
    State("show-cs-ig-muni-state", "data"),
    prevent_initial_call=True,
)
def toggle_cs_ig_muni_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["ig_muni"],
            "borderWidth": "2px",
            "color": CREDIT_SPREAD_BUTTON_COLORS["ig_muni"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-cs-hy-muni-state", "data"),
    Output("cs-hy-muni-yield-btn", "className"),
    Output("cs-hy-muni-yield-btn", "style"),
    Input("cs-hy-muni-yield-btn", "n_clicks"),
    State("show-cs-hy-muni-state", "data"),
    prevent_initial_call=True,
)
def toggle_cs_hy_muni_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["hy_muni"],
            "borderWidth": "2px",
            "color": CREDIT_SPREAD_BUTTON_COLORS["hy_muni"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-cs-aaa-clo-state", "data"),
    Output("cs-aaa-clo-yield-btn", "className"),
    Output("cs-aaa-clo-yield-btn", "style"),
    Input("cs-aaa-clo-yield-btn", "n_clicks"),
    State("show-cs-aaa-clo-state", "data"),
    prevent_initial_call=True,
)
def toggle_cs_aaa_clo_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["aaa_clo"],
            "borderWidth": "2px",
            "color": CREDIT_SPREAD_BUTTON_COLORS["aaa_clo"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-cs-senior-loans-state", "data"),
    Output("cs-senior-loans-yield-btn", "className"),
    Output("cs-senior-loans-yield-btn", "style"),
    Input("cs-senior-loans-yield-btn", "n_clicks"),
    State("show-cs-senior-loans-state", "data"),
    prevent_initial_call=True,
)
def toggle_cs_senior_loans_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["senior_loans"],
            "borderWidth": "2px",
            "color": CREDIT_SPREAD_BUTTON_COLORS["senior_loans"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-cs-agency-mbs-state", "data"),
    Output("cs-agency-mbs-yield-btn", "className"),
    Output("cs-agency-mbs-yield-btn", "style"),
    Input("cs-agency-mbs-yield-btn", "n_clicks"),
    State("show-cs-agency-mbs-state", "data"),
    prevent_initial_call=True,
)
def toggle_cs_agency_mbs_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "transparent",
            "backgroundColor": "transparent",
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["agency_mbs"],
            "borderWidth": "2px",
            "color": CREDIT_SPREAD_BUTTON_COLORS["agency_mbs"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-vol-vix-state", "data"),
    Output("show-vol-vxn-state", "data"),
    Output("show-vol-rvx-state", "data"),
    Output("show-vol-vxeem-state", "data"),
    Output("show-vol-skew-state", "data"),
    Output("show-vol-gvz-state", "data"),
    Output("show-vol-ovx-state", "data"),
    Output("show-vol-stlfsi-state", "data"),
    Output("show-vol-hy_oas-state", "data"),
    Output("show-vol-move-state", "data"),
    Output("show-vol-dxy-state", "data"),
    Output("show-vol-cnn_fear_greed-state", "data"),
    Output("vol-vix-btn", "className"),
    Output("vol-vxn-btn", "className"),
    Output("vol-rvx-btn", "className"),
    Output("vol-vxeem-btn", "className"),
    Output("vol-skew-btn", "className"),
    Output("vol-gvz-btn", "className"),
    Output("vol-ovx-btn", "className"),
    Output("vol-stlfsi-btn", "className"),
    Output("vol-hy_oas-btn", "className"),
    Output("vol-move-btn", "className"),
    Output("vol-dxy-btn", "className"),
    Output("vol-cnn_fear_greed-btn", "className"),
    Output("vol-vix-btn", "style"),
    Output("vol-vxn-btn", "style"),
    Output("vol-rvx-btn", "style"),
    Output("vol-vxeem-btn", "style"),
    Output("vol-skew-btn", "style"),
    Output("vol-gvz-btn", "style"),
    Output("vol-ovx-btn", "style"),
    Output("vol-stlfsi-btn", "style"),
    Output("vol-hy_oas-btn", "style"),
    Output("vol-move-btn", "style"),
    Output("vol-dxy-btn", "style"),
    Output("vol-cnn_fear_greed-btn", "style"),
    Input("vol-vix-btn", "n_clicks"),
    Input("vol-vxn-btn", "n_clicks"),
    Input("vol-rvx-btn", "n_clicks"),
    Input("vol-vxeem-btn", "n_clicks"),
    Input("vol-skew-btn", "n_clicks"),
    Input("vol-gvz-btn", "n_clicks"),
    Input("vol-ovx-btn", "n_clicks"),
    Input("vol-stlfsi-btn", "n_clicks"),
    Input("vol-hy_oas-btn", "n_clicks"),
    Input("vol-move-btn", "n_clicks"),
    Input("vol-dxy-btn", "n_clicks"),
    Input("vol-cnn_fear_greed-btn", "n_clicks"),
    State("show-vol-vix-state", "data"),
    State("show-vol-vxn-state", "data"),
    State("show-vol-rvx-state", "data"),
    State("show-vol-vxeem-state", "data"),
    State("show-vol-skew-state", "data"),
    State("show-vol-gvz-state", "data"),
    State("show-vol-ovx-state", "data"),
    State("show-vol-stlfsi-state", "data"),
    State("show-vol-hy_oas-state", "data"),
    State("show-vol-move-state", "data"),
    State("show-vol-dxy-state", "data"),
    State("show-vol-cnn_fear_greed-state", "data"),
    prevent_initial_call=True,
)
def toggle_volatility_buttons(
    _vix_clicks: int,
    _vxn_clicks: int,
    _rvx_clicks: int,
    _vxeem_clicks: int,
    _skew_clicks: int,
    _gvz_clicks: int,
    _ovx_clicks: int,
    _stlfsi_clicks: int,
    _hy_oas_clicks: int,
    _move_clicks: int,
    _dxy_clicks: int,
    _cnn_fear_greed_clicks: int,
    show_vol_vix_state: bool,
    show_vol_vxn_state: bool,
    show_vol_rvx_state: bool,
    show_vol_vxeem_state: bool,
    show_vol_skew_state: bool,
    show_vol_gvz_state: bool,
    show_vol_ovx_state: bool,
    show_vol_stlfsi_state: bool,
    show_vol_hy_oas_state: bool,
    show_vol_move_state: bool,
    show_vol_dxy_state: bool,
    show_vol_cnn_fear_greed_state: bool,
):
    key_order = ["vix", "vxn", "rvx", "vxeem", "skew", "gvz", "ovx", "stlfsi", "hy_oas", "move", "dxy", "cnn_fear_greed"]
    id_to_key = {f"vol-{k}-btn": k for k in key_order}
    current = {
        "vix": bool(show_vol_vix_state),
        "vxn": bool(show_vol_vxn_state),
        "rvx": bool(show_vol_rvx_state),
        "vxeem": bool(show_vol_vxeem_state),
        "skew": bool(show_vol_skew_state),
        "gvz": bool(show_vol_gvz_state),
        "ovx": bool(show_vol_ovx_state),
        "stlfsi": bool(show_vol_stlfsi_state),
        "hy_oas": bool(show_vol_hy_oas_state),
        "move": bool(show_vol_move_state),
        "dxy": bool(show_vol_dxy_state),
        "cnn_fear_greed": bool(show_vol_cnn_fear_greed_state),
    }

    trigger = callback_context.triggered_id
    clicked_key = id_to_key.get(str(trigger), "")
    if not clicked_key:
        return no_update

    # Single-select behavior: selecting a new button deselects any previously selected one.
    # Clicking the currently selected button toggles the selection off.
    if current.get(clicked_key, False) and sum(current.values()) == 1:
        active_key = ""
    else:
        active_key = clicked_key

    new_states = {k: (k == active_key) for k in key_order}

    classes = []
    styles = []
    for key in key_order:
        is_active = new_states[key]
        classes.append("maturity-btn maturity-btn-active" if is_active else "maturity-btn")
        if is_active:
            color = VOLATILITY_BUTTON_COLORS[key]
            styles.append(
                {
                    "background": "transparent",
                    "backgroundColor": "transparent",
                    "backgroundImage": "none",
                    "borderColor": color,
                    "borderWidth": "2px",
                    "color": color,
                }
            )
        else:
            styles.append({})

    state_values = [new_states[k] for k in key_order]
    return tuple(state_values + classes + styles)


@app.callback(
    Output("refresh-progress-msg", "children"),
    Output("refresh-progress-bar", "value"),
    Output("refresh-progress-wrap", "style"),
    Output("refresh-progress-interval", "disabled"),
    Output("refresh-token", "data"),
    Output("dataset-range-label", "children"),
    Input("refresh-btn", "n_clicks"),
    Input("refresh-progress-interval", "n_intervals"),
    State("refresh-token", "data"),
    prevent_initial_call=True,
)
def handle_refresh(n_clicks: int, _n_intervals: int, token: int):
    trigger = callback_context.triggered_id

    if trigger == "refresh-btn":
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update, no_update
        started = start_refresh_worker()
        if not started:
            return "Refresh already running...", no_update, {"display": "block"}, False, no_update, no_update
        return "Starting refresh...", "0", {"display": "block"}, False, no_update, no_update

    with refresh_lock:
        state = dict(refresh_state)

    if state["running"]:
        return state["message"], str(state["progress"]), {"display": "block"}, False, no_update, no_update

    if state["done"]:
        with refresh_lock:
            refresh_state["done"] = False
        refreshed_min, refreshed_max = get_date_bounds()
        range_text = f"Downloaded range: {refreshed_min.strftime('%b-%Y')} to {refreshed_max.strftime('%b-%Y')}"
        return state["message"], str(state["progress"]), {"display": "none"}, True, (token or 0) + 1, range_text

    return no_update, no_update, {"display": "none"}, True, no_update, no_update


@app.callback(
    Output("indicator-graph", "figure"),
    Output("latest-indicators", "children"),
    Output("warning-message", "children"),
    Input("selected-maturities", "data"),
    Input("show-spread-state", "data"),
    Input("show-us-ig-corp-state", "data"),
    Input("show-aaa-corp-state", "data"),
    Input("show-us-hy-corp-state", "data"),
    Input("show-ig-muni-state", "data"),
    Input("show-hy-muni-state", "data"),
    Input("show-aaa-clo-state", "data"),
    Input("show-senior-loans-state", "data"),
    Input("show-agency-mbs-state", "data"),
    Input("show-cs-us-ig-corp-state", "data"),
    Input("show-cs-aaa-corp-state", "data"),
    Input("show-cs-us-hy-corp-state", "data"),
    Input("show-cs-ig-muni-state", "data"),
    Input("show-cs-hy-muni-state", "data"),
    Input("show-cs-aaa-clo-state", "data"),
    Input("show-cs-senior-loans-state", "data"),
    Input("show-cs-agency-mbs-state", "data"),
    Input("show-vol-vix-state", "data"),
    Input("show-vol-vxn-state", "data"),
    Input("show-vol-rvx-state", "data"),
    Input("show-vol-vxeem-state", "data"),
    Input("show-vol-skew-state", "data"),
    Input("show-vol-gvz-state", "data"),
    Input("show-vol-ovx-state", "data"),
    Input("show-vol-stlfsi-state", "data"),
    Input("show-vol-hy_oas-state", "data"),
    Input("show-vol-ig_oas-state", "data"),
    Input("show-vol-move-state", "data"),
    Input("show-vol-dxy-state", "data"),
    Input("show-vol-cnn_fear_greed-state", "data"),
    Input("vol-band-select", "value"),
    Input("vol-median-select", "value"),
    Input("cs-baseline-tenor", "value"),
    Input("show-fed-rate", "value"),
    Input("show-inflation", "value"),
    Input("show-unemployment", "value"),
    Input("show-u6-unemployment", "value"),
    Input("show-unemp-ind", "value"),
    Input("show-crisis-dotcom", "value"),
    Input("show-crisis-gfc", "value"),
    Input("show-crisis-eu-debt", "value"),
    Input("show-crisis-covid", "value"),
    Input("show-crisis-us-banking", "value"),
    Input("start-date", "date"),
    Input("end-date", "date"),
    Input("refresh-token", "data"),
)
def update_visuals(
    selected_maturities,
    show_spread_state,
    show_us_ig_corp_state,
    show_aaa_corp_state,
    show_us_hy_corp_state,
    show_ig_muni_state,
    show_hy_muni_state,
    show_aaa_clo_state,
    show_senior_loans_state,
    show_agency_mbs_state,
    show_cs_us_ig_corp_state,
    show_cs_aaa_corp_state,
    show_cs_us_hy_corp_state,
    show_cs_ig_muni_state,
    show_cs_hy_muni_state,
    show_cs_aaa_clo_state,
    show_cs_senior_loans_state,
    show_cs_agency_mbs_state,
    show_vol_vix_state,
    show_vol_vxn_state,
    show_vol_rvx_state,
    show_vol_vxeem_state,
    show_vol_skew_state,
    show_vol_gvz_state,
    show_vol_ovx_state,
    show_vol_stlfsi_state,
    show_vol_hy_oas_state,
    show_vol_ig_oas_state,
    show_vol_move_state,
    show_vol_dxy_state,
    show_vol_cnn_fear_greed_state,
    vol_band_mode,
    vol_median_mode,
    cs_baseline_tenor,
    show_fed_rate_val,
    show_inflation_val,
    show_unemployment_val,
    show_u6_unemployment_val,
    show_unemp_ind_val,
    show_crisis_dotcom_val,
    show_crisis_gfc_val,
    show_crisis_eu_debt_val,
    show_crisis_covid_val,
    show_crisis_us_banking_val,
    start_date_str,
    end_date_str,
    _refresh_token,
):
    show_yields = True
    show_spread = bool(show_spread_state)
    show_us_ig_corp = bool(show_us_ig_corp_state)
    show_aaa_corp = bool(show_aaa_corp_state)
    show_us_hy_corp = bool(show_us_hy_corp_state)
    show_ig_muni = bool(show_ig_muni_state)
    show_hy_muni = bool(show_hy_muni_state)
    show_aaa_clo = bool(show_aaa_clo_state)
    show_senior_loans = bool(show_senior_loans_state)
    show_agency_mbs = bool(show_agency_mbs_state)
    show_cs_us_ig_corp = bool(show_cs_us_ig_corp_state)
    show_cs_aaa_corp = bool(show_cs_aaa_corp_state)
    show_cs_us_hy_corp = bool(show_cs_us_hy_corp_state)
    show_cs_ig_muni = bool(show_cs_ig_muni_state)
    show_cs_hy_muni = bool(show_cs_hy_muni_state)
    show_cs_aaa_clo = bool(show_cs_aaa_clo_state)
    show_cs_senior_loans = bool(show_cs_senior_loans_state)
    show_cs_agency_mbs = bool(show_cs_agency_mbs_state)
    show_vol_vix = bool(show_vol_vix_state)
    show_vol_vxn = bool(show_vol_vxn_state)
    show_vol_rvx = bool(show_vol_rvx_state)
    show_vol_vxeem = bool(show_vol_vxeem_state)
    show_vol_skew = bool(show_vol_skew_state)
    show_vol_gvz = bool(show_vol_gvz_state)
    show_vol_ovx = bool(show_vol_ovx_state)
    show_vol_stlfsi = bool(show_vol_stlfsi_state)
    show_vol_hy_oas = bool(show_vol_hy_oas_state)
    show_vol_ig_oas = False
    show_vol_move = bool(show_vol_move_state)
    show_vol_dxy = bool(show_vol_dxy_state)
    show_vol_cnn_fear_greed = bool(show_vol_cnn_fear_greed_state)
    show_fed_rate = "on" in (show_fed_rate_val or [])
    show_inflation = "on" in (show_inflation_val or [])
    show_unemployment = "on" in (show_unemployment_val or [])
    show_u6_unemployment = "on" in (show_u6_unemployment_val or [])
    show_unemp_ind = "on" in (show_unemp_ind_val or [])
    show_crisis_dotcom = "on" in (show_crisis_dotcom_val or [])
    show_crisis_gfc = "on" in (show_crisis_gfc_val or [])
    show_crisis_eu_debt = "on" in (show_crisis_eu_debt_val or [])
    show_crisis_covid = "on" in (show_crisis_covid_val or [])
    show_crisis_us_banking = "on" in (show_crisis_us_banking_val or [])

    selected_maturities = selected_maturities or []

    ust_df = load_and_process_csv("data/ust.csv")
    if ust_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Treasury data found. Ensure data/ust.csv exists.")
        with chart_dataset_lock:
            chart_dataset_cache["csv"] = "date\n"
        return fig, [], "No Treasury data found."

    for col in SERIES.values():
        if col in ust_df.columns:
            ust_df[col] = pd.to_numeric(ust_df[col], errors="coerce")

    treasury_dates = ust_df["DATE"]

    bond_yields_df = load_and_process_csv("data/bondyields.csv")
    if not bond_yields_df.empty:
        for col in CREDIT_YIELD_COLS.values():
            if col in bond_yields_df.columns:
                bond_yields_df[col] = pd.to_numeric(bond_yields_df[col], errors="coerce")
    else:
        bond_yields_df = pd.DataFrame()

    fed_df = load_and_process_csv("data/fedrate.csv")
    if not fed_df.empty:
        fed_df["FED_RATE"] = pd.to_numeric(fed_df["FED_RATE"], errors="coerce")
        fed_monthly = align_to_month_end(fed_df[["DATE", "FED_RATE"]])
        fed_monthly = align_to_treasury_daily_calendar(fed_monthly, ["FED_RATE"], treasury_dates)
    else:
        fed_monthly = pd.DataFrame()

    infl_df = load_and_process_csv("data/inflation.csv")
    if not infl_df.empty:
        infl_df["PCE_YoY"] = pd.to_numeric(infl_df["PCE_YoY"], errors="coerce")
        infl_monthly = align_to_month_end(infl_df[["DATE", "PCE_YoY"]])
        infl_monthly = align_to_treasury_daily_calendar(infl_monthly, ["PCE_YoY"], treasury_dates)
    else:
        infl_monthly = pd.DataFrame()

    unrate_df = load_and_process_csv("data/unemployment.csv")
    if not unrate_df.empty:
        if "U6RATE" not in unrate_df.columns:
            unrate_df["U6RATE"] = pd.NA
        for col in ["UNRATE", "U6RATE", "NROU"]:
            unrate_df[col] = pd.to_numeric(unrate_df[col], errors="coerce")
        unrate_df["UNEMP_INDICATOR"] = unrate_df["UNRATE"] - unrate_df["NROU"]
        unrate_monthly = align_to_month_end(unrate_df[["DATE", "UNRATE", "U6RATE", "UNEMP_INDICATOR"]])
        unrate_monthly = align_to_treasury_daily_calendar(
            unrate_monthly,
            ["UNRATE", "U6RATE", "UNEMP_INDICATOR"],
            treasury_dates,
        )
    else:
        unrate_monthly = pd.DataFrame()
    vol_df = load_and_process_csv("data/volatility.csv")
    if not vol_df.empty:
        for z_col in VOLATILITY_DATA_COLS.values():
            if z_col in vol_df.columns:
                vol_df[z_col] = pd.to_numeric(vol_df[z_col], errors="coerce")
        vol_cols = [c for c in VOLATILITY_DATA_COLS.values() if c in vol_df.columns]
        vol_resampled = (
            vol_df[["DATE"] + vol_cols]
            .set_index("DATE")
            .resample("W")
            .mean()
            .dropna(how="all")
            .reset_index()
        )
    else:
        vol_df = pd.DataFrame()
        vol_resampled = pd.DataFrame()

    min_date = max(ust_df["DATE"].min().date(), pd.to_datetime("2000-01-01").date())
    max_date = max(ust_df["DATE"].max().date(), pd.Timestamp.today().normalize().date())

    start_date = pd.to_datetime(start_date_str).date() if start_date_str else max(min_date, pd.to_datetime("2020-01-01").date())
    end_date = pd.to_datetime(end_date_str).date() if end_date_str else max_date

    start_date = max(min_date, min(max_date, start_date))
    end_date = max(min_date, min(max_date, end_date))

    warning = ""
    if start_date > end_date:
        warning = "Start date must be before end date. Resetting to full range."
        start_date, end_date = min_date, max_date

    delta = end_date - start_date
    if delta.days < 183:
        freq = None
        freq_label = "Daily"
    elif delta.days <= 731:
        freq = "W"
        freq_label = "Weekly"
    else:
        freq = "ME"
        freq_label = "Monthly"

    if freq:
        ust_resampled = (
            ust_df[["DATE"] + list(SERIES.values())]
            .set_index("DATE")
            .resample(freq)
            .mean()
            .dropna(how="all")
            .reset_index()
        )
        if not bond_yields_df.empty:
            bond_cols = [c for c in CREDIT_YIELD_COLS.values() if c in bond_yields_df.columns]
            bond_resampled = (
                bond_yields_df[["DATE"] + bond_cols]
                .set_index("DATE")
                .resample(freq)
                .mean()
                .dropna(how="all")
                .reset_index()
            )
        else:
            bond_resampled = pd.DataFrame()
    else:
        ust_resampled = ust_df
        bond_resampled = bond_yields_df

    plot_ust = filter_by_date(ust_resampled, start_date, end_date)
    plot_ust["SPREAD_10Y_2Y"] = plot_ust["BC_10YEAR"] - plot_ust["BC_2YEAR"]
    plot_bond_yields = filter_by_date(bond_resampled, start_date, end_date) if not bond_resampled.empty else pd.DataFrame()
    plot_vol = filter_by_date(vol_resampled, start_date, end_date) if not vol_resampled.empty else pd.DataFrame()

    plot_fed = filter_by_date(fed_monthly, start_date, end_date)
    if not plot_fed.empty:
        plot_fed["rate_changed"] = plot_fed["FED_RATE"].diff().abs() > 0.001

    plot_infl = filter_by_date(infl_monthly, start_date, end_date)

    if (show_unemployment or show_u6_unemployment or show_unemp_ind) and not unrate_monthly.empty:
        plot_unrate = filter_by_date(unrate_monthly, start_date, end_date)
    else:
        plot_unrate = pd.DataFrame()

    fig = build_figure(
        plot_ust=plot_ust,
        plot_bond_yields=plot_bond_yields,
        plot_fed=plot_fed,
        plot_infl=plot_infl,
        plot_unrate=plot_unrate,
        plot_vol=plot_vol,
        selected_maturities=selected_maturities,
        show_yields=show_yields,
        show_spread=show_spread,
        show_us_ig_corp=show_us_ig_corp,
        show_aaa_corp=show_aaa_corp,
        show_us_hy_corp=show_us_hy_corp,
        show_ig_muni=show_ig_muni,
        show_hy_muni=show_hy_muni,
        show_aaa_clo=show_aaa_clo,
        show_senior_loans=show_senior_loans,
        show_agency_mbs=show_agency_mbs,
        show_cs_us_ig_corp=show_cs_us_ig_corp,
        show_cs_aaa_corp=show_cs_aaa_corp,
        show_cs_us_hy_corp=show_cs_us_hy_corp,
        show_cs_ig_muni=show_cs_ig_muni,
        show_cs_hy_muni=show_cs_hy_muni,
        show_cs_aaa_clo=show_cs_aaa_clo,
        show_cs_senior_loans=show_cs_senior_loans,
        show_cs_agency_mbs=show_cs_agency_mbs,
        cs_baseline_tenor=cs_baseline_tenor,
        show_fed_rate=show_fed_rate,
        show_inflation=show_inflation,
        show_unemployment=show_unemployment,
        show_u6_unemployment=show_u6_unemployment,
        show_unemp_ind=show_unemp_ind,
        show_vol_vix=show_vol_vix,
        show_vol_vxn=show_vol_vxn,
        show_vol_rvx=show_vol_rvx,
        show_vol_vxeem=show_vol_vxeem,
        show_vol_skew=show_vol_skew,
        show_vol_gvz=show_vol_gvz,
        show_vol_ovx=show_vol_ovx,
        show_vol_stlfsi=show_vol_stlfsi,
        show_vol_hy_oas=show_vol_hy_oas,
        show_vol_ig_oas=show_vol_ig_oas,
        show_vol_move=show_vol_move,
        show_vol_dxy=show_vol_dxy,
        show_vol_cnn_fear_greed=show_vol_cnn_fear_greed,
        vol_band_mode=vol_band_mode,
        vol_median_mode=vol_median_mode,
    )
    fig.update_xaxes(range=[start_date.isoformat(), end_date.isoformat()])
    if show_crisis_dotcom:
        fig.add_vrect(
            x0="2000-03-01",
            x1="2002-03-31",
            fillcolor="rgba(239, 68, 68, 0.14)",
            opacity=0.28,
            line_color="rgba(185, 28, 28, 0.75)",
            line_width=1,
            layer="above",
        )
    if show_crisis_gfc:
        fig.add_vrect(
            x0="2007-08-01",
            x1="2009-08-31",
            fillcolor="rgba(239, 68, 68, 0.14)",
            opacity=0.28,
            line_color="rgba(185, 28, 28, 0.75)",
            line_width=1,
            layer="above",
        )
    if show_crisis_eu_debt:
        fig.add_vrect(
            x0="2010-01-01",
            x1="2013-12-31",
            fillcolor="rgba(239, 68, 68, 0.14)",
            opacity=0.28,
            line_color="rgba(185, 28, 28, 0.75)",
            line_width=1,
            layer="above",
        )
    if show_crisis_covid:
        fig.add_vrect(
            x0="2020-01-01",
            x1="2020-05-31",
            fillcolor="rgba(239, 68, 68, 0.14)",
            opacity=0.28,
            line_color="rgba(185, 28, 28, 0.75)",
            line_width=1,
            layer="above",
        )
    if show_crisis_us_banking:
        fig.add_vrect(
            x0="2023-03-01",
            x1="2023-07-31",
            fillcolor="rgba(239, 68, 68, 0.14)",
            opacity=0.28,
            line_color="rgba(185, 28, 28, 0.75)",
            line_width=1,
            layer="above",
        )

    export_series: list[tuple[str, pd.Series]] = []

    if show_yields and selected_maturities:
        for maturity in selected_maturities:
            ust_col = SERIES.get(maturity)
            if not ust_col or ust_col not in ust_df.columns:
                continue
            s_df = filter_by_date(ust_df[["DATE", ust_col]], start_date, end_date)
            s = pd.Series(pd.to_numeric(s_df[ust_col], errors="coerce").values, index=s_df["DATE"], name=f"Yield: {maturity}")
            export_series.append((f"Yield: {maturity}", s))

    if show_spread and {"BC_10YEAR", "BC_2YEAR"}.issubset(ust_df.columns):
        s_df = filter_by_date(ust_df[["DATE", "BC_10YEAR", "BC_2YEAR"]], start_date, end_date).copy()
        spread_vals = pd.to_numeric(s_df["BC_10YEAR"], errors="coerce") - pd.to_numeric(s_df["BC_2YEAR"], errors="coerce")
        export_series.append(("10Y-2Y Spread", pd.Series(spread_vals.values, index=s_df["DATE"])))

    bond_flags = [
        ("us_ig_corp", show_us_ig_corp, "IG CORP"),
        ("aaa_corp", show_aaa_corp, "AAA CORP"),
        ("us_hy_corp", show_us_hy_corp, "HY CORP"),
        ("ig_muni", show_ig_muni, "IG MUNI"),
        ("hy_muni", show_hy_muni, "HY MUNI"),
        ("aaa_clo", show_aaa_clo, "AAA_CLO"),
        ("senior_loans", show_senior_loans, "SENIOR LOANS"),
        ("agency_mbs", show_agency_mbs, "AGENCY MBS"),
    ]
    if not bond_yields_df.empty:
        for key, enabled, label in bond_flags:
            col = CREDIT_YIELD_COLS[key]
            if not enabled or col not in bond_yields_df.columns:
                continue
            s_df = filter_by_date(bond_yields_df[["DATE", col]], start_date, end_date)
            s = pd.Series(pd.to_numeric(s_df[col], errors="coerce").values, index=s_df["DATE"], name=label)
            export_series.append((label, s))

    baseline_tenor = cs_baseline_tenor if cs_baseline_tenor in SERIES else "10Y"
    baseline_col = SERIES[baseline_tenor]
    cs_flags = [
        ("us_ig_corp", show_cs_us_ig_corp, "IG CORP"),
        ("aaa_corp", show_cs_aaa_corp, "AAA CORP"),
        ("us_hy_corp", show_cs_us_hy_corp, "HY CORP"),
        ("ig_muni", show_cs_ig_muni, "IG MUNI"),
        ("hy_muni", show_cs_hy_muni, "HY MUNI"),
        ("aaa_clo", show_cs_aaa_clo, "AAA_CLO"),
        ("senior_loans", show_cs_senior_loans, "SENIOR LOANS"),
        ("agency_mbs", show_cs_agency_mbs, "AGENCY MBS"),
    ]
    if (
        not bond_yields_df.empty
        and baseline_col in ust_df.columns
    ):
        base_df = filter_by_date(ust_df[["DATE", baseline_col]], start_date, end_date).sort_values("DATE")
        for key, enabled, label in cs_flags:
            col = CREDIT_YIELD_COLS[key]
            if not enabled or col not in bond_yields_df.columns:
                continue
            y_df = filter_by_date(bond_yields_df[["DATE", col]], start_date, end_date).sort_values("DATE")
            if y_df.empty or base_df.empty:
                continue
            merged = pd.merge_asof(y_df, base_df, on="DATE", direction="backward")
            spread_vals = pd.to_numeric(merged[col], errors="coerce") - pd.to_numeric(merged[baseline_col], errors="coerce")
            export_series.append((f"{label}-{baseline_tenor}", pd.Series(spread_vals.values, index=merged["DATE"])))

    if show_fed_rate and not fed_df.empty and "FED_RATE" in fed_df.columns:
        s_df = filter_by_date(fed_df[["DATE", "FED_RATE"]], start_date, end_date)
        export_series.append(("Fed Rate", pd.Series(pd.to_numeric(s_df["FED_RATE"], errors="coerce").values, index=s_df["DATE"])))

    if show_inflation and not infl_df.empty and "PCE_YoY" in infl_df.columns:
        s_df = filter_by_date(infl_df[["DATE", "PCE_YoY"]], start_date, end_date)
        export_series.append(("Core PCE Inflation", pd.Series(pd.to_numeric(s_df["PCE_YoY"], errors="coerce").values, index=s_df["DATE"])))

    if show_unemployment and not unrate_df.empty and "UNRATE" in unrate_df.columns:
        s_df = filter_by_date(unrate_df[["DATE", "UNRATE"]], start_date, end_date)
        export_series.append(("Unemployment Rate", pd.Series(pd.to_numeric(s_df["UNRATE"], errors="coerce").values, index=s_df["DATE"])))

    if show_u6_unemployment and not unrate_df.empty and "U6RATE" in unrate_df.columns:
        s_df = filter_by_date(unrate_df[["DATE", "U6RATE"]], start_date, end_date)
        export_series.append(("U-6 Unemployment Rate", pd.Series(pd.to_numeric(s_df["U6RATE"], errors="coerce").values, index=s_df["DATE"])))

    if show_unemp_ind and not unrate_df.empty and {"UNRATE", "NROU"}.issubset(unrate_df.columns):
        s_df = filter_by_date(unrate_df[["DATE", "UNRATE", "NROU"]], start_date, end_date).copy()
        indicator_vals = pd.to_numeric(s_df["UNRATE"], errors="coerce") - pd.to_numeric(s_df["NROU"], errors="coerce")
        export_series.append(("Unemp. Indicator (U3-NROU)", pd.Series(indicator_vals.values, index=s_df["DATE"])))
    if not vol_df.empty:
        vol_flags = {
            "vix": show_vol_vix,
            "vxn": show_vol_vxn,
            "rvx": show_vol_rvx,
            "vxeem": show_vol_vxeem,
            "skew": show_vol_skew,
            "gvz": show_vol_gvz,
            "ovx": show_vol_ovx,
            "stlfsi": show_vol_stlfsi,
            "hy_oas": show_vol_hy_oas,
            "ig_oas": show_vol_ig_oas,
            "move": show_vol_move,
            "dxy": show_vol_dxy,
            "cnn_fear_greed": show_vol_cnn_fear_greed,
        }
        for key, enabled in vol_flags.items():
            z_col = VOLATILITY_DATA_COLS[key]
            if not enabled or z_col not in vol_df.columns:
                continue
            s_df = filter_by_date(vol_df[["DATE", z_col]], start_date, end_date)
            s = pd.Series(pd.to_numeric(s_df[z_col], errors="coerce").values, index=s_df["DATE"], name=key)
            export_series.append((key, s))

    latest_cards = [
        indicator_card("2Y T-bill yield", latest_non_null_value(ust_df, "BC_2YEAR")),
        indicator_card("10Y T-bill yield", latest_non_null_value(ust_df, "BC_10YEAR")),
        indicator_card("Core Inflation", latest_non_null_value(infl_monthly, "PCE_YoY")),
        indicator_card("Fed Rate", latest_non_null_value(fed_monthly, "FED_RATE")),
        indicator_card("U-3 Unemployment rate", latest_non_null_value(unrate_monthly, "UNRATE")),
        indicator_card("U3-NROU", latest_non_null_value(unrate_monthly, "UNEMP_INDICATOR")),
    ]

    with chart_dataset_lock:
        chart_dataset_cache["csv"] = series_items_to_csv_text(export_series)

    return fig, latest_cards, warning


if __name__ == "__main__":
    app.run(debug=True)
