import subprocess
import sys
import threading
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from flask import Response

BASE_DIR = Path(__file__).resolve().parent

SERIES = {
    "1Y": "BC_1YEAR",
    "2Y": "BC_2YEAR",
    "5Y": "BC_5YEAR",
    "7Y": "BC_7YEAR",
    "10Y": "BC_10YEAR",
    "20Y": "BC_20YEAR",
    "30Y": "BC_30YEAR",
}
SOFR_COL = "SOFR"
MATURITY_PRESETS = list(SERIES.keys())
CS_BASELINE_PRESETS = ["1Y", "2Y", "5Y", "10Y", "30Y"]

PRESETS = ["YTD", "1W", "1M", "3M", "6M", "1Y", "5Y", "10Y", "15Y", "20Y"]
YIELD_COLORS = {
    "SOFR": "#1B8F3A",
    "1Y": "#B7D7F5",
    "2Y": "#8EBFEA",
    "5Y": "#5B9EDB",
    "7Y": "#3F88CC",
    "10Y": "#2E73B8",
    "20Y": "#1C629E",
    "30Y": "#124A80",
}
CREDIT_YIELD_COLS = {
    "us_ig_corp": "IG_CORP:LQD",
    "aaa_corp": "AAA_CORP:QLTA",
    "us_hy_corp": "HY_CORP:HYG",
    "ig_muni": "IG_MUNIS:MUB",
    "hy_muni": "HY_MUNIS:HYD",
    "aaa_clo": "AAA_CLO:JAAA",
    "senior_loans": "SENIOR_LOANS:BKLN",
    "agency_mbs": "AGENCY_MBS:MBB",
    "em_sov_hard": "EM_SOV_HARD:EMB",
    "em_sov_local": "EM_SOV_LOCAL:ELD",
    "money_market": "MONEY_MARKET:SGOV",
    "tips_10y": "CMBS:CMBS",
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
    "em_sov_hard": "#0E5A36",
    "em_sov_local": "#166E44",
    "money_market": "#2A8A56",
    "tips_10y": "#3BA66B",
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
REFRESH_DATA_FILES = [
    "data/fedrate.csv",
    "data/inflation.csv",
    "data/ust.csv",
    "data/bondyields.csv",
    "data/volatility.csv",
    "data/unemployment.csv",
]

refresh_lock = threading.Lock()
refresh_state = {
    "running": False,
    "done": False,
    "ok": True,
    "message": "",
    "progress": 0,
}
AUTO_REFRESH_HOURS = {8, 14}
AUTO_REFRESH_WINDOW_MINUTES = 10
auto_refresh_lock = threading.Lock()
auto_refresh_last_slot = ""
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


def maybe_start_scheduled_refresh() -> bool:
    global auto_refresh_last_slot

    now = datetime.now()
    if now.hour not in AUTO_REFRESH_HOURS:
        return False
    if now.minute >= AUTO_REFRESH_WINDOW_MINUTES:
        return False

    slot = now.strftime("%Y-%m-%d %H")
    with auto_refresh_lock:
        if auto_refresh_last_slot == slot:
            return False
        started = start_refresh_worker()
        if started:
            auto_refresh_last_slot = slot
            return True
        with refresh_lock:
            if refresh_state["running"]:
                auto_refresh_last_slot = slot
        return started


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


def get_last_refresh_label() -> str:
    latest_mtime = None
    for rel_path in REFRESH_DATA_FILES:
        path = BASE_DIR / rel_path
        if not path.exists():
            continue
        mtime = path.stat().st_mtime
        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
    if latest_mtime is None:
        return "Last: --/-- --:--"
    dt = datetime.fromtimestamp(latest_mtime)
    return f"Last: {dt.strftime('%m/%d %H:%M')}"


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
    elif preset == "15Y":
        start = now - pd.DateOffset(years=15)
    elif preset == "20Y":
        start = now - pd.DateOffset(years=20)
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
    marks = {0: min_date.strftime("%Y"), total_days: "26"}
    for year in [2010, 2015, 2020]:
        year_start = pd.Timestamp(year=year, month=1, day=1)
        if year_start < min_date or year_start > max_date:
            continue
        marks[int((year_start - min_date).days)] = f"{year % 100:02d}"
    return dict(sorted(marks.items()))


def build_figure(
    plot_ust: pd.DataFrame,
    plot_ust_raw: pd.DataFrame,
    plot_bond_yields: pd.DataFrame,
    plot_fed: pd.DataFrame,
    plot_infl: pd.DataFrame,
    plot_unrate: pd.DataFrame,
    plot_vol: pd.DataFrame,
    selected_maturities: list[str],
    show_yields: bool,
    show_spread: bool,
    show_sofr: bool,
    show_yield_curve: bool,
    show_us_ig_corp: bool,
    show_aaa_corp: bool,
    show_us_hy_corp: bool,
    show_ig_muni: bool,
    show_hy_muni: bool,
    show_aaa_clo: bool,
    show_senior_loans: bool,
    show_agency_mbs: bool,
    show_em_sov_hard: bool,
    show_em_sov_local: bool,
    show_money_market: bool,
    show_tips_10y: bool,
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
    y1_vals: list[float] = []
    y2_vals: list[float] = []

    if show_yields and selected_maturities:
        for maturity in selected_maturities:
            y1_vals.extend(plot_ust[SERIES[maturity]].dropna().tolist())
    if show_spread and not plot_ust.empty:
        y1_vals.extend(plot_ust["SPREAD_10Y_2Y"].dropna().tolist())
    if show_sofr and not plot_fed.empty and SOFR_COL in plot_fed.columns:
        y1_vals.extend(pd.to_numeric(plot_fed[SOFR_COL], errors="coerce").dropna().tolist())
    if (
        show_us_ig_corp
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["us_ig_corp"] in plot_bond_yields.columns
    ):
        y1_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["us_ig_corp"]].dropna().tolist())
    if (
        show_aaa_corp
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["aaa_corp"] in plot_bond_yields.columns
    ):
        y1_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["aaa_corp"]].dropna().tolist())
    if (
        show_us_hy_corp
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["us_hy_corp"] in plot_bond_yields.columns
    ):
        y1_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["us_hy_corp"]].dropna().tolist())
    if (
        show_ig_muni
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["ig_muni"] in plot_bond_yields.columns
    ):
        y1_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["ig_muni"]].dropna().tolist())
    if (
        show_hy_muni
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["hy_muni"] in plot_bond_yields.columns
    ):
        y1_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["hy_muni"]].dropna().tolist())
    if (
        show_aaa_clo
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["aaa_clo"] in plot_bond_yields.columns
    ):
        y1_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["aaa_clo"]].dropna().tolist())
    if (
        show_senior_loans
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["senior_loans"] in plot_bond_yields.columns
    ):
        y1_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["senior_loans"]].dropna().tolist())
    if (
        show_agency_mbs
        and not plot_bond_yields.empty
        and CREDIT_YIELD_COLS["agency_mbs"] in plot_bond_yields.columns
    ):
        y1_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS["agency_mbs"]].dropna().tolist())
    for key, enabled in [
        ("em_sov_hard", show_em_sov_hard),
        ("em_sov_local", show_em_sov_local),
        ("money_market", show_money_market),
        ("tips_10y", show_tips_10y),
    ]:
        if (
            enabled
            and not plot_bond_yields.empty
            and CREDIT_YIELD_COLS[key] in plot_bond_yields.columns
        ):
            y1_vals.extend(plot_bond_yields[CREDIT_YIELD_COLS[key]].dropna().tolist())
    if show_inflation and not plot_infl.empty:
        y1_vals.extend(plot_infl["PCE_YoY"].dropna().tolist())
    if show_unemployment and not plot_unrate.empty:
        y1_vals.extend(plot_unrate["UNRATE"].dropna().tolist())
    if show_u6_unemployment and not plot_unrate.empty and "U6RATE" in plot_unrate.columns:
        y1_vals.extend(plot_unrate["U6RATE"].dropna().tolist())
    if show_unemp_ind and not plot_unrate.empty:
        y1_vals.extend(plot_unrate["UNEMP_INDICATOR"].dropna().tolist())
    if show_fed_rate and not plot_fed.empty:
        y1_vals.extend(plot_fed["FED_RATE"].dropna().tolist())
    if not plot_vol.empty:
        selected_band_mode = str(vol_band_mode or "").strip()
        if selected_band_mode not in {"25_75", "10_90"}:
            selected_band_mode = "none"
        selected_median_mode = str(vol_median_mode or "").strip()
        if selected_median_mode not in {"none", "1m_mean", "3m_mean", "6m_mean", "1y_mean", "3y_mean", "10y_mean", "15y_mean"}:
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
                y2_vals.extend(pd.to_numeric(plot_vol[z_col], errors="coerce").dropna().tolist())

    if y1_vals:
        v_min, v_max = min(y1_vals), max(y1_vals)
        pad = 0.05 * (v_max - v_min) if v_max != v_min else 1.0
        y1_range = [v_min - pad, v_max + pad]
    else:
        y1_range = [0, 10]

    if y2_vals:
        v2_min, v2_max = min(y2_vals), max(y2_vals)
        pad2 = 0.05 * (v2_max - v2_min) if v2_max != v2_min else 1.0
        y2_range = [v2_min - pad2, v2_max + pad2]
    else:
        y2_range = list(y1_range)

    fig = go.Figure()

    if show_yield_curve:
        curve_cols = [SERIES[m] for m in MATURITY_PRESETS if SERIES[m] in plot_ust_raw.columns]
        curve_labels = [m for m in MATURITY_PRESETS if SERIES[m] in plot_ust_raw.columns]
        curve_df = (
            plot_ust_raw[["DATE"] + curve_cols]
            .copy()
            .dropna(how="all", subset=curve_cols)
            .sort_values("DATE")
        )
        if curve_df.empty or not curve_cols:
            fig.update_layout(
                template="plotly_white",
                title="No Treasury data available for yield curve view.",
                height=450,
                margin={"t": 36, "b": 40, "l": 52, "r": 52},
            )
            return fig

        curve_df["DATE"] = pd.to_datetime(curve_df["DATE"], errors="coerce")
        curve_df = curve_df.dropna(subset=["DATE"]).set_index("DATE").sort_index()
        if curve_df.empty:
            fig.update_layout(
                template="plotly_white",
                title="No Treasury data available for yield curve view.",
                height=450,
                margin={"t": 36, "b": 40, "l": 52, "r": 52},
            )
            return fig

        latest_dt = curve_df.index.max()
        row = curve_df.loc[latest_dt]
        y_vals = [pd.to_numeric(row.get(col), errors="coerce") for col in curve_cols]
        curve_vals = [float(v) for v in y_vals if pd.notna(v)]
        point_labels = [f"{float(v):.2f}%" if pd.notna(v) else "" for v in y_vals]
        fig.add_trace(
            go.Scatter(
                x=curve_labels,
                y=y_vals,
                name=f"Current ({latest_dt.date().isoformat()})",
                mode="lines+markers+text",
                text=point_labels,
                textposition="top center",
                textfont={"size": 11, "color": "#1f2a37"},
                line={"color": "#1F5F93", "width": 2.6},
                marker={"size": 7},
                hovertemplate="%{fullData.name}<br>%{x}: %{y:.2f}%<extra></extra>",
            )
        )

        if curve_vals:
            cmin, cmax = min(curve_vals), max(curve_vals)
            cpad = 0.08 * (cmax - cmin) if cmax != cmin else 0.5
            curve_range = [cmin - cpad, cmax + cpad]
        else:
            curve_range = [0, 10]

        fig.update_layout(
            template="plotly_white",
            font={"family": "Plus Jakarta Sans, Segoe UI, Arial, sans-serif", "size": 13, "color": "#1f2a37"},
            xaxis_title="Maturity",
            xaxis={
                "type": "category",
                "showgrid": False,
                "linecolor": "rgba(15, 23, 42, 0.25)",
            },
            yaxis={
                "title": "Yield (%)",
                "side": "left",
                "range": curve_range,
                "tickformat": ".1f",
                "showgrid": True,
                "gridcolor": "rgba(15, 23, 42, 0.08)",
                "zeroline": True,
                "zerolinecolor": "rgba(15, 23, 42, 0.28)",
                "linecolor": "rgba(15, 23, 42, 0.25)",
            },
            yaxis2={"visible": False},
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
        return fig

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
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
    if show_sofr and not plot_fed.empty and SOFR_COL in plot_fed.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_fed["DATE"],
                y=plot_fed[SOFR_COL],
                name="SOFR 30D Avg",
                line={"color": "#1B8F3A", "width": 2},
                hovertemplate="%{fullData.name}<br>%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
            )
        )
    for key, enabled, label in [
        ("em_sov_hard", show_em_sov_hard, "EM SOV HARD BOND YIELD"),
        ("em_sov_local", show_em_sov_local, "EM SOV LOCAL BOND YIELD"),
        ("money_market", show_money_market, "MONEY MARKET BOND YIELD"),
        ("tips_10y", show_tips_10y, "CMBS BOND YIELD"),
    ]:
        col = CREDIT_YIELD_COLS[key]
        if enabled and not plot_bond_yields.empty and col in plot_bond_yields.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_bond_yields["DATE"],
                    y=plot_bond_yields[col],
                    name=label,
                    line={"color": BOND_LINE_COLORS[key], "width": 1.6},
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
                "1m_mean": ("30D", "1M"),
                "3m_mean": ("91D", "3M"),
                "6m_mean": ("183D", "6M"),
                "1y_mean": ("365D", "1Y"),
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
                    connectgaps=True,
                    hovertemplate=f"{display_label}<br>%{{x|%b %d, %Y}}<br>%{{y:.2f}}<extra></extra>",
                )
            )
            if selected_median_mode.endswith("_mean"):
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
                        connectgaps=True,
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
                        connectgaps=True,
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
                        connectgaps=True,
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
                        connectgaps=True,
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
                        connectgaps=True,
                        hovertemplate=f"{display_label} {window_label} P75<br>%{{x|%b %d, %Y}}<br>%{{y:.2f}}<extra></extra>",
                    )
                )
        if vix_plotted:
            fig.add_hrect(y0=0, y1=15, yref="y2", fillcolor="rgba(22, 163, 74, 0.12)", line_width=0, layer="below")
            fig.add_hrect(y0=25, y1=35, yref="y2", fillcolor="rgba(194, 65, 12, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=35, y1=50, yref="y2", fillcolor="rgba(185, 28, 28, 0.14)", line_width=0, layer="below")
            fig.add_hrect(y0=15, y1=25, yref="y2", fillcolor="rgba(234, 179, 8, 0.14)", line_width=0, layer="below")
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
            fig.add_hrect(y0=20, y1=30, yref="y2", fillcolor="rgba(234, 179, 8, 0.14)", line_width=0, layer="below")
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
            fig.add_hrect(y0=22, y1=32, yref="y2", fillcolor="rgba(234, 179, 8, 0.14)", line_width=0, layer="below")
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
            fig.add_hrect(y0=130, y1=145, yref="y2", fillcolor="rgba(234, 179, 8, 0.14)", line_width=0, layer="below")
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
            fig.add_hrect(y0=18, y1=28, yref="y2", fillcolor="rgba(234, 179, 8, 0.14)", line_width=0, layer="below")
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
            fig.add_hrect(y0=20, y1=30, yref="y2", fillcolor="rgba(234, 179, 8, 0.14)", line_width=0, layer="below")
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
            fig.add_hrect(y0=35, y1=50, yref="y2", fillcolor="rgba(234, 179, 8, 0.14)", line_width=0, layer="below")
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
            fig.add_hrect(y0=-0.5, y1=0.5, yref="y2", fillcolor="rgba(234, 179, 8, 0.14)", line_width=0, layer="below")
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
            fig.add_hrect(y0=3, y1=6, yref="y2", fillcolor="rgba(234, 179, 8, 0.14)", line_width=0, layer="below")
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
            fig.add_hrect(y0=80, y1=100, yref="y2", fillcolor="rgba(234, 179, 8, 0.14)", line_width=0, layer="below")
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
            fig.add_hrect(y0=90, y1=100, yref="y2", fillcolor="rgba(234, 179, 8, 0.14)", line_width=0, layer="below")
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
            fig.add_hrect(y0=45, y1=55, yref="y2", fillcolor="rgba(234, 179, 8, 0.14)", line_width=0, layer="below")
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
            y2_range = [min(y2_range[0], 0.0), max(y2_range[1], 100.0)]

    fig.update_layout(
        template="plotly_white",
        font={"family": "Plus Jakarta Sans, Segoe UI, Arial, sans-serif", "size": 13, "color": "#1f2a37"},
        xaxis={
            "showgrid": True,
            "gridcolor": "rgba(15, 23, 42, 0.08)",
            "linecolor": "rgba(15, 23, 42, 0.25)",
        },
        yaxis={
            "title": "Yield (%)",
            "side": "left",
            "range": y1_range,
            "tickformat": ".1f",
            "showgrid": True,
            "gridcolor": "rgba(15, 23, 42, 0.08)",
            "zeroline": True,
            "zerolinecolor": "rgba(15, 23, 42, 0.28)",
            "linecolor": "rgba(15, 23, 42, 0.25)",
        },
        yaxis2={
            "title": "",
            "side": "right",
            "overlaying": "y",
            "showgrid": False,
            "range": y2_range,
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


def _vix_dot_style(color: str) -> dict:
    return {
        "display": "inline-block",
        "width": "12px",
        "height": "12px",
        "borderRadius": "50%",
        "backgroundColor": color,
        "border": "1px solid rgba(15, 23, 42, 0.22)",
    }


def _vix_color_from_value(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "#D1D5DB"
    if v < 15:
        return "#16A34A"  # Low Vol / Complacency
    if v < 25:
        return "#EAB308"  # Normal / Watchful
    if v < 35:
        return "#C2410C"  # Elevated Stress
    if v < 50:
        return "#B91C1C"  # High Stress / Fear
    return "#7F1D1D"      # Extreme Stress / Panic


def _vxn_color_from_value(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "#D1D5DB"
    if v < 20:
        return "#16A34A"  # Low Vol / Complacency
    if v < 30:
        return "#EAB308"  # Normal / Watchful
    if v < 40:
        return "#C2410C"  # Elevated Stress
    if v < 55:
        return "#B91C1C"  # High Stress / Fear
    return "#7F1D1D"      # Extreme Stress / Panic


def _rvx_color_from_value(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "#D1D5DB"
    if v < 22:
        return "#16A34A"  # Low Vol / Complacency
    if v < 32:
        return "#EAB308"  # Normal / Watchful
    if v < 42:
        return "#C2410C"  # Elevated Stress
    if v < 58:
        return "#B91C1C"  # High Stress / Fear
    return "#7F1D1D"      # Extreme Stress / Panic


def _vxeem_color_from_value(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "#D1D5DB"
    if v < 18:
        return "#16A34A"  # Low Vol / Complacency
    if v < 28:
        return "#EAB308"  # Normal / Watchful
    if v < 40:
        return "#C2410C"  # Elevated Stress
    if v < 55:
        return "#B91C1C"  # High Stress / Fear
    return "#7F1D1D"      # Extreme Stress / Panic


def _skew_color_from_value(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "#D1D5DB"
    if v < 130:
        return "#16A34A"  # Low Tail-Risk Concern
    if v < 145:
        return "#EAB308"  # Watchful
    if v < 160:
        return "#C2410C"  # Elevated Tail-Risk Pricing
    if v < 175:
        return "#B91C1C"  # High Tail-Risk Concern
    return "#7F1D1D"      # Extreme Tail-Risk Pricing


def _move_color_from_value(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "#D1D5DB"
    if v < 80:
        return "#16A34A"  # Low Rates Vol
    if v < 100:
        return "#EAB308"  # Watchful
    if v < 120:
        return "#C2410C"  # Elevated Uncertainty
    if v < 140:
        return "#B91C1C"  # High Stress
    return "#7F1D1D"      # Extreme Stress


def _hy_oas_color_from_value(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "#D1D5DB"
    if v < 3:
        return "#16A34A"  # Low Credit Risk / Complacency
    if v < 6:
        return "#EAB308"  # Normal / Watchful
    if v < 8:
        return "#C2410C"  # Elevated Credit Stress
    if v < 10:
        return "#B91C1C"  # High Credit Stress
    return "#7F1D1D"      # Extreme Credit Stress


def _dxy_color_from_value(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "#D1D5DB"
    if v < 90:
        return "#B91C1C"  # USD Weak Regime
    if v < 100:
        return "#EAB308"  # Neutral USD
    if v < 105:
        return "#C2410C"  # Moderately Strong USD
    if v < 110:
        return "#16A34A"  # Strong USD
    return "#14532D"      # Extreme USD Strength


def _cnn_fng_color_from_value(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "#D1D5DB"
    if v < 25:
        return "#B91C1C"  # Extreme Fear
    if v < 45:
        return "#C2410C"  # Fear
    if v < 55:
        return "#EAB308"  # Neutral
    if v < 75:
        return "#16A34A"  # Greed
    return "#14532D"      # Extreme Greed


def _stlfsi_color_from_value(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "#D1D5DB"
    if v < -0.5:
        return "#16A34A"  # Easy Conditions
    if v < 0.5:
        return "#EAB308"  # Normal
    if v < 2.0:
        return "#C2410C"  # Elevated Stress
    if v < 3.0:
        return "#B91C1C"  # High Stress
    return "#7F1D1D"      # Extreme Systemic Stress


def _gvz_color_from_value(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "#D1D5DB"
    if v < 20:
        return "#16A34A"  # Low Vol / Complacency
    if v < 30:
        return "#EAB308"  # Normal / Watchful
    if v < 40:
        return "#C2410C"  # Elevated Stress
    if v < 55:
        return "#B91C1C"  # High Stress / Fear
    return "#7F1D1D"      # Extreme Stress / Panic


def _ovx_color_from_value(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "#D1D5DB"
    if v < 35:
        return "#16A34A"  # Low Vol / Complacency
    if v < 50:
        return "#EAB308"  # Normal / Watchful
    if v < 70:
        return "#C2410C"  # Elevated Stress
    if v < 90:
        return "#B91C1C"  # High Stress / Fear
    return "#7F1D1D"      # Extreme Stress / Panic


def _asof_value(series: pd.Series, target: pd.Timestamp) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    s = s.sort_index()
    s = s.loc[:target]
    if s.empty:
        return None
    return float(s.iloc[-1])


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


def two_line_button_label(label: str, value_id: str):
    return html.Span(
        [
            html.Span(label, className="btn-main-label"),
            html.Span(id=value_id, className="btn-subvalue"),
        ],
        className="btn-stack",
    )


min_dt, max_dt = get_date_bounds()
timeline_min_dt = max(min_dt, pd.to_datetime("2005-01-01"))
default_start_str, default_end_str = compute_preset_range("1Y", timeline_min_dt, max_dt)
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
        dcc.Store(id="active-preset", data="1Y"),
        dcc.Store(id="selected-maturities", data=[]),
        dcc.Store(id="show-spread-state", data=False),
        dcc.Store(id="show-yield-curve-state", data=False),
        dcc.Store(id="show-us-ig-corp-state", data=False),
        dcc.Store(id="show-aaa-corp-state", data=False),
        dcc.Store(id="show-us-hy-corp-state", data=False),
        dcc.Store(id="show-ig-muni-state", data=False),
        dcc.Store(id="show-hy-muni-state", data=False),
        dcc.Store(id="show-aaa-clo-state", data=False),
        dcc.Store(id="show-senior-loans-state", data=False),
        dcc.Store(id="show-agency-mbs-state", data=False),
        dcc.Store(id="show-em-sov-hard-state", data=False),
        dcc.Store(id="show-em-sov-local-state", data=False),
        dcc.Store(id="show-money-market-state", data=False),
        dcc.Store(id="show-tips-10y-state", data=False),
        dcc.Store(id="show-vol-vix-state", data=True),
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
        dcc.Interval(id="auto-refresh-interval", interval=60_000, n_intervals=0, disabled=False),
        html.Div(
            [
                html.H1("Key Economic Indicators", className="page-title"),
                html.Div(id="warning-message", className="warning-message"),
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
                        html.Button(
                            [
                                html.Span("Refresh Data", className="refresh-btn-main"),
                                html.Span("Last: --/-- --:--", id="refresh-last-line", className="refresh-btn-sub"),
                            ],
                            id="refresh-btn",
                            n_clicks=0,
                            className="primary-btn chart-action-btn",
                        ),
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
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Macro indicators", className="row-tag"),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Checklist(
                                                    id="show-fed-rate",
                                                    options=[{"label": "Federal Reserve Rate", "value": "on"}],
                                                    value=[],
                                                    className="macro-checklist-inner",
                                                ),
                                                html.Div(id="fed-rate-current-value", className="macro-option-subvalue"),
                                            ],
                                            className="control-group macro-metric-group",
                                        ),
                                        html.Div(
                                            [
                                                dcc.Checklist(
                                                    id="show-sofr",
                                                    options=[{"label": "SOFR 30D Avg", "value": "on"}],
                                                    value=[],
                                                    className="macro-checklist-inner",
                                                ),
                                                html.Div(id="sofr-current-value", className="macro-option-subvalue"),
                                            ],
                                            className="control-group macro-metric-group",
                                        ),
                                        html.Div(
                                            [
                                                dcc.Checklist(
                                                    id="show-inflation",
                                                    options=[{"label": "Core PCE Inflation", "value": "on"}],
                                                    value=[],
                                                    className="macro-checklist-inner",
                                                ),
                                                html.Div(id="inflation-current-value", className="macro-option-subvalue"),
                                            ],
                                            className="control-group macro-metric-group",
                                        ),
                                        html.Div(
                                            [
                                                dcc.Checklist(
                                                    id="show-unemployment",
                                                    options=[{"label": "U-3 Unemployment Rate", "value": "on"}],
                                                    value=[],
                                                    className="macro-checklist-inner",
                                                ),
                                                html.Div(id="unemployment-current-value", className="macro-option-subvalue"),
                                            ],
                                            className="control-group macro-metric-group",
                                        ),
                                        html.Div(
                                            [
                                                dcc.Checklist(
                                                    id="show-u6-unemployment",
                                                    options=[{"label": "U-6 Unemployment Rate", "value": "on"}],
                                                    value=[],
                                                    className="macro-checklist-inner",
                                                ),
                                                html.Div(id="u6-unemployment-current-value", className="macro-option-subvalue"),
                                            ],
                                            className="control-group macro-metric-group",
                                        ),
                                        html.Div(
                                            [
                                                dcc.Checklist(
                                                    id="show-unemp-ind",
                                                    options=[{"label": "U3-NROU Unemployment", "value": "on"}],
                                                    value=[],
                                                    className="macro-checklist-inner",
                                                ),
                                                html.Div(id="unemp-ind-current-value", className="macro-option-subvalue"),
                                            ],
                                            className="control-group macro-metric-group",
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
                html.Div(style={"height": "8px"}),
                dcc.Graph(id="indicator-graph"),
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
                                                html.Button(two_line_button_label("10Y-2Y", "yield-spread-current"), id="spread-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button("YIELD CURVE", id="yield-curve-btn", n_clicks=0, className="maturity-btn"),
                                                *[
                                                    html.Button(
                                                        two_line_button_label(m, f"yield-maturity-{m}-current"),
                                                        id=f"maturity-{m}",
                                                        n_clicks=0,
                                                        className="maturity-btn",
                                                    )
                                                    for m in MATURITY_PRESETS
                                                ],
                                            ],
                                            className="maturity-grid",
                                        ),
                                    ],
                                    className="yield-row",
                                ),
                            ],
                            className="below-chart-controls treasury-controls-box",
                        ),
                        html.Div(
                            [
                                html.Div("", className="control-label"),
                                html.Div(
                                    [
                                        html.Div("Bond Yields", className="row-tag"),
                                        html.Div(
                                            [
                                                html.Button(two_line_button_label("MONEY MARKET", "yield-money-market-current"), id="money-market-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button(two_line_button_label("IG CORP", "yield-us-ig-corp-current"), id="ig-corp-spread-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button(two_line_button_label("AAA CORP", "yield-aaa-corp-current"), id="aaa-corp-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button(two_line_button_label("AAA CLO", "yield-aaa-clo-current"), id="aaa-clo-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button(two_line_button_label("CMBS", "yield-tips-10y-current"), id="tips-10y-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button(two_line_button_label("AGENCY MBS", "yield-agency-mbs-current"), id="agency-mbs-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button(two_line_button_label("IG MUNI", "yield-ig-muni-current"), id="ig-muni-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button(two_line_button_label("SENIOR LOANS", "yield-senior-loans-current"), id="senior-loans-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button(two_line_button_label("HY CORP", "yield-us-hy-corp-current"), id="ig-muni-spread-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button(two_line_button_label("HY MUNI", "yield-hy-muni-current"), id="hy-muni-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button(two_line_button_label("EM SOV HARD", "yield-em-sov-hard-current"), id="em-sov-hard-yield-btn", n_clicks=0, className="maturity-btn"),
                                                html.Button(two_line_button_label("EM SOV LOCAL", "yield-em-sov-local-current"), id="em-sov-local-yield-btn", n_clicks=0, className="maturity-btn"),
                                            ],
                                            className="spread-grid",
                                        ),
                                    ],
                                    className="spread-row",
                                ),
                            ],
                            className="below-chart-controls bond-controls-box",
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
                                                        {"label": "1M average overlay", "value": "1m_mean"},
                                                        {"label": "3M average overlay", "value": "3m_mean"},
                                                        {"label": "6M average overlay", "value": "6m_mean"},
                                                        {"label": "1Y average overlay", "value": "1y_mean"},
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
                                                        (
                                                            [
                                                                html.Span(
                                                                    [
                                                                        html.Span(
                                                                            id=f"{col}-status-dot-1",
                                                                            style={
                                                                                "display": "inline-block",
                                                                                "width": "12px",
                                                                                "height": "12px",
                                                                                "borderRadius": "50%",
                                                                                "backgroundColor": "#9CA3AF",
                                                                                "border": "1px solid rgba(15, 23, 42, 0.35)",
                                                                            },
                                                                        ),
                                                                        html.Span(
                                                                            id=f"{col}-status-dot-2",
                                                                            style={
                                                                                "display": "inline-block",
                                                                                "width": "12px",
                                                                                "height": "12px",
                                                                                "borderRadius": "50%",
                                                                                "backgroundColor": "#9CA3AF",
                                                                                "border": "1px solid rgba(15, 23, 42, 0.35)",
                                                                            },
                                                                        ),
                                                                        html.Span(
                                                                            id=f"{col}-status-dot-3",
                                                                            style={
                                                                                "display": "inline-block",
                                                                                "width": "12px",
                                                                                "height": "12px",
                                                                                "borderRadius": "50%",
                                                                                "backgroundColor": "#9CA3AF",
                                                                                "border": "1px solid rgba(15, 23, 42, 0.35)",
                                                                            },
                                                                        ),
                                                                        html.Span(
                                                                            id=f"{col}-status-dot-4",
                                                                            style={
                                                                                "display": "inline-block",
                                                                                "width": "12px",
                                                                                "height": "12px",
                                                                                "borderRadius": "50%",
                                                                                "backgroundColor": "#9CA3AF",
                                                                                "border": "1px solid rgba(15, 23, 42, 0.35)",
                                                                            },
                                                                        ),
                                                                        html.Span(
                                                                            id=f"{col}-status-dot-5",
                                                                            style={
                                                                                "display": "inline-block",
                                                                                "width": "12px",
                                                                                "height": "12px",
                                                                                "borderRadius": "50%",
                                                                                "backgroundColor": "#9CA3AF",
                                                                                "border": "1px solid rgba(15, 23, 42, 0.35)",
                                                                            },
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "display": "inline-flex",
                                                                        "alignItems": "center",
                                                                        "gap": "4px",
                                                                        "marginRight": "8px",
                                                                    },
                                                                ),
                                                                html.Span(
                                                                    [
                                                                        html.Span(VOLATILITY_BUTTON_LABELS.get(col, col)),
                                                                        html.Span(id=f"vol-{col}-current-value", className="vol-btn-subvalue"),
                                                                    ],
                                                                    className="vol-btn-text-stack",
                                                                ),
                                                            ]
                                                            if col in {"vix", "vxn", "rvx", "vxeem"}
                                                            else VOLATILITY_BUTTON_LABELS.get(col, col)
                                                        ),
                                                        id=f"vol-{col}-btn",
                                                        n_clicks=0,
                                                        className=("maturity-btn maturity-btn-active" if col == "vix" else "maturity-btn"),
                                                        style=(
                                                            {
                                                                "background": "#fff",
                                                                "backgroundColor": "#fff",
                                                                "backgroundImage": "none",
                                                                "borderColor": VOLATILITY_BUTTON_COLORS["vix"],
                                                                "borderWidth": "2px",
                                                                "color": VOLATILITY_BUTTON_COLORS["vix"],
                                                            }
                                                            if col == "vix"
                                                            else {}
                                                        ),
                                                        title=VOLATILITY_HOVER_LABELS.get(col, col),
                                                    )
                                                    for col in ["vix", "vxn", "rvx", "vxeem"]
                                                ],
                                                *[
                                                    html.Button(
                                                        (
                                                            [
                                                                html.Span(
                                                                    [
                                                                        html.Span(
                                                                            id=f"{col}-status-dot-1",
                                                                            style={
                                                                                "display": "inline-block",
                                                                                "width": "12px",
                                                                                "height": "12px",
                                                                                "borderRadius": "50%",
                                                                                "backgroundColor": "#9CA3AF",
                                                                                "border": "1px solid rgba(15, 23, 42, 0.35)",
                                                                            },
                                                                        ),
                                                                        html.Span(
                                                                            id=f"{col}-status-dot-2",
                                                                            style={
                                                                                "display": "inline-block",
                                                                                "width": "12px",
                                                                                "height": "12px",
                                                                                "borderRadius": "50%",
                                                                                "backgroundColor": "#9CA3AF",
                                                                                "border": "1px solid rgba(15, 23, 42, 0.35)",
                                                                            },
                                                                        ),
                                                                        html.Span(
                                                                            id=f"{col}-status-dot-3",
                                                                            style={
                                                                                "display": "inline-block",
                                                                                "width": "12px",
                                                                                "height": "12px",
                                                                                "borderRadius": "50%",
                                                                                "backgroundColor": "#9CA3AF",
                                                                                "border": "1px solid rgba(15, 23, 42, 0.35)",
                                                                            },
                                                                        ),
                                                                        html.Span(
                                                                            id=f"{col}-status-dot-4",
                                                                            style={
                                                                                "display": "inline-block",
                                                                                "width": "12px",
                                                                                "height": "12px",
                                                                                "borderRadius": "50%",
                                                                                "backgroundColor": "#9CA3AF",
                                                                                "border": "1px solid rgba(15, 23, 42, 0.35)",
                                                                            },
                                                                        ),
                                                                        html.Span(
                                                                            id=f"{col}-status-dot-5",
                                                                            style={
                                                                                "display": "inline-block",
                                                                                "width": "12px",
                                                                                "height": "12px",
                                                                                "borderRadius": "50%",
                                                                                "backgroundColor": "#9CA3AF",
                                                                                "border": "1px solid rgba(15, 23, 42, 0.35)",
                                                                            },
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "display": "inline-flex",
                                                                        "alignItems": "center",
                                                                        "gap": "4px",
                                                                        "marginRight": "8px",
                                                                    },
                                                                ),
                                                                html.Span(
                                                                    [
                                                                        html.Span(VOLATILITY_BUTTON_LABELS.get(col, col)),
                                                                        html.Span(id=f"vol-{col}-current-value", className="vol-btn-subvalue"),
                                                                    ],
                                                                    className="vol-btn-text-stack",
                                                                ),
                                                            ]
                                                            if col in {"skew", "move", "hy_oas", "dxy", "cnn_fear_greed"}
                                                            else VOLATILITY_BUTTON_LABELS.get(col, col)
                                                        ),
                                                        id=f"vol-{col}-btn",
                                                        n_clicks=0,
                                                        className="maturity-btn",
                                                        title=VOLATILITY_HOVER_LABELS.get(col, col),
                                                    )
                                                    for col in ["skew", "move", "hy_oas", "dxy", "cnn_fear_greed"]
                                                ],
                                                html.Button(
                                                    [
                                                        html.Span(
                                                            [
                                                                html.Span(id="stlfsi-status-dot-1", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                                html.Span(id="stlfsi-status-dot-2", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                                html.Span(id="stlfsi-status-dot-3", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                                html.Span(id="stlfsi-status-dot-4", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                                html.Span(id="stlfsi-status-dot-5", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                            ],
                                                            style={"display": "inline-flex", "alignItems": "center", "gap": "4px", "marginRight": "8px"},
                                                        ),
                                                        html.Span(
                                                            [
                                                                html.Span(VOLATILITY_BUTTON_LABELS.get("stlfsi", "stlfsi")),
                                                                html.Span(id="vol-stlfsi-current-value", className="vol-btn-subvalue"),
                                                            ],
                                                            className="vol-btn-text-stack",
                                                        ),
                                                    ],
                                                    id="vol-stlfsi-btn",
                                                    n_clicks=0,
                                                    className="maturity-btn",
                                                    title=VOLATILITY_HOVER_LABELS.get("stlfsi", "stlfsi"),
                                                ),
                                                html.Button(
                                                    [
                                                        html.Span(
                                                            [
                                                                html.Span(id="gvz-status-dot-1", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                                html.Span(id="gvz-status-dot-2", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                                html.Span(id="gvz-status-dot-3", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                                html.Span(id="gvz-status-dot-4", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                                html.Span(id="gvz-status-dot-5", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                            ],
                                                            style={"display": "inline-flex", "alignItems": "center", "gap": "4px", "marginRight": "8px"},
                                                        ),
                                                        html.Span(
                                                            [
                                                                html.Span(VOLATILITY_BUTTON_LABELS.get("gvz", "gvz")),
                                                                html.Span(id="vol-gvz-current-value", className="vol-btn-subvalue"),
                                                            ],
                                                            className="vol-btn-text-stack",
                                                        ),
                                                    ],
                                                    id="vol-gvz-btn",
                                                    n_clicks=0,
                                                    className="maturity-btn",
                                                    title=VOLATILITY_HOVER_LABELS.get("gvz", "gvz"),
                                                ),
                                                html.Button(
                                                    [
                                                        html.Span(
                                                            [
                                                                html.Span(id="ovx-status-dot-1", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                                html.Span(id="ovx-status-dot-2", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                                html.Span(id="ovx-status-dot-3", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                                html.Span(id="ovx-status-dot-4", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                                html.Span(id="ovx-status-dot-5", style={"display": "inline-block", "width": "12px", "height": "12px", "borderRadius": "50%", "backgroundColor": "#9CA3AF", "border": "1px solid rgba(15, 23, 42, 0.35)"}),
                                                            ],
                                                            style={"display": "inline-flex", "alignItems": "center", "gap": "4px", "marginRight": "8px"},
                                                        ),
                                                        html.Span(
                                                            [
                                                                html.Span(VOLATILITY_BUTTON_LABELS.get("ovx", "ovx")),
                                                                html.Span(id="vol-ovx-current-value", className="vol-btn-subvalue"),
                                                            ],
                                                            className="vol-btn-text-stack",
                                                        ),
                                                    ],
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


STATUS_DOT_KEYS = [
    "vix",
    "vxn",
    "rvx",
    "vxeem",
    "skew",
    "move",
    "hy_oas",
    "dxy",
    "cnn_fear_greed",
    "stlfsi",
    "gvz",
    "ovx",
]

STATUS_DOT_COLOR_FNS = {
    "vix": _vix_color_from_value,
    "vxn": _vxn_color_from_value,
    "rvx": _rvx_color_from_value,
    "vxeem": _vxeem_color_from_value,
    "skew": _skew_color_from_value,
    "move": _move_color_from_value,
    "hy_oas": _hy_oas_color_from_value,
    "dxy": _dxy_color_from_value,
    "cnn_fear_greed": _cnn_fng_color_from_value,
    "stlfsi": _stlfsi_color_from_value,
    "gvz": _gvz_color_from_value,
    "ovx": _ovx_color_from_value,
}


@app.callback(
    Output("fed-rate-current-value", "children"),
    Output("inflation-current-value", "children"),
    Output("unemployment-current-value", "children"),
    Output("u6-unemployment-current-value", "children"),
    Output("unemp-ind-current-value", "children"),
    Output("sofr-current-value", "children"),
    Input("refresh-token", "data"),
)
def update_macro_current_values(_refresh_token):
    def _format(series: pd.Series, suffix: str = "%") -> str:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return "--"
        return f"{float(s.iloc[-1]):.2f}{suffix}"

    fed_df = load_and_process_csv("data/fedrate.csv")
    fed_value = _format(fed_df["FED_RATE"]) if (not fed_df.empty and "FED_RATE" in fed_df.columns) else "--"

    infl_df = load_and_process_csv("data/inflation.csv")
    infl_value = _format(infl_df["PCE_YoY"]) if (not infl_df.empty and "PCE_YoY" in infl_df.columns) else "--"

    unrate_df = load_and_process_csv("data/unemployment.csv")
    if not unrate_df.empty and "UNRATE" in unrate_df.columns:
        unrate_value = _format(unrate_df["UNRATE"])
    else:
        unrate_value = "--"

    if not unrate_df.empty and "U6RATE" in unrate_df.columns:
        u6_value = _format(unrate_df["U6RATE"])
    else:
        u6_value = "--"

    if not unrate_df.empty and {"UNRATE", "NROU"}.issubset(unrate_df.columns):
        unemp_indicator = pd.to_numeric(unrate_df["UNRATE"], errors="coerce") - pd.to_numeric(unrate_df["NROU"], errors="coerce")
        unemp_ind_value = _format(unemp_indicator, suffix="")
    else:
        unemp_ind_value = "--"

    sofr_value = _format(fed_df[SOFR_COL]) if (not fed_df.empty and SOFR_COL in fed_df.columns) else "--"

    return fed_value, infl_value, unrate_value, u6_value, unemp_ind_value, sofr_value


@app.callback(
    Output("yield-spread-current", "children"),
    *[Output(f"yield-maturity-{m}-current", "children") for m in MATURITY_PRESETS],
    Output("yield-us-ig-corp-current", "children"),
    Output("yield-aaa-corp-current", "children"),
    Output("yield-us-hy-corp-current", "children"),
    Output("yield-ig-muni-current", "children"),
    Output("yield-hy-muni-current", "children"),
    Output("yield-aaa-clo-current", "children"),
    Output("yield-senior-loans-current", "children"),
    Output("yield-agency-mbs-current", "children"),
    Output("yield-em-sov-hard-current", "children"),
    Output("yield-em-sov-local-current", "children"),
    Output("yield-money-market-current", "children"),
    Output("yield-tips-10y-current", "children"),
    Input("refresh-token", "data"),
)
def update_yield_button_current_values(_refresh_token):
    def _fmt_value(v: float | None) -> str:
        if v is None or pd.isna(v):
            return "--"
        return f"{float(v):.2f}%"

    ust_df = load_and_process_csv("data/ust.csv")
    bond_df = load_and_process_csv("data/bondyields.csv")

    spread_value = None
    maturity_values: list[float | None] = []
    if not ust_df.empty and {"BC_10YEAR", "BC_2YEAR"}.issubset(ust_df.columns):
        ten = pd.to_numeric(ust_df["BC_10YEAR"], errors="coerce")
        two = pd.to_numeric(ust_df["BC_2YEAR"], errors="coerce")
        spread_series = (ten - two).dropna()
        if not spread_series.empty:
            spread_value = float(spread_series.iloc[-1])

    if not ust_df.empty:
        for m in MATURITY_PRESETS:
            col = SERIES[m]
            if col in ust_df.columns:
                s = pd.to_numeric(ust_df[col], errors="coerce").dropna()
                maturity_values.append(float(s.iloc[-1]) if not s.empty else None)
            else:
                maturity_values.append(None)
    else:
        maturity_values = [None for _ in MATURITY_PRESETS]

    def _bond_latest(key: str) -> float | None:
        if bond_df.empty:
            return None
        col = CREDIT_YIELD_COLS[key]
        if col not in bond_df.columns:
            return None
        date_col = "DATE" if "DATE" in bond_df.columns else ("date" if "date" in bond_df.columns else None)
        if date_col is None:
            return None
        s_df = bond_df[[date_col, col]].copy()
        s_df[date_col] = pd.to_datetime(s_df[date_col], errors="coerce")
        s_df[col] = pd.to_numeric(s_df[col], errors="coerce")
        s = (
            s_df.dropna(subset=[date_col])
            .set_index(date_col)[col]
            .resample("W-FRI")
            .last()
            .dropna()
        )
        if s.empty:
            return None
        return float(s.iloc[-1])

    return (
        _fmt_value(spread_value),
        *[_fmt_value(v) for v in maturity_values],
        _fmt_value(_bond_latest("us_ig_corp")),
        _fmt_value(_bond_latest("aaa_corp")),
        _fmt_value(_bond_latest("us_hy_corp")),
        _fmt_value(_bond_latest("ig_muni")),
        _fmt_value(_bond_latest("hy_muni")),
        _fmt_value(_bond_latest("aaa_clo")),
        _fmt_value(_bond_latest("senior_loans")),
        _fmt_value(_bond_latest("agency_mbs")),
        _fmt_value(_bond_latest("em_sov_hard")),
        _fmt_value(_bond_latest("em_sov_local")),
        _fmt_value(_bond_latest("money_market")),
        _fmt_value(_bond_latest("tips_10y")),
    )


@app.callback(
    *[Output(f"vol-{k}-current-value", "children") for k in STATUS_DOT_KEYS],
    Input("refresh-token", "data"),
)
def update_volatility_button_current_values(_refresh_token):
    vol_df = load_and_process_csv("data/volatility.csv")
    if vol_df.empty:
        return tuple("--" for _ in STATUS_DOT_KEYS)

    out = []
    for key in STATUS_DOT_KEYS:
        if key not in vol_df.columns:
            out.append("--")
            continue
        s = pd.to_numeric(vol_df[key], errors="coerce").dropna()
        out.append(f"{float(s.iloc[-1]):.2f}" if not s.empty else "--")
    return tuple(out)


@app.callback(
    Output("refresh-last-line", "children"),
    Input("refresh-token", "data"),
)
def update_refresh_last_line(_refresh_token):
    return get_last_refresh_label()


@app.callback(
    [Output(f"{k}-status-dot-{i}", "style") for k in STATUS_DOT_KEYS for i in range(1, 6)],
    Input("refresh-token", "data"),
)
def update_vix_status_dots(_refresh_token):
    vol_df = load_and_process_csv("data/volatility.csv")
    gray = _vix_dot_style("#D1D5DB")
    yellow = _vix_dot_style("#EAB308")
    if vol_df.empty:
        return [gray for _ in range(len(STATUS_DOT_KEYS) * 5)]

    def _with_flash(style: dict) -> dict:
        return {
            **style,
            "animationName": "statusDotFlash",
            "animationDuration": "2.2s",
            "animationTimingFunction": "ease-in-out",
            "animationIterationCount": "infinite",
        }

    def _flash_if_red(style: dict) -> dict:
        color = str(style.get("backgroundColor", "")).strip().lower()
        return _with_flash(style) if color == "#dc2626" else style

    def _series_from_col(col: str) -> pd.Series:
        if col not in vol_df.columns:
            return pd.Series(dtype="float64")
        return pd.Series(
            pd.to_numeric(vol_df[col], errors="coerce").values,
            index=pd.to_datetime(vol_df["DATE"], errors="coerce"),
        ).dropna()

    def _five_colors(series: pd.Series, color_fn):
        if series.empty:
            styles = [gray, gray, gray, gray, gray]
            styles[4] = _flash_if_red(styles[4])
            return styles
        latest_ts = pd.Timestamp(series.index.max())
        vals = [
            _asof_value(series, latest_ts - pd.DateOffset(weeks=4)),
            _asof_value(series, latest_ts - pd.DateOffset(weeks=3)),
            _asof_value(series, latest_ts - pd.DateOffset(weeks=2)),
            _asof_value(series, latest_ts - pd.DateOffset(weeks=1)),
            _asof_value(series, latest_ts),
        ]
        styles = [_vix_dot_style(color_fn(v)) for v in vals]
        styles[4] = _flash_if_red(styles[4])
        return styles

    def _five_colors_wow(series: pd.Series, invert_red_green: bool = False):
        if series.empty:
            styles = [yellow, yellow, yellow, yellow, yellow]
            styles[4] = _flash_if_red(styles[4])
            return styles
        s = series.sort_index()
        weekly = s.resample("W-FRI").last().dropna()
        if len(weekly) < 6:
            styles = [yellow, yellow, yellow, yellow, yellow]
            styles[4] = _flash_if_red(styles[4])
            return styles
        anchors = list(weekly.tail(6).index)  # [w5, w4, w3, w2, w1, w0]
        vals = []
        for a in anchors:
            window = s.loc[(s.index >= a - pd.Timedelta(days=27)) & (s.index <= a)]
            vals.append(float(window.mean()) if not window.empty else float("nan"))
        deltas = [
            vals[1] - vals[0],  # circle 1: 4W ago vs 5W ago
            vals[2] - vals[1],  # circle 2: 3W ago vs 4W ago
            vals[3] - vals[2],  # circle 3: 2W ago vs 3W ago
            vals[4] - vals[3],  # circle 4: 1W ago vs 2W ago
            vals[5] - vals[4],  # circle 5: most recent vs 1W ago
        ]

        def _delta_style(d):
            if pd.isna(d) or d == 0:
                return yellow
            if invert_red_green:
                if d > 0:
                    return _vix_dot_style("#16A34A")
                return _vix_dot_style("#DC2626")
            if d > 0:
                return _vix_dot_style("#DC2626")
            return _vix_dot_style("#16A34A")

        styles = [_delta_style(d) for d in deltas]
        styles[4] = _flash_if_red(styles[4])
        return styles

    out = []
    for key in STATUS_DOT_KEYS:
        out.extend(_five_colors_wow(_series_from_col(key), invert_red_green=(key in {"dxy", "cnn_fear_greed"})))
    return out


@app.callback(
    Output("start-date", "date"),
    Output("end-date", "date"),
    Output("timeline-slider", "value"),
    Output("active-preset", "data"),
    [Input(f"preset-{p}", "n_clicks") for p in PRESETS],
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
    start_date_str = rest[0]
    end_date_str = rest[1]
    slider_range = rest[2]
    relayout_data = rest[3]

    if str(trigger).startswith("preset-"):
        preset = str(trigger).replace("preset-", "")
        start_date, end_date = compute_preset_range(preset, timeline_min_dt, max_dt)
        slider_value = [
            date_to_timeline_idx(start_date, timeline_min_dt, max_dt),
            date_to_timeline_idx(end_date, timeline_min_dt, max_dt),
        ]
        return start_date, end_date, slider_value, preset

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
                    "background": "#fff",
                    "backgroundColor": "#fff",
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
            "background": "#fff",
            "backgroundColor": "#fff",
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
    Output("show-yield-curve-state", "data"),
    Output("yield-curve-btn", "className"),
    Output("yield-curve-btn", "style"),
    Input("yield-curve-btn", "n_clicks"),
    State("show-yield-curve-state", "data"),
    prevent_initial_call=True,
)
def toggle_yield_curve_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "#fff",
            "backgroundColor": "#fff",
            "backgroundImage": "none",
            "borderColor": "#1F5F93",
            "borderWidth": "2px",
            "color": "#1F5F93",
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
            "background": "#fff",
            "backgroundColor": "#fff",
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
            "background": "#fff",
            "backgroundColor": "#fff",
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
            "background": "#fff",
            "backgroundColor": "#fff",
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
            "background": "#fff",
            "backgroundColor": "#fff",
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
            "background": "#fff",
            "backgroundColor": "#fff",
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
            "background": "#fff",
            "backgroundColor": "#fff",
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
            "background": "#fff",
            "backgroundColor": "#fff",
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
    Output("show-em-sov-hard-state", "data"),
    Output("em-sov-hard-yield-btn", "className"),
    Output("em-sov-hard-yield-btn", "style"),
    Input("em-sov-hard-yield-btn", "n_clicks"),
    State("show-em-sov-hard-state", "data"),
    prevent_initial_call=True,
)
def toggle_em_sov_hard_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "#fff",
            "backgroundColor": "#fff",
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["em_sov_hard"],
            "borderWidth": "2px",
            "color": BOND_LINE_COLORS["em_sov_hard"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-em-sov-local-state", "data"),
    Output("em-sov-local-yield-btn", "className"),
    Output("em-sov-local-yield-btn", "style"),
    Input("em-sov-local-yield-btn", "n_clicks"),
    State("show-em-sov-local-state", "data"),
    prevent_initial_call=True,
)
def toggle_em_sov_local_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "#fff",
            "backgroundColor": "#fff",
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["em_sov_local"],
            "borderWidth": "2px",
            "color": BOND_LINE_COLORS["em_sov_local"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-money-market-state", "data"),
    Output("money-market-yield-btn", "className"),
    Output("money-market-yield-btn", "style"),
    Input("money-market-yield-btn", "n_clicks"),
    State("show-money-market-state", "data"),
    prevent_initial_call=True,
)
def toggle_money_market_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "#fff",
            "backgroundColor": "#fff",
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["money_market"],
            "borderWidth": "2px",
            "color": BOND_LINE_COLORS["money_market"],
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-tips-10y-state", "data"),
    Output("tips-10y-yield-btn", "className"),
    Output("tips-10y-yield-btn", "style"),
    Input("tips-10y-yield-btn", "n_clicks"),
    State("show-tips-10y-state", "data"),
    prevent_initial_call=True,
)
def toggle_tips_10y_button(_n_clicks: int, current_state: bool):
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    style = (
        {
            "background": "#fff",
            "backgroundColor": "#fff",
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["tips_10y"],
            "borderWidth": "2px",
            "color": BOND_LINE_COLORS["tips_10y"],
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
                    "background": "#fff",
                    "backgroundColor": "#fff",
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
    Input("auto-refresh-interval", "n_intervals"),
    State("refresh-token", "data"),
    prevent_initial_call=True,
)
def handle_refresh(n_clicks: int, _n_intervals: int, _auto_n_intervals: int, token: int):
    trigger = callback_context.triggered_id

    if trigger == "refresh-btn":
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update, no_update
        started = start_refresh_worker()
        if not started:
            return "Refresh already running...", no_update, {"display": "block"}, False, no_update, no_update
        return "Starting refresh...", "0", {"display": "block"}, False, no_update, no_update
    if trigger == "auto-refresh-interval":
        maybe_start_scheduled_refresh()

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
    Output("warning-message", "children"),
    Input("selected-maturities", "data"),
    Input("show-spread-state", "data"),
    Input("show-yield-curve-state", "data"),
    Input("show-us-ig-corp-state", "data"),
    Input("show-aaa-corp-state", "data"),
    Input("show-us-hy-corp-state", "data"),
    Input("show-ig-muni-state", "data"),
    Input("show-hy-muni-state", "data"),
    Input("show-aaa-clo-state", "data"),
    Input("show-senior-loans-state", "data"),
    Input("show-agency-mbs-state", "data"),
    Input("show-em-sov-hard-state", "data"),
    Input("show-em-sov-local-state", "data"),
    Input("show-money-market-state", "data"),
    Input("show-tips-10y-state", "data"),
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
    Input("show-sofr", "value"),
    Input("show-fed-rate", "value"),
    Input("show-inflation", "value"),
    Input("show-unemployment", "value"),
    Input("show-u6-unemployment", "value"),
    Input("show-unemp-ind", "value"),
    Input("start-date", "date"),
    Input("end-date", "date"),
    Input("refresh-token", "data"),
)
def update_visuals(
    selected_maturities,
    show_spread_state,
    show_yield_curve_state,
    show_us_ig_corp_state,
    show_aaa_corp_state,
    show_us_hy_corp_state,
    show_ig_muni_state,
    show_hy_muni_state,
    show_aaa_clo_state,
    show_senior_loans_state,
    show_agency_mbs_state,
    show_em_sov_hard_state,
    show_em_sov_local_state,
    show_money_market_state,
    show_tips_10y_state,
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
    show_sofr_val,
    show_fed_rate_val,
    show_inflation_val,
    show_unemployment_val,
    show_u6_unemployment_val,
    show_unemp_ind_val,
    start_date_str,
    end_date_str,
    _refresh_token,
):
    show_yields = True
    show_spread = bool(show_spread_state)
    show_sofr = "on" in (show_sofr_val or [])
    show_yield_curve = bool(show_yield_curve_state)
    show_us_ig_corp = bool(show_us_ig_corp_state)
    show_aaa_corp = bool(show_aaa_corp_state)
    show_us_hy_corp = bool(show_us_hy_corp_state)
    show_ig_muni = bool(show_ig_muni_state)
    show_hy_muni = bool(show_hy_muni_state)
    show_aaa_clo = bool(show_aaa_clo_state)
    show_senior_loans = bool(show_senior_loans_state)
    show_agency_mbs = bool(show_agency_mbs_state)
    show_em_sov_hard = bool(show_em_sov_hard_state)
    show_em_sov_local = bool(show_em_sov_local_state)
    show_money_market = bool(show_money_market_state)
    show_tips_10y = bool(show_tips_10y_state)
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

    selected_maturities = selected_maturities or []

    ust_df = load_and_process_csv("data/ust.csv")
    if ust_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Treasury data found. Ensure data/ust.csv exists.")
        with chart_dataset_lock:
            chart_dataset_cache["csv"] = "date\n"
        return fig, "No Treasury data found."

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
        value_cols = []
        if "FED_RATE" in fed_df.columns:
            fed_df["FED_RATE"] = pd.to_numeric(fed_df["FED_RATE"], errors="coerce")
            value_cols.append("FED_RATE")
        if SOFR_COL in fed_df.columns:
            fed_df[SOFR_COL] = pd.to_numeric(fed_df[SOFR_COL], errors="coerce")
            value_cols.append(SOFR_COL)
        if value_cols:
            fed_monthly = align_to_month_end(fed_df[["DATE"] + value_cols])
            fed_monthly = align_to_treasury_daily_calendar(fed_monthly, value_cols, treasury_dates)
        else:
            fed_monthly = pd.DataFrame()
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
    if delta.days <= 365:
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
    plot_ust_raw = filter_by_date(ust_df, start_date, end_date)
    plot_ust["SPREAD_10Y_2Y"] = plot_ust["BC_10YEAR"] - plot_ust["BC_2YEAR"]
    plot_bond_yields = filter_by_date(bond_resampled, start_date, end_date) if not bond_resampled.empty else pd.DataFrame()
    any_vol_selected = any(
        [
            show_vol_vix,
            show_vol_vxn,
            show_vol_rvx,
            show_vol_vxeem,
            show_vol_skew,
            show_vol_gvz,
            show_vol_ovx,
            show_vol_stlfsi,
            show_vol_hy_oas,
            show_vol_ig_oas,
            show_vol_move,
            show_vol_dxy,
            show_vol_cnn_fear_greed,
        ]
    )
    use_daily_vol = any_vol_selected and delta.days <= 365
    vol_source = vol_df if use_daily_vol else vol_resampled
    plot_vol = filter_by_date(vol_source, start_date, end_date) if not vol_source.empty else pd.DataFrame()

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
        plot_ust_raw=plot_ust_raw,
        plot_bond_yields=plot_bond_yields,
        plot_fed=plot_fed,
        plot_infl=plot_infl,
        plot_unrate=plot_unrate,
        plot_vol=plot_vol,
        selected_maturities=selected_maturities,
        show_yields=show_yields,
        show_spread=show_spread,
        show_sofr=show_sofr,
        show_yield_curve=show_yield_curve,
        show_us_ig_corp=show_us_ig_corp,
        show_aaa_corp=show_aaa_corp,
        show_us_hy_corp=show_us_hy_corp,
        show_ig_muni=show_ig_muni,
        show_hy_muni=show_hy_muni,
        show_aaa_clo=show_aaa_clo,
        show_senior_loans=show_senior_loans,
        show_agency_mbs=show_agency_mbs,
        show_em_sov_hard=show_em_sov_hard,
        show_em_sov_local=show_em_sov_local,
        show_money_market=show_money_market,
        show_tips_10y=show_tips_10y,
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
    if not show_yield_curve:
        fig.update_xaxes(range=[start_date.isoformat(), end_date.isoformat()])
        fig.add_vrect(
            x0="2000-03-01",
            x1="2002-03-31",
            fillcolor="rgba(59, 130, 246, 0.12)",
            opacity=0.26,
            line_color="rgba(37, 99, 235, 0.55)",
            line_width=1,
            layer="above",
        )
        fig.add_vrect(
            x0="2007-08-01",
            x1="2009-08-31",
            fillcolor="rgba(59, 130, 246, 0.12)",
            opacity=0.26,
            line_color="rgba(37, 99, 235, 0.55)",
            line_width=1,
            layer="above",
        )
        fig.add_vrect(
            x0="2010-01-01",
            x1="2013-12-31",
            fillcolor="rgba(59, 130, 246, 0.12)",
            opacity=0.26,
            line_color="rgba(37, 99, 235, 0.55)",
            line_width=1,
            layer="above",
        )
        fig.add_vrect(
            x0="2020-01-01",
            x1="2020-05-31",
            fillcolor="rgba(59, 130, 246, 0.12)",
            opacity=0.26,
            line_color="rgba(37, 99, 235, 0.55)",
            line_width=1,
            layer="above",
        )
        fig.add_vrect(
            x0="2023-03-01",
            x1="2023-07-31",
            fillcolor="rgba(59, 130, 246, 0.12)",
            opacity=0.26,
            line_color="rgba(37, 99, 235, 0.55)",
            line_width=1,
            layer="above",
        )
        fig.add_vrect(
            x0="2025-04-01",
            x1="2025-04-30",
            fillcolor="rgba(59, 130, 246, 0.12)",
            opacity=0.26,
            line_color="rgba(37, 99, 235, 0.55)",
            line_width=1,
            layer="above",
        )
        fig.add_annotation(x="2001-03-16", y=1.0, xref="x", yref="paper", text="Dot Com Bubble", showarrow=False, xanchor="center", yanchor="bottom", yshift=6, font={"size": 11, "color": "#1D4ED8"})
        fig.add_annotation(x="2008-08-15", y=1.0, xref="x", yref="paper", text="Global Financial Crisis", showarrow=False, xanchor="center", yanchor="bottom", yshift=6, font={"size": 11, "color": "#1D4ED8"})
        fig.add_annotation(x="2012-01-01", y=1.0, xref="x", yref="paper", text="EU Debt Crisis", showarrow=False, xanchor="center", yanchor="bottom", yshift=6, font={"size": 11, "color": "#1D4ED8"})
        fig.add_annotation(x="2020-03-16", y=1.0, xref="x", yref="paper", text="COVID-19 Recession", showarrow=False, xanchor="center", yanchor="bottom", yshift=6, font={"size": 11, "color": "#1D4ED8"})
        fig.add_annotation(x="2023-05-16", y=1.0, xref="x", yref="paper", text="US Regional Banking Crisis", showarrow=False, xanchor="center", yanchor="bottom", yshift=6, font={"size": 11, "color": "#1D4ED8"})
        fig.add_annotation(x="2025-04-15", y=1.0, xref="x", yref="paper", text="Liberation Day", showarrow=False, xanchor="center", yanchor="bottom", yshift=6, font={"size": 11, "color": "#1D4ED8"})

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
    if show_sofr and SOFR_COL in fed_df.columns:
        s_df = filter_by_date(fed_df[["DATE", SOFR_COL]], start_date, end_date)
        sofr_vals = pd.to_numeric(s_df[SOFR_COL], errors="coerce")
        export_series.append(("SOFR 30D Avg", pd.Series(sofr_vals.values, index=s_df["DATE"])))

    bond_flags = [
        ("us_ig_corp", show_us_ig_corp, "IG CORP"),
        ("aaa_corp", show_aaa_corp, "AAA CORP"),
        ("us_hy_corp", show_us_hy_corp, "HY CORP"),
        ("ig_muni", show_ig_muni, "IG MUNI"),
        ("hy_muni", show_hy_muni, "HY MUNI"),
        ("aaa_clo", show_aaa_clo, "AAA_CLO"),
        ("senior_loans", show_senior_loans, "SENIOR LOANS"),
        ("agency_mbs", show_agency_mbs, "AGENCY MBS"),
        ("em_sov_hard", show_em_sov_hard, "EM SOV HARD"),
        ("em_sov_local", show_em_sov_local, "EM SOV LOCAL"),
        ("money_market", show_money_market, "MONEY MARKET"),
        ("tips_10y", show_tips_10y, "CMBS"),
    ]
    if not bond_yields_df.empty:
        for key, enabled, label in bond_flags:
            col = CREDIT_YIELD_COLS[key]
            if not enabled or col not in bond_yields_df.columns:
                continue
            s_df = filter_by_date(bond_yields_df[["DATE", col]], start_date, end_date)
            s = pd.Series(pd.to_numeric(s_df[col], errors="coerce").values, index=s_df["DATE"], name=label)
            export_series.append((label, s))

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

    with chart_dataset_lock:
        chart_dataset_cache["csv"] = series_items_to_csv_text(export_series)

    return fig, warning


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8050")),
        debug=False,
    )
