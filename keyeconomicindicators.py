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
    "gvz",
    "ovx",
    "stlfsi",
    "hy_oas",
    "ig_oas",
    "move",
]
VOLATILITY_DATA_COLS = {col: col for col in VOLATILITY_COLS}
VOLATILITY_BUTTON_COLORS = {
    "vix": "#BE185D",
    "vxn": "#0891B2",
    "gvz": "#D97706",
    "ovx": "#DC2626",
    "stlfsi": "#6D28D9",
    "hy_oas": "#B45309",
    "ig_oas": "#0F766E",
    "move": "#374151",
}
VOLATILITY_HOVER_LABELS = {
    "vix": "VIX (Equity Volatility)",
    "vxn": "VXN (NASDAQ-100 Volatility)",
    "gvz": "Gold Volatility",
    "ovx": "Oil Volatility",
    "stlfsi": "Fed Financial Stress Index",
    "hy_oas": "High Yield Corporate OAS",
    "ig_oas": "IG Corporate OAS",
    "move": "Bond Volatility",
}
VOLATILITY_BUTTON_LABELS = {
    "vix": "VIX (BROAD EQUITIES)",
    "vxn": "VXN (NASDAQ-100)",
    "gvz": "GVZ (GOLD)",
    "ovx": "OVX (OIL)",
    "stlfsi": "STLFSI (FED STRESS INDEX)",
    "hy_oas": "HY OAS (HY BOND SPREADS)",
    "ig_oas": "IG OAS (IG BOND SPREADS)",
    "move": "MOVE (BONDS)",
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
    show_vol_gvz: bool,
    show_vol_ovx: bool,
    show_vol_stlfsi: bool,
    show_vol_hy_oas: bool,
    show_vol_ig_oas: bool,
    show_vol_move: bool,
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
        if selected_median_mode not in {"10y", "3y"}:
            selected_median_mode = "10y"
        vol_flags = {
            "vix": show_vol_vix,
            "vxn": show_vol_vxn,
            "gvz": show_vol_gvz,
            "ovx": show_vol_ovx,
            "stlfsi": show_vol_stlfsi,
            "hy_oas": show_vol_hy_oas,
            "ig_oas": show_vol_ig_oas,
            "move": show_vol_move,
        }
        for key, enabled in vol_flags.items():
            z_col = VOLATILITY_DATA_COLS[key]
            if enabled and z_col in plot_vol.columns:
                all_vals.extend(pd.to_numeric(plot_vol[z_col], errors="coerce").dropna().tolist())

    if all_vals:
        v_min, v_max = min(all_vals), max(all_vals)
        # Force zero to remain visible in all chart states.
        v_min = min(v_min, 0.0)
        v_max = max(v_max, 0.0)
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
        changes = plot_fed[plot_fed["rate_changed"]]
        if not changes.empty:
            fig.add_trace(
                go.Scatter(
                    x=changes["DATE"],
                    y=changes["FED_RATE"],
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
        vol_flags = {
            "vix": show_vol_vix,
            "vxn": show_vol_vxn,
            "gvz": show_vol_gvz,
            "ovx": show_vol_ovx,
            "stlfsi": show_vol_stlfsi,
            "hy_oas": show_vol_hy_oas,
            "ig_oas": show_vol_ig_oas,
            "move": show_vol_move,
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
            if selected_median_mode == "3y":
                rolling_window = "1096D"
                window_label = "3Y"
            else:
                rolling_window = "3652D"
                window_label = "10Y"
            vol_median = vol_series.rolling(rolling_window, min_periods=1).median()
            median_name = f"{display_label} {window_label} MEDIAN"
            median_hover = f"{display_label} {window_label} Median"
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
            fig.add_trace(
                go.Scatter(
                    x=vol_median.index,
                    y=vol_median.values,
                    name=median_name,
                    mode="lines",
                    line={"color": VOLATILITY_BUTTON_COLORS[key], "width": 2.0, "dash": "dot"},
                    yaxis="y2",
                    hovertemplate=f"{median_hover}<br>%{{x|%b %d, %Y}}<br>%{{y:.2f}}<extra></extra>",
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
        dcc.Store(id="show-vol-gvz-state", data=False),
        dcc.Store(id="show-vol-ovx-state", data=False),
        dcc.Store(id="show-vol-stlfsi-state", data=False),
        dcc.Store(id="show-vol-hy_oas-state", data=False),
        dcc.Store(id="show-vol-ig_oas-state", data=False),
        dcc.Store(id="show-vol-move-state", data=False),
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
                                                html.Button("AAA_CLO", id="aaa-clo-yield-btn", n_clicks=0, className="maturity-btn"),
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
                                                html.Button("AAA_CLO", id="cs-aaa-clo-yield-btn", n_clicks=0, className="maturity-btn"),
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
                                        html.Div("Volatility Indicators", className="row-tag"),
                                        dcc.Dropdown(
                                            id="vol-band-select",
                                            options=[
                                                {"label": "Select band", "value": "none"},
                                                {"label": "Elevated: x > p75", "value": "25_75"},
                                                {"label": "Stress: x > p90", "value": "10_90"},
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
                                                {"label": "10Y median overlay", "value": "10y"},
                                                {"label": "3Y median overlay", "value": "3y"},
                                            ],
                                            value="10y",
                                            clearable=False,
                                            searchable=False,
                                            persistence=False,
                                            className="vol-median-dropdown",
                                        ),
                                        html.Div(
                                            [
                                                *[
                                                    html.Button(
                                                        VOLATILITY_BUTTON_LABELS.get(col, col.upper()),
                                                        id=f"vol-{col}-btn",
                                                        n_clicks=0,
                                                        className="maturity-btn",
                                                        title=VOLATILITY_HOVER_LABELS.get(col, col),
                                                    )
                                                    for col in VOLATILITY_COLS
                                                ],
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
    Input("start-date", "date"),
    Input("end-date", "date"),
    Input("timeline-slider", "value"),
    prevent_initial_call=True,
)
def apply_preset(*args):
    trigger = callback_context.triggered_id
    if not trigger:
        return no_update, no_update, no_update, no_update

    start_date_str = args[-3]
    end_date_str = args[-2]
    slider_range = args[-1]

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
            text_color = "#0F172A" if maturity in {"1Y", "2Y", "5Y"} else "#FFFFFF"
            styles.append(
                {
                    "background": color,
                    "backgroundColor": color,
                    "backgroundImage": "none",
                    "borderColor": color,
                    "color": text_color,
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
            "background": "#0B3A63",
            "backgroundColor": "#0B3A63",
            "backgroundImage": "none",
            "borderColor": "#0B3A63",
            "color": "#FFFFFF",
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
            "background": BOND_LINE_COLORS["us_ig_corp"],
            "backgroundColor": BOND_LINE_COLORS["us_ig_corp"],
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["us_ig_corp"],
            "color": "#FFFFFF",
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
            "background": BOND_LINE_COLORS["aaa_corp"],
            "backgroundColor": BOND_LINE_COLORS["aaa_corp"],
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["aaa_corp"],
            "color": "#FFFFFF",
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
            "background": BOND_LINE_COLORS["us_hy_corp"],
            "backgroundColor": BOND_LINE_COLORS["us_hy_corp"],
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["us_hy_corp"],
            "color": "#FFFFFF",
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
            "background": BOND_LINE_COLORS["ig_muni"],
            "backgroundColor": BOND_LINE_COLORS["ig_muni"],
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["ig_muni"],
            "color": "#FFFFFF",
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
            "background": BOND_LINE_COLORS["hy_muni"],
            "backgroundColor": BOND_LINE_COLORS["hy_muni"],
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["hy_muni"],
            "color": "#FFFFFF",
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
            "background": BOND_LINE_COLORS["aaa_clo"],
            "backgroundColor": BOND_LINE_COLORS["aaa_clo"],
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["aaa_clo"],
            "color": "#FFFFFF",
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
            "background": BOND_LINE_COLORS["senior_loans"],
            "backgroundColor": BOND_LINE_COLORS["senior_loans"],
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["senior_loans"],
            "color": "#FFFFFF",
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
            "background": BOND_LINE_COLORS["agency_mbs"],
            "backgroundColor": BOND_LINE_COLORS["agency_mbs"],
            "backgroundImage": "none",
            "borderColor": BOND_LINE_COLORS["agency_mbs"],
            "color": "#FFFFFF",
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
            "background": CREDIT_SPREAD_BUTTON_COLORS["us_ig_corp"],
            "backgroundColor": CREDIT_SPREAD_BUTTON_COLORS["us_ig_corp"],
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["us_ig_corp"],
            "color": "#FFFFFF",
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
            "background": CREDIT_SPREAD_BUTTON_COLORS["aaa_corp"],
            "backgroundColor": CREDIT_SPREAD_BUTTON_COLORS["aaa_corp"],
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["aaa_corp"],
            "color": "#FFFFFF",
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
            "background": CREDIT_SPREAD_BUTTON_COLORS["us_hy_corp"],
            "backgroundColor": CREDIT_SPREAD_BUTTON_COLORS["us_hy_corp"],
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["us_hy_corp"],
            "color": "#FFFFFF",
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
            "background": CREDIT_SPREAD_BUTTON_COLORS["ig_muni"],
            "backgroundColor": CREDIT_SPREAD_BUTTON_COLORS["ig_muni"],
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["ig_muni"],
            "color": "#FFFFFF",
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
            "background": CREDIT_SPREAD_BUTTON_COLORS["hy_muni"],
            "backgroundColor": CREDIT_SPREAD_BUTTON_COLORS["hy_muni"],
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["hy_muni"],
            "color": "#FFFFFF",
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
            "background": CREDIT_SPREAD_BUTTON_COLORS["aaa_clo"],
            "backgroundColor": CREDIT_SPREAD_BUTTON_COLORS["aaa_clo"],
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["aaa_clo"],
            "color": "#FFFFFF",
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
            "background": CREDIT_SPREAD_BUTTON_COLORS["senior_loans"],
            "backgroundColor": CREDIT_SPREAD_BUTTON_COLORS["senior_loans"],
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["senior_loans"],
            "color": "#FFFFFF",
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
            "background": CREDIT_SPREAD_BUTTON_COLORS["agency_mbs"],
            "backgroundColor": CREDIT_SPREAD_BUTTON_COLORS["agency_mbs"],
            "backgroundImage": "none",
            "borderColor": CREDIT_SPREAD_BUTTON_COLORS["agency_mbs"],
            "color": "#FFFFFF",
        }
        if new_state
        else {}
    )
    return new_state, cls, style


def _toggle_volatility_button(current_state: bool, series_key: str) -> tuple[bool, str, dict]:
    new_state = not bool(current_state)
    cls = "maturity-btn maturity-btn-active" if new_state else "maturity-btn"
    color = VOLATILITY_BUTTON_COLORS[series_key]
    style = (
        {
            "background": color,
            "backgroundColor": color,
            "backgroundImage": "none",
            "borderColor": color,
            "color": "#FFFFFF",
        }
        if new_state
        else {}
    )
    return new_state, cls, style


@app.callback(
    Output("show-vol-vix-state", "data"),
    Output("vol-vix-btn", "className"),
    Output("vol-vix-btn", "style"),
    Input("vol-vix-btn", "n_clicks"),
    State("show-vol-vix-state", "data"),
    prevent_initial_call=True,
)
def toggle_vol_vix_button(_n_clicks: int, current_state: bool):
    return _toggle_volatility_button(current_state, "vix")


@app.callback(
    Output("show-vol-vxn-state", "data"),
    Output("vol-vxn-btn", "className"),
    Output("vol-vxn-btn", "style"),
    Input("vol-vxn-btn", "n_clicks"),
    State("show-vol-vxn-state", "data"),
    prevent_initial_call=True,
)
def toggle_vol_vxn_button(_n_clicks: int, current_state: bool):
    return _toggle_volatility_button(current_state, "vxn")


@app.callback(
    Output("show-vol-gvz-state", "data"),
    Output("vol-gvz-btn", "className"),
    Output("vol-gvz-btn", "style"),
    Input("vol-gvz-btn", "n_clicks"),
    State("show-vol-gvz-state", "data"),
    prevent_initial_call=True,
)
def toggle_vol_gvz_button(_n_clicks: int, current_state: bool):
    return _toggle_volatility_button(current_state, "gvz")


@app.callback(
    Output("show-vol-ovx-state", "data"),
    Output("vol-ovx-btn", "className"),
    Output("vol-ovx-btn", "style"),
    Input("vol-ovx-btn", "n_clicks"),
    State("show-vol-ovx-state", "data"),
    prevent_initial_call=True,
)
def toggle_vol_ovx_button(_n_clicks: int, current_state: bool):
    return _toggle_volatility_button(current_state, "ovx")


@app.callback(
    Output("show-vol-stlfsi-state", "data"),
    Output("vol-stlfsi-btn", "className"),
    Output("vol-stlfsi-btn", "style"),
    Input("vol-stlfsi-btn", "n_clicks"),
    State("show-vol-stlfsi-state", "data"),
    prevent_initial_call=True,
)
def toggle_vol_stlfsi_button(_n_clicks: int, current_state: bool):
    return _toggle_volatility_button(current_state, "stlfsi")


@app.callback(
    Output("show-vol-hy_oas-state", "data"),
    Output("vol-hy_oas-btn", "className"),
    Output("vol-hy_oas-btn", "style"),
    Input("vol-hy_oas-btn", "n_clicks"),
    State("show-vol-hy_oas-state", "data"),
    prevent_initial_call=True,
)
def toggle_vol_hy_oas_button(_n_clicks: int, current_state: bool):
    return _toggle_volatility_button(current_state, "hy_oas")


@app.callback(
    Output("show-vol-ig_oas-state", "data"),
    Output("vol-ig_oas-btn", "className"),
    Output("vol-ig_oas-btn", "style"),
    Input("vol-ig_oas-btn", "n_clicks"),
    State("show-vol-ig_oas-state", "data"),
    prevent_initial_call=True,
)
def toggle_vol_ig_oas_button(_n_clicks: int, current_state: bool):
    return _toggle_volatility_button(current_state, "ig_oas")


@app.callback(
    Output("show-vol-move-state", "data"),
    Output("vol-move-btn", "className"),
    Output("vol-move-btn", "style"),
    Input("vol-move-btn", "n_clicks"),
    State("show-vol-move-state", "data"),
    prevent_initial_call=True,
)
def toggle_vol_move_button(_n_clicks: int, current_state: bool):
    return _toggle_volatility_button(current_state, "move")


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
    Input("show-vol-gvz-state", "data"),
    Input("show-vol-ovx-state", "data"),
    Input("show-vol-stlfsi-state", "data"),
    Input("show-vol-hy_oas-state", "data"),
    Input("show-vol-ig_oas-state", "data"),
    Input("show-vol-move-state", "data"),
    Input("vol-band-select", "value"),
    Input("vol-median-select", "value"),
    Input("cs-baseline-tenor", "value"),
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
    show_vol_gvz_state,
    show_vol_ovx_state,
    show_vol_stlfsi_state,
    show_vol_hy_oas_state,
    show_vol_ig_oas_state,
    show_vol_move_state,
    vol_band_mode,
    vol_median_mode,
    cs_baseline_tenor,
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
    show_vol_gvz = bool(show_vol_gvz_state)
    show_vol_ovx = bool(show_vol_ovx_state)
    show_vol_stlfsi = bool(show_vol_stlfsi_state)
    show_vol_hy_oas = bool(show_vol_hy_oas_state)
    show_vol_ig_oas = bool(show_vol_ig_oas_state)
    show_vol_move = bool(show_vol_move_state)
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
    if not plot_fed.empty and len(plot_fed) > 1:
        plot_fed["rate_changed"] = plot_fed["FED_RATE"].diff().abs() > 0.001
        plot_fed.iloc[0, plot_fed.columns.get_loc("rate_changed")] = True
    elif not plot_fed.empty:
        plot_fed["rate_changed"] = True

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
        show_vol_gvz=show_vol_gvz,
        show_vol_ovx=show_vol_ovx,
        show_vol_stlfsi=show_vol_stlfsi,
        show_vol_hy_oas=show_vol_hy_oas,
        show_vol_ig_oas=show_vol_ig_oas,
        show_vol_move=show_vol_move,
        vol_band_mode=vol_band_mode,
        vol_median_mode=vol_median_mode,
    )
    fig.update_xaxes(range=[start_date.isoformat(), end_date.isoformat()])

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
            "gvz": show_vol_gvz,
            "ovx": show_vol_ovx,
            "stlfsi": show_vol_stlfsi,
            "hy_oas": show_vol_hy_oas,
            "ig_oas": show_vol_ig_oas,
            "move": show_vol_move,
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
