"""③ Portfolio Analysis: allocation, rebalancing, lump sum or monthly DCA, compared with one asset."""
import pandas as pd
import streamlit as st

import charts
import common
import compute as cp

CUSTOM = "カスタム"
DEFAULT_PRESET = "US 60/40"
REBALANCE = {"なし（Buy & Hold）": "none", "年1回（Annual）": "annual", "四半期ごと（Quarterly）": "quarterly",
             "毎月（Monthly）": "monthly"}
METHODS = {"一括投資（Lump Sum）": "lump_sum", "毎月定額積立（Monthly DCA）": "dca"}
PORTFOLIO = "ポートフォリオ"
CONTRIBUTED = "拠出累計"


def weight_key(ticker: str) -> str:
    return f"pf_w_{ticker}"


def _apply_preset(tickers: list[str]) -> None:
    name = st.session_state["_pf_preset"]
    if name == CUSTOM:
        return
    weights = cp.presets()[name]
    for t in tickers:
        st.session_state[f"_{weight_key(t)}"] = round(weights.get(t, 0) * 100)


def _normalize(tickers: list[str]) -> None:
    current = {t: st.session_state[f"_{weight_key(t)}"] for t in tickers}
    for t, value in cp.normalize_percent(current).items():
        st.session_state[f"_{weight_key(t)}"] = value
    st.session_state["_pf_preset"] = CUSTOM


def _mark_custom() -> None:
    st.session_state["_pf_preset"] = CUSTOM


@st.cache_data(show_spinner=False)
def simulate(start, end, weights: tuple, rebalance: str, method: str, amount: float):
    p = cp.trading_days(common.prices(), start, end)
    return cp.simulate(p, dict(weights), rebalance, method, amount)


def comparison(portfolio: dict, benchmark: dict, method: str, benchmark_name: str) -> pd.DataFrame:
    rows = [("最終資産", "final_value", "money"), ("拠出総額", "total_contributed", "money"),
            ("拠出額に対するリターン", "return_on_contributions", "pct"),
            ("CAGR（時間加重）", "cagr", "pct"), ("年率ボラティリティ", "annual_volatility", "pct"),
            ("最大DD（時間加重）", "max_drawdown", "pct"), ("Sharpe", "sharpe_ratio", "num"),
            ("最大DD（拠出を考慮した金額ベース）", "money_max_drawdown", "pct"),
            ("リバランス回数", "rebalances", "int"), ("年間回転率（片道）", "annual_turnover", "pct")]
    if method == "lump_sum":
        rows = [r for r in rows if r[1] != "money_max_drawdown"]

    def fmt(value, kind):
        if kind == "money":
            return f"{value:,.0f}"
        if kind == "pct":
            return common.fmt_pct(value)
        if kind == "int":
            return f"{value:,}"
        return common.fmt_num(value)

    return pd.DataFrame({
        PORTFOLIO: [fmt(portfolio[k], kind) for _, k, kind in rows],
        benchmark_name: [fmt(benchmark[k], kind) for _, k, kind in rows],
    }, index=[label for label, _, _ in rows])


def render() -> None:
    st.title("③ Portfolio Analysis（ポートフォリオ）")
    prices = common.prices()
    tickers = list(prices.columns)
    presets = cp.presets()
    default = presets[DEFAULT_PRESET]

    with st.container(border=True):
        st.markdown("**配分**")
        st.selectbox("プリセット（config/portfolios.toml）", list(presets) + [CUSTOM],
                     key=common.persistent("pf_preset", DEFAULT_PRESET), on_change=_apply_preset, args=(tickers,))
        cols = st.columns(len(tickers))
        percent = {}
        for col, t in zip(cols, tickers):
            percent[t] = col.slider(f"{t}（%）", 0, 100, step=1, on_change=_mark_custom,
                                    key=common.persistent(weight_key(t), round(default.get(t, 0) * 100)))
        total = sum(percent.values())
        left, right = st.columns([3, 1])
        left.markdown(f"合計：**{total}%**")
        right.button("合計を 100% に合わせる", on_click=_normalize, args=(tickers,), disabled=total == 0)

        c1, c2, c3, c4 = st.columns(4)
        rebalance_label = c1.selectbox("リバランス", list(REBALANCE),
                                       key=common.persistent("pf_rebalance", "年1回（Annual）"))
        method_label = c2.radio("投資方法", list(METHODS), key=common.persistent("pf_method", "一括投資（Lump Sum）"))
        method = METHODS[method_label]
        amount = c3.number_input("一括投資額" if method == "lump_sum" else "毎月の積立額", min_value=1.0,
                                 step=1000.0, format="%.0f", key=common.persistent("pf_amount", 10_000.0),
                                 help="正規化した金額（USD 建て）。実際の資産額ではありません。")
        benchmark = c4.selectbox("ベンチマーク（単一資産）", tickers, key=common.persistent("pf_benchmark", "VOO"))
    common.remember("pf_preset", "pf_rebalance", "pf_method", "pf_amount", "pf_benchmark",
                    *(weight_key(t) for t in tickers))

    problems = cp.weight_problems(percent)
    if problems:
        for problem in problems:
            st.error(problem)
        return
    p = common.period_prices(min_days=21)
    if p is None:
        return
    s = common.settings()
    weights = tuple((t, w) for t, w in percent.items() if w > 0)
    rebalance = REBALANCE[rebalance_label]
    result, summary = simulate(s.start, s.end, weights, rebalance, method, amount)
    bench_result, bench_summary = simulate(s.start, s.end, ((benchmark, 100),), "none", method, amount)

    mix = " / ".join(f"{t} {w}%" for t, w in weights)
    st.caption(f"{p.index[0].date()} 〜 {p.index[-1].date()}。{mix}、リバランス {rebalance_label}、{method_label}。"
               "取引は各イベント日の終値、手数料・税は含みません（Phase 4 と同じ前提）。")

    cards = st.columns(5)
    for col, (label, key, fmt) in zip(cards, [
            ("最終資産", "final_value", lambda v: f"{v:,.0f}"), ("CAGR（時間加重）", "cagr", common.fmt_pct),
            ("年率ボラティリティ", "annual_volatility", common.fmt_pct), ("最大DD", "max_drawdown", common.fmt_pct),
            ("Sharpe", "sharpe_ratio", common.fmt_num)]):
        diff = summary[key] - bench_summary[key]
        # ASCII sign: st.metric picks the arrow direction from a leading "-"
        delta = f"{diff:+,.0f}" if key == "final_value" else (f"{diff:+.2f}" if key == "sharpe_ratio"
                                                               else f"{diff:+.2%}")
        col.metric(label, fmt(summary[key]), delta=f"{delta}（対 {benchmark}）",
                   delta_color="inverse" if key == "annual_volatility" else "normal", border=True)

    bench_name = f"ベンチマーク（{benchmark}）"
    colors = {PORTFOLIO: charts.SERIES_COLORS[0], bench_name: charts.MUTED, CONTRIBUTED: "#52514e"}
    # the benchmark is drawn first so that the portfolio line stays on top
    value = pd.DataFrame({bench_name: bench_result.value, PORTFOLIO: result.value})
    if method == "dca":
        value[CONTRIBUTED] = result.contributions
    left, right = st.columns([3, 2])
    with left:
        st.subheader("資産額の推移")
        st.plotly_chart(charts.lines(value, colors, "資産額", ",.0f", dashed=(CONTRIBUTED,)))
    with right:
        st.subheader("ポートフォリオとベンチマーク")
        st.dataframe(comparison(summary, bench_summary, method, benchmark))
        st.caption("時間加重の指標は拠出の影響を除いた運用成績。「拠出額に対するリターン」は拠出総額に対する損益。")

    left, right = st.columns(2)
    with left:
        st.subheader("ドローダウン（時間加重）")
        dd = pd.DataFrame({bench_name: cp.nav_drawdown(bench_result), PORTFOLIO: cp.nav_drawdown(result)})
        st.plotly_chart(charts.lines(dd, colors, "高値からの下落率", ".0%"))
    with right:
        st.subheader("配分の推移")
        drift = result.weights(p)
        st.plotly_chart(charts.stacked_area(drift, charts.ASSET_COLORS, "比率"))
        if len(weights) > 1:
            d = cp.diversification(p, dict(weights))
            st.caption(f"比率を一定とした場合のボラティリティ {common.fmt_pct(d['portfolio_volatility'])}"
                       f"（各資産の加重平均 {common.fmt_pct(d['weighted_average_volatility'])}、"
                       f"分散投資で {common.fmt_pct(d['volatility_reduction'], 1)} 低下）。")

    common.data_view(value, "資産額", "portfolio_value.csv")
    common.data_view(drift, "配分の推移", "portfolio_weights.csv")


if __name__ == "__main__":
    render()
