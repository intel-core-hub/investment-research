"""⑤ Statistical Analysis: Sharpe uncertainty, bootstrap with Holm correction, risk and distribution."""
import pandas as pd
import streamlit as st

import charts
import common
import compute as cp

CONFIDENCES = {"90%": 0.90, "95%": 0.95, "99%": 0.99}
RESAMPLES = [1000, 2000, 5000, 10000]
STATISTICS = {"Sharpe": "sharpe_ratio", "CAGR（幾何平均）": "cagr", "平均リターン（年率）": "mean_return_annualized"}
METHODS = {"bootstrap_iid": "1日単位", "bootstrap_block": "ブロック"}


def asset_label(ticker: str) -> str:
    return f"{ticker}（資産そのもの）"


@st.cache_data(show_spinner="ブートストラップを計算中…")
def bootstrap(returns: pd.DataFrame, rf: pd.Series, benchmark: str, resamples: int, seed: int, confidence: float,
              block_size: int) -> pd.DataFrame:
    return cp.bootstrap_table(returns, rf, benchmark, resamples, seed, confidence, block_size)


def render() -> None:
    st.title("⑤ Statistical Analysis（統計的検証）")
    s = common.settings()
    bt = common.backtest_settings()
    boot_cfg = cp.load_config("statistics")["bootstrap"]
    names = common.strategy_names()
    candidates = names + [asset_label(t) for t in s.assets]

    with st.container(border=True):
        chosen = st.multiselect("比較する系列（④ の戦略とサイドバーの資産）", candidates,
                                key=common.persistent("st_series", names))
        chosen = [c for c in chosen if c in candidates]
        c1, c2, c3, c4, c5 = st.columns(5)
        benchmark = c1.selectbox("ベンチマーク", chosen or ["−"],
                                 index=chosen.index(names[0]) if names[0] in chosen else 0, key="st_benchmark")
        focus = c2.selectbox("詳しく見る系列", chosen or ["−"], key="st_focus")
        confidence = CONFIDENCES[c3.selectbox("信頼水準", list(CONFIDENCES), index=1, key="st_confidence")]
        resamples = c4.select_slider("ブートストラップ回数", RESAMPLES, key=common.persistent("st_resamples",
                                                                                      boot_cfg["resamples"]))
        block = c5.number_input("ブロック長（取引日）", min_value=2, max_value=126, step=1,
                                key=common.persistent("st_block", boot_cfg["block_size"]))
    common.remember("st_series", "st_resamples", "st_block")
    if not chosen:
        st.warning("系列を 1 つ以上選んでください。")
        return

    strategies = [c for c in chosen if c in names]
    equities = {}
    if strategies:
        try:
            results = common.run_strategies(s.start, s.end, tuple(strategies), bt)
        except ValueError as e:
            st.error(f"バックテストを実行できません：{e}。サイドバーで期間を広げてください。")
            return
        equities.update({n: r.equity for n, r in results.items()})
    period = cp.trading_days(common.prices(), s.start, s.end)
    for t in s.assets:
        if asset_label(t) in chosen:
            equities[asset_label(t)] = period[t]
    returns = cp.aligned_returns(pd.DataFrame(equities))[chosen]
    if len(returns) < 30:
        st.warning(f"共通の日次リターンが {len(returns)} 日しかありません。30 日以上になるよう期間を広げてください。")
        return
    rf = cp.risk_free_returns(common.prices())
    st.caption(f"全系列に共通の {returns.index[0].date()} 〜 {returns.index[-1].date()}（日次リターン {len(returns):,} 日）。"
               f"戦略は ④ の設定（初期資金 {bt.initial_cash:,.0f}、手数料 {bt.commission_rate:.3%}、"
               f"スリッページ {bt.slippage_rate:.3%}）で実行。無リスク金利は BIL。")
    alpha = 1 - confidence

    # --- Sharpe -------------------------------------------------------------------------------
    st.subheader("Sharpe ratio の不確実性")
    sharpe = cp.sharpe_table(returns, rf, confidence)
    common.table(sharpe[["sharpe_ratio", "se_lo", "se_mertens", "ci_lower_mertens", "ci_upper_mertens", "psr",
                         "observations"]],
                 labels={"sharpe_ratio": "Sharpe", "se_lo": "SE（Lo, i.i.d.）", "se_mertens": "SE（Mertens, 非正規）",
                         "ci_lower_mertens": f"{confidence:.0%} 区間 下限", "ci_upper_mertens": f"{confidence:.0%} 区間 上限",
                         "psr": "PSR（真の Sharpe > 0 の確率）", "observations": "日数"},
                 decimals={c: 3 for c in ("sharpe_ratio", "se_lo", "se_mertens", "ci_lower_mertens",
                                          "ci_upper_mertens", "psr")})
    st.caption("年率換算。Mertens の標準誤差は歪度と尖度を考慮します。PSR は Bailey & López de Prado (2012)。")

    boot = bootstrap(returns, rf, benchmark, resamples, boot_cfg["seed"], confidence, int(block))
    left, right = st.columns(2)
    with left:
        st.markdown(f"**各系列の Sharpe と {confidence:.0%} 区間（Mertens）**")
        st.plotly_chart(charts.intervals(sharpe, "sharpe_ratio", "ci_lower_mertens", "ci_upper_mertens", "Sharpe"))
    with right:
        st.markdown(f"**{focus} の区間：手法による違い**")
        methods = pd.DataFrame([
            {"method": "Lo（i.i.d.）", "estimate": sharpe.loc[focus, "sharpe_ratio"],
             "lower": sharpe.loc[focus, "ci_lower_lo"], "upper": sharpe.loc[focus, "ci_upper_lo"]},
            {"method": "Mertens", "estimate": sharpe.loc[focus, "sharpe_ratio"],
             "lower": sharpe.loc[focus, "ci_lower_mertens"], "upper": sharpe.loc[focus, "ci_upper_mertens"]},
        ] + [{"method": f"ブートストラップ（{METHODS[r.method]}）", "estimate": r.estimate, "lower": r.ci_lower,
              "upper": r.ci_upper}
             for r in boot[(boot.strategy == focus) & (boot.statistic == "sharpe_ratio")].itertuples()]
        ).set_index("method")
        st.plotly_chart(charts.intervals(methods, "estimate", "lower", "upper", "Sharpe"))

    # --- bootstrap vs benchmark -------------------------------------------------------------------
    st.subheader(f"ベンチマーク（{benchmark}）との差：ブートストラップと Holm 補正")
    statistic_label = st.segmented_control("指標", list(STATISTICS), key=common.persistent("st_statistic", "Sharpe"))
    common.remember("st_statistic")
    statistic = STATISTICS.get(statistic_label, "sharpe_ratio")
    diff = boot[boot.statistic == f"{statistic}_difference_vs_benchmark"].copy()
    if diff.empty:
        st.info("ベンチマーク以外の系列を選ぶと、差の検定を表示します。")
    else:
        diff["method"] = diff["method"].map(METHODS)
        diff["significant"] = diff["p_value_holm"].lt(alpha).map({True: "区別できる", False: "区別できない"})
        view = diff.set_index(["strategy", "method"])[["estimate", "ci_lower", "ci_upper", "p_value", "p_value_holm",
                                                       "significant"]]
        is_pct = statistic != "sharpe_ratio"
        common.table(view, labels={"estimate": "差（系列 − ベンチマーク）", "ci_lower": "区間 下限",
                                   "ci_upper": "区間 上限", "p_value": "p 値", "p_value_holm": "p 値（Holm 補正後）",
                                   "significant": f"判定（有意水準 {alpha:.0%}）"},
                     percent=("estimate", "ci_lower", "ci_upper") if is_pct else (),
                     decimals={"estimate": 3, "ci_lower": 3, "ci_upper": 3, "p_value": 3, "p_value_holm": 3})
        block_rows = diff[diff.method == METHODS["bootstrap_block"]].set_index("strategy")
        st.plotly_chart(charts.intervals(block_rows, "estimate", "ci_lower", "ci_upper",
                                         f"{statistic_label} の差（ブロック・ブートストラップ）"))
        st.caption(f"日付を全系列で共通に再標本化（{resamples:,} 回、seed {boot_cfg['seed']}）。ブロック長 {int(block)} 日は"
                   "自己相関を残すため。p 値は両側。Holm 補正は、同じ手法・指標でベンチマークと比べた"
                   f"{len(block_rows)} 件をまとめて補正します。")

    # --- mean return, risk and distribution ---------------------------------------------------------
    st.subheader("平均リターンの信頼区間：通常と Newey-West")
    means = cp.mean_return_table(returns, confidence)
    means["method"] = means["method"].map({"normal_iid": "通常（i.i.d.）", "newey_west": "Newey-West（HAC）"})
    common.table(means.set_index(["series", "method"])[["estimate", "standard_error", "ci_lower", "ci_upper", "lag"]],
                 labels={"estimate": "平均（年率）", "standard_error": "標準誤差", "ci_lower": "区間 下限",
                         "ci_upper": "区間 上限", "lag": "ラグ"},
                 percent=("estimate", "standard_error", "ci_lower", "ci_upper"))

    st.subheader("リスクとリターン分布")
    risk = cp.risk_table(returns)
    common.table(risk, labels={"var_95": "VaR 95%", "cvar_95": "CVaR 95%", "var_99": "VaR 99%", "cvar_99": "CVaR 99%",
                               "skewness": "歪度", "excess_kurtosis": "超過尖度", "observations": "日数"},
                 percent=("var_95", "cvar_95", "var_99", "cvar_99"),
                 decimals={"skewness": 2, "excess_kurtosis": 2})
    st.caption("ヒストリカル法の 1 日 VaR / CVaR（損失を正の数で表示）。超過尖度が正なら、正規分布より裾が厚い。")
    color = {**charts.ASSET_COLORS, **common.strategy_colors()}.get(focus.split("（")[0], charts.SERIES_COLORS[0])
    markers = {f"VaR {int(c * 100)}%": -risk.loc[focus, f"var_{int(c * 100)}"] for c in (0.95, 0.99)
               if pd.notna(risk.loc[focus, f"var_{int(c * 100)}"])}
    st.markdown(f"**{focus} の日次リターン分布**")
    st.plotly_chart(charts.histogram(returns[focus], markers, color))

    common.data_view(returns, "日次リターン", "statistics_returns.csv")
    common.data_view(boot, "ブートストラップの全結果", "statistics_bootstrap.csv")


if __name__ == "__main__":
    render()
