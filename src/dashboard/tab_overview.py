"""① Overview: headline metrics, growth and drawdowns of the assets chosen in the sidebar."""
import streamlit as st

import charts
import common
import compute as cp
import metrics as m

KPIS = [("total_return", "累積リターン", lambda v: common.fmt_pct(v, 1)), ("cagr", "CAGR", common.fmt_pct),
        ("volatility_annualized", "年率ボラ", common.fmt_pct), ("max_drawdown", "最大DD", common.fmt_pct),
        ("sharpe_ratio", "Sharpe", common.fmt_num), ("sortino_ratio", "Sortino", common.fmt_num),
        ("calmar_ratio", "Calmar", common.fmt_num)]


def render() -> None:
    st.title("① Overview（概要）")
    assets = common.need_assets()
    if assets is None:
        return
    prices = common.period_prices(min_days=3)
    if prices is None:
        return
    p = prices[assets]
    st.caption(f"{p.index[0].date()} 〜 {p.index[-1].date()}（{len(p):,} 取引日）。Sharpe・Sortino は BIL を"
               "無リスク金利とし、定義できない場合（BIL 自身など）は「−」。指標の定義は Phase 6 と同じ。")

    kpis = cp.asset_kpis(p, prices[cp.RISK_FREE])
    for ticker in assets:
        with st.container(border=True):
            cols = st.columns([0.6] + [1] * len(KPIS))
            cols[0].markdown(f"**{ticker}**")
            for col, (key, label, fmt) in zip(cols[1:], KPIS):
                col.metric(label, fmt(kpis.loc[ticker, key]))
    common.data_view(kpis, "主要指標", "overview_kpis.csv")

    left, right = st.columns(2)
    with left:
        st.subheader("累積リターン")
        scale = st.segmented_control("縦軸", ["線形（累積リターン）", "対数（資産の伸び）"],
                                     key=common.persistent("overview_scale", "線形（累積リターン）"))
        common.remember("overview_scale")
        if scale == "対数（資産の伸び）":
            fig = charts.lines(cp.growth(p), charts.ASSET_COLORS, "開始時を 1 とした価値", ".2f", log=True)
        else:
            fig = charts.lines(cp.growth(p) - 1, charts.ASSET_COLORS, "累積リターン", ".0%")
        st.plotly_chart(fig)
    with right:
        st.subheader("ドローダウン（Underwater）")
        st.caption("それまでの最高値からの下落率。")
        dd = m.drawdowns(p)
        st.plotly_chart(charts.lines(dd, charts.ASSET_COLORS, "高値からの下落率", ".0%"))
    common.data_view(cp.growth(p), "資産の伸び（開始時 = 1）", "overview_growth.csv")
    common.data_view(dd, "ドローダウン", "overview_drawdown.csv")


if __name__ == "__main__":
    render()
