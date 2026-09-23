"""② Asset Analysis: cumulative, rolling and drawdown series, yearly returns and correlation."""
import streamlit as st

import charts
import common
import compute as cp
import metrics as m

WINDOWS = {"3か月（63日）": 63, "6か月（126日）": 126, "1年（252日）": 252}
FREQUENCIES = {"日次": "daily", "月次": "monthly"}


def render() -> None:
    st.title("② Asset Analysis（資産分析）")
    assets = common.need_assets()
    if assets is None:
        return
    p = common.period_prices(assets, min_days=3)
    if p is None:
        return
    st.caption(f"{p.index[0].date()} 〜 {p.index[-1].date()}（{len(p):,} 取引日）。資産と期間はサイドバーで選びます。")

    window_label = st.segmented_control("ローリングの窓", list(WINDOWS),
                                        key=common.persistent("asset_window", "1年（252日）"))
    common.remember("asset_window")
    window = WINDOWS.get(window_label, 252)

    top_left, top_right = st.columns(2)
    with top_left:
        st.subheader("累積リターン")
        cumulative = m.cumulative_returns(p)
        st.plotly_chart(charts.lines(cumulative, charts.ASSET_COLORS, "累積リターン", ".0%"))
    with top_right:
        st.subheader("ドローダウン")
        drawdown = m.drawdowns(p)
        st.plotly_chart(charts.lines(drawdown, charts.ASSET_COLORS, "高値からの下落率", ".0%"))

    if len(p) <= window:
        st.warning(f"ローリング指標には {window + 1} 取引日以上が必要です（選んだ期間は {len(p)} 日）。"
                   "期間を広げるか、窓を短くしてください。")
        rolling_return = rolling_vol = None
    else:
        bottom_left, bottom_right = st.columns(2)
        with bottom_left:
            st.subheader("ローリング・リターン（年率）")
            st.caption(f"各日から過去 {window} 取引日のリターンを年率換算。")
            rolling_return = m.rolling_return(p, window)
            st.plotly_chart(charts.lines(rolling_return, charts.ASSET_COLORS, "年率リターン", ".0%"))
        with bottom_right:
            st.subheader("ローリング・ボラティリティ（年率）")
            st.caption(f"各日から過去 {window} 取引日の日次リターンの標準偏差を年率換算。")
            rolling_vol = m.rolling_volatility(p, window)
            st.plotly_chart(charts.lines(rolling_vol, charts.ASSET_COLORS, "年率ボラティリティ", ".0%"))

    st.subheader("年別リターン")
    st.caption("最初と最後の年は、期間に含まれる日だけのリターン。")
    yearly = m.period_returns(p, "YE")
    yearly.index = yearly.index.year.rename("年")
    common.table(yearly, percent=tuple(yearly.columns))

    st.subheader("相関行列")
    if len(assets) < 2:
        st.info("相関行列には資産を 2 つ以上選んでください。")
        corr = None
    else:
        freq_label = st.segmented_control("リターンの頻度", list(FREQUENCIES),
                                          key=common.persistent("asset_corr_freq", "日次"))
        common.remember("asset_corr_freq")
        frequency = FREQUENCIES.get(freq_label, "daily")
        corr = cp.correlation(p, frequency)
        if corr.isna().all().all():
            st.warning("この期間では相関を計算できません（月次なら 3 か月以上必要です）。")
        else:
            st.plotly_chart(charts.heatmap(corr, height=120 + 50 * len(assets)))

    common.data_view(cumulative, "累積リターン", "assets_cumulative.csv")
    common.data_view(drawdown, "ドローダウン", "assets_drawdown.csv")
    if rolling_return is not None:
        common.data_view(rolling_return, "ローリング・リターン", "assets_rolling_return.csv")
        common.data_view(rolling_vol, "ローリング・ボラティリティ", "assets_rolling_volatility.csv")
    if corr is not None:
        common.data_view(corr, "相関行列", "assets_correlation.csv")


if __name__ == "__main__":
    render()
