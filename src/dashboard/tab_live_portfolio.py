"""⑦ Live Portfolio: holdings, value, profit and system status of the real (or demo) account."""
import pandas as pd
import streamlit as st

import charts
import common
import live_common as lc

import bridge  # noqa: E402  (src/live, put on sys.path by live_common)
import real_ledger as rl  # noqa: E402

VALUE, DEPOSITS = "評価額", "入金累計"


def render() -> None:
    st.title("⑦ Live Portfolio（実資産）")
    acc = lc.account()
    if not lc.need_market(acc):
        return
    try:
        # also rewrites the derived holdings.csv, cash.csv and valuations.csv
        result = bridge.recalculate(acc.store, acc.market)
        tx, state, valuations = result["transactions"], result["state"], result["valuations"]
    except rl.LedgerError as e:
        st.error(f"取引履歴に矛盾があり、残高を計算できません。transactions.csv を確認してください。\n\n{e}")
        return

    status = bridge.status(acc.store, acc.market, acc.today)
    if tx.empty:
        st.info("まだ取引がありません。⑧ Today's Decision で入金を記録すると、ここに表示されます。")
    else:
        close = acc.market.close.iloc[-1]
        fx_rate = float(acc.market.fx.dropna().iloc[-1])
        s = rl.summary(state, close, fx_rate)
        twr = valuations["growth"].iloc[-1] - 1 if len(valuations) else float("nan")
        cards = st.columns(6)
        for col, (label, value, help_text) in zip(cards, [
                ("評価額", f"{s['total_value_jpy']:,.0f} 円", "保有資産の評価額と現金の合計"),
                ("現金", f"{s['cash_jpy']:,.0f} 円", "評価額のうち、まだ投資していない円"),
                ("入金累計", f"{s['net_deposits_jpy']:,.0f} 円", "入金 − 出金"),
                ("損益", f"{s['total_pnl_jpy']:+,.0f} 円", "評価額 − 入金累計（実現・含み・配当・コストを含む）"),
                ("含み損益", f"{s['unrealized_pnl_jpy']:+,.0f} 円", "保有中の資産の評価額 − 取得原価"),
                ("時間加重リターン", common.fmt_pct(twr), "入出金の影響を除いた運用成績")]):
            col.metric(label, value, help=help_text, border=True)
        st.caption(f"価格は {acc.market.as_of.date()} の終値、USD/JPY {fx_rate:.2f}（{acc.market.fx.dropna().index[-1].date()}）。"
                   f"実現損益 {s['realized_pnl_jpy']:+,.0f} 円、配当 {s['total_dividends_jpy']:,.0f} 円、"
                   f"手数料 {s['total_fees_jpy']:,.0f} 円、税金 {s['total_taxes_jpy']:,.0f} 円。"
                   "時間加重リターンは入出金の影響を除いた運用成績。")

        st.subheader("保有資産")
        holdings = rl.holdings_table(state, close, fx_rate)
        common.table(holdings[["quantity", "price", "value_jpy", "weight", "cost_basis_jpy", "average_cost_jpy",
                               "unrealized_pnl_jpy", "unrealized_return"]],
                     labels={"quantity": "数量", "price": "価格（USD）", "value_jpy": "評価額（円）", "weight": "比率",
                             "cost_basis_jpy": "取得原価（円）", "average_cost_jpy": "平均取得単価（円）",
                             "unrealized_pnl_jpy": "含み損益（円）", "unrealized_return": "含み損益率"},
                     percent=("weight", "unrealized_return"), money=("value_jpy", "cost_basis_jpy",
                                                                     "average_cost_jpy", "unrealized_pnl_jpy"),
                     decimals={"quantity": 4, "price": 2})
        st.caption("取得原価は手数料を含む移動平均（円）。比率は現金を含む評価額に対する割合。")

        left, right = st.columns([3, 2])
        with left:
            st.subheader("評価額と入金累計")
            frame = valuations[["total_jpy", "net_deposits_jpy"]].rename(columns={"total_jpy": VALUE,
                                                                                   "net_deposits_jpy": DEPOSITS})
            st.plotly_chart(charts.lines(frame, {VALUE: charts.SERIES_COLORS[0], DEPOSITS: charts.MUTED},
                                         "円", ",.0f", dashed=(DEPOSITS,)))
        with right:
            st.subheader("資産の比率")
            weights = pd.DataFrame({"asset": [*holdings.index, "現金"],
                                    "weight": [*holdings["weight"], s["cash_jpy"] / s["total_value_jpy"]]})
            fig = charts.horizontal_bars(weights, "weight", "asset", None, "比率")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig)
        common.data_view(valuations, "評価額の推移", "live_valuations.csv")

    st.subheader("システムの状態")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("価格データ", str(status["market_as_of"]), f"{status['market_age_business_days']} 営業日前",
              delta_color="off", delta_arrow="off", border=True)
    c2.metric("承認待ちの提案", status["pending_decisions"], border=True)
    c3.metric("未約定の注文", status["open_orders"], border=True)
    c4.metric("緊急停止", "停止中" if status["emergency_stop"] else "解除", border=True)
    last = status["last_proposal"]
    if last:
        st.caption(f"最新の提案：{last['proposed_at']}（データ {last['as_of']}、{last['strategy']}、"
                   f"リスクチェック {last['risk']}、判断 {last['decision']}）")
    if status["errors"]:
        with st.expander(f"リスクチェックで止めた記録（{len(status['errors'])} 件）"):
            errors = pd.DataFrame([{"timestamp": e["timestamp"], "as_of": e.get("as_of"),
                                    "理由": " / ".join(f["detail"] for f in e["failures"])} for e in status["errors"]])
            st.dataframe(errors.iloc[::-1], hide_index=True)
    if not tx.empty:
        with st.expander(f"取引履歴（{len(tx)} 件）"):
            st.dataframe(tx.assign(date=tx["date"].dt.date), hide_index=True)


if __name__ == "__main__":
    render()
