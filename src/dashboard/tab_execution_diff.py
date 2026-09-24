"""⑨ Research vs Real: execution costs, backtest vs paper vs real, attribution and taxes."""
import pandas as pd
import streamlit as st

import charts
import common
import compute as cp
import live_common as lc

import attribution as at  # noqa: E402  (src/live, put on sys.path by live_common)
import audit  # noqa: E402
import bridge  # noqa: E402
import execution_analyzer as ea  # noqa: E402
import fx_and_taxes as fxt  # noqa: E402
import paper_engine as paper  # noqa: E402
import real_ledger as rl  # noqa: E402
import signal_pipeline as sp  # noqa: E402

from engine import CostModel  # noqa: E402

SERIES = {"backtest": "バックテスト", "paper": "ペーパー（全シグナルに従う）", "real": "実運用"}
COLORS = {SERIES["backtest"]: charts.MUTED, SERIES["paper"]: charts.SERIES_COLORS[1], SERIES["real"]: charts.SERIES_COLORS[0]}
EFFECT_LABELS = {"asset": "資産のリターン（目標比率）", "allocation": "配分の差（目標とのずれ）",
                 "rebalancing": "リバランス（前回売買後の値動きによるずれ）", "fx": "為替", "dividend": "配当",
                 "fee_tax": "手数料・税金", "execution": "約定（終値・仲値との差）"}
COST_LABELS = {"date": "約定日", "asset": "資産", "side": "売買", "quantity": "数量", "price_actual": "約定価格",
               "price_backtest": "バックテストの価格（終値）", "fx_applied": "適用レート", "fx_mid": "仲値",
               "price_cost_jpy": "価格差のコスト", "fees_jpy": "手数料", "fx_cost_jpy": "為替コスト",
               "execution_cost_jpy": "執行コスト合計", "delay_days": "遅れ（営業日）"}


def cost_table(frame: pd.DataFrame) -> None:
    common.table(frame[list(COST_LABELS)], labels=COST_LABELS, hide_index=True,
                 money=("price_cost_jpy", "fees_jpy", "fx_cost_jpy", "execution_cost_jpy"),
                 decimals={"quantity": 4, "price_actual": 2, "price_backtest": 2, "fx_applied": 2, "fx_mid": 2})


def render() -> None:
    st.title("⑨ Research vs Real（研究と実運用の差）")
    acc = lc.account()
    if not lc.need_market(acc):
        return
    try:
        result = bridge.recalculate(acc.store, acc.market)
    except rl.LedgerError as e:
        st.error(f"取引履歴に矛盾があります：{e}")
        return
    tx, paper_tx = result["transactions"], result["paper_transactions"]
    if tx.empty:
        st.info("まだ取引がありません。")
        return

    queue = audit.load_queue(acc.store)
    real_costs = ea.execution_costs(tx, acc.market, dict(zip(queue["order_id"], queue["date"])))
    log = paper.load(acc.store)
    paper_costs = ea.execution_costs(paper_tx, acc.market, dict(zip(log["order_id"], log["signal_date"])))

    # --- execution --------------------------------------------------------------------------------
    st.subheader("執行コスト：バックテストの想定との差")
    st.caption("執行コスト = |約定価格 − バックテストの価格| × 数量 × 仲値 + 手数料 + 為替コスト（数量 × 約定価格 × |適用レート − 仲値|）。"
               "バックテストはシグナルの翌営業日の終値で約定する前提なので、約定日の終値と比べます。"
               "遅れは、その前提の日から何営業日後に約定したか。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("実運用の執行コスト合計", f"{real_costs['execution_cost_jpy'].sum():,.0f} 円", border=True)
    c2.metric("うち為替コスト", f"{real_costs['fx_cost_jpy'].sum():,.0f} 円", border=True)
    c3.metric("うち手数料", f"{real_costs['fees_jpy'].sum():,.0f} 円", border=True)
    c4.metric("平均の遅れ", f"{real_costs['delay_days'].mean():.1f} 営業日" if len(real_costs) else "−", border=True)
    tab_real, tab_paper = st.tabs([f"実運用（{len(real_costs)} 件）", f"ペーパー（{len(paper_costs)} 件）"])
    with tab_real:
        cost_table(real_costs)
    with tab_paper:
        cost_table(paper_costs)
        st.caption("ペーパーは翌営業日の終値に設定のスリッページ・為替スプレッド・手数料を加えて約定させるので、"
                   "差は設定したコストそのものです。")

    # --- performance ----------------------------------------------------------------------------
    st.subheader("バックテスト vs ペーパー vs 実運用")
    spec = sp.strategy_spec(acc.settings)
    real_r = result["valuations"]["daily_return"]
    paper_r = result["paper_valuations"]["daily_return"]
    series = {SERIES["real"]: real_r, SERIES["paper"]: paper_r}
    ratios = {SERIES["real"]: ea.cost_ratio(real_costs, result["valuations"]),
              SERIES["paper"]: ea.cost_ratio(paper_costs, result["paper_valuations"])}
    if spec["type"] != "ml":
        bt_cfg = cp.load_config("backtest")
        costs = CostModel(commission_rate=bt_cfg["costs"]["commission_rate"],
                          commission_min=bt_cfg["costs"]["commission_min"], slippage_rate=bt_cfg["costs"]["slippage_rate"])
        start = real_r.dropna().index[0]
        bt_r, bt_ratio = ea.backtest_returns(acc.market, spec, start, acc.market.as_of, costs)
        series = {SERIES["backtest"]: bt_r, **series}
        ratios[SERIES["backtest"]] = bt_ratio
    else:
        st.caption("戦略が ML のため、バックテストの行は Phase 7 のレポートを参照してください。")
    table = ea.comparison(series, ratios)
    common.table(table, labels={"start": "開始", "end": "終了", "days": "日数", "total_return": "累積リターン",
                                "cagr": "CAGR", "volatility": "年率ボラティリティ", "max_drawdown": "最大DD",
                                "sharpe_ratio": "Sharpe（無リスク 0）", "cost_ratio": "コスト ÷ 平均資産"},
                 percent=("total_return", "cagr", "volatility", "max_drawdown", "cost_ratio"),
                 decimals={"sharpe_ratio": 2})
    common_start = table["start"].max()
    growth = pd.DataFrame({k: (1 + v.loc[pd.Timestamp(common_start):].fillna(0)).cumprod()
                           for k, v in series.items()}).dropna()
    st.plotly_chart(charts.lines(growth, COLORS, "1 円あたりの価値（時間加重、円建て）", ".3f"))
    st.caption("すべて円建ての時間加重リターン（入出金の影響を除く）。バックテストは同じ戦略を Phase 5 のエンジンで常に全額投資し、"
               "同じ為替で円に換算。実運用とペーパーは、整数株で買えない現金を持つ分だけ差が出ます。"
               "1 年未満の CAGR は短期の値動きを年率に引き延ばした値なので、参考程度に見てください。")

    # --- attribution ------------------------------------------------------------------------------
    st.subheader("損益の要因分解（実運用）")
    daily = at.daily_attribution(tx, acc.market, audit.approved_targets(acc.store))
    totals = at.totals(daily)
    effects = pd.DataFrame({"effect": [EFFECT_LABELS[e] for e in at.EFFECTS], "jpy": [totals[e] for e in at.EFFECTS]})
    left, right = st.columns([2, 3])
    with left:
        fig = charts.horizontal_bars(effects, "jpy", "effect", None, "円")
        fig.update_xaxes(tickformat=",.0f")
        st.plotly_chart(fig)
        st.caption(f"合計 {totals[at.EFFECTS].sum():+,.0f} 円 = 損益（評価額 − 入金累計）{totals['profit']:+,.0f} 円"
                   f"（説明できない差 {totals['unexplained']:+.6f} 円）。")
    with right:
        cumulative = daily[at.EFFECTS].cumsum().rename(columns=EFFECT_LABELS)
        colors = dict(zip(EFFECT_LABELS.values(), charts.SERIES_COLORS + [charts.MUTED]))
        st.plotly_chart(charts.lines(cumulative, colors, "累積（円）", ",.0f"))
    with st.expander("要因の定義"):
        st.markdown("- **資産のリターン**：その時点の目標比率（承認した提案）で持っていた場合の値動き（ドル建て）\n"
                    "- **配分の差**：直前の売買の後の比率と目標比率の差による値動き（整数株、買いのみ、却下・遅れ）\n"
                    "- **リバランス**：直前の売買の後の値動きで比率がずれた分\n"
                    "- **為替**：保有資産の円換算の変化\n"
                    "- **配当**・**手数料・税金**：台帳の記録そのもの\n"
                    "- **約定**：実際の約定価格・レートと、その日の終値・仲値との差\n\n"
                    "7 つの合計は、毎日の損益（評価額の変化 − 入出金）と一致します（テストで確認）。")

    # --- taxes -------------------------------------------------------------------------------------
    st.subheader("税金と経費（概算）")
    taxes = fxt.TaxSettings.from_config(acc.settings)
    state = result["state"]
    yearly = fxt.annual_capital_gains_tax(state.realized, taxes)
    close = acc.market.close.iloc[-1]
    fx_rate = float(acc.market.fx.dropna().iloc[-1])
    holdings = rl.holdings_table(state, close, fx_rate)
    ratios_cfg = acc.settings["expense_ratios"]
    drag = pd.DataFrame({"評価額（円）": holdings["value_jpy"],
                         "経費率（年）": [ratios_cfg.get(a, float("nan")) for a in holdings.index]})
    drag["年間の経費（概算、円）"] = drag["評価額（円）"] * drag["経費率（年）"]
    left, right = st.columns(2)
    with left:
        st.markdown(f"**売却益の税金（口座：{acc.settings['account']['type']}）**")
        if yearly.empty:
            st.caption("売却の記録はありません。")
        else:
            st.dataframe(yearly, hide_index=True)
        st.caption("特定口座は同じ年の損益を通算し、20.315%。NISA は非課税。配当は米国で 10% 源泉徴収され、"
                   "特定口座ではさらに残りに 20.315%（外国税額控除は含めていない）。")
    with right:
        st.markdown("**経費率（ETF の価格にすでに含まれる）**")
        common.table(drag, percent=("経費率（年）",), money=("評価額（円）", "年間の経費（概算、円）"))
        st.caption("経費率は ETF の価格から毎日差し引かれているので、損益から二重に引かない。参考として年額を示す。")


if __name__ == "__main__":
    render()
