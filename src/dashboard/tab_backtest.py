"""④ Backtest: the Phase 5 strategies with adjustable capital, costs and parameters."""
import pandas as pd
import streamlit as st

import charts
import common
import compute as cp
import metrics as m

METRICS = {"cagr": "CAGR", "volatility_annualized": "年率ボラティリティ", "max_drawdown": "最大DD",
           "sharpe_ratio": "Sharpe", "sortino_ratio": "Sortino", "calmar_ratio": "Calmar", "win_rate": "勝率",
           "profit_factor": "Profit Factor", "closed_trades": "完結した売買", "transactions": "約定回数",
           "total_costs": "コスト合計", "final_equity": "最終資産"}
NO_ROUND_TRIPS = {"buy_and_hold": "最初に買ったまま保有し続けるため、売買が完結しません。",
                  "periodic_rebalance": "比率を調整するだけでポジションを閉じないため、売買の入口と出口がありません。"}
TRANSACTION_LABELS = {"date": "約定日", "signal_date": "シグナル日", "ticker": "銘柄", "side": "売買",
                      "quantity": "数量", "close": "終値", "fill_price": "約定価格", "gross_value": "約定金額",
                      "commission": "手数料", "slippage_cost": "スリッページ", "cash_after": "約定後の現金"}


def settings_form(defaults: cp.BacktestSettings) -> cp.BacktestSettings:
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        cash = c1.number_input("初期資金", min_value=100.0, step=1000.0, format="%.0f",
                               key=common.persistent("bt_cash", defaults.initial_cash),
                               help="正規化した金額（USD 建て）。実際の資産額ではありません。")
        commission = c2.number_input("手数料（約定金額の %）", min_value=0.0, max_value=5.0, step=0.01, format="%.3f",
                                     key=common.persistent("bt_commission", defaults.commission_rate * 100))
        slippage = c3.number_input("スリッページ（%）", min_value=0.0, max_value=5.0, step=0.01, format="%.3f",
                                   key=common.persistent("bt_slippage", defaults.slippage_rate * 100),
                                   help="買いは終値より高く、売りは終値より安く約定します。")
        with st.expander("戦略のパラメータ"):
            p1, p2, p3 = st.columns(3)
            window = p1.number_input("トレンド：移動平均の日数", min_value=2, max_value=252, step=10,
                                     key=common.persistent("bt_ma_window", defaults.ma_window))
            lookback = p2.number_input("モメンタム：過去リターンの日数", min_value=20, max_value=252, step=21,
                                       key=common.persistent("bt_lookback", defaults.momentum_lookback))
            top_n = p3.number_input("モメンタム：保有する資産数", min_value=1, max_value=4, step=1,
                                    key=common.persistent("bt_top_n", defaults.momentum_top_n))
            st.caption("取引開始前に 252 取引日の履歴を確保するため、日数の上限は 252 です。")
    common.remember("bt_cash", "bt_commission", "bt_slippage", "bt_ma_window", "bt_lookback", "bt_top_n")
    return cp.BacktestSettings(initial_cash=float(cash), commission_rate=commission / 100,
                               slippage_rate=slippage / 100, ma_window=int(window),
                               momentum_lookback=int(lookback), momentum_top_n=int(top_n))


def render() -> None:
    st.title("④ Backtest（バックテスト）")
    st.caption("Phase 5 のエンジン：終値でシグナルを判断し、翌取引日の終値で約定（手数料・スリッページ込み）。"
               "ここで設定した初期資金・コスト・パラメータは ⑤ と ⑥ でも使います。")
    names = common.strategy_names()
    chosen = st.multiselect("戦略", names, key=common.persistent("bt_strategies", names))
    common.remember("bt_strategies")
    settings = settings_form(cp.default_backtest_settings())
    st.session_state[common.BACKTEST_SETTINGS_KEY] = settings
    if not chosen:
        st.warning("戦略を 1 つ以上選んでください。")
        return

    s = common.settings()
    try:
        results = common.run_strategies(s.start, s.end, tuple(chosen), settings)
    except ValueError as e:
        st.error(f"バックテストを実行できません：{e}。サイドバーで期間を広げてください。")
        return
    days = next(iter(results.values())).equity.index
    if days[0] > s.start:
        st.info(f"取引開始日は {days[0].date()} です（戦略が必要とする 252 取引日の履歴を確保するため）。")
    if len(days) < 21:
        st.warning(f"取引期間が {len(days)} 日しかありません。指標はほとんど意味を持ちません。")

    table = cp.backtest_table(results, settings, cp.risk_free_returns(common.prices()))
    st.subheader("主要指標")
    common.table(table[list(METRICS)], labels=METRICS,
                 percent=("cagr", "volatility_annualized", "max_drawdown", "win_rate"),
                 decimals={"sharpe_ratio": 2, "sortino_ratio": 2, "calmar_ratio": 2, "profit_factor": 2},
                 money=("total_costs", "final_equity"))
    st.caption("勝率・Profit Factor は、ポジションを完全に売り切った売買（往復）だけで計算します。"
               "Buy & Hold と定期リバランスには往復の売買がないため計算せず、「None」と表示します。")

    colors = common.strategy_colors()
    equity = pd.DataFrame({n: r.equity for n, r in results.items()})
    left, right = st.columns(2)
    with left:
        st.subheader("資産曲線（Equity Curve）")
        log = st.toggle("対数目盛", key=common.persistent("bt_log", False))
        common.remember("bt_log")
        st.plotly_chart(charts.lines(equity, colors, "資産額", ",.0f", log=log))
    with right:
        st.subheader("ドローダウン")
        drawdown = m.drawdowns(equity)
        st.plotly_chart(charts.lines(drawdown, colors, "高値からの下落率", ".0%"))

    st.subheader("トレードログ")
    name = st.selectbox("戦略", chosen, key="bt_log_strategy")
    result = results[name]
    tx = result.transactions
    tab_fills, tab_trips = st.tabs([f"約定（{len(tx)} 件）", "往復の売買"])
    with tab_fills:
        common.table(tx, labels=TRANSACTION_LABELS, money=("gross_value", "cash_after"),
                     decimals={"quantity": 4, "close": 2, "fill_price": 2, "commission": 2, "slippage_cost": 2},
                     hide_index=True)
        st.download_button("約定ログを CSV でダウンロード", tx.to_csv(index=False).encode("utf-8"),
                           file_name=f"transactions_{name}.csv", mime="text/csv")
    with tab_trips:
        if not table.loc[name, "trade_stats_applicable"]:
            st.info(f"この戦略は往復の売買を集計しません。{NO_ROUND_TRIPS[table.loc[name, 'strategy_type']]}")
        else:
            common.table(cp.round_trips(result), percent=("return",), money=("cost", "proceeds", "profit"),
                         labels={"ticker": "銘柄", "entry_date": "買い", "exit_date": "売り切り",
                                 "holding_days": "保有日数", "cost": "買いの総額", "proceeds": "売りの総額",
                                 "profit": "損益", "return": "リターン"}, hide_index=True)

    common.data_view(equity, "資産曲線", "backtest_equity.csv")


if __name__ == "__main__":
    render()
