"""⑧ Today's Decision: the only place where a proposal becomes an order (human in the loop).

    [今日の提案を作成] -> risk check -> [承認] / [却下] -> order queue
    -> place the order at the broker yourself -> [約定を記録] -> ledger
Nothing on this page sends an order to a broker.
"""
import json

import pandas as pd
import streamlit as st

import common
import live_common as lc

import audit  # noqa: E402  (src/live, put on sys.path by live_common)
import bridge  # noqa: E402
import real_ledger as rl  # noqa: E402
import risk_controller as rc  # noqa: E402

SIDES = {"buy": "買い", "sell": "売り"}
CASH_ACTIONS = {"DEPOSIT": "入金", "WITHDRAW": "出金", "DIVIDEND": "配当（税引前の額と源泉税）", "FEE": "手数料",
                "TAX": "税金"}


def emergency_switch(acc: lc.Account) -> bool:
    active = rc.emergency_stop_active(acc.store)
    with st.container(border=True):
        if active:
            st.error(f"緊急停止中：{acc.store.emergency_stop.read_text(encoding='utf-8').strip()}。"
                     "提案はすべて STOP になり、承認もできません。", icon=":material/block:")
            confirm = st.checkbox("停止の理由が解消したことを確認した", key="stop_clear_confirm")
            if st.button("緊急停止を解除", disabled=not confirm):
                rc.clear_emergency_stop(acc.store)
                st.rerun()
        else:
            left, right = st.columns([3, 1])
            reason = left.text_input("緊急停止の理由", placeholder="例：データがおかしい、相場が急変した",
                                     label_visibility="collapsed")
            if right.button("緊急停止", type="primary"):
                rc.set_emergency_stop(acc.store, reason or "理由の記入なし")
                st.rerun()
            st.caption("緊急停止中は、提案の作成でも承認でも処理を止めます（ファイル EMERGENCY_STOP で管理）。")
    return active


def proposal_view(acc: lc.Account, record: dict, decision: str) -> None:
    risk = record["risk"]
    status = risk["status"]
    st.markdown(f"**提案 `{record['decision_id']}`**：{record['strategy']}、データ {record['as_of']}、"
                f"リスクチェック **{status}**、判断 **{decision}**")
    signal = {k: v for k, v in record["signal"].items() if k != "type"}
    if signal:
        st.caption("シグナルの詳細：" + json.dumps(signal, ensure_ascii=False))
    assets = sorted(set(record["target_weights"]) | set(record["current_weights"]) |
                    {o["asset"] for o in record["orders"]})
    orders = {o["asset"]: o for o in record["orders"]}
    close = acc.market.close.loc[:pd.Timestamp(record["as_of"])].iloc[-1]
    table = pd.DataFrame([{
        "資産": a, "現在の比率": record["current_weights"].get(a, 0.0), "目標の比率": record["target_weights"].get(a, 0.0),
        "売買": SIDES.get(orders[a]["side"], "") if a in orders else "なし",
        "数量": orders[a]["quantity"] if a in orders else 0.0,
        "価格（USD、直近終値）": orders[a]["price"] if a in orders else close.get(a),
        "概算金額（円）": orders[a]["quantity"] * orders[a]["price"] * orders[a]["fx_rate"] if a in orders else 0.0,
    } for a in assets])
    common.table(table.set_index("資産"), percent=("現在の比率", "目標の比率"), money=("概算金額（円）",),
                 decimals={"数量": 4, "価格（USD、直近終値）": 2})
    st.caption("現在の比率は、承認済みで未約定の注文も約定したとみなした比率（現金を含む）。概算金額は手数料・為替スプレッド抜き。")
    for note in record.get("notes", []):
        st.info(note)
    checks = pd.DataFrame(risk["checks"]).rename(columns={"name": "チェック", "passed": "合格", "detail": "内容"})
    with st.expander(f"リスクチェック（{len(checks)} 項目、{status}）", expanded=status != "PASS"):
        st.dataframe(checks, hide_index=True)
    st.caption(f"data_version {record['data_version']}・strategy_version {record['strategy_version']}・"
               f"git_commit {str(record['git_commit'])[:10]}{'（未コミットの変更あり）' if record['git_dirty'] else ''}")


def decision_buttons(acc: lc.Account, record: dict, stopped: bool) -> None:
    note = st.text_input("判断のメモ（監査ログに残ります）", key=f"note_{record['decision_id']}")
    left, right, _ = st.columns([1, 1, 4])
    can_approve = record["risk"]["status"] == "PASS" and not stopped
    if left.button("承認（Approve）", type="primary", disabled=not can_approve, key=f"approve_{record['decision_id']}"):
        try:
            audit.decide(acc.store, record["decision_id"], True, note)
            st.session_state["live_message"] = ("success", "承認しました。注文キューに入れた注文を、証券会社で自分で発注してください。")
        except audit.AuditError as e:
            st.session_state["live_message"] = ("error", f"承認できません：{e}")
        st.rerun()
    if right.button("却下（Reject）", key=f"reject_{record['decision_id']}"):
        audit.decide(acc.store, record["decision_id"], False, note)
        st.session_state["live_message"] = ("info", "却下しました（監査ログに記録）。")
        st.rerun()


def fill_form(acc: lc.Account, tx: pd.DataFrame) -> None:
    queue = audit.queue_status(acc.store, tx)
    open_orders = queue[queue["status"] == "OPEN"]
    st.subheader(f"未約定の注文（{len(open_orders)} 件）")
    if open_orders.empty:
        st.caption("承認済みで、まだ約定を記録していない注文はありません。")
        return
    common.table(open_orders.set_index("order_id")[["date", "asset", "side", "quantity", "est_price_usd", "est_fx",
                                                     "est_value_jpy"]],
                 labels={"date": "シグナル日", "asset": "資産", "side": "売買", "quantity": "数量",
                         "est_price_usd": "想定価格（USD）", "est_fx": "想定レート", "est_value_jpy": "概算（円）"},
                 money=("est_value_jpy",), decimals={"quantity": 4, "est_price_usd": 2, "est_fx": 2})
    st.markdown("**約定を記録**（証券会社の約定内容をそのまま入力）")
    # outside the form, so the defaults below follow the chosen order
    order_id = st.selectbox("注文", list(open_orders["order_id"]))
    row = open_orders.set_index("order_id").loc[order_id]
    with st.form(f"fill_{order_id}"):
        c1, c2, c3 = st.columns(3)
        date = c1.date_input("約定日", value=acc.today.date())
        quantity = c2.number_input("数量", min_value=0.0, value=float(row["quantity"]), step=1.0)
        price = c3.number_input("約定価格（USD）", min_value=0.0, value=float(row["est_price_usd"]), format="%.4f")
        c4, c5, c6 = st.columns(3)
        fx_rate = c4.number_input("適用された為替レート（円/USD）", min_value=0.0, value=float(row["est_fx"]),
                                  format="%.4f")
        fees = c5.number_input("手数料（円、税込）", min_value=0.0, value=0.0)
        tax = c6.number_input("源泉徴収税（円、売りのみ）", min_value=0.0, value=0.0)
        note = st.text_input("メモ", key="fill_note")
        if st.form_submit_button("約定を記録", type="primary"):
            try:
                bridge.record_fill(acc.store, order_id, date, quantity, price, fx_rate,
                                   acc.settings["account"]["type"], fees_jpy=fees, tax_jpy=tax, note=note)
                st.session_state["live_message"] = ("success", f"{order_id} の約定を台帳に記録しました。")
                st.rerun()
            except (bridge.BridgeError, rl.LedgerError, ValueError) as e:
                st.error(f"記録できません（台帳は変更していません）：{e}")


def cash_form(acc: lc.Account) -> None:
    with st.form("cash"):
        st.markdown("**入出金・配当・手数料・税金を記録**")
        c1, c2, c3 = st.columns(3)
        action = c1.selectbox("種類", list(CASH_ACTIONS), format_func=CASH_ACTIONS.get)
        date = c2.date_input("日付", value=acc.today.date())
        amount = c3.number_input("金額（円）", min_value=0.0, step=1000.0, format="%.0f")
        c4, c5, _ = st.columns(3)
        asset = c4.text_input("資産（配当のみ）")
        tax = c5.number_input("源泉徴収税（円、配当のみ）", min_value=0.0, format="%.0f")
        note = st.text_input("メモ", key="cash_note")
        if st.form_submit_button("記録"):
            try:
                bridge.record_cash(acc.store, action, date, amount, acc.settings["account"]["type"],
                                   asset=asset.upper(), tax_jpy=tax if action == "DIVIDEND" else 0.0, note=note)
                st.session_state["live_message"] = ("success", f"{CASH_ACTIONS[action]} {amount:,.0f} 円を記録しました。")
                st.rerun()
            except (bridge.BridgeError, rl.LedgerError, ValueError) as e:
                st.error(f"記録できません（台帳は変更していません）：{e}")


def render() -> None:
    st.title("⑧ Today's Decision（承認）")
    st.caption("システムが作った提案を、リスクチェックの後で人が確認して承認・却下します。承認しても自動では発注しません。"
               "証券会社で自分で発注し、約定内容をここで記録します。")
    acc = lc.account()
    if acc.is_demo and st.button("デモ口座を初期状態に戻す"):
        lc.reset_demo()
        st.rerun()
    message = st.session_state.pop("live_message", None)
    if message:
        getattr(st, message[0])(message[1])
    if not lc.need_market(acc):
        return
    stopped = emergency_switch(acc)

    try:
        tx = rl.load_transactions(acc.store.transactions)
        rl.rebuild(tx)
    except rl.LedgerError as e:
        st.error(f"取引履歴に矛盾があるため、提案を作れません：{e}")
        return

    st.subheader("今日の提案")
    if st.button(f"今日の提案を作成（{acc.settings['signal']['strategy']}、データ {acc.market.as_of.date()}）"):
        with st.spinner("データ確認 → シグナル → 注文数量 → リスクチェック…"):
            proposal, recorded = bridge.propose(acc.store, acc.settings, acc.market, acc.today)
        st.session_state["live_message"] = (
            "info", f"提案 {proposal.decision_id}（{proposal.status}）を" + ("記録しました。" if recorded else
                                                                         "作成済みです（同じ提案は二重に記録しません）。"))
        st.rerun()

    decisions = audit.decision_table(acc.store)
    records = audit.proposals(acc.store)
    pending = decisions[decisions["decision"] == audit.PENDING] if len(decisions) else decisions
    if len(pending):
        for decision_id in pending["decision_id"]:
            with st.container(border=True):
                proposal_view(acc, records[decision_id], audit.PENDING)
                decision_buttons(acc, records[decision_id], stopped)
    elif len(decisions):
        latest = decisions.iloc[-1]
        st.caption("承認待ちの提案はありません。最新の提案：")
        with st.container(border=True):
            proposal_view(acc, records[latest["decision_id"]], latest["decision"])
    else:
        st.caption("まだ提案がありません。")

    fill_form(acc, tx)
    cash_form(acc)

    st.subheader("判断の履歴（監査ログ）")
    if len(decisions):
        st.dataframe(decisions.iloc[::-1], hide_index=True)
        st.caption("audit_logs/decisions.jsonl は追記だけで、書き換えません。提案（PROPOSED）と判断（APPROVED / REJECTED）は"
                   "別の行で、どのデータ・設定・コードから出た提案かを後から追えます。")


if __name__ == "__main__":
    render()
