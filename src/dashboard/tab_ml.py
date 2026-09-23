"""⑥ Machine Learning: Phase 7 models end to end (prediction -> Phase 5 backtest -> Phase 6 statistics)."""
import json

import pandas as pd
import streamlit as st

import charts
import common
import compute as cp

MODEL_LABELS = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
                "hist_gradient_boosting": "HistGradientBoosting（勾配ブースティング）"}
FEATURE_SETS = {"base": "Base（価格・出来高の 10 個）", "extended": "Extended（Phase 6 の指標・他資産との比較を加えた 19 個）"}
MODES = {"テスト期間（2023年〜、1回だけの評価）": "test", "walk-forward（2016年〜、毎年再学習）": "walk_forward"}
METRICS = {"roc_auc": "ROC-AUC", "accuracy": "正解率", "precision": "適合率", "recall": "再現率", "f1": "F1",
           "brier_score": "Brier", "log_loss": "log loss", "predicted_positive_rate": "上昇と予測した割合",
           "positive_rate": "実際に上昇した割合", "observations": "行数"}
INVESTMENT = [("cagr", "CAGR", "pct"), ("volatility_annualized", "年率ボラティリティ", "pct"),
              ("max_drawdown", "最大DD", "pct"), ("sharpe_ratio", "Sharpe", "num"),
              ("share_of_decisions_invested", "VOO を持った判断の割合", "pct"), ("trades", "約定回数", "int"),
              ("total_costs", "コスト合計", "money"), ("var_95", "VaR 95%（1日）", "pct"),
              ("cvar_95", "CVaR 95%", "pct"), ("var_99", "VaR 99%", "pct"), ("cvar_99", "CVaR 99%", "pct"),
              ("final_equity", "最終資産", "money")]
FORMATS = {"pct": common.fmt_pct, "num": common.fmt_num, "int": lambda v: f"{int(v):,}",
           "money": lambda v: f"{v:,.0f}"}


@st.cache_resource(show_spinner="機械学習のライブラリを読み込み中…")
def ml():
    """ml_compute is imported here, not at the top, so the page title shows while scikit-learn loads."""
    import ml_compute
    return ml_compute


def rule_colors() -> dict[str, str]:
    """Same colors as the Phase 7 report figures."""
    return {ml().BUY_AND_HOLD: "#eda100", "momentum_252d": "#e87ba4", "moving_average_200d": "#008300"}


@st.cache_resource(show_spinner="特徴量とラベルを作成中…")
def setup():
    return ml().load_setup()


@st.cache_data(show_spinner="検証期間でハイパーパラメータを比較中…")
def search(model: str, fset: str) -> pd.DataFrame:
    cfg, data = setup()
    return ml().validation_search(data, cfg, model, fset)


@st.cache_resource(show_spinner="モデルを学習中…")
def predict(model: str, fset: str, params: str):
    cfg, data = setup()
    return ml().predict(data, cfg, model, fset, json.loads(params))


@st.cache_data(show_spinner="permutation importance を計算中…")
def importance(model: str, fset: str, params: str) -> pd.DataFrame:
    cfg, data = setup()
    return ml().importance(data, cfg, predict(model, fset, params)[0], fset)


@st.cache_data(show_spinner="walk-forward：年ごとに学習し直しています…")
def walk_forward(model: str, fset: str):
    cfg, data = setup()
    return ml().walk_forward(data, cfg, model, fset)


@st.cache_data(show_spinner=False)
def baselines() -> dict:
    cfg, data = setup()
    return ml().baseline_score_table(data, cfg)


@st.cache_data(show_spinner="予測を売買シグナルに変換してバックテスト中…")
def backtest(name: str, score: pd.Series, threshold: float, settings: cp.BacktestSettings):
    cfg, data = setup()
    start, end = score.index[0], data.last
    scores = {name: score, **ml().rule_scores(data, cfg, score.index)}
    return ml().backtest(data, cfg, scores, start, end, settings.initial_cash, settings.commission_rate,
                       settings.slippage_rate, threshold)


@st.cache_data(show_spinner="ブートストラップを計算中…")
def bootstrap(returns: pd.DataFrame, rf: pd.Series, resamples: int, seed: int, confidence: float, block: int):
    return cp.bootstrap_table(returns, rf, ml().BUY_AND_HOLD, resamples, seed, confidence, block)


def params_label(row) -> str:
    params = ", ".join(f"{k}={v}" for k, v in json.loads(row.hyperparameters).items())
    note = "（検証期間で選択）" if row.selected else ("（walk-forward の既定値）" if not row.in_grid else "")
    return f"{params}：検証 ROC-AUC {row.validation_roc_auc:.3f}{note}"


def render() -> None:
    st.title("⑥ Machine Learning（機械学習）")
    cfg, data = setup()
    target = cfg["target"]
    st.caption(f"予測対象：{target['asset']} の {target['horizon_days']} 営業日後のリターンがプラスか（y = 1）。"
               f"期間分割は Phase 7 と同じ固定の分割で、サイドバーの期間は使いません。予測が「上昇」の月は "
               f"{target['asset']}、それ以外は {target['risk_free']} を持ちます。")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        model = c1.selectbox("モデル", list(MODEL_LABELS), format_func=MODEL_LABELS.get,
                             key=common.persistent("ml_model", "logistic_regression"))
        fset = c2.radio("特徴量セット", list(FEATURE_SETS), format_func=FEATURE_SETS.get,
                        key=common.persistent("ml_fset", "base"))
        mode = MODES[c3.radio("評価", list(MODES), key=common.persistent("ml_mode", list(MODES)[0]))]
        common.remember("ml_model", "ml_fset", "ml_mode")

        default = json.dumps(cfg["models"][model]["default"], sort_keys=True)
        if mode == "test":
            table = search(model, fset)
            options = list(table.hyperparameters)
            chosen = table.loc[table.selected, "hyperparameters"].iloc[0]
            labels = {r.hyperparameters: params_label(r) for r in table.itertuples()}
            params = st.selectbox("ハイパーパラメータ（学習期間で学習、検証期間の ROC-AUC で比較）", options,
                                  index=options.index(chosen), format_func=labels.get, key=f"ml_params_{model}_{fset}")
            if params != chosen:
                st.warning("検証期間で選ばれた値と違う値を選んでいます。テスト期間の結果を見ながら選び直すと、"
                           "テスト期間を選択に使うことになり、未使用データでの検証ではなくなります。")
        else:
            params = default
            st.info(f"walk-forward では既定のハイパーパラメータ（{default}）を使います。2020〜2022 年の検証期間で"
                    "選んだ値を使うと、それより前の年の予測に将来の情報が入るためです。")
        threshold = st.slider("しきい値（予測確率がこれを超えたら VOO を持つ）", 0.30, 0.70, step=0.01,
                              key=common.persistent("ml_threshold", cfg["trading"]["threshold"]))
        common.remember("ml_threshold")
        if threshold != cfg["trading"]["threshold"]:
            st.warning(f"Phase 7 のしきい値は {cfg['trading']['threshold']} に固定です。結果を見ながら変えると、"
                       "それ自体が過学習（データののぞき見）になります。")

    name = f"ML: {ml().model_label(model, fset)}"
    color = ml().ml.COLORS[f"ml:{model}"]
    _, scores = predict(model, fset, params)

    # --- splits ---------------------------------------------------------------------------------
    st.subheader("期間分割と予測確率")
    segments = {s: (data.rows[s][0], data.rows[s][-1]) for s in ml().ml.SEGMENTS}
    st.plotly_chart(charts.probability_by_segment(scores, segments, threshold, color))
    if mode == "walk_forward":
        st.caption("この図は最終の期間分割（既定のハイパーパラメータ）での予測です。walk-forward の予測は下の年別の図と"
                   "バックテストで使います。")
    split = data.split_table()
    common.table(split.set_index("segment"), percent=("positive_rate",),
                 labels={"start": "開始", "end": "終了", "rows": "行数", "evaluated_rows": "評価に使う行",
                         "rows_without_usable_label": "ラベルを使わない行", "positive_rate": "上昇の割合",
                         "model_fitted_on": "予測したモデルの学習データ", "fit_rows": "学習に使った行"})
    st.caption(f"各区間の境界の手前 {target['horizon_days']} 行は、ラベルが次の区間の価格で決まるため学習・評価に使いません"
               "（purge）。標準化は学習データだけで fit します。")

    # --- prediction metrics -----------------------------------------------------------------------
    st.subheader("予測性能")
    metrics = ml().segment_metrics(data, {name: scores, **baselines()}, threshold)
    ml_rows = metrics[metrics.predictor == name].set_index("segment")
    base_rows = metrics[(metrics.predictor != name) & (metrics.segment == "test")].set_index("predictor")
    base_rows.index = base_rows.index.str.replace("baseline:", "ベースライン：")
    view = pd.concat([ml_rows.rename(index=lambda s: f"{name}（{s}）"), base_rows.rename(index=lambda s: f"{s}（test）")])
    common.table(view[list(METRICS)], labels=METRICS,
                 percent=("accuracy", "precision", "recall", "predicted_positive_rate", "positive_rate"),
                 decimals={"roc_auc": 3, "f1": 3, "brier_score": 3, "log_loss": 3})
    gap = ml_rows.loc["train", "roc_auc"] - ml_rows.loc["validation", "roc_auc"]
    st.caption(f"学習期間と検証期間の ROC-AUC の差は {gap:.3f}（大きいほど過学習）。上昇の割合が約 7 割あるため、"
               "正解率は「常に上昇」と予測するだけでも高くなります。予測の良さは ROC-AUC（0.5 = 当てずっぽう）で見てください。")

    wf_scores, wf_table = (walk_forward(model, fset) if mode == "walk_forward" else (None, None))
    if wf_table is not None:
        st.markdown("**walk-forward：年ごとの ROC-AUC**")
        auc = wf_table.set_index("year")["roc_auc"]
        st.plotly_chart(charts.bars(auc, "ROC-AUC", 0.5, color))
        st.caption(f"平均 {auc.mean():.3f} ± {auc.std():.3f}、0.5 を超えた年は {(auc > 0.5).mean():.0%}。"
                   "各年は前年末までにラベルが確定した行だけで学習しています。")
        common.data_view(wf_table, "walk-forward の年別指標", "ml_walk_forward.csv")

    # --- feature importance -------------------------------------------------------------------------
    st.subheader("特徴量の重要度（検証期間の permutation importance）")
    imp = importance(model, fset, params)
    st.plotly_chart(charts.horizontal_bars(imp, "importance_mean", "feature", "importance_std",
                                           "ROC-AUC の低下（特徴量を並べ替えた時）", color))
    st.caption("学習期間で学習したモデルについて、検証期間で各特徴量をランダムに並べ替え、ROC-AUC がどれだけ下がるかを 10 回平均。"
               "マイナスは、並べ替えた方が良くなった（学習期間の関係が検証期間では逆に働いた）ことを示します。"
               "重要度は「使われていた」ことを示すだけで、因果関係ではありません。")

    # --- end to end ---------------------------------------------------------------------------------
    st.subheader("予測 → バックテスト → 統計検証")
    bt = common.backtest_settings()
    score = scores.loc[data.rows["test"]] if mode == "test" else wf_scores
    equities, invest = backtest(name, score, threshold, bt)
    st.caption(f"{equities.index[0].date()} 〜 {equities.index[-1].date()}。月初に判断して翌日に約定（Phase 5 のエンジン）。"
               f"初期資金 {bt.initial_cash:,.0f}、手数料 {bt.commission_rate:.3%}、スリッページ {bt.slippage_rate:.3%}"
               "（④ Backtest の設定）。ベースラインは同じ条件で実行。")
    rules = rule_colors()
    # baselines first so that the model's line is drawn on top (it often overlaps Buy & Hold)
    st.plotly_chart(charts.lines(equities[[*rules, name]], {name: color, **rules}, "資産額", ",.0f"))
    st.dataframe(pd.DataFrame({key: [FORMATS[kind](invest.loc[p, key]) for p in invest.index]
                               for key, _, kind in INVESTMENT}, index=invest.index)
                 .rename(columns={key: label for key, label, _ in INVESTMENT}))

    stats_cfg = cp.load_config("statistics")
    confidence = stats_cfg["confidence_level"]
    returns = cp.aligned_returns(equities)
    sharpe = cp.sharpe_table(returns, data.rf, confidence)
    boot = bootstrap(returns, data.rf, stats_cfg["bootstrap"]["resamples"], stats_cfg["bootstrap"]["seed"],
                     confidence, stats_cfg["bootstrap"]["block_size"])
    diff = boot[(boot.statistic == "sharpe_ratio_difference_vs_benchmark") & (boot.method == "bootstrap_block")]
    diff = diff.set_index("strategy")
    stats = sharpe[["sharpe_ratio", "ci_lower_mertens", "ci_upper_mertens", "psr"]].join(
        diff[["estimate", "ci_lower", "ci_upper", "p_value", "p_value_holm"]])
    common.table(stats, labels={"sharpe_ratio": "Sharpe", "ci_lower_mertens": f"{confidence:.0%} 区間 下限",
                                "ci_upper_mertens": f"{confidence:.0%} 区間 上限", "psr": "PSR",
                                "estimate": "Buy & Hold との差", "ci_lower": "差の区間 下限", "ci_upper": "差の区間 上限",
                                "p_value": "p 値", "p_value_holm": "p 値（Holm 補正後）"},
                 decimals={c: 3 for c in stats.columns})
    st.caption("Sharpe の区間は Mertens の標準誤差。差の区間と p 値は 21 日ブロックのブートストラップ"
               f"（{stats_cfg['bootstrap']['resamples']:,} 回）で、ここに表示した Buy & Hold との 3 件の比較をまとめて"
               "Holm 補正します。Phase 7 のレポートは全予測の 10 件をまとめて補正したため、補正後の p 値はレポートより"
               "小さくなることがあります（補正前の p 値と差の区間は同じ）。")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.intervals(sharpe, "sharpe_ratio", "ci_lower_mertens", "ci_upper_mertens", "Sharpe"))
    with right:
        st.plotly_chart(charts.intervals(diff, "estimate", "ci_lower", "ci_upper", "Sharpe の差（対 Buy & Hold）"))
    verdict = diff.loc[name, "p_value_holm"]
    if verdict < 1 - confidence:
        direction = "上回って" if diff.loc[name, "estimate"] > 0 else "下回って"
        st.success(f"{name} は Buy & Hold を Sharpe で{direction}いると言えます（Holm 補正後 p = {verdict:.3f}）。")
    else:
        st.info(f"{name} と Buy & Hold の Sharpe の差は、統計的に区別できません（Holm 補正後 p = {verdict:.3f}）。")
    common.data_view(equities, "資産曲線", "ml_equity.csv")


if __name__ == "__main__":
    render()
