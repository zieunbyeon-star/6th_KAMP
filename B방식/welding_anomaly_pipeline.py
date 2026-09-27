"""
KAMP 로봇용접 - 라벨 없는 2단계 이상탐지 + 라벨 독립 검증 파이프라인

[문제 인식]
  - 타점 단위 라벨이 없다 → 임의 라벨링(A방식)은 근거가 없다
  - 일자 단위 불량 수(3~8건/일)는 우연 변동과 구분되지 않는다 (카이제곱 동질성 검정)
    → 어떤 모델이든 '일자 불량률과의 상관'으로 성능을 증명할 수 없다
  => 모델은 비지도로 학습하고(B방식 유지), 검증은 라벨에 의존하지 않는 방법으로 한다
     일자 라벨은 '모순 여부 확인 + 필요 표본 수 산정'에만 사용한다

  - 데이터 무결성 점검 결과, 98% 타점이 다른 날에도 똑같이 반복되는 20타점 구간에 속함
    (공정 이탈 291타점 블록은 4일에 걸쳐 완전히 동일) → 증강/재조합 데이터로 판단
    → 일자 라벨과 타점 데이터의 물리적 연결을 가정할 수 없음, 유효 표본은 행 수보다 훨씬 작음
    → 학습·홀드아웃은 중복 제거 후 수행 (누수 방지)

[파이프라인]
  0. 데이터 로드 + 무결성 점검 (반복 구간 탐지)
  1. 라벨 신호 점검: 일자 불량률 동질성 검정 + 검정력 분석(필요 일수)
  2. 탐지 모델 (2단계)
     Stage 1  Robust SPC 규칙 (median/MAD) → 큰 공정 이탈 (해석 가능)
     Stage 2  Isolation Forest (Stage 1 정상 구간으로만 학습) → 미세 복합 이상
  3. 라벨 독립 검증
     3-1 홀드아웃 오경보율
     3-2 시나리오별 합성 이탈 주입 → 최소 탐지 가능 변화량(MDS)
     3-3 시드 안정성
  4. 약라벨 일관성 확인 (일자 단위, 조건부 이항검정)
  5. 실패조건 분석
  6. 안정 공정조건 범위

사용법
  python welding_anomaly_pipeline.py --data Welding_Data_Set_01.xlsx --out results
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
DATE = "working time"
SPEC = {  # 'data set' 시트의 설비 수집 범위
    "weld force(bar)": (1.0, 12.0),
    "weld current(kA)": (12.0, 18.0),
    "weld Voltage(v)": (1.5, 3.5),
    "weld time(ms)": (30, 120),
}
# 합성 이탈 시나리오: (변수, 방향, 물리적 의미)
SCENARIOS = [
    ("weld force(bar)", +1, "과가압 → 파임"),
    ("weld current(kA)", -1, "전류 저하 → 용접부족"),
    ("weld current(kA)", +1, "전류 과다 → 과용접/크랙"),
    ("weld Voltage(v)", -1, "전압 강하 → 접촉불량"),
    ("weld time(ms)", -1, "통전시간 부족 → 용접부족"),
]
SEED = 42
Z_SPC = 6.0          # Stage 1 임계 (robust z)
IF_QUANTILE = 0.997  # Stage 2 임계 (학습 점수 분포 상위 0.3% ≈ 3σ)


# ================================================================ 0. 데이터
def load_data(path):
    raw = pd.read_excel(path, sheet_name="Raw data")
    res = pd.read_excel(path, sheet_name="result")
    raw[DATE] = pd.to_datetime(raw[DATE])
    res[DATE] = pd.to_datetime(res[DATE])
    for c in ["Thickness 1(mm)", "Thickness 2(mm)"]:
        assert raw[c].nunique() == 1, f"{c}가 상수가 아님 - input 재검토 필요"
    defects = res.pivot_table(index=DATE, columns="defect type", values="defect",
                              aggfunc="sum").fillna(0)
    defects.columns = [f"type{int(c)}" for c in defects.columns]
    defects["total"] = defects.sum(axis=1)
    return raw, defects


def audit_replay(raw, k=20):
    """
    연속 k타점(4변수 완전 일치)이 데이터 안에서 2회 이상 등장하는지 검사.
    실측 연속값에서 20타점이 우연히 완전 일치할 확률은 사실상 0 → 반복 = 복제/재조합 흔적
    """
    rows = [tuple(r) for r in raw[FEATURES].values]
    wins = [tuple(rows[i:i + k]) for i in range(len(rows) - k + 1)]
    cnt = {}
    for w in wins:
        cnt[w] = cnt.get(w, 0) + 1
    replay = np.zeros(len(rows), bool)
    for i, w in enumerate(wins):
        if cnt[w] > 1:
            replay[i:i + k] = True
    return {"n_windows": len(wins), "n_unique_windows": len(cnt),
            "replayed_row_frac": replay.mean(),
            "n_unique_rows": int((~raw[FEATURES].duplicated()).sum())}


# ================================================================ 1. 라벨 신호 점검
def homogeneity_test(k, n):
    """모든 날의 불량률이 같다는 귀무가설 (Pearson 카이제곱)"""
    p = k.sum() / n.sum()
    e = n * p
    stat = ((k - e) ** 2 / e).sum()
    return stat, 1 - chi2.cdf(stat, len(k) - 1)


def power_analysis(base_rate, n_per_day, rr_list=(1.5, 2.0, 3.0),
                   days_list=(8, 20, 40, 80, 160), exposed_frac=0.5,
                   n_sim=2000, alpha=0.05, seed=SEED):
    """
    '이탈 발생일의 불량률이 RR배 높다'를 일자 단위 데이터로 검출하려면 며칠이 필요한가
    (시뮬레이션 기반 검정력)
    """
    rng = np.random.default_rng(seed)
    rows = []
    for rr in rr_list:
        for D in days_list:
            n_exp = int(D * exposed_frac)
            n = np.full(D, n_per_day)
            rate = np.r_[np.full(n_exp, base_rate * rr), np.full(D - n_exp, base_rate)]
            share = n[:n_exp].sum() / n.sum()
            hits = 0
            for _ in range(n_sim):
                k = rng.binomial(n, rate)
                if k.sum() == 0:
                    continue
                if binomtest(k[:n_exp].sum(), k.sum(), share,
                             alternative="greater").pvalue < alpha:
                    hits += 1
            rows.append({"rate_ratio": rr, "days": D, "power": hits / n_sim})
    return pd.DataFrame(rows)


# ================================================================ 2. 탐지 모델
class TwoStageDetector:
    """
    Stage 1: robust z = |x - median| / (1.4826·MAD), 어느 변수든 Z_SPC 초과 시 이탈
             → 큰 공정 이탈을 설명 가능한 형태로 분리
    Stage 2: Stage 1 정상 구간만으로 Isolation Forest 학습 (novelty detection)
             → 개별 변수로는 정상이지만 조합이 비정상인 타점 탐지
    라벨은 어디에도 사용하지 않는다.
    """

    def __init__(self, z_thr=Z_SPC, if_q=IF_QUANTILE, seed=SEED):
        self.z_thr, self.if_q, self.seed = z_thr, if_q, seed

    def _robust_z(self, X):
        return np.abs(X - self.med_) / self.scale_

    def fit(self, X):
        self.med_ = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.abs(X - self.med_), axis=0)
        # 통전시간처럼 이산값이라 MAD=0인 변수는 표준편차로 대체
        self.scale_ = np.where(mad > 0, mad, X.std(axis=0))
        normal = ~(self._robust_z(X) > self.z_thr).any(axis=1)
        self.scaler_ = StandardScaler().fit(X[normal])
        Xn = self.scaler_.transform(X[normal])
        self.iforest_ = IsolationForest(n_estimators=300, random_state=self.seed,
                                        n_jobs=-1).fit(Xn)
        self.thr_ = np.quantile(-self.iforest_.score_samples(Xn), self.if_q)
        return self

    def predict(self, X):
        z = self._robust_z(X)
        s1 = (z > self.z_thr).any(axis=1)
        score = -self.iforest_.score_samples(self.scaler_.transform(X))
        s2 = (~s1) & (score > self.thr_)
        return pd.DataFrame({
            "stage1_spc": s1,
            "stage2_if": s2,
            "flag": s1 | s2,
            "if_score": score,
            "max_robust_z": z.max(axis=1),
            "driver": np.array(FEATURES)[z.argmax(axis=1)],  # 가장 크게 벗어난 변수
        })


# ================================================================ 3. 라벨 독립 검증
def holdout_false_alarm(X, seed=SEED):
    """
    정상 구간을 7:3으로 나눠 학습 안 한 정상 데이터에서의 오경보율 측정
    X는 중복 제거된 행이어야 함 (중복이 train/test 양쪽에 들어가면 오경보율이 낙관적으로 나옴)
    """
    det0 = TwoStageDetector(seed=seed).fit(X)
    normal_idx = np.where(~det0.predict(X)["stage1_spc"].values)[0]
    tr, te = train_test_split(normal_idx, test_size=0.3, random_state=seed)
    det = TwoStageDetector(seed=seed).fit(X[tr])
    return det, tr, te, det.predict(X[te])["flag"].mean()


def minimum_detectable_shift(det, X, te, k_grid=np.arange(0.5, 10.5, 0.5),
                             n=500, target=0.9, seed=SEED):
    """
    홀드아웃 정상 타점에 시나리오별로 kσ 이탈을 주입 → 탐지율 곡선
    MDS = 탐지율이 target 이상이 되는 최소 k
    """
    rng = np.random.default_rng(seed)
    sigma = X[te].std(axis=0)
    curves, mds = [], []
    for feat, sign, meaning in SCENARIOS:
        j = FEATURES.index(feat)
        lo, hi = SPEC[feat]
        base = X[rng.choice(te, n)]
        found = np.nan
        for k in k_grid:
            Xs = base.copy()
            Xs[:, j] = np.clip(Xs[:, j] + sign * k * sigma[j], lo, hi)
            rate = det.predict(Xs)["flag"].mean()
            curves.append({"scenario": meaning, "label_en": f"{feat.split('(')[0].strip()} {'+' if sign > 0 else '-'}",
                           "k_sigma": k, "detect_rate": rate})
            if np.isnan(found) and rate >= target:
                found = k
        mds.append({"scenario": meaning, "feature": feat,
                    "MDS_k_sigma": found,
                    "MDS_abs": found * sigma[j] if not np.isnan(found) else np.nan})
    return pd.DataFrame(curves), pd.DataFrame(mds)


def seed_stability(X, n_seeds=20):
    """시드별 탐지 결과의 일치도 (Jaccard, 기준 시드 대비)"""
    ref = TwoStageDetector(seed=0).fit(X).predict(X)["flag"].values
    out = []
    for s in range(1, n_seeds):
        f = TwoStageDetector(seed=s).fit(X).predict(X)["flag"].values
        out.append((ref & f).sum() / max((ref | f).sum(), 1))
    return np.array(out)


# ================================================================ 4. 약라벨 일관성
def daily_table(raw, pred, defects):
    d = pd.concat([raw[[DATE]].reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    t = d.groupby(DATE).agg(n_prod=("flag", "size"), n_flag=("flag", "sum"),
                            n_spc=("stage1_spc", "sum"), n_if=("stage2_if", "sum"))
    t = t.join(defects, how="left")
    t["has_result"] = t["total"].notna()
    t["flag_rate"] = t["n_flag"] / t["n_prod"]
    t["defect_rate"] = t["total"] / t["n_prod"]
    return t


def exposure_test(v):
    """
    이탈 발생일(Stage 1 이탈 있음) vs 미발생일의 불량 비율
    귀무가설: 불량은 생산량 비율대로 분포 (조건부 이항검정)
    """
    exp = v["n_spc"] > 0
    k_exp, k_all = int(v.loc[exp, "total"].sum()), int(v["total"].sum())
    share = v.loc[exp, "n_prod"].sum() / v["n_prod"].sum()
    p = binomtest(k_exp, k_all, share, alternative="greater").pvalue
    return {
        "days_exposed": int(exp.sum()), "days_not": int((~exp).sum()),
        "rate_exposed": v.loc[exp, "total"].sum() / v.loc[exp, "n_prod"].sum(),
        "rate_not": v.loc[~exp, "total"].sum() / v.loc[~exp, "n_prod"].sum(),
        "defects_on_nonexposed_days": int(v.loc[~exp, "total"].sum()),
        "defects_total": k_all, "p_value": p,
    }


# ================================================================ 6. 안정 공정조건
def stable_window(raw, pred):
    ok = raw.loc[~pred["flag"].values, FEATURES]
    ng = raw.loc[pred["stage1_spc"].values, FEATURES]
    return pd.DataFrame({
        "spec_min": [SPEC[f][0] for f in FEATURES],
        "spec_max": [SPEC[f][1] for f in FEATURES],
        "stable_mean": ok.mean(),
        "stable_LCL(-3σ)": ok.mean() - 3 * ok.std(),
        "stable_UCL(+3σ)": ok.mean() + 3 * ok.std(),
        "stable_P0.5": ok.quantile(0.005),
        "stable_P99.5": ok.quantile(0.995),
        "excursion_median": ng.median() if len(ng) else np.nan,
        "excursion_max": ng.max() if len(ng) else np.nan,
    }, index=FEATURES)


# ================================================================ 시각화
def plot_timeline(raw, pred, out):
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    x = np.arange(len(raw))
    for ax, f in zip(axes, FEATURES):
        ax.plot(x, raw[f], lw=0.4, color="0.5")
        m1, m2 = pred["stage1_spc"].values, pred["stage2_if"].values
        ax.scatter(x[m1], raw[f][m1], s=4, c="C3", label="Stage1 SPC")
        ax.scatter(x[m2], raw[f][m2], s=8, c="C1", label="Stage2 IF")
        ax.set_ylabel(f, fontsize=8)
    axes[0].legend(loc="upper right", fontsize=8)
    for b in np.where(raw[DATE].diff().dt.days.fillna(0) > 0)[0]:
        for ax in axes:
            ax.axvline(b, color="k", lw=0.5, ls=":")
    axes[-1].set_xlabel("weld sequence (dotted = day boundary)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "timeline.png"), dpi=150)
    plt.close(fig)


def plot_mds(curves, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for s, g in curves.groupby("label_en"):  # 그래프는 한글 폰트 없이도 깨지지 않게 영문
        ax.plot(g["k_sigma"], g["detect_rate"], "o-", ms=3, label=s)
    ax.axhline(0.9, ls="--", c="k", lw=0.8)
    ax.set_xlabel("injected shift (k·σ)"); ax.set_ylabel("detection rate")
    ax.set_title("Minimum detectable shift by scenario")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mds_curves.png"), dpi=150)
    plt.close(fig)


def plot_power(pw, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    for rr, g in pw.groupby("rate_ratio"):
        ax.plot(g["days"], g["power"], "o-", label=f"RR={rr}")
    ax.axhline(0.8, ls="--", c="k", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("number of days"); ax.set_ylabel("power")
    ax.set_title("Days needed to validate with daily labels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "power_analysis.png"), dpi=150)
    plt.close(fig)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Welding_Data_Set_01.xlsx")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw, defects = load_data(args.data)
    X = raw[FEATURES].values.astype(float)
    uniq = ~raw[FEATURES].duplicated().values
    Xu = X[uniq]  # 학습/홀드아웃용 (중복 제거)

    # ---- 0. 무결성 점검
    au = audit_replay(raw)
    print("=" * 70)
    print(f"[0] 무결성: 반복 20타점 구간에 속한 타점 {au['replayed_row_frac']:.1%}, "
          f"고유 20타점 구간 {au['n_unique_windows']}/{au['n_windows']}, "
          f"고유 행 {au['n_unique_rows']}/{len(raw)}")
    print("    → 증강/재조합 데이터 가능성: 일자 라벨-타점 연결 가정 불가, 중복 제거 후 학습")

    # ---- 1. 라벨 신호 점검
    prod = raw.groupby(DATE).size()
    lab = defects.join(prod.rename("n_prod"), how="inner")
    stat, p_h = homogeneity_test(lab["total"].values, lab["n_prod"].values)
    base = lab["total"].sum() / lab["n_prod"].sum()
    print(f"\n[1] 일자 불량률 동질성: chi2={stat:.2f}, p={p_h:.3f} "
          f"(n={len(lab)}일, 총 불량 {int(lab['total'].sum())}건, 평균 {base:.4%})")
    print("    → p>0.05면 일자 간 불량률 차이가 우연 변동과 구분되지 않음")
    pw = power_analysis(base, int(prod.mean()))
    pw.to_csv(os.path.join(args.out, "power_analysis.csv"), index=False)
    plot_power(pw, args.out)
    print(pw.pivot(index="days", columns="rate_ratio", values="power").round(2).to_string())

    # ---- 2. 모델
    det = TwoStageDetector().fit(Xu)   # 고유 행으로 학습
    pred = det.predict(X)              # 전체 타점 점수화
    print(f"\n[2] Stage1(SPC) {pred['stage1_spc'].sum()}타점, "
          f"Stage2(IF) {pred['stage2_if'].sum()}타점, 총 {pred['flag'].mean():.2%}")
    print("    Stage1 주요 원인 변수:",
          pred.loc[pred["stage1_spc"], "driver"].value_counts().to_dict())

    # ---- 3. 라벨 독립 검증
    det_h, tr, te, far = holdout_false_alarm(Xu)
    print(f"\n[3-1] 홀드아웃 오경보율(중복 제거) = {far:.3%}")
    curves, mds = minimum_detectable_shift(det_h, Xu, te)
    curves.to_csv(os.path.join(args.out, "mds_curves.csv"), index=False)
    mds.to_csv(os.path.join(args.out, "mds.csv"), index=False)
    plot_mds(curves, args.out)
    print("[3-2] 최소 탐지 가능 변화량 (탐지율 90%)\n", mds.round(3).to_string(index=False))
    jac = seed_stability(Xu)
    print(f"[3-3] 시드 안정성 Jaccard: mean={jac.mean():.3f}, min={jac.min():.3f}")

    # ---- 4. 약라벨 일관성
    t = daily_table(raw, pred, defects)
    t.to_csv(os.path.join(args.out, "daily_table.csv"))
    v = t[t["has_result"]]  # result 누락일(3/27) 제외
    ex = exposure_test(v)
    print(f"\n[4] 이탈일 불량률 {ex['rate_exposed']:.3%} vs 비이탈일 {ex['rate_not']:.3%}, "
          f"p={ex['p_value']:.3f}")

    # ---- 5. 실패조건
    print("\n[5] 실패조건")
    print(f"  - 전체 불량 {ex['defects_total']}건 중 {ex['defects_on_nonexposed_days']}건"
          f"({ex['defects_on_nonexposed_days']/ex['defects_total']:.0%})이 "
          "공정 이탈이 없는 날 발생 → 측정 변수 밖 요인(전극 마모, 표면상태, 소재 편차 등)")
    for _, r in mds.iterrows():
        msg = (f"{r['MDS_k_sigma']:.1f}σ (≈{r['MDS_abs']:.3f}) 미만 변화는 놓침"
               if not np.isnan(r["MDS_k_sigma"]) else "수집 범위 내 탐지율 90% 미달")
        print(f"  - {r['scenario']}: {msg}")

    # ---- 6. 안정 공정조건
    sw = stable_window(raw, pred)
    sw.to_csv(os.path.join(args.out, "stable_window.csv"))
    print("\n[6] 안정 공정조건 (이상 제외 타점 기준)\n", sw.round(3).T.to_string())

    plot_timeline(raw, pred, args.out)
    pd.concat([raw.reset_index(drop=True), pred], axis=1).to_csv(
        os.path.join(args.out, "scored_points.csv"), index=False)
    print(f"\n[done] 결과 저장: {args.out}/")


if __name__ == "__main__":
    main()
