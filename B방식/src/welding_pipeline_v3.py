"""
KAMP 로봇용접 v3 - 구간 단위 중복 제거 + 가설 B 검정 + 알고리즘 비교 + 실패조건 대안 ablation

[v2 대비 변경]
  0. 행 단위 중복 제거 → 구간(연속 20타점) 단위 중복 제거
     - 우연 일치 여부를 셔플 귀무모형으로 검정 (측정 해상도 문제 대응)
  1. 가설 B(일자 라벨이 타점 데이터와 연결되지 않음) 검정 추가
  2. 알고리즘 비교: Isolation Forest / Robust Mahalanobis(MCD) / LOF
     - 동일한 오경보율(FAR)로 임계값을 맞춘 뒤 탐지율 비교 (공정 비교)
  3. 실패조건 대안 ablation: 물리 파생변수 → 시퀀스 변수 → EWMA 를 하나씩 추가
  4. 시드 안정성: 무작위성이 있는 Stage 2 모델만 측정
  5. SPC 임계값 trade-off: MDS가 임계값에 의해 결정됨을 보임
  6. 출력: 타점별 위험 점수와 검사 우선순위 (scored_points.csv)

사용법
  python welding_pipeline_v3.py --data Welding_Data_Set_01.xlsx --out results_v3
"""
import argparse
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, spearmanr
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

F = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
FORCE, CUR, VOLT, TIME = F
DATE = "working time"
RUN = "run_id"          # 파일 내 연속 구간 번호 (시퀀스 계산 단위)
SPEC = {FORCE: (1.0, 12.0), CUR: (12.0, 18.0), VOLT: (1.5, 3.5), TIME: (30, 120)}
K = 20                  # 구간 중복 판정 길이
Z_SPC = 6.0             # Stage 1 robust z 임계값
TARGET_FAR = 0.01       # Stage 2(+EWMA) 오경보율 목표 (학습셋 기준)
SEED = 42
N_TREES = 1000        # IF 트리 수 (300개일 때 경계 판정이 시드에 따라 흔들림)


# ====================================================================== 0. 데이터
def load_data(path):
    raw = pd.read_excel(path, sheet_name="Raw data")
    res = pd.read_excel(path, sheet_name="result")
    raw[DATE] = pd.to_datetime(raw[DATE])
    res[DATE] = pd.to_datetime(res[DATE])
    for c in ["Thickness 1(mm)", "Thickness 2(mm)"]:
        assert raw[c].nunique() == 1, f"{c}가 상수가 아님"
    raw[RUN] = raw[DATE].ne(raw[DATE].shift()).cumsum()  # 파일 순서 기준 연속 구간
    defects = res.pivot_table(index=DATE, columns="defect type", values="defect",
                              aggfunc="sum").fillna(0)
    defects.columns = [f"type{int(c)}" for c in defects.columns]
    defects["total"] = defects.sum(axis=1)
    return raw, defects


# ====================================================================== 1. 구간 중복 제거
def segment_dedup(raw, k=K):
    """
    파일 순서대로 연속 k타점 창을 훑으며, 이전에 이미 나온 창이면 그 k타점을 '복제'로 표시.
    최초 등장 구간만 남긴다. → 우연히 값이 같은 개별 행은 보존된다.
    """
    rows = [tuple(r) for r in raw[F].values]
    seen, replay = set(), np.zeros(len(rows), bool)
    for i in range(len(rows) - k + 1):
        w = tuple(rows[i:i + k])
        if w in seen:
            replay[i:i + k] = True
        else:
            seen.add(w)
    return ~replay


def chance_null(raw, ks=(3, 5, 10, 20), seed=SEED):
    """
    귀무모형: 각 날짜 안에서 행 순서를 섞는다 (값·해상도·분포는 그대로, 순서만 파괴).
    측정 해상도 때문에 우연히 같아지는 것이라면 섞은 데이터에서도 반복 구간이 비슷하게 나와야 한다.
    """
    rng = np.random.default_rng(seed)
    shuffled = raw.groupby(DATE, group_keys=False)[F].apply(
        lambda g: g.iloc[rng.permutation(len(g))]).values
    out = []
    for k in ks:
        for name, arr in [("original", raw[F].values), ("shuffled", shuffled)]:
            rows = [tuple(r) for r in arr]
            cnt = {}
            for i in range(len(rows) - k + 1):
                w = tuple(rows[i:i + k])
                cnt[w] = cnt.get(w, 0) + 1
            rep = sum(v for v in cnt.values() if v > 1)
            out.append({"k": k, "data": name, "windows_in_repeats": rep})
    return pd.DataFrame(out).pivot(index="k", columns="data", values="windows_in_repeats")


# ====================================================================== 2. 가설 B 검정
def day_windows(raw, k=K):
    out = {}
    for d, g in raw.groupby(DATE):
        rows = [tuple(r) for r in g[F].values]
        out[d] = {tuple(rows[i:i + k]) for i in range(len(rows) - k + 1)}
    return out


def mantel_test(raw, defects, k=K):
    """
    라벨이 타점 데이터와 연결돼 있다면: 타점 내용이 비슷한 두 날은 불량률도 비슷해야 한다.
    통계량 = Spearman(내용 유사도, |불량률 차이|)  → 연결 가설 하에서는 음수 기대
    p값 = 불량률의 날짜 배정을 모든 순열(8!)로 섞어 계산 (단측)
    """
    prod = raw.groupby(DATE).size()
    lab = defects.join(prod.rename("n"), how="inner")
    days = list(lab.index)
    rate = (lab["total"] / lab["n"]).values
    W = day_windows(raw, k)
    pairs = list(itertools.combinations(range(len(days)), 2))
    sim = np.array([len(W[days[i]] & W[days[j]]) / len(W[days[i]] | W[days[j]])
                    for i, j in pairs])

    def stat(r):
        return spearmanr(sim, [abs(r[i] - r[j]) for i, j in pairs])[0]

    obs = stat(rate)
    perm = np.array([stat(rate[list(p)]) for p in itertools.permutations(range(len(days)))])
    sim_mat = pd.DataFrame(np.eye(len(days)), index=days, columns=days)
    for (i, j), s in zip(pairs, sim):
        sim_mat.iloc[i, j] = sim_mat.iloc[j, i] = s
    return obs, np.mean(perm <= obs + 1e-12), sim_mat


def homogeneity_test(raw, defects):
    prod = raw.groupby(DATE).size()
    lab = defects.join(prod.rename("n"), how="inner")
    k, n = lab["total"].values, lab["n"].values
    e = n * k.sum() / n.sum()
    stat = ((k - e) ** 2 / e).sum()
    return stat, 1 - chi2.cdf(stat, len(k) - 1)


# ====================================================================== 3. 특징
def build_features(df, feature_set):
    """
    base    : 원변수 4개
    physics : + 동저항 R=V/I [mΩ], 입열량 Q=V·I·t [J], I²t [kA²·ms]
    seq     : + 직전 20타점 대비 편차, 20타점 이동표준편차 (가압력/전류/전압)
    """
    X = df[F].copy()
    if feature_set in ("physics", "physics+seq"):
        X["R_mohm"] = df[VOLT] / df[CUR]
        X["Q_J"] = df[VOLT] * df[CUR] * df[TIME]
        X["I2t"] = df[CUR] ** 2 * df[TIME]
    if feature_set in ("seq", "physics+seq"):
        g = df.groupby(RUN)
        for c in [FORCE, CUR, VOLT]:
            base = g[c].transform(lambda s: s.rolling(20, min_periods=5).mean().shift(1))
            X[f"{c}_dev"] = (df[c] - base).fillna(0.0)
            X[f"{c}_rstd"] = g[c].transform(lambda s: s.rolling(20, min_periods=5).std()) \
                .fillna(0.0)
    return X


# ====================================================================== 4. 탐지 모델
class SPC:
    """Stage 1: robust z = |x - median| / (1.4826·MAD) > Z_SPC (원변수 4개)"""

    def __init__(self, z=Z_SPC):
        self.z = z

    def fit(self, X):
        self.med = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.abs(X - self.med), axis=0)
        self.scale = np.where(mad > 0, mad, X.std(axis=0))  # 통전시간(MAD=0) 대체
        return self

    def zscore(self, X):
        return np.abs(X - self.med) / self.scale

    def flag(self, X):
        return (self.zscore(X) > self.z).any(axis=1)


class Detector:
    """Stage 2 후보. score가 클수록 이상."""

    def __init__(self, algo, seed=SEED):
        self.algo, self.seed = algo, seed

    def fit(self, X):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        if self.algo == "IF":
            self.m = IsolationForest(n_estimators=N_TREES, random_state=self.seed, n_jobs=-1).fit(Z)
        elif self.algo == "MCD":
            self.m = MinCovDet(random_state=self.seed).fit(Z)
        elif self.algo == "LOF":
            self.m = LocalOutlierFactor(n_neighbors=35, novelty=True).fit(Z)
        return self

    def score(self, X):
        Z = self.sc.transform(X)
        if self.algo == "IF":
            return -self.m.score_samples(Z)
        if self.algo == "MCD":
            return self.m.mahalanobis(Z)
        return -self.m.score_samples(Z)


class EWMA:
    """
    Stage 3: 작은 변화가 지속되는 것을 누적 탐지 (가압력/전류/전압).
    z_t = λ·x_t + (1-λ)·z_{t-1}, 구간(run)마다 초기화. 통계량 = 변수별 |z|/σ_z 의 최대값
    """

    def __init__(self, lam=0.1):
        self.lam = lam
        self.cols = [FORCE, CUR, VOLT]

    def fit(self, df):
        self.mu = df[self.cols].mean().values
        self.sd = df[self.cols].std().values
        return self

    def score(self, df):
        x = (df[self.cols].values - self.mu) / self.sd
        z = np.zeros_like(x)
        runs = df[RUN].values
        for i in range(len(x)):
            prev = z[i - 1] if i > 0 and runs[i] == runs[i - 1] else 0.0
            z[i] = self.lam * x[i] + (1 - self.lam) * prev
        sz = np.sqrt(self.lam / (2 - self.lam))
        return np.abs(z / sz).max(axis=1)


def calibrate(train_scores, target=TARGET_FAR):
    """여러 점수를 같은 분위수로 자를 때, 학습셋 합집합 오경보율이 target이 되는 분위수 탐색"""
    lo, hi = 0.9, 1.0
    for _ in range(40):
        q = (lo + hi) / 2
        thr = [np.quantile(s, q) for s in train_scores]
        far = np.mean(np.any([s > t for s, t in zip(train_scores, thr)], axis=0))
        lo, hi = (q, hi) if far > target else (lo, q)
    return [np.quantile(s, hi) for s in train_scores]


# ====================================================================== 5. 실험 프레임
class System:
    """SPC + Stage2(algo, feature_set) [+ EWMA]"""

    def __init__(self, algo, feature_set, use_ewma, seed=SEED):
        self.algo, self.fs, self.use_ewma, self.seed = algo, feature_set, use_ewma, seed

    def fit(self, df_seq, train_mask, spc):
        self.spc = spc
        X = build_features(df_seq, self.fs).values
        self.det = Detector(self.algo, self.seed).fit(X[train_mask])
        tr = [self.det.score(X[train_mask])]
        if self.use_ewma:
            self.ew = EWMA().fit(df_seq[train_mask])
            tr.append(self.ew.score(df_seq)[train_mask])
        self.thr = calibrate(tr)
        return self

    def scores(self, df_seq):
        X = build_features(df_seq, self.fs).values
        sc = [self.det.score(X)]
        if self.use_ewma:
            sc.append(self.ew.score(df_seq))
        return self.spc.flag(df_seq[F].values), sc

    def predict(self, df_seq, thr=None):
        thr = self.thr if thr is None else thr
        s1, sc = self.scores(df_seq)
        s2 = sc[0] > thr[0]
        s3 = sc[1] > thr[1] if self.use_ewma else np.zeros(len(df_seq), bool)
        return s1, s2, s3


def make_splits(orig, spc):
    """원본 구간을 연속 10블록으로 나눠 3블록을 홀드아웃 (시계열 누수 방지)"""
    blocks = np.array_split(np.arange(len(orig)), 10)
    hold = np.zeros(len(orig), bool)
    for b in (2, 5, 8):
        hold[blocks[b]] = True
    normal = ~spc.flag(orig[F].values)
    return normal & ~hold, normal & hold


def scenarios(sigma):
    """합성 이탈 시나리오 (σ = 학습 정상 데이터 표준편차)"""
    return [
        ("과가압(+3σ)", "point", {FORCE: +3}),
        ("전류저하(-3σ)", "point", {CUR: -3}),
        ("전류과다(+3σ)", "point", {CUR: +3}),
        ("전압강하(-3σ)", "point", {VOLT: -3}),
        ("통전부족(-3σ)", "point", {TIME: -3}),
        ("복합(전류·전압·시간 각 -2σ)", "point", {CUR: -2, VOLT: -2, TIME: -2}),
        ("드리프트(전류 -1.5σ 지속)", "drift", {CUR: -1.5}),
    ]


def evaluate(system, orig, train, hold, spc, n_rep=10, seed=SEED, target=TARGET_FAR):
    """
    공정 비교 원칙: 모든 시스템을 '홀드아웃 정상 데이터에서 오경보율 1%'가 되도록 임계값을 맞춘 뒤
    합성 이탈 탐지율을 비교한다 (ROC 곡선의 같은 FAR 지점에서 비교하는 것과 같음).
    FAR_train_thr 는 학습셋 기준 임계값을 그대로 썼을 때의 홀드아웃 오경보율 (일반화 지표).
    """
    rng = np.random.default_rng(seed)
    system.fit(orig, train, spc)
    s1, s2, s3 = system.predict(orig)
    res = {"FAR_train_thr": np.mean((s1 | s2 | s3)[hold])}
    _, sc = system.scores(orig)
    thr_h = calibrate([s[hold] for s in sc], target)  # 홀드아웃 정상 기준 FAR 고정
    sigma = orig.loc[train, F].std()
    hold_idx = np.where(hold)[0]
    for name, kind, shift in scenarios(sigma):
        hits = []
        for rep in range(n_rep):
            df = orig.copy()
            if kind == "point":
                idx = hold_idx[rep::25]  # 25타점 간격의 고립 이상
                eval_idx = idx
            else:
                start = rng.choice(hold_idx[:-120])
                idx = np.arange(start, start + 100)
                eval_idx = idx[50:]      # 누적형 탐지를 고려해 후반 50타점에서 평가
            for c, k in shift.items():
                lo, hi = SPEC[c]
                df.loc[df.index[idx], c] = np.clip(df.loc[df.index[idx], c] + k * sigma[c], lo, hi)
            f1, f2, f3 = system.predict(df, thr_h)
            hits.append(np.mean((f1 | f2 | f3)[eval_idx]))
        res[name] = np.mean(hits)
    return res


def daily_link(system, raw, defects):
    """Stage2+3 일자별 탐지율 vs 일자 불량률 (SPC 제외 — SPC는 가압력 블록에 지배됨)"""
    s1, s2, s3 = system.predict(raw)
    d = pd.DataFrame({DATE: raw[DATE].values, "f": s2 | s3}).groupby(DATE)["f"].agg(["mean", "size"])
    lab = defects.join(d, how="inner")
    x, y = lab["mean"].values, (lab["total"] / lab["size"]).values
    if np.std(x) == 0:
        return np.nan, np.nan
    obs = spearmanr(x, y)[0]
    perm = [spearmanr(x, y[list(p)])[0] for p in itertools.permutations(range(len(y)))]
    return obs, np.mean(np.nan_to_num(perm) >= obs - 1e-12)


# ====================================================================== 6. 부가 분석
def seed_stability(orig, train, spc, algo, fs, n=20):
    """
    무작위성이 있는 Stage 2만 비교 (seed 0 기준).
    평가 대상 = 원본 타점 중 SPC 정상 구간 (Stage 2가 실제로 판정하는 영역).
    복제 행까지 포함하면 경계값 하나가 뒤집힐 때 수백 행이 함께 뒤집혀 불안정성이 과장됨.
    """
    region = ~spc.flag(orig[F].values)
    X = build_features(orig, fs).values[region]
    ref = System(algo, fs, False, seed=0).fit(orig, train, spc)
    ref_s = ref.det.score(X)
    ref_f = ref_s > ref.thr[0]
    rows = []
    for s in range(1, n):
        m = System(algo, fs, False, seed=s).fit(orig, train, spc)
        sc = m.det.score(X)
        f = sc > m.thr[0]
        rows.append({"seed": s, "score_spearman": spearmanr(ref_s, sc)[0],
                     "stage2_jaccard": (ref_f & f).sum() / max((ref_f | f).sum(), 1),
                     "n_stage2": int(f.sum())})
    return pd.DataFrame(rows)


def spc_tradeoff(orig, z_grid=(3, 4, 5, 6)):
    """SPC 단독: 임계값 z에 따라 오경보율과 최소 탐지 변화량(MDS)이 함께 움직임을 확인"""
    base_normal = ~SPC(Z_SPC).fit(orig[F].values).flag(orig[F].values)
    Xn = orig.loc[base_normal, F].values
    sigma = Xn.std(axis=0)
    rows = []
    for z in z_grid:
        spc = SPC(z).fit(orig[F].values)
        row = {"z": z, "FAR_on_normal": spc.flag(Xn).mean()}
        for c, sign in [(FORCE, +1), (CUR, -1), (VOLT, -1)]:
            j = F.index(c)
            for k in np.arange(0.5, 12.5, 0.5):
                Xs = Xn.copy()
                Xs[:, j] += sign * k * sigma[j]
                if spc.flag(Xs).mean() >= 0.9:
                    row[f"MDS_{c.split('(')[0].strip()}_ksigma"] = k
                    break
        rows.append(row)
    return pd.DataFrame(rows)


def stable_window(orig, normal_mask):
    ok = orig.loc[normal_mask, F]
    return pd.DataFrame({
        "spec_min": [SPEC[c][0] for c in F], "spec_max": [SPEC[c][1] for c in F],
        "mean": ok.mean(), "LCL(-3σ)": ok.mean() - 3 * ok.std(),
        "UCL(+3σ)": ok.mean() + 3 * ok.std(),
        "P0.5": ok.quantile(0.005), "P99.5": ok.quantile(0.995)}, index=F)


# ====================================================================== 시각화
def plot_similarity(sim, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim.values, cmap="viridis", vmin=0, vmax=1)
    lab = [d.strftime("%m-%d") for d in sim.index]
    ax.set_xticks(range(len(lab)), lab, rotation=45); ax.set_yticks(range(len(lab)), lab)
    ax.set_title("Day-to-day content similarity (shared 20-weld windows)")
    fig.colorbar(im)
    fig.tight_layout(); fig.savefig(os.path.join(out, "day_similarity.png"), dpi=150); plt.close(fig)


def plot_origin(raw, keep, out):
    fig, ax = plt.subplots(figsize=(13, 3))
    x = np.arange(len(raw))
    ax.plot(x, raw[CUR], lw=0.4, color="0.6")
    ax.scatter(x[keep], raw[CUR][keep], s=1, c="C0", label="original (first occurrence)")
    for b in np.where(raw[RUN].diff().fillna(0) > 0)[0]:
        ax.axvline(b, color="k", lw=0.5, ls=":")
    ax.set_ylabel("current (kA)"); ax.set_xlabel("row in file (dotted = run boundary)")
    ax.legend(loc="upper right"); ax.set_title("Only the first rows are original; the rest are replays")
    fig.tight_layout(); fig.savefig(os.path.join(out, "original_vs_replay.png"), dpi=150); plt.close(fig)


def plot_ablation(abl, out):
    cols = [c for c in abl.columns if c not in ("config", "FAR_train_thr", "daily_rho", "daily_p")]
    fig, ax = plt.subplots(figsize=(11, 4))
    w = 0.8 / len(abl)
    for i, (_, r) in enumerate(abl.iterrows()):
        ax.bar(np.arange(len(cols)) + i * w, r[cols].values.astype(float), w, label=r["config"])
    ax.set_xticks(np.arange(len(cols)) + 0.4 - w / 2,
                  [f"S{i+1}" for i in range(len(cols))])
    ax.set_ylabel("detection rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Ablation (S1-S7 = scenarios in ablation.csv)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "ablation.png"), dpi=150); plt.close(fig)


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Welding_Data_Set_01.xlsx")
    ap.add_argument("--out", default="results_v3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 200)

    raw, defects = load_data(args.data)

    # ---------------- [0] 구간 중복 제거
    keep = segment_dedup(raw)
    orig = raw[keep].reset_index(drop=True)
    per_day = pd.DataFrame({"n_rows": raw.groupby(DATE).size(),
                            "n_original": pd.Series(keep, index=raw.index).groupby(raw[DATE]).sum()})
    per_day["original_frac"] = per_day["n_original"] / per_day["n_rows"]
    per_day = per_day.join(defects["total"].rename("defects"))
    per_day.to_csv(os.path.join(args.out, "integrity_by_day.csv"))
    null = chance_null(raw)
    null.to_csv(os.path.join(args.out, "chance_null.csv"))
    print("=" * 72)
    print(f"[0] 구간 중복 제거: 원본 {len(orig)}타점 / 전체 {len(raw)}타점 "
          f"({len(orig)/len(raw):.1%})")
    print("    셔플 귀무모형 (반복 창에 속한 창 수):\n", null.to_string())
    print("    일자별 원본 비율:\n", per_day.round(3).to_string())
    plot_origin(raw, keep, args.out)

    # ---------------- [1] 가설 B
    stat, p_h = homogeneity_test(raw, defects)
    rho_m, p_m, sim = mantel_test(raw, defects)
    sim.to_csv(os.path.join(args.out, "day_similarity.csv"))
    plot_similarity(sim, args.out)
    nolink = per_day.dropna(subset=["defects"])
    zero_orig = nolink[nolink["n_original"] == 0]
    print(f"\n[1] 가설 B 검정")
    print(f"  (a) 구조: 라벨 있는 {len(nolink)}일 중 {len(zero_orig)}일은 원본 타점이 0개인데 "
          f"불량 {int(zero_orig['defects'].sum())}건이 기록됨")
    print(f"  (b) 동질성: chi2={stat:.2f}, p={p_h:.3f}")
    print(f"  (c) Mantel: rho(내용 유사도, |불량률 차|)={rho_m:.3f}, p={p_m:.3f} "
          "(연결 가설이면 유의한 음수 기대)")
    pd.DataFrame([{"homog_chi2": stat, "homog_p": p_h, "mantel_rho": rho_m, "mantel_p": p_m,
                   "days_zero_original": len(zero_orig),
                   "defects_on_zero_original_days": zero_orig["defects"].sum()}]) \
        .to_csv(os.path.join(args.out, "hypothesis_b.csv"), index=False)

    # ---------------- 공통 분할
    spc = SPC().fit(orig[F].values)
    train, hold = make_splits(orig, spc)
    print(f"\n    학습 {train.sum()} / 홀드아웃 {hold.sum()} 타점 (원본·SPC 정상, 연속 블록 분할)")

    # ---------------- [2] 알고리즘 비교
    rows = []
    for fs in ["base", "physics"]:
        for algo in ["IF", "MCD", "LOF"]:
            r = evaluate(System(algo, fs, False), orig, train, hold, spc)
            rows.append({"algo": algo, "features": fs, **r})
    algo_df = pd.DataFrame(rows)
    algo_df.to_csv(os.path.join(args.out, "algo_comparison.csv"), index=False)
    print("\n[2] 알고리즘 비교 (홀드아웃 FAR 1%로 맞춘 탐지율)\n", algo_df.round(3).to_string(index=False))
    point_cols = [c for c in algo_df.columns if c not in ("algo", "features", "FAR_train_thr")
                  and "드리프트" not in c]
    best = algo_df.assign(m=algo_df[point_cols].mean(axis=1)).groupby("algo")["m"].mean().idxmax()
    print(f"    → 선택 알고리즘: {best} (점 이상 시나리오 평균 탐지율 기준)")

    # ---------------- [3] Ablation
    configs = [("A0 base", "base", False), ("A1 +physics", "physics", False),
               ("A2 +seq", "physics+seq", False), ("A3 +EWMA", "physics+seq", True)]
    rows = []
    for name, fs, ew in configs:
        sysm = System(best, fs, ew)
        r = evaluate(sysm, orig, train, hold, spc)
        sysm.fit(orig, train, spc)
        rho, p = daily_link(sysm, raw, defects)
        rows.append({"config": name, **r, "daily_rho": rho, "daily_p": p})
    abl = pd.DataFrame(rows)
    abl.to_csv(os.path.join(args.out, "ablation.csv"), index=False)
    plot_ablation(abl, args.out)
    print("\n[3] Ablation (알고리즘 고정, 대안 하나씩 추가)\n", abl.round(3).to_string(index=False))

    # 기준 FAR(1%)을 바꿔도 순위가 유지되는지 확인
    rows = []
    for far in (0.005, 0.01, 0.02, 0.05):
        for name, fs, ew in configs:
            r = evaluate(System(best, fs, ew), orig, train, hold, spc, target=far)
            pts = [v for k, v in r.items() if k != "FAR_train_thr" and "드리프트" not in k]
            rows.append({"FAR": far, "config": name, "point_mean_detect": np.mean(pts)})
    fs_df = pd.DataFrame(rows).pivot(index="config", columns="FAR", values="point_mean_detect")
    fs_df.to_csv(os.path.join(args.out, "far_sensitivity.csv"))
    print("\n[3-1] 기준 FAR별 점 이상 평균 탐지율 (순위 유지 여부 확인)\n", fs_df.round(3).to_string())

    # ---------------- [4] 시드 안정성 (Stage 2만)
    rows = []
    for algo in ["IF", "MCD"]:
        ss = seed_stability(orig, train, spc, algo, "physics")
        ss["algo"] = algo
        rows.append(ss)
    ss = pd.concat(rows)
    ss.to_csv(os.path.join(args.out, "seed_stability.csv"), index=False)
    print("\n[4] 시드 안정성 (Stage 2만, seed 0 대비 19회)\n",
          ss.groupby("algo")[["score_spearman", "stage2_jaccard", "n_stage2"]]
          .agg(["mean", "min"]).round(3).to_string())

    # ---------------- [5] SPC 임계값 trade-off
    tr = spc_tradeoff(orig)
    tr.to_csv(os.path.join(args.out, "spc_tradeoff.csv"), index=False)
    print("\n[5] SPC 임계값에 따른 오경보율·MDS\n", tr.round(4).to_string(index=False))

    # ---------------- [6] 최종 시스템 + 안정 조건
    # 점 이상 시나리오 평균 탐지율이 가장 높은 구성을 최종 채택 (동점이면 단순한 쪽)
    pc = [c for c in abl.columns if c not in ("config", "FAR_train_thr", "daily_rho", "daily_p")
          and "드리프트" not in c]
    best_i = int(np.argmax(abl[pc].mean(axis=1).round(3).values))
    final_name, final_fs, final_ew = configs[best_i]
    print(f"\n    → 최종 구성: {final_name}")
    final = System(best, final_fs, final_ew).fit(orig, ~spc.flag(orig[F].values), spc)
    s1, s2, s3 = final.predict(raw)
    o1, o2, o3 = final.predict(orig)
    sw = stable_window(orig, ~(o1 | o2 | o3))
    sw.to_csv(os.path.join(args.out, "stable_window.csv"))
    print(f"\n[6] 최종({best}, {final_fs}, EWMA={final_ew}) 전체 타점: "
          f"SPC {s1.sum()} / Stage2 {s2.sum()} / EWMA {s3.sum()}")
    print("    안정 공정조건 (원본 타점 중 이상 제외)\n", sw.round(3).T.to_string())

    # ---------------- [7] 검사 우선순위: SPC 이탈(robust z 큰 순) → Stage 2 점수 큰 순
    _, sc_all = final.scores(raw)
    zmax = spc.zscore(raw[F].values).max(axis=1)
    pri = raw.assign(original=keep, stage1_spc=s1, stage2=s2, stage3_ewma=s3,
                     max_robust_z=zmax, risk_score=sc_all[0],
                     risk_percentile=pd.Series(sc_all[0]).rank(pct=True).values,
                     deviating_var=np.array(F)[spc.zscore(raw[F].values).argmax(axis=1)])
    pri = pri.sort_values(["stage1_spc", "max_robust_z", "risk_score"],
                          ascending=[False, False, False])
    pri.insert(0, "priority", np.arange(1, len(pri) + 1))
    pri.to_csv(os.path.join(args.out, "scored_points.csv"), index=False)
    top = pri[~pri["stage1_spc"]].head(10)
    print("\n[7] 검사 우선순위 (SPC 이탈 다음 순서, Stage 2 점수 상위 10)\n",
          top[["priority", DATE, "idx", *F, "risk_score", "risk_percentile"]]
          .round(3).to_string(index=False))
    print(f"\n[done] 결과 저장: {args.out}/")


if __name__ == "__main__":
    main()