"""Слабая калибровка на тесте: calibration-in-the-large и calibration slope. Вход: results/cmp_test_predictions.csv.
slope: коэффициент b в logit P(y=1) = a + b*logit(p); intercept (CITL): a при b=1 (offset).
Интервалы: стратифицированный бутстреп, 2000 повторов, random_state=42."""
import numpy as np, pandas as pd
from scipy.optimize import minimize

def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))

def nll(params, X, y, off):
    z = X @ params + off
    return np.sum(np.logaddexp(0, z) - y * z)

def grad(params, X, y, off):
    z = X @ params + off
    return X.T @ (1 / (1 + np.exp(-z)) - y)

def fit_slope(l, y):
    X = np.column_stack([np.ones_like(l), l])
    r = minimize(nll, np.zeros(2), args=(X, y, 0.0), jac=grad, method="BFGS")
    return r.x  # a, b

def fit_citl(l, y):
    X = np.ones((len(l), 1))
    r = minimize(nll, np.zeros(1), args=(X, y, l), jac=grad, method="BFGS")
    return r.x[0]

def stats(l, y):
    a, b = fit_slope(l, y)
    return fit_citl(l, y), b

def run(name, p, y, B=2000, seed=42):
    l = logit(p)
    citl, slope = stats(l, y)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    res = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        res.append(stats(l[idx], y[idx]))
    res = np.array(res)
    lo = np.percentile(res, 2.5, axis=0); hi = np.percentile(res, 97.5, axis=0)
    share = float(np.mean(res[:, 1] > 20))
    line = (f"{name}: n={len(y)}, calibration-in-the-large = {citl:.3f} [{lo[0]:.3f}; {hi[0]:.3f}], "
            f"calibration slope = {slope:.3f} [{lo[1]:.3f}; {hi[1]:.3f}]")
    return line, share

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    t = pd.read_csv("results/cmp_test_predictions.csv", index_col=0)
    y = t["y_true"].to_numpy(float)
    lines = []
    shares = []
    for name, col in (("Собственная логистическая регрессия, тест", "y_prob_own_lr"),
                      ("Случайный лес, тест", "y_prob_random_forest")):
        line, share = run(name, t[col].to_numpy(), y)
        lines.append(line); shares.append(share)
    o = pd.read_csv("results/l2_oof_predictions.csv", index_col=0)
    yo = o["y_true"].to_numpy(float)
    oof = np.array([stats(logit(o[c].to_numpy()), yo) for c in o.columns if c.startswith("y_prob_oof_r")])
    lines.append(f"OOF (собственная ЛР, {len(yo)} объектов, {len(oof)} повторов): "
                 f"calibration slope {oof[:, 1].min():.2f}-{oof[:, 1].max():.2f}, "
                 f"calibration-in-the-large {oof[:, 0].min():.2f}-{oof[:, 0].max():.2f}.")
    lines.append(f"Примечание: верхняя граница бутстреп-интервала slope не ограничена: в {shares[0]:.0%} "
                 f"повторов (регрессия) и {shares[1]:.0%} (лес) выборка почти полностью разделима по logit(p).")
    text = "\n".join(lines)
    print(text)
    with open("results/calibration_check.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
