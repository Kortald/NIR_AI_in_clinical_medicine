
# Графики для отчёта.
# Каждая функция строит один рисунок и сохраняет его (PNG 300 dpi + PDF).
# Данные приходят готовыми: обучение здесь не выполняется.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.evaluation import clopper_pearson

# Цвета (текст всегда в цветах текста, а не серий)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e4e3df"
AXIS = "#b9b8b3"
SURFACE = "#ffffff"

BLUE = "#2a78d6"
BLUE_DARK = "#184f95"
ORANGE = "#eb6834"
GREEN = "#1baf7a"

FIGURE_WIDTH_TWO_PANELS = 7.2
FIGURE_HEIGHT = 3.3


# Общий стиль всех рисунков
def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 9,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": SURFACE,
    })


def _save(fig, out_stem):
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    paths = []
    for suffix in (".png", ".pdf"):
        path = out_stem.with_suffix(suffix)
        fig.savefig(path)
        paths.append(path)

    plt.close(fig)
    return paths


def _note(fig, text, y=-0.01):
    if text:
        fig.text(0.01, y, text, fontsize=7.5, color=INK_MUTED,
                 ha="left", va="top")


# Названия признаков на русском для подписей
_MEASURES = {
    "radius": "радиус",
    "texture": "текстура",
    "perimeter": "периметр",
    "area": "площадь",
    "smoothness": "гладкость",
    "compactness": "компактность",
    "concavity": "вогнутость",
    "concave points": "число вогнутых точек",
    "symmetry": "симметрия",
    "fractal dimension": "фрактальная размерность",
}
_PREFIXES = {"mean": "среднее", "error": "ошибка", "worst": "наихудшее"}


def russian_feature_name(name):
    for prefix, prefix_ru in _PREFIXES.items():
        if name.startswith(prefix + " "):
            measure = name[len(prefix) + 1:]
            return f"{_MEASURES.get(measure, measure)} ({prefix_ru})"
        if name.endswith(" " + prefix):
            measure = name[: -len(prefix) - 1]
            return f"{_MEASURES.get(measure, measure)} ({prefix_ru})"
    return name


# Рисунок 1. Подбор регуляризации: log-loss и ROC-AUC от C
def plot_logloss_vs_c(summary, selected_c, n_splits, out_stem, note=None):
    apply_style()

    data = summary.sort_values("C")
    c = data["C"].to_numpy()

    fig, (ax_loss, ax_auc) = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH_TWO_PANELS, FIGURE_HEIGHT)
    )

    panels = (
        (ax_loss, "log_loss", "Log-loss (меньше — лучше)", np.argmin),
        (ax_auc, "roc_auc", "ROC-AUC (больше — лучше)", np.argmax),
    )

    for ax, metric, title, best_fn in panels:
        mean = data[f"{metric}_mean"].to_numpy()
        se = (data[f"{metric}_std"] / np.sqrt(n_splits)).to_numpy()
        best = int(best_fn(mean))

        ax.fill_between(c, mean - se, mean + se, color=BLUE, alpha=0.12,
                        linewidth=0)
        ax.plot(c, mean, color=BLUE, marker="o", markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2)

        # Граница правила одной стандартной ошибки (только для log-loss:
        # по нему выбирается C; ROC-AUC на панели справа показан для справки)
        if metric == "log_loss":
            limit = mean[best] + se[best]
            ax.axhline(limit, color=INK_MUTED, linewidth=0.9,
                       linestyle=(0, (4, 3)))

        ax.axvline(selected_c, color=ORANGE, linewidth=1.6)

        ax.set_xscale("log")
        ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1, 3])
        ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1", "3"])
        ax.minorticks_off()
        ax.set_xlabel("C (чем больше, тем слабее регуляризация)")
        ax.set_title(title, loc="left")

        # Оптимум по среднему
        ax.plot([c[best]], [mean[best]], marker="o", markersize=7,
                markerfacecolor=BLUE, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=5)

    # Подписи на левой панели
    y0, y1 = ax_loss.get_ylim()
    best_loss = int(np.argmin(data["log_loss_mean"].to_numpy()))
    ax_loss.annotate(
        f"минимум\nC = {c[best_loss]:.2f}",
        xy=(c[best_loss], data["log_loss_mean"].to_numpy()[best_loss]),
        xytext=(c[best_loss] * 1.0, y0 + 0.50 * (y1 - y0)),
        ha="center", fontsize=8, color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
    )
    ax_loss.text(selected_c * 0.93, y0 + 0.93 * (y1 - y0),
                 f"выбрано\nC = {selected_c:.2f}",
                 fontsize=8, color=INK_SECONDARY, ha="right", va="top")
    ax_loss.text(0.0105, limit_label_y(data, n_splits), "граница\nодной ошибки",
                 fontsize=7.5, color=INK_MUTED, va="bottom")

    ax_auc.text(selected_c * 0.93, ax_auc.get_ylim()[0]
                + 0.06 * (ax_auc.get_ylim()[1] - ax_auc.get_ylim()[0]),
                "ROC-AUC почти не\nразличает значения C",
                fontsize=8, color=INK_SECONDARY, ha="right", va="bottom")

    legend_items = [
        Line2D([0], [0], color=BLUE, marker="o", markersize=4.5,
               label="среднее по фолдам ± σ/√5"),
        Line2D([0], [0], color=ORANGE, label="выбранное C"),
        Line2D([0], [0], color=INK_MUTED, linestyle=(0, (4, 3)),
               linewidth=0.9, label="граница правила одной ошибки"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.09))

    fig.tight_layout()
    _note(fig, note, y=-0.17)
    return _save(fig, out_stem)


def limit_label_y(data, n_splits):
    mean = data["log_loss_mean"].to_numpy()
    se = (data["log_loss_std"] / np.sqrt(n_splits)).to_numpy()
    best = int(np.argmin(mean))
    return mean[best] + se[best] + 0.002


# Рисунок 2. ROC и кривая точность–полнота с рабочей точкой
def plot_roc_pr(y_true, y_prob, threshold, out_stem, note=None):
    apply_style()

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    predicted = y_prob >= threshold
    sensitivity = predicted[y_true == 1].mean()
    specificity = (~predicted[y_true == 0]).mean()
    precision_at = (y_true[predicted] == 1).mean() if predicted.any() else np.nan
    prevalence = y_true.mean()

    fig, (ax_roc, ax_pr) = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH_TWO_PANELS, 3.5)
    )

    # ROC
    ax_roc.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=0.9,
                linestyle=(0, (1, 2)))
    ax_roc.plot(fpr, tpr, color=BLUE)
    ax_roc.plot([1 - specificity], [sensitivity], marker="o", markersize=8,
                color=ORANGE, markeredgecolor=SURFACE, markeredgewidth=1.5,
                zorder=5)
    ax_roc.annotate(
        f"порог {threshold:.2f}\nчувствительность {sensitivity:.2f}\n"
        f"специфичность {specificity:.2f}",
        xy=(1 - specificity, sensitivity), xytext=(0.42, 0.28),
        fontsize=8, color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
    )
    ax_roc.text(0.97, 0.05, f"ROC-AUC = {roc_auc:.3f}", ha="right",
                fontsize=9, color=INK)
    ax_roc.set_xlabel("Доля ложных тревог (1 − специфичность)")
    ax_roc.set_ylabel("Чувствительность")
    ax_roc.set_title("ROC-кривая", loc="left")
    ax_roc.set_xlim(-0.01, 1.0)
    ax_roc.set_ylim(0, 1.02)

    # PR
    ax_pr.axhline(prevalence, color=INK_MUTED, linewidth=0.9,
                  linestyle=(0, (1, 2)))
    ax_pr.plot(recall, precision, color=BLUE)
    ax_pr.plot([sensitivity], [precision_at], marker="o", markersize=8,
               color=ORANGE, markeredgecolor=SURFACE, markeredgewidth=1.5,
               zorder=5)
    ax_pr.text(0.03, prevalence + 0.02,
               f"случайный классификатор\n(доля злокачественных {prevalence:.2f})",
               fontsize=7.5, color=INK_MUTED, va="bottom")
    ax_pr.annotate(
        f"рабочая точка\n(порог {threshold:.2f})",
        xy=(sensitivity, precision_at), xytext=(0.25, 0.62),
        fontsize=8, color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8},
    )
    ax_pr.text(0.03, 0.05, f"AUPRC = {auprc:.3f}", fontsize=9, color=INK)
    ax_pr.set_xlabel("Полнота (чувствительность)")
    ax_pr.set_ylabel("Точность (доля верных среди тревог)")
    ax_pr.set_title("Кривая точность–полнота", loc="left")
    ax_pr.set_xlim(0, 1.01)
    ax_pr.set_ylim(0, 1.02)

    for ax in (ax_roc, ax_pr):
        ax.set_box_aspect(1)

    fig.tight_layout()
    _note(fig, note)
    return _save(fig, out_stem)


# Рисунок 3. Калибровочная кривая и распределение вероятностей
def plot_calibration(y_true, y_prob, out_stem, n_bins=10, note=None):
    footnote = (
        "Корзины с равным числом объектов; отрезки — 95% ДИ Клоппера–Пирсона."
    )
    note = f"{note}\n{footnote}" if note else footnote
    apply_style()

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Корзины с равным числом объектов
    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1]), 0, len(edges) - 2)

    mean_pred, observed, low, high, counts = [], [], [], [], []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if not mask.any():
            continue
        k, n = int(y_true[mask].sum()), int(mask.sum())
        lo, hi = clopper_pearson(k, n)
        mean_pred.append(y_prob[mask].mean())
        observed.append(k / n)
        low.append(lo)
        high.append(hi)
        counts.append(n)

    mean_pred = np.array(mean_pred)
    observed = np.array(observed)
    yerr = np.array([observed - np.array(low), np.array(high) - observed])

    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(4.6, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08},
    )

    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=0.9,
            linestyle=(0, (4, 3)))
    ax.errorbar(mean_pred, observed, yerr=yerr, color=BLUE, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
                capsize=2.5, elinewidth=1.0)
    ax.text(0.62, 0.22, "идеальная\nкалибровка", fontsize=8, color=INK_MUTED,
            ha="left")
    ax.text(0.03, 0.92,
            f"Brier = {brier_score_loss(y_true, y_prob):.3f}\n"
            f"log-loss = {log_loss(y_true, y_prob):.3f}",
            fontsize=8.5, color=INK, va="top")
    ax.set_ylabel("Наблюдаемая доля злокачественных")
    ax.set_title("Калибровочная кривая", loc="left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, 1.05)

    ax_hist.hist(y_prob, bins=np.linspace(0, 1, 21), color=BLUE, alpha=0.55,
                 edgecolor=SURFACE, linewidth=0.8)
    ax_hist.set_xlabel("Предсказанная вероятность злокачественности")
    ax_hist.set_ylabel("Объектов")

    _note(fig, note)
    return _save(fig, out_stem)


# Рисунок 4. Чувствительность и специфичность от порога
def plot_threshold_curves(y_true, y_prob, threshold, min_sensitivity,
                          out_stem, note=None):
    apply_style()

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    grid = np.linspace(0, 1, 201)
    positives = y_prob[y_true == 1]
    negatives = y_prob[y_true == 0]
    sensitivity = np.array([(positives >= t).mean() for t in grid])
    specificity = np.array([(negatives < t).mean() for t in grid])

    sens_at = (positives >= threshold).mean()
    spec_at = (negatives < threshold).mean()

    fig, ax = plt.subplots(figsize=(5.6, 3.6))

    ax.axhline(min_sensitivity, color=INK_MUTED, linewidth=0.9,
               linestyle=(0, (1, 2)))

    ax.axvline(threshold, color=ORANGE, linewidth=1.6)

    ax.plot(grid, sensitivity, color=BLUE, label="Чувствительность")
    ax.plot(grid, specificity, color=BLUE_DARK, linestyle=(0, (5, 2.5)),
            label="Специфичность")

    for value in (sens_at, spec_at):
        ax.plot([threshold], [value], marker="o", markersize=7, color=ORANGE,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)

    ax.text(threshold + 0.015, 0.30,
            f"выбранный порог {threshold:.2f}\n"
            f"чувствительность {sens_at:.3f}\n"
            f"специфичность {spec_at:.3f}",
            fontsize=8, color=INK_SECONDARY, va="top")

    ax.set_xlabel("Порог классификации")
    ax.set_ylabel("Значение")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_title("Выбор порога на out-of-fold предсказаниях", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=INK_MUTED, linewidth=0.9,
                          linestyle=(0, (1, 2))))
    labels.append(f"Требование: чувствительность ≥ {min_sensitivity:.2f}")
    ax.legend(handles, labels, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.20), columnspacing=1.2,
              handlelength=2.2)

    fig.tight_layout()
    _note(fig, note)
    return _save(fig, out_stem)


# Рисунок 5. Коэффициенты модели (стандартизованные признаки)
def plot_coefficients(coef_table, out_stem, top_k=10, note=None):
    apply_style()

    table = coef_table[coef_table["feature"] != "(intercept)"].copy()
    order = table["coef_standardized"].abs().sort_values(ascending=False).index
    table = table.loc[order].head(top_k).iloc[::-1]

    values = table["coef_standardized"].to_numpy()
    colors = [BLUE if v >= 0 else ORANGE for v in values]
    labels = [russian_feature_name(name) for name in table["feature"]]

    fig, ax = plt.subplots(figsize=(6.2, 0.38 * top_k + 1.5))

    positions = np.arange(len(values))
    ax.barh(positions, values, height=0.55, color=colors)
    ax.axvline(0, color=AXIS, linewidth=1.0)

    span = np.abs(values).max()
    for pos, value in zip(positions, values):
        offset = 0.02 * span
        ax.text(value + (offset if value >= 0 else -offset), pos,
                f"{value:+.2f}", va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8, color=INK_SECONDARY)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(min(values.min(), 0) * 1.25 - 0.02 * span,
                max(values.max(), 0) * 1.25 + 0.02 * span)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Коэффициент при стандартизованном признаке")
    ax.set_title(f"Признаки с наибольшим весом (топ {top_k})", loc="left")
    legend_items = []
    if (values >= 0).any():
        legend_items.append(Patch(color=BLUE, label="в сторону злокачественной"))
    if (values < 0).any():
        legend_items.append(Patch(color=ORANGE, label="в сторону доброкачественной"))
    ax.legend(handles=legend_items, loc="upper center", ncol=2,
              bbox_to_anchor=(0.4, -0.16))

    fig.tight_layout()
    _note(fig, note)
    return _save(fig, out_stem)


# Рисунок 6. Сходимость градиентного спуска итоговой модели
def plot_convergence(history, tol, out_stem, note=None):
    apply_style()

    epochs = np.arange(1, len(history) + 1)

    fig, (ax_loss, ax_grad) = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH_TWO_PANELS, FIGURE_HEIGHT)
    )

    ax_loss.plot(epochs, history["loss"], color=BLUE)
    ax_loss.set_xscale("log")
    ax_loss.set_xlabel("Эпоха")
    ax_loss.set_ylabel("Потеря (с учётом L2-штрафа)")
    ax_loss.set_title("Функция потерь", loc="left")
    ax_loss.minorticks_off()

    ax_grad.plot(epochs, history["grad_norm"], color=BLUE)
    ax_grad.axhline(tol, color=ORANGE, linewidth=1.4)
    exponent = int(round(np.log10(tol)))
    ax_grad.text(epochs[0] * 1.15, tol * 1.5,
                 f"порог остановки tol = $10^{{{exponent}}}$",
                 fontsize=8, color=INK_SECONDARY, ha="left", va="bottom")
    ax_grad.set_xscale("log")
    ax_grad.set_yscale("log")
    ax_grad.set_xlabel("Эпоха")
    ax_grad.set_ylabel("Норма градиента")
    ax_grad.set_title("Норма градиента", loc="left")
    ax_grad.minorticks_off()

    fig.tight_layout()
    _note(fig, note)
    return _save(fig, out_stem)


# Названия моделей сравнения и их оформление (цвет + тип линии + маркер,
# чтобы кривые различались и в чёрно-белой печати)
MODEL_LABELS = {
    "own_lr": "Собственная (градиентный спуск)",
    "sklearn_lr": "sklearn LogisticRegression",
    "random_forest": "Случайный лес",
}
# Короткие названия для легенды рисунка
MODEL_SHORT = {
    "own_lr": "Собственная",
    "sklearn_lr": "sklearn LR",
    "random_forest": "Случайный лес",
}
_MODEL_STYLES = {
    "own_lr": (BLUE, "-", "o"),
    "sklearn_lr": (GREEN, (0, (5, 2)), "s"),
    "random_forest": (ORANGE, (0, (1, 1.5)), "^"),
}


# Рисунок 7. Сравнение моделей: ROC и калибровка на одних и тех же данных
def plot_model_comparison(y_true, probs, out_stem, n_bins=10, note=None):
    footnote = ("ROC показана увеличенным фрагментом (доля ложных тревог до 0.25). "
                "Корзины калибровки с равным числом объектов.")
    note = f"{note}\n{footnote}" if note else footnote
    apply_style()

    y_true = np.asarray(y_true, dtype=int)

    fig, (ax_roc, ax_cal) = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH_TWO_PANELS, 3.8)
    )

    ax_roc.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=0.9,
                linestyle=(0, (1, 2)))
    ax_cal.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=0.9,
                linestyle=(0, (4, 3)))

    for model, prob in probs.items():
        color, linestyle, marker = _MODEL_STYLES[model]
        prob = np.asarray(prob, dtype=float)

        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        ax_roc.plot(fpr, tpr, color=color, linestyle=linestyle,
                    label=f"{MODEL_SHORT[model]}, AUC {auc:.3f}")

        edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
        bin_ids = np.clip(np.digitize(prob, edges[1:-1]), 0, len(edges) - 2)
        mean_pred, observed = [], []
        for b in range(len(edges) - 1):
            mask = bin_ids == b
            if mask.any():
                mean_pred.append(prob[mask].mean())
                observed.append(y_true[mask].mean())
        ax_cal.plot(mean_pred, observed, color=color, linestyle=linestyle,
                    marker=marker, markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=1.0,
                    label=f"{MODEL_SHORT[model]}, "
                          f"Brier {brier_score_loss(y_true, prob):.3f}")

    ax_roc.set_xlabel("Доля ложных тревог (1 − специфичность)")
    ax_roc.set_ylabel("Чувствительность")
    ax_roc.set_title("ROC-кривые (фрагмент)", loc="left")
    ax_roc.set_xlim(-0.005, 0.25)
    ax_roc.set_ylim(0.75, 1.005)
    ax_roc.legend(loc="lower right", fontsize=7.5)

    ax_cal.set_xlabel("Предсказанная вероятность")
    ax_cal.set_ylabel("Наблюдаемая доля злокачественных")
    ax_cal.set_title("Калибровка", loc="left")
    ax_cal.set_xlim(-0.02, 1.02)
    ax_cal.set_ylim(-0.03, 1.05)
    ax_cal.legend(loc="upper left", fontsize=7.5)

    for ax in (ax_roc, ax_cal):
        ax.set_box_aspect(1)

    fig.tight_layout()
    _note(fig, note)
    return _save(fig, out_stem)
