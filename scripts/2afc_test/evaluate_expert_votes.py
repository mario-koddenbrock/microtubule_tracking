import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

LIKERT_CSV = 'results/expert_validation/likert_ratings.csv'
AFC_CSV = 'results/expert_validation/2afc_choices.csv'
RESULTS_DIR = 'results/expert_validation'
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# Seaborn style
# -------------------------
sns.set_theme(style='whitegrid', context='talk')
PALETTE = sns.color_palette('Set2')
DIVERGING = sns.diverging_palette(260, 10, as_cmap=True)

# -------------------------
# Load data
# -------------------------
likert = pd.read_csv(LIKERT_CSV)
afc = pd.read_csv(AFC_CSV)

likert = likert.rename(columns={
    "user": "user_name",
    "expert": "user_name",
    "rating": "rating_score",
    "image": "image_filename",
    "type": "image_type"
})
afc = afc.rename(columns={
    "user": "user_name",
    "expert": "user_name",
})

if "rating_score" in likert.columns:
    likert["rating_score"] = pd.to_numeric(likert["rating_score"], errors="coerce")

# -------------------------
# Helpers
# -------------------------
def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

def clopper_pearson_ci(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    lower = stats.beta.ppf(alpha/2, k, n-k+1)
    upper = stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return (lower, upper)

def cliffs_delta(x, y):
    x = np.array(x); y = np.array(y)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    gt = sum((xi > yi) for xi in x for yi in y)
    lt = sum((xi < yi) for xi in x for yi in y)
    return (gt - lt) / (n1 * n2)

def roc_auc_from_scores(scores, labels):
    s = pd.Series(scores); l = pd.Series(labels)
    df = pd.DataFrame({"s": s, "l": l}).dropna()
    if df["l"].nunique() != 2:
        return np.nan
    real = df[df.l == 1]["s"]; synth = df[df.l == 0]["s"]
    if len(real) == 0 or len(synth) == 0:
        return np.nan
    U, _ = stats.mannwhitneyu(real, synth, alternative="two-sided")
    return U / (len(real) * len(synth))

# -------------------------
# 1) LIKERT ANALYSIS
# -------------------------
# Per-expert Likert distributions
bins = np.arange(0.5, 5.6, 1.0)
if "user_name" in likert and len(likert["user_name"].unique()) > 0:
    g = sns.displot(
        data=likert.dropna(subset=["rating_score"]),
        x="rating_score",
        col="user_name",
        col_wrap=3,
        bins=bins,
        discrete=True,
        facet_kws=dict(sharex=True, sharey=False),
        color=PALETTE[0]
    )
    g.set_axis_labels("Likert rating", "Count")
    g.set_titles("Likert distribution — {col_name}")
    save_fig(os.path.join(RESULTS_DIR, "likert_hist_per_expert.png"))

# Overall by type
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(
    data=likert.dropna(subset=["rating_score"]),
    x="rating_score", hue="image_type",
    multiple="dodge", bins=bins, shrink=0.9,
    palette=PALETTE, edgecolor="black", ax=ax
)
sns.kdeplot(
    data=likert.dropna(subset=["rating_score"]),
    x="rating_score", hue="image_type",
    common_norm=False, fill=False, ax=ax, palette=PALETTE, lw=2
)
ax.set_title("Likert distribution by image_type (overall)")
ax.set_xlabel("Likert rating"); ax.set_ylabel("Count")
save_fig(os.path.join(RESULTS_DIR, "likert_hist_by_type_overall.png"))

# Per-expert split by type
for expert, sub in likert.groupby("user_name"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(
        data=sub, x="image_type", y="rating_score",
        inner=None, palette=PALETTE, ax=ax, cut=0
    )
    sns.boxplot(
        data=sub, x="image_type", y="rating_score",
        width=0.25, showcaps=True, fliersize=2, boxprops={'zorder': 3},
        ax=ax, color="white"
    )
    sns.stripplot(
        data=sub, x="image_type", y="rating_score",
        color="k", size=3, alpha=0.3, dodge=True, ax=ax
    )
    ax.set_title(f"Likert by type — {expert}")
    ax.set_xlabel(""); ax.set_ylabel("Rating")
    save_fig(os.path.join(RESULTS_DIR, f"likert_by_type_{expert}.png"))

# Stats
real_scores = likert.loc[likert.image_type=="real", "rating_score"].dropna()
synth_scores = likert.loc[likert.image_type=="synthetic", "rating_score"].dropna()
mw_stat, mw_p = (np.nan, np.nan)
delta = np.nan; auc = np.nan
if len(real_scores) > 0 and len(synth_scores) > 0:
    mw_stat, mw_p = stats.mannwhitneyu(real_scores, synth_scores, alternative="two-sided")
    delta = cliffs_delta(real_scores, synth_scores)
    auc = roc_auc_from_scores(pd.concat([real_scores, synth_scores]),
                              [1]*len(real_scores) + [0]*len(synth_scores))

# Inter-expert agreement
experts = sorted(likert["user_name"].dropna().unique().tolist())
pivot = likert.pivot_table(index="image_filename", columns="user_name", values="rating_score", aggfunc="mean")

for corr_method, fname, title in [
    ("spearman", "likert_agreement_spearman.png", "Inter-expert agreement (Spearman)"),
    ("kendall", "likert_agreement_kendall.png", "Inter-expert agreement (Kendall)")
]:
    corr = pivot.corr(method=corr_method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        corr, mask=mask, vmin=-1, vmax=1, cmap=DIVERGING, center=0,
        square=True, cbar_kws={"shrink": .75, "label": "Correlation"},
        linewidths=.5, annot=True, fmt=".2f", ax=ax
    )
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    save_fig(os.path.join(RESULTS_DIR, fname))

# Likert AUC per expert
expert_likert_auc = []
for expert, sub in likert.groupby("user_name"):
    a = roc_auc_from_scores(sub["rating_score"], (sub["image_type"]=="real").astype(int).values)
    expert_likert_auc.append({"user_name": expert, "likert_auc_real_vs_synth": a})
expert_likert_auc = pd.DataFrame(expert_likert_auc)

if len(expert_likert_auc):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    order = expert_likert_auc.sort_values("likert_auc_real_vs_synth", ascending=False)
    sns.barplot(
        data=order,
        y="user_name", x="likert_auc_real_vs_synth",
        hue="user_name", dodge=False, legend=False,
        palette=PALETTE, ax=ax
    )
    ax.set_xlabel("Likert AUC (real > synthetic)")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    for i, v in enumerate(order["likert_auc_real_vs_synth"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    ax.set_title("Who separates real vs synthetic (Likert AUC)?")
    save_fig(os.path.join(RESULTS_DIR, "likert_auc_per_expert.png"))

# -------------------------
# 2) 2AFC ANALYSIS
# -------------------------
afc["pair_kind"] = np.where(
    afc["image1_type"] == afc["image2_type"],
    afc["image1_type"],
    "mixed"
)
afc["is_correct"] = np.where(
    afc["pair_kind"]=="mixed",
    (afc["chosen_image_type"]=="real").astype(int),
    np.nan
)
afc["chose_left"] = np.where(
    afc["pair_kind"]!="mixed",
    (afc["chosen_image_filename"]==afc["image1_filename"]).astype(int),
    np.nan
)

mixed = afc[afc["pair_kind"]=="mixed"].copy()
overall_n = mixed["is_correct"].notna().sum()
overall_k = mixed["is_correct"].sum()
overall_acc = (overall_k / overall_n) if overall_n else np.nan
overall_ci = clopper_pearson_ci(int(overall_k), int(overall_n)) if overall_n else (np.nan, np.nan)
overall_pval = stats.binomtest(int(overall_k), int(overall_n), p=0.5, alternative="two-sided").pvalue if overall_n else np.nan

per_expert_rows = []
for expert, sub in mixed.groupby("user_name"):
    n = sub["is_correct"].notna().sum()
    k = sub["is_correct"].sum()
    acc = k/n if n else np.nan
    lo, hi = clopper_pearson_ci(int(k), int(n)) if n else (np.nan, np.nan)
    pval = stats.binomtest(int(k), int(n), p=0.5, alternative="two-sided").pvalue if n else np.nan
    per_expert_rows.append({
        "user_name": expert, "n_trials": n, "k_correct": int(k),
        "accuracy": acc, "ci_low": lo, "ci_high": hi, "binom_p_vs_0.5": pval
    })
per_expert_acc = pd.DataFrame(per_expert_rows).sort_values("accuracy", ascending=False)

if len(per_expert_acc):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=per_expert_acc,
        y="user_name", x="accuracy",
        hue="user_name", dodge=False, legend=False,
        palette=PALETTE, ax=ax
    )
    for i, r in per_expert_acc.iterrows():
        ax.errorbar(
            x=r["accuracy"], y=i,
            xerr=[[r["accuracy"] - r["ci_low"]], [r["ci_high"] - r["accuracy"]]],
            fmt="none", ecolor="k", elinewidth=1.2, capsize=4
        )
        ax.text(min(r["ci_high"]+0.02, 1.02), i, f"{r['accuracy']:.2f}", va="center")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("2AFC accuracy (95% CI)")
    ax.set_ylabel("")
    ax.set_title("Which experts are fooled? (higher = better at spotting real)")
    save_fig(os.path.join(RESULTS_DIR, "afc_accuracy_per_expert.png"))

mixed["synthetic_was_chosen"] = (mixed["chosen_image_type"]=="synthetic").astype(int)
fool_rate_overall = mixed["synthetic_was_chosen"].mean() if len(mixed) else np.nan

fool_rows = []
for expert, sub in mixed.groupby("user_name"):
    fr = sub["synthetic_was_chosen"].mean()
    fool_rows.append({"user_name": expert, "fool_rate_synth_chosen": fr, "n_trials": len(sub)})
per_expert_fool = pd.DataFrame(fool_rows).sort_values("fool_rate_synth_chosen", ascending=False)

if len(per_expert_fool):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(
        data=per_expert_fool,
        y="user_name", x="fool_rate_synth_chosen",
        hue="user_name", dodge=False, legend=False,
        palette=PALETTE, ax=ax
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fool rate (synthetic chosen in mixed pairs)")
    ax.set_ylabel("")
    for i, v in enumerate(per_expert_fool["fool_rate_synth_chosen"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    ax.set_title("Which experts are tricked by synthetic images?")
    save_fig(os.path.join(RESULTS_DIR, "afc_fool_rate_per_expert.png"))

controls = afc[afc["pair_kind"]!="mixed"].copy()
pos_rows = []
for expert, sub in controls.groupby("user_name"):
    n = sub["chose_left"].notna().sum()
    left_rate = sub["chose_left"].mean() if n else np.nan
    pval = stats.binomtest(int(sub["chose_left"].sum()), int(n), p=0.5).pvalue if n else np.nan
    pos_rows.append({"user_name": expert, "left_bias_rate_controls": left_rate, "n_control_trials": n, "p_vs_0.5": pval})
per_expert_posbias = pd.DataFrame(pos_rows).sort_values("left_bias_rate_controls", ascending=False)

if len(per_expert_posbias):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    sns.barplot(
        data=per_expert_posbias,
        y="user_name", x="left_bias_rate_controls",
        hue="user_name", dodge=False, legend=False,
        palette=PALETTE, ax=ax
    )
    ax.axvline(0.5, ls="--", color="k", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Chose-left rate (controls)")
    ax.set_ylabel("")
    for i, r in per_expert_posbias.iterrows():
        star = " *" if pd.notna(r["p_vs_0.5"]) and r["p_vs_0.5"] < 0.05 else ""
        ax.text(r["left_bias_rate_controls"] + 0.01, i,
                f"{r['left_bias_rate_controls']:.2f}{star}", va="center")
    ax.set_title("Control trials: position bias (left/right)")
    save_fig(os.path.join(RESULTS_DIR, "control_position_bias.png"))

# -------------------------
# 3) CROSS-LINK
# -------------------------
cross = per_expert_fool.merge(expert_likert_auc, on="user_name", how="left")
rho, rho_p = stats.spearmanr(
    cross["fool_rate_synth_chosen"], cross["likert_auc_real_vs_synth"],
    nan_policy="omit"
) if len(cross) > 1 else (np.nan, np.nan)

if len(cross):
    jp = sns.jointplot(
        data=cross,
        x="likert_auc_real_vs_synth",
        y="fool_rate_synth_chosen",
        kind="reg", scatter_kws=dict(s=40, alpha=0.8),
        height=5, color=PALETTE[1]
    )
    jp.ax_joint.set_xlim(0, 1); jp.ax_joint.set_ylim(0, 1)
    jp.ax_joint.set_xlabel("Likert AUC (real vs synthetic separation)")
    jp.ax_joint.set_ylabel("Fool rate (synthetic chosen)")
    jp.ax_joint.set_title(f"Do Likert separations predict who gets fooled?\nSpearman ρ={rho:.2f}, p={rho_p:.3f}")
    jp.figure.subplots_adjust(top=0.88)
    jp.figure.savefig(os.path.join(RESULTS_DIR, "likert_auc_vs_foolrate.png"), dpi=220, bbox_inches="tight")
    plt.close(jp.figure)

# -------------------------
# 4) PER-IMAGE
# -------------------------
mixed_pairs = mixed.copy()
def synthetic_image_filename(row):
    if row["image1_type"]=="synthetic":
        return row["image1_filename"]
    else:
        return row["image2_filename"]

mixed_pairs["synthetic_image"] = mixed_pairs.apply(synthetic_image_filename, axis=1)
mixed_pairs["synthetic_won"] = (mixed_pairs["chosen_image_type"]=="synthetic").astype(int)

per_synth_image = mixed_pairs.groupby("synthetic_image")["synthetic_won"].agg(["mean","count"]).reset_index()
per_synth_image = per_synth_image.rename(columns={"mean":"foolability", "count":"n_trials"}).sort_values("foolability", ascending=False)

likert_by_image = likert.groupby(["image_filename","image_type"])["rating_score"].agg(["mean","std","count"]).reset_index()
likert_by_image = likert_by_image.rename(columns={"mean":"mean_rating","std":"std_rating","count":"n_ratings"})

synth_likert = likert_by_image[likert_by_image["image_type"]=="synthetic"].copy()
per_synth_image = per_synth_image.merge(
    synth_likert[["image_filename","mean_rating","n_ratings"]],
    left_on="synthetic_image", right_on="image_filename", how="left"
).drop(columns=["image_filename"])

if len(per_synth_image):
    topN = per_synth_image.head(15).copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=topN,
        y="synthetic_image", x="foolability",
        color=PALETTE[2], ax=ax
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Foolability (win rate vs real in mixed pairs)")
    ax.set_ylabel("")
    for i, r in enumerate(topN.itertuples()):
        ax.text(r.foolability + 0.01, i, f"{r.foolability:.2f} (n={r.n_trials})", va="center")
    ax.set_title("Most convincing synthetic images (Top 15)")
    save_fig(os.path.join(RESULTS_DIR, "per_synthetic_image_foolability_top15.png"))

# -------------------------
# 5) REPORTING
# -------------------------
overall_row = {
    "overall_likert_real_mean": float(real_scores.mean()) if len(real_scores)>0 else np.nan,
    "overall_likert_synth_mean": float(synth_scores.mean()) if len(synth_scores)>0 else np.nan,
    "likert_MannWhitneyU": float(mw_stat) if not np.isnan(mw_stat) else np.nan,
    "likert_MW_pvalue": float(mw_p) if not np.isnan(mw_p) else np.nan,
    "likert_cliffs_delta": float(delta) if not np.isnan(delta) else np.nan,
    "likert_auc_real_vs_synth": float(auc) if not np.isnan(auc) else np.nan,
    "afc_overall_accuracy": float(overall_acc) if not np.isnan(overall_acc) else np.nan,
    "afc_overall_ci_low": float(overall_ci[0]) if not np.isnan(overall_ci[0]) else np.nan,
    "afc_overall_ci_high": float(overall_ci[1]) if not np.isnan(overall_ci[1]) else np.nan,
    "afc_overall_p_vs_0.5": float(overall_pval) if not np.isnan(overall_pval) else np.nan,
    "afc_overall_fool_rate": float(fool_rate_overall) if not np.isnan(fool_rate_overall) else np.nan,
    "n_experts": int(len(experts)),
    "n_likert": int(len(likert)),
    "n_afc": int(len(afc)),
    "n_afc_mixed": int(len(mixed)),
    "n_afc_controls": int(len(controls)),
}
summary_overall = pd.DataFrame([overall_row])

per_expert_all = per_expert_acc.merge(per_expert_fool[["user_name","fool_rate_synth_chosen"]], on="user_name", how="left")
per_expert_all = per_expert_all.merge(expert_likert_auc, on="user_name", how="left")
per_expert_all = per_expert_all.merge(per_expert_posbias[["user_name","left_bias_rate_controls","n_control_trials","p_vs_0.5"]], on="user_name", how="left")

summary_overall.to_csv(os.path.join(RESULTS_DIR, "summary_overall.csv"), index=False)
per_expert_all.to_csv(os.path.join(RESULTS_DIR, "summary_per_expert.csv"), index=False)
per_synth_image.to_csv(os.path.join(RESULTS_DIR, "summary_per_synthetic_image.csv"), index=False)
likert_by_image.to_csv(os.path.join(RESULTS_DIR, "likert_by_image.csv"), index=False)

# -------------------------
# 6) REPORT
# -------------------------
def pct(x): return f"{100*x:.1f}%" if pd.notna(x) else "n/a"
def fmt(x, nd=3): return f"{x:.{nd}f}" if pd.notna(x) else "n/a"

top_tricked = per_expert_all.sort_values("fool_rate_synth_chosen", ascending=False).head(5)
top_discriminators = per_expert_all.sort_values("likert_auc_real_vs_synth", ascending=False).head(5)

report_lines = []
report_lines.append(f"# Can we trick the experts?\n")
report_lines.append(f"_Auto-report generated {datetime.now().isoformat(timespec='seconds')}_\n")
report_lines.append("## TL;DR\n")
report_lines.append(f"- **2AFC overall accuracy** on mixed pairs: **{pct(overall_acc)}** (95% CI {pct(overall_ci[0])}–{pct(overall_ci[1])}; p vs 50/50 chance = {fmt(overall_pval)}).")
report_lines.append(f"- **Fool rate** (synthetic chosen in mixed pairs): **{pct(fool_rate_overall)}**.")
report_lines.append(f"- **Likert separation** (AUC real > synthetic): **{fmt(auc)}** (Cliff's delta {fmt(delta)}; Mann–Whitney p={fmt(mw_p)}).")
report_lines.append("")
report_lines.append("## Which experts get tricked?\n")
for _, r in top_tricked.iterrows():
    report_lines.append(f"- **{r['user_name']}**: fool rate {pct(r['fool_rate_synth_chosen'])}, 2AFC accuracy {pct(r['accuracy'])} (n={int(r['n_trials'])}).")
report_lines.append("\n## Who can tell apart real vs fake from Likert alone?\n")
for _, r in top_discriminators.iterrows():
    report_lines.append(f"- **{r['user_name']}**: Likert AUC {fmt(r['likert_auc_real_vs_synth'])}; 2AFC acc {pct(r['accuracy'])}.")
report_lines.append("\n## What do the control tests reveal?\n")
report_lines.append("- Control trials are pairs where **both images are the same type** (both real or both synthetic).")
if len(controls):
    mean_left = per_expert_posbias["left_bias_rate_controls"].mean()
    report_lines.append(f"- Average **left/right position bias** in controls: chose-left rate {pct(mean_left)} across experts.")
    biased = per_expert_posbias[per_expert_posbias["p_vs_0.5"] < 0.05]
    if len(biased):
        names = ", ".join(biased["user_name"].tolist())
        report_lines.append(f"- Experts showing **significant position bias** (p<0.05): {names}.")
    else:
        report_lines.append("- No expert shows a statistically significant position bias (p<0.05).")
else:
    report_lines.append("- No control trials found in the dataset.")
report_lines.append("\n## Figures\n")
figs = [
    "likert_hist_per_expert.png",
    "likert_hist_by_type_overall.png",
    "likert_agreement_spearman.png",
    "likert_agreement_kendall.png",
    "likert_auc_per_expert.png",
    "afc_accuracy_per_expert.png",
    "afc_fool_rate_per_expert.png",
    "control_position_bias.png",
    "likert_auc_vs_foolrate.png",
    "per_synthetic_image_foolability_top15.png",
]
for f in figs: report_lines.append(f"- {f}")
report_lines.append("\n## Do Likert separations predict who gets fooled?\n")
report_lines.append(f"- Spearman correlation between per-expert Likert AUC and fool rate: ρ={fmt(rho)}, p={fmt(rho_p)}.")
report_lines.append("\n## Which synthetic images are most convincing?\n")
if len(per_synth_image):
    head = per_synth_image.head(10)
    for _, r in head.iterrows():
        report_lines.append(f"- `{r['synthetic_image']}`: foolability {pct(r['foolability'])} over {int(r['n_trials'])} trials; mean Likert {fmt(r['mean_rating'])} (n={int(r['n_ratings']) if pd.notna(r['n_ratings']) else 0}).")
else:
    report_lines.append("- No mixed 2AFC trials found to assess synthetic-image foolability.")

report_path = os.path.join(RESULTS_DIR, "can_we_trick_the_experts_report.md")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
