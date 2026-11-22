"""
Figures and headline numbers for the report (PLAN 6.1, experiments E1-E9).

    python -m glassbox.src.experiments.figures

Reads only committed results under glassbox/results/ (no model is loaded, a few
seconds) and writes:
    glassbox/results/figures/E*.png      one figure per experiment
    glassbox/results/figures/summary.json   every headline number, with its source file
    glassbox/results/figures/tables.md      the tables (E1 OOD, E2 leakage, E9 replication)

All numbers are realistic-dataset TEST split unless the label says otherwise.
Error bars are the bootstrap 95% CIs stored with each result (Wilson intervals
for the name-origin rates, which are plain proportions of 30).
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("glassbox/results")
OUT = R / "figures"

# Reference categorical palette (validated: CVD and normal-vision separation pass;
# aqua is below 3:1 on the surface, so every bar carries a direct value label)
GLASSBOX, RAW, NOGUARD = "#2a78d6", "#eb6834", "#1baf7a"
BEFORE = "#b5b4ae"          # "before a fix" states: neutral grey, not a series hue
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def load(rel):
    return json.loads((R / rel).read_text())


def err(value, ci):
    """matplotlib yerr from a value and its [lo, hi] interval."""
    return [[value - ci[0]], [ci[1] - value]]


def wilson(p, n, z=1.96):
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def label_bars(ax, bars, fmt="{:.3f}", inside=False):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)


# ------------------------------------------------------------------ E1 redaction
def e1(summary, tables):
    ev = load("redaction/realistic_t0.50.json")["evaluation"]
    types = ["NAME", "ADDRESS", "PHONE_NUMBER", "PATIENT_ID", "DATE_OF_BIRTH", "EMAIL"]
    f1 = [ev["per_type"][t]["f1"] for t in types]
    masked = [ev["masking"]["per_type"][t]["fully_masked_rate"] for t in types]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = range(len(types))
    w = 0.38
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], f1, w, color=GLASSBOX, label="Detection F1")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], masked, w, color=NOGUARD, label="Value fully masked")
    label_bars(ax, b1), label_bars(ax, b2)
    pretty = {"NAME": "Name", "ADDRESS": "Address", "PHONE_NUMBER": "Phone", "PATIENT_ID": "Patient ID",
              "DATE_OF_BIRTH": "Date of birth", "EMAIL": "Email"}
    ax.set_xticks(list(x), [pretty[t] for t in types])
    ax.set_ylim(0.9, 1.012)
    ax.set_title("E1a  Redaction quality per PII type (2,500 notes, threshold 0.5)", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    ax.set_ylabel("rate (axis starts at 0.9)")
    save(fig, "E1a_redaction_per_type.png")

    ood = {k: load(f"robustness/ood_redaction_{k}.json") for k in ["a_prefix", "a_after_fixes", "b_prefix", "b"]}
    rows = ["| PII type | Set A before fixes | Set A after fixes | Set B (held out) before | Set B after |", "|---|---|---|---|---|"]
    for t in types:
        cells = [f"{ood[k]['fully_masked'][t]['rate']:.3f}" for k in ood]
        rows.append(f"| {t} (n A={ood['a_prefix']['fully_masked'][t]['n']}, B={ood['b']['fully_masked'][t]['n']}) | " + " | ".join(cells) + " |")
    rows.append("| **All values** | " + " | ".join(f"**{ood[k]['overall_fully_masked']:.3f}**" for k in ood) + " |")
    tables.append("## E1b  Redaction on out-of-distribution note formats (share of values fully masked)\n\n"
                  "Set A was used to develop fixes; set B was written and committed before any fix and evaluated once. "
                  "240 notes each.\n\n" + "\n".join(rows))

    bias = {k.split(", ")[1].rstrip("]"): v for k, v in ood["b"]["fully_masked"].items() if k.startswith("NAME[patient")}
    order = sorted(bias, key=lambda k: bias[k]["rate"])
    names = {"tr_TR": "Turkish", "vi_VN": "Vietnamese", "es_ES": "Spanish", "nl_NL": "Dutch", "da_DK": "Danish",
             "en_GB": "English (UK)", "en_IE": "English (IE)", "sv_SE": "Swedish"}
    rates = [bias[k]["rate"] for k in order]
    cis = [wilson(bias[k]["rate"], bias[k]["n"]) for k in order]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh([names.get(k, k) for k in order], rates, 0.6, color=GLASSBOX,
                   xerr=[[r - c[0] for r, c in zip(rates, cis)], [c[1] - r for r, c in zip(rates, cis)]],
                   error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    for b, r in zip(bars, rates):
        ax.annotate(f"{r:.2f}", (0.02, b.get_y() + b.get_height() / 2), va="center", fontsize=8.5, color="white",
                    fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", visible=False)
    ax.set_title("E1c  Patient names fully masked, by name origin (held-out set B, n = 30 each)")
    ax.set_xlabel("share fully masked (Wilson 95% CI)")
    save(fig, "E1c_name_origin_bias.png")

    summary["E1"] = {"source": "redaction/realistic_t0.50.json, robustness/ood_redaction_*.json",
                     "overall_f1": round(ev["overall"]["f1"], 4),
                     "fully_masked_rate": round(ev["masking"]["overall"]["fully_masked_rate"], 4),
                     "char_leakage": round(ev["masking"]["overall"]["char_leakage"], 4),
                     "per_type_f1": {t: round(v, 4) for t, v in zip(types, f1)},
                     "ood_fully_masked": {k: v["overall_fully_masked"] for k, v in ood.items()},
                     "name_origin_fully_masked_set_b": {names.get(k, k): bias[k]["rate"] for k in order}}


# ------------------------------------------------------------------ E2 leakage
def e2(summary, tables):
    la = load("retrieval/realistic_t0.50.json")["leak_audit"]
    rows = ["| PII type | Values | Redacted index: fully visible | Redacted index: any part visible | Raw index: any part visible |",
            "|---|---|---|---|---|"]
    for t, v in la["redacted"]["per_type"].items():
        rows.append(f"| {t} | {v['n']:,} | {v['full_leak_rate']:.2%} | {v['partial_leak_rate']:.2%} | "
                    f"{la['raw']['per_type'][t]['partial_leak_rate']:.0%} |")
    o = la["redacted"]["overall"]
    rows.append(f"| **All** | {o['n']:,} | **{o['full_leak_rate']:.2%}** | **{o['partial_leak_rate']:.2%}** | "
                f"{la['raw']['overall']['partial_leak_rate']:.0%} |")
    scan = load("experiments/log_pii_scan.json")["totals"]["realistic"]
    tables.append("## E2  Leakage: ground-truth PII values visible in the retrieval index\n\n" + "\n".join(rows) +
                  f"\n\nLogs and traces of the redacted system ({scan['privacy']['records']:,} records): "
                  f"**{scan['privacy']['full_value']}** of {scan['pii_values_searched']:,} PII values found "
                  f"(raw-baseline positive control: {scan['baseline']['full_value']:,} hits).")
    summary["E2"] = {"source": "retrieval/realistic_t0.50.json (leak_audit), experiments/log_pii_scan.json",
                     "index_full_leak_rate": o["full_leak_rate"], "index_partial_leak_rate": o["partial_leak_rate"],
                     "raw_index_partial_leak_rate": la["raw"]["overall"]["partial_leak_rate"],
                     "log_scan_privacy_records": scan["privacy"]["records"],
                     "log_scan_privacy_hits": scan["privacy"]["full_value"],
                     "log_scan_baseline_hits": scan["baseline"]["full_value"]}


# ------------------------------------------------------------------ E3 utility
def e3(summary):
    runs = load("retrieval/realistic_t0.50.json")["runs"]
    gen = load("generation/realistic_t0.50.json")["configs"]
    metrics = []
    for cfg, run in [("redacted", "redacted-t0.50-section/hybrid/qa"), ("raw", "raw-section/hybrid/qa")]:
        r, g = runs[run]["test"], gen[cfg]["test"]
        metrics.append({
            "evidence": (r["answerable"]["evidence@auto"], r["answerable_evidence@auto_ci95"]),
            "grounded": (g["answerable"]["grounded_accuracy"], g["answerable_grounded_accuracy_ci95"]),
            "accuracy": (g["answerable"]["accuracy"], g["answerable_accuracy_ci95"]),
        })
    labels = {"evidence": "Answer in retrieved context", "accuracy": "Answer correct", "grounded": "Grounded accuracy\n(correct + cited from the right note)"}
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    w = 0.36
    for j, (name, colour, m) in enumerate([("GlassBox (redacted)", GLASSBOX, metrics[0]), ("Raw baseline (no privacy)", RAW, metrics[1])]):
        xs = [i + (j - 0.5) * (w + 0.02) for i in range(len(labels))]
        vals = [m[k][0] for k in labels]
        bars = ax.bar(xs, vals, w, color=colour, label=name,
                      yerr=[[v - m[k][1][0] for v, k in zip(vals, labels)], [m[k][1][1] - v for v, k in zip(vals, labels)]],
                      error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
        for b, v, k in zip(bars, vals, labels):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, m[k][1][1]), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(labels)), list(labels.values()))
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("rate (axis starts at 0.5)")
    ax.set_title("E3  Utility with and without privacy (189 answerable test questions, 95% CI)")
    ax.legend(loc="lower right")
    save(fig, "E3_utility_redacted_vs_raw.png")
    summary["E3"] = {"source": "retrieval/realistic_t0.50.json, generation/realistic_t0.50.json",
                     **{f"{cfg}_{k}": {"value": m[k][0], "ci95": m[k][1]} for cfg, m in zip(["redacted", "raw"], metrics) for k in labels}}


# ------------------------------------------------------------------ E4 threshold sweep
def e4(summary):
    s = load("experiments/threshold_sweep.json")
    ts = [v["threshold"] for v in s.values()]
    leak = [v["redaction"]["masking"]["char_leakage"] * 100 for v in s.values()]
    over = [v["redaction"]["over_redactions"] for v in s.values()]
    ga = [v["generation_test"]["answerable"]["grounded_accuracy"] for v in s.values()]
    gci = [v["generation_test"]["answerable_grounded_accuracy_ci95"] for v in s.values()]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharex=True)
    panels = [(axes[0], leak, "PII characters left visible (%)", "{:.2f}"),
              (axes[1], over, "Over-redactions (clinical text removed)", "{:,.0f}"),
              (axes[2], ga, "Grounded accuracy (95% CI)", "{:.3f}")]
    for ax, ys, title, fmt in panels:
        if ys is ga:
            ax.fill_between(ts, [c[0] for c in gci], [c[1] for c in gci], color=GLASSBOX, alpha=0.15, linewidth=0)
        ax.plot(ts, ys, color=GLASSBOX, linewidth=2, marker="o", markersize=5, clip_on=False, zorder=3)
        ax.axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_title(title, fontsize=10)
        for x, y in [(ts[0], ys[0]), (0.5, ys[ts.index(0.5)]), (ts[-1], ys[-1])]:
            ax.annotate(fmt.format(y), (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5)
        ax.set_xlabel("redaction confidence threshold")
        ax.margins(y=0.25)
        if ys is not ga:
            ax.set_ylim(bottom=0)
    axes[0].annotate("chosen 0.5", (0.5, axes[0].get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                     fontsize=8.5, color=INK2)
    fig.suptitle("E4  Privacy-utility trade-off across redaction thresholds (one full re-run per threshold)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "E4_threshold_sweep.png")
    summary["E4"] = {"source": "experiments/threshold_sweep.json",
                     "by_threshold": {f"{t:.2f}": {"char_leakage": round(l / 100, 4), "over_redactions": o,
                                                  "grounded_accuracy": g, "ci95": c}
                                      for t, l, o, g, c in zip(ts, leak, over, ga, gci)}}


# ------------------------------------------------------------------ E5 identity protection
def e5(summary):
    gen = load("generation/realistic_t0.50.json")["configs"]
    rows = []
    for cfg, name, colour in [("redacted", "GlassBox", GLASSBOX), ("redacted_noguard", "Redacted, guard off", NOGUARD),
                              ("raw", "Raw baseline", RAW)]:
        idt = gen[cfg]["test"]["identity"]
        n = idt["n"]
        rows.append((name, colour, round((1 - idt["abstained"]) * n), round(idt["grounded_accuracy"] * n), n))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    groups = ["Answered", "Gave the person's true value,\ncited from their own note"]
    w = 0.26
    for j, (name, colour, answered, leaked, n) in enumerate(rows):
        xs = [i + (j - 1) * (w + 0.02) for i in range(2)]
        bars = ax.bar(xs, [answered, leaked], w, color=colour, label=name)
        for b, v in zip(bars, [answered, leaked]):
            ax.annotate(f"{v}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9)
    ax.set_xticks(range(2), groups)
    ax.set_ylim(0, 92)
    ax.set_ylabel(f"identity questions (of {rows[0][4]})")
    ax.set_title("E5  Questions that name a person (84 test questions)")
    ax.legend(loc="upper right")
    save(fig, "E5_identity_protection.png")
    summary["E5"] = {"source": "generation/realistic_t0.50.json (test, identity)",
                     **{name: {"answered": a, "true_value_own_note": l, "n": n} for name, _, a, l, n in rows},
                     "note": "Guard off: every answer came from another patient's note (false attribution 1.0)."}


# ------------------------------------------------------------------ E6 retrieval ablation
def e6(summary):
    steps = load("retrieval/realistic_t0.50_selection_ablation.json")
    names = [s["step"] for s in steps]
    vals = [s["test"]["answerable_evidence@auto"] for s in steps]
    cis = [s["test"]["answerable_evidence@auto_ci95"] for s in steps]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    xs = range(len(steps))
    ax.fill_between(xs, [c[0] for c in cis], [c[1] for c in cis], color=GLASSBOX, alpha=0.15, linewidth=0, step="mid")
    ax.step(xs, vals, where="mid", color=GLASSBOX, linewidth=2)
    ax.plot(xs, vals, "o", color=GLASSBOX, markersize=6)
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_xticks(list(xs), [n.replace(" (final)", "\n(final)").replace("+ ", "+ ").replace(" chunks", "\nchunks")
                             .replace(" grouping", "\ngrouping").replace(" redaction", "\nredaction")
                             .replace(" answer", "\nanswer") for n in names], fontsize=8.5)
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("answer in retrieved context")
    ax.set_title("E6  Retrieval design, one step at a time (189 answerable test questions, 95% CI)")
    save(fig, "E6_retrieval_ablation.png")
    summary["E6"] = {"source": "retrieval/realistic_t0.50_selection_ablation.json",
                     "steps": {n: {"evidence": v, "ci95": c} for n, v, c in zip(names, vals, cis)}}


# ------------------------------------------------------------------ E7 verification
def e7(summary):
    v = load("verification/verifier_eval.json")
    t = v["test"]
    kinds = [k for k in t["per_kind"] if k != "genuine"]
    caught = [1 - t["per_kind"][k]["said_supported"] for k in kinds]
    order = sorted(range(len(kinds)), key=lambda i: caught[i])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    bars = ax1.barh([kinds[i].replace("_", " ") + f"  (n={t['per_kind'][kinds[i]]['n']})" for i in order],
                    [caught[i] for i in order], 0.6, color=GLASSBOX)
    for b, i in zip(bars, order):
        ax1.annotate(f"{caught[i]:.1%}", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(-4, 0),
                     textcoords="offset points", ha="right", va="center", fontsize=8.5, color="white", fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.grid(axis="y", visible=False)
    ax1.set_title(f"E7a  Corrupted claims caught: {t['corruptions_caught']:.1%} overall")
    n_corrupted = sum(t["per_kind"][k]["n"] for k in kinds)
    ax1.set_xlabel(f"share not labelled SUPPORTED (test, {n_corrupted} corrupted claims)")

    av = v["real_answers_test"]["answer_verdicts"]
    names = [k for k in ["VERIFIED", "NOT_VERIFIED"] if k in av]
    vals = [av[k]["grounded_accuracy"] for k in names]
    bars = ax2.bar([n.replace("_", " ").title() + f"\n(n={av[n]['n']})" for n in names], vals, 0.5,
                   color=[GLASSBOX, BEFORE])
    label_bars(ax2, bars)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("E7b  Is the verdict a useful trust signal?")
    ax2.set_ylabel("grounded accuracy of those answers")
    fig.tight_layout()
    save(fig, "E7_verification.png")
    summary["E7"] = {"source": "verification/verifier_eval.json",
                     "corruptions_caught": t["corruptions_caught"], "ci95": t["corruptions_caught_ci95"],
                     "false_alarm_on_clean_genuine": t["false_alarm_on_genuine_clean"],
                     "false_alarm_on_genuine_strict": t["false_alarm_on_genuine"],
                     "per_kind_caught": {k: round(c, 4) for k, c in zip(kinds, caught)},
                     "grounded_accuracy_by_verdict": {k: av[k] for k in av},
                     "guard_off_identity_claims": v["guard_off_identity_claims"]}


# ------------------------------------------------------------------ E8 robustness
def e8(summary):
    p = {k: load(f"robustness/paraphrases_{k}.json")["overall"] for k in ["a_prefix", "a", "b_prefix", "b"]}
    template = p["b"]["evidence_original"]
    gen = load("generation/realistic_t0.50.json")["configs"]["redacted"]["test"]
    cats = ["Template wording", "Set A (development)", "Set B (held out)"]
    before = [None, p["a_prefix"]["evidence_paraphrase"], p["b_prefix"]["evidence_paraphrase"]]
    before_ci = [None, p["a_prefix"]["evidence_paraphrase_ci95"], p["b_prefix"]["evidence_paraphrase_ci95"]]
    after = [template, p["a"]["evidence_paraphrase"], p["b"]["evidence_paraphrase"]]
    after_ci = [load("retrieval/realistic_t0.50.json")["runs"]["redacted-t0.50-section/hybrid/qa"]["test"]["answerable_evidence@auto_ci95"],
                p["a"]["evidence_paraphrase_ci95"], p["b"]["evidence_paraphrase_ci95"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    w = 0.36
    for j, (vals, cis, colour, name) in enumerate([(before, before_ci, BEFORE, "Before the robustness fixes"),
                                                   (after, after_ci, GLASSBOX, "After (final system)")]):
        xs = [i + (j - 0.5) * (w + 0.02) if before[i] is not None else i for i in range(3)]
        for x, v, c in zip(xs, vals, cis):
            if v is None:
                continue
            ax1.bar(x, v, w, color=colour, label=name if x == xs[-1] else None,
                    yerr=err(v, c), error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
            ax1.annotate(f"{v:.3f}", (x, c[1]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
    ax1.set_xticks(range(3), cats)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("answer in retrieved context")
    ax1.set_title("E8a  Retrieval on reworded questions (189 each, 95% CI)")
    ax1.legend(loc="upper right", fontsize=8.5)
    ga = [(gen["answerable"]["grounded_accuracy"], gen["answerable_grounded_accuracy_ci95"]),
          (p["b"]["grounded_accuracy"], p["b"]["grounded_accuracy_ci95"])]
    bars = ax2.bar(["Template wording", "Set B (held out)"], [g[0] for g in ga], 0.5, color=GLASSBOX,
                   yerr=[[g[0] - g[1][0] for g in ga], [g[1][1] - g[0] for g in ga]],
                   error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    for b, g in zip(bars, ga):
        ax2.annotate(f"{g[0]:.3f}", (b.get_x() + b.get_width() / 2, g[1][1]), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8.5)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("grounded accuracy")
    ax2.set_title("E8b  End to end, final system")
    fig.tight_layout()
    save(fig, "E8_robustness.png")
    summary["E8"] = {"source": "robustness/paraphrases_*.json",
                     "evidence": {"template": template, **{k: p[k]["evidence_paraphrase"] for k in p}},
                     "grounded_accuracy": {"template": ga[0][0], "set_b": ga[1][0], "set_b_ci95": ga[1][1]}}


# ------------------------------------------------------------------ E9 synthetic replication
def e9(summary, tables):
    files = {c: R / "pipeline" / f"synthetic_{c}.json" for c in ["redacted", "raw"]}
    real = {c: load(f"pipeline/realistic_{c}.json")["summary"] for c in ["redacted", "raw"]}
    if not all(f.exists() for f in files.values()):
        tables.append("## E9  Synthetic replication\n\nNot run yet: `python -m glassbox.run --qa-set --dataset synthetic "
                      "[--config raw]`.")
        summary["E9"] = {"status": "not run"}
        return
    syn = {c: json.loads(f.read_text())["summary"] for c, f in files.items()}

    def row(label, key, fmt="{:.3f}"):
        cells = []
        for s in (real["redacted"], syn["redacted"], real["raw"], syn["raw"]):
            v = s.get(key)
            cells.append("n/a" if v is None else fmt.format(v))
        return f"| {label} | " + " | ".join(cells) + " |"

    def ci(s):
        c = s.get("answerable_grounded_accuracy_ci95")
        return f"[{c[0]:.3f}, {c[1]:.3f}]" if c else ""

    rows = ["| Metric (test split) | Realistic GlassBox | Synthetic GlassBox | Realistic raw | Synthetic raw |",
            "|---|---|---|---|---|",
            f"| Questions (answerable / identity) | {real['redacted']['n_answerable']} / {real['redacted']['n_identity']} | "
            f"{syn['redacted']['n_answerable']} / {syn['redacted']['n_identity']} | {real['raw']['n_answerable']} / "
            f"{real['raw']['n_identity']} | {syn['raw']['n_answerable']} / {syn['raw']['n_identity']} |",
            row("Grounded accuracy (answerable)", "answerable_grounded_accuracy"),
            "| 95% CI | " + " | ".join(ci(s) for s in (real["redacted"], syn["redacted"], real["raw"], syn["raw"])) + " |",
            row("Identity questions answered", "identity_answered", "{:.0f}")]
    tables.append("## E9  Replication on the second dataset (250 synthetic notes, different generator)\n\n" + "\n".join(rows))
    summary["E9"] = {"source": "pipeline/synthetic_*.json, pipeline/realistic_*.json",
                     **{f"synthetic_{c}": {k: syn[c].get(k) for k in ["n", "n_answerable", "n_identity",
                                                                    "answerable_grounded_accuracy",
                                                                    "answerable_grounded_accuracy_ci95",
                                                                    "identity_answered", "verdicts"]}
                        for c in syn}}


def main():
    summary, tables = {}, []
    for f in (e1, e2):
        f(summary, tables)
    for f in (e3, e4, e5, e6, e7, e8):
        f(summary)
    e9(summary, tables)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "tables.md").write_text("# GlassBox result tables\n\nGenerated by `python -m glassbox.src.experiments.figures` "
                                   "from `glassbox/results/`. Do not edit by hand.\n\n" + "\n\n".join(tables) + "\n")
    print(f"Wrote {len(list(OUT.glob('*.png')))} figures, summary.json and tables.md to {OUT}")


if __name__ == "__main__":
    main()
