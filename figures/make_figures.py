"""Renders the data figures from outputs/figure_data/*.csv (written by analysis/figure_data.py)."""
from jit_repair import paths as jp  # noqa: E402
import csv, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style as S
S.apply()
H = jp.outputs("figures")
D = jp.outputs("figure_data")


def rd(name):
    return list(csv.DictReader(open(D / name)))


def fig2():
    rows = rd("fig2_x7_confirmation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(S.W15, 46 * S.MM), sharey=True)
    titles = {"RAND_U": "vs random allocation", "CONF_sum_U": "vs confidence-based allocation"}
    order = ["Llama-3.1-8B", "Mistral-7B", "Qwen3-8B", "Qwen3-14B", "Pooled (4 models)"]
    for ax, comp in zip(axes, ("RAND_U", "CONF_sum_U")):
        sub = {r["model"]: r for r in rows if r["comparator"] == comp}
        for i, m in enumerate(order):
            r = sub[m]; y = len(order) - 1 - i
            d, lo, hi = 100 * float(r["diff"]), 100 * float(r["lo95"]), 100 * float(r["hi95"])
            if m.startswith("Pooled"):
                lo, hi = 100 * float(r["lo975"]), 100 * float(r["hi975"])
                ax.plot([lo, hi], [y, y], color=S.INK, lw=1.4, solid_capstyle="butt")
                ax.plot(d, y, marker="D", ms=5, color=S.INK, mec="white", mew=0.6)
            else:
                ax.plot([lo, hi], [y, y], color=S.GATED, lw=1.2, solid_capstyle="butt")
                ax.plot(d, y, marker="o", ms=4.5, color=S.GATED, mec="white", mew=0.6)
        ax.axvline(0, color=S.INK2, lw=0.6)
        ax.axhline(0.5, color=S.GRID, lw=0.6)
        ax.set_title(titles[comp])
        ax.set_xlabel("Accuracy difference (points)")
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order[::-1])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.22, top=0.86, wspace=0.12)
    S.save(fig, str(H / "fig2_x7_confirmation"))


def fig4():
    rows = rd("fig4_vote_gain_flagged_vs_passed.csv")
    fig, ax = plt.subplots(figsize=(S.W1, 80 * S.MM))
    groups = []
    for r in rows:
        if not groups or groups[-1][0] != r["bench"]:
            groups.append((r["bench"], []))
        groups[-1][1].append(r)
    y = 0; yt, yl = [], []
    for bench, rs in groups:
        ax.text(-0.02, y, bench, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7, fontweight="bold", color=S.INK)
        y -= 1
        for r in rs:
            for tag, col, mk in (("passed", S.REF, "s"), ("flagged", S.GATED, "o")):
                g, lo, hi = 100 * float(r[f"{tag}_gain"]), 100 * float(r[f"{tag}_lo"]), 100 * float(r[f"{tag}_hi"])
                ax.plot([lo, hi], [y, y], color=col, lw=1.0, solid_capstyle="butt")
                ax.plot(g, y, marker=mk, ms=4.0, color=col, mec="white", mew=0.5, zorder=3)
            yt.append(y); yl.append(r["model"]); y -= 1
        y -= 0.35
    ax.axvline(0, color=S.INK2, lw=0.6)
    ax.set_yticks(yt); ax.set_yticklabels(yl)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_ylim(y + 0.6, 0.6)
    ax.set_xlabel("Accuracy change from 3-sample voting (points)")
    ax.set_xlim(-11, 28); ax.set_xticks([-10, 0, 10, 20])
    ax.text(27, yt[-2], "flagged by the gate", color=S.GATED, ha="right", va="center", fontsize=7)
    ax.text(27, yt[-1], "passed by the gate", color=S.REF, ha="right", va="center", fontsize=7)
    fig.subplots_adjust(left=0.36, right=0.98, bottom=0.11, top=0.99)
    S.save(fig, str(H / "fig4_flagged_vs_passed"))


def fig5():
    rows = rd("fig5_drspider_frontier.csv")
    models = ["Llama-3.1-8B", "Mistral-7B", "Qwen3-8B"]
    SPAN = 12.0
    fig, axes = plt.subplots(1, 3, figsize=(S.W2, 60 * S.MM))
    for ax, m in zip(axes, models):
        sub = {r["arm"]: r for r in rows if r["model"] == m}
        def pts(prefix):
            ks = sorted([k for k in sub if k.startswith(prefix + "@")], key=lambda k: int(k.split("@")[1]))
            return ([float(sub[k]["mean_calls"]) for k in ks], [100 * float(sub[k]["accuracy"]) for k in ks], [k.split("@")[1] for k in ks])
        x, y, lab = pts("vote")
        ax.plot(x, y, "-s", color=S.UNIFORM, ms=3.6, mec="white", mew=0.5, label="self-consistency")
        xg, yg, labg = pts("gated_vote")
        ax.plot(xg, yg, "-o", color=S.GATED, ms=4.0, mec="white", mew=0.5, label="gated voting")
        if m == models[0]:
            for xi, yi, li in zip(xg, yg, labg):
                ax.annotate(f"N={li}" if li == "1" else li, (xi, yi), textcoords="offset points", xytext=(0, 5), fontsize=6.5, color=S.GATED, ha="center")
        j = sub["jit"]; ax.plot(float(j["mean_calls"]), 100 * float(j["accuracy"]), "D", color=S.REVISE, ms=4.4, mec="white", mew=0.5, label="revision on flagged outputs")
        g = sub["greedy"]; ax.plot(1.0, 100 * float(g["accuracy"]), "^", color=S.REF, ms=4.6, mec="white", mew=0.5, label="greedy")
        allv = y + yg + [100 * float(j["accuracy"]), 100 * float(g["accuracy"])]
        mid = (max(allv) + min(allv)) / 2
        ax.set_ylim(mid - SPAN / 2, mid + SPAN / 2)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"]); ax.set_xlim(0.9, 9)
        ax.set_title(m); ax.set_xlabel("Mean model calls per question")
        ax.grid(axis="y")
    axes[0].set_ylabel("Execution accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=4, bbox_to_anchor=(0.995, 1.0), handlelength=1.8, columnspacing=1.4)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.17, top=0.80, wspace=0.24)
    S.save(fig, str(H / "fig5_drspider_frontier"))


def fig6():
    rows = rd("fig6_programs_vs_thinking.csv")
    panels = [("Qwen3-14B", "CONF", "Qwen3-14B, four-label"), ("Qwen3-14B", "D5", "Qwen3-14B, depth-5"), ("Qwen3-8B", "D5", "Qwen3-8B, depth-5")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.W1, 46 * S.MM), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    for i, (m, st, lab) in enumerate(panels):
        yv = len(panels) - 1 - i
        sub = {r["comparison"]: r for r in rows if r["model"] == m and r["set"] == st}
        t = sub["JIT-think"]; c = sub["JIT-cot"]
        acc = [(100 * float(t["jit_acc"]), S.GATED, "o"), (100 * float(t["baseline_acc"]), S.UNIFORM, "s"), (100 * float(c["baseline_acc"]), S.REF, "^")]
        tok = [(float(t["jit_tokens"]), S.GATED, "o"), (float(t["baseline_tokens"]), S.UNIFORM, "s"), (float(c["baseline_tokens"]), S.REF, "^")]
        a1.plot([min(v for v, _, _ in acc), max(v for v, _, _ in acc)], [yv, yv], color=S.GRID, lw=1.6, zorder=1)
        a2.plot([min(v for v, _, _ in tok), max(v for v, _, _ in tok)], [yv, yv], color=S.GRID, lw=1.6, zorder=1)
        for v, col, mk in acc:
            a1.plot(v, yv, marker=mk, color=col, ms=5, mec="white", mew=0.5, zorder=3)
        for v, col, mk in tok:
            a2.plot(v, yv, marker=mk, color=col, ms=5, mec="white", mew=0.5, zorder=3)
    a1.set_yticks(range(len(panels))); a1.set_yticklabels([p[2] for p in panels][::-1])
    a1.tick_params(axis="y", length=0); a1.spines["left"].set_visible(False); a2.spines["left"].set_visible(False); a2.tick_params(axis="y", length=0)
    a1.set_xlim(50, 102); a1.set_xticks([60, 80, 100]); a1.set_xlabel("Accuracy (%)")
    a1.get_xticklabels()[-1].set_horizontalalignment("right")
    a2.set_xscale("log"); a2.set_xlim(90, 1600); a2.set_xticks([100, 300, 1000]); a2.set_xticklabels(["100", "300", "1000"])
    a2.set_xlabel("Tokens per question")
    a1.set_ylim(-0.6, len(panels) - 0.2)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", color=S.GATED, ms=5, label="gated program"),
          Line2D([], [], marker="s", ls="", color=S.UNIFORM, ms=5, label="thinking"),
          Line2D([], [], marker="^", ls="", color=S.REF, ms=5, label="step-by-step")]
    fig.legend(handles=hs, loc="upper center", ncol=3, bbox_to_anchor=(0.62, 1.0), handletextpad=0.3, columnspacing=1.0)
    fig.subplots_adjust(left=0.33, right=0.98, bottom=0.22, top=0.86, wspace=0.16)
    S.save(fig, str(H / "fig6_programs_vs_thinking"))


def fig3():
    """Fix-break ledger: what each policy fixes (right of zero) and breaks (left of zero), pooled over cells."""
    from matplotlib.colors import to_rgba
    rows = rd("fig3_ledger.csv")
    style = {"self-consistency": (S.UNIFORM, None), "gated voting, winner replaces": (S.GATED, "////"),
             "JIT-Repair, vote spend": (S.GATED, None), "JIT-Repair, revise spend": (S.REVISE, None)}
    fig = plt.figure(figsize=(S.W1, 60 * S.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.0], left=0.305, right=0.995, bottom=0.17, top=0.985, wspace=0.03)
    short = {"self-consistency": "self-consistency", "gated voting, winner replaces": "gated, no selector",
             "JIT-Repair, vote spend": "JIT-Repair, vote", "JIT-Repair, revise spend": "JIT-Repair, revise"}
    ax, ac = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    y, yt, yl, heads = 0.0, [], [], []
    for task, head in (("SQL", "SQL"), ("ProofWriter", "ProofWriter")):
        rs = [r for r in rows if r["task"] == task]
        heads.append((y, f"{head} ({int(rs[0]['n']):,})")); y -= 1
        for r in rs:
            col, hatch = style[r["policy"]]
            f, b = int(r["fixes"]), int(r["breaks"])
            ax.barh(y, f, height=0.64, color="white" if hatch else col, edgecolor=col, hatch=hatch, lw=0.8)
            ax.barh(y, -b, height=0.64, color=to_rgba(col, 0.28), edgecolor=col, lw=0.8)
            ax.text(f + 30, y, f"{f:,}", va="center", ha="left", fontsize=7, color=S.INK)
            ax.text(-b - 30, y, f"{b:,}", va="center", ha="right", fontsize=7, color=S.INK)
            if r["extra_cost"]:
                ac.text(0.0, y, f"{int(r['extra_cost']):,}", va="center", ha="left", fontsize=7, color=S.INK)
                ac.text(0.0, y - 0.36, r["cost_unit"], va="center", ha="left", fontsize=6, color=S.INK2)
            yt.append(y); yl.append(short[r["policy"]]); y -= 1
        y -= 0.35
    for a_ in (ax, ac):
        a_.set_ylim(y + 0.6, 0.6)
    for yh, txt in heads:
        ax.text(-0.02, yh, txt, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7, fontweight="bold", color=S.INK)
    ac.text(0.0, heads[0][0], "extra cost", va="center", ha="left", fontsize=7, fontweight="bold", color=S.INK)
    ac.axis("off"); ac.set_xlim(0, 1)
    ax.axvline(0, color=S.INK2, lw=0.6)
    ax.set_yticks(yt); ax.set_yticklabels(yl); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(-800, 1640); ax.set_xticks([-500, 0, 500, 1000]); ax.set_xticklabels(["500", "0", "500", "1000"])
    ax.set_xlabel("questions broken  |  questions fixed")
    S.save(fig, str(H / "fig3_ledger"))


def figS1():
    """Allocation curves: accuracy gain against the share of questions that receive the vote, held-out BIRD."""
    rows = rd("figS1_allocation_curves.csv")
    models = ["Llama-3.1-8B", "Mistral-7B", "Qwen3-8B", "Qwen3-14B"]
    fig, axes = plt.subplots(1, 4, figsize=(S.W2, 50 * S.MM), sharey=False)
    for ax, m in zip(axes, models):
        sub = [r for r in rows if r["model"] == m]
        def series(score, spend):
            s = [r for r in sub if r["score"] == score and r["spend"] == spend]
            return [float(r["coverage"]) for r in s], [100 * float(r["gain"]) for r in s]
        x, y = series("random", "winner replaces"); ax.plot(x, y, "--", color=S.REF, lw=1.0, label="random allocation")
        x, y = series("confidence (sum)", "winner replaces"); ax.plot(x, y, "-", color=S.CONF, lw=1.2, label="confidence, winner replaces")
        x, y = series("gate", "winner replaces"); ax.plot(x, y, "-", color=S.GATED, lw=1.6, label="gate, winner replaces")
        x, y = series("gate", "behind selector"); ax.plot(x, y, ":", color=S.GATED, lw=1.6, label="gate, behind selector")
        rho = float(sub[0]["rho"]); ax.axvline(rho, color=S.GRID, lw=1.0, zorder=0)
        ax.text(rho, 1.0, " $\\rho$", transform=ax.get_xaxis_transform(), va="top", ha="left", fontsize=7, color=S.INK2)
        ax.set_title(m); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xlabel("Share of questions voted on"); ax.grid(axis="y")
    axes[0].set_ylabel("Accuracy gain over greedy (points)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=2.0, columnspacing=1.2)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.2, top=0.78, wspace=0.32)
    S.save(fig, str(H / "figS1_allocation_curves"))


if __name__ == "__main__":
    fig2(); fig3(); fig4(); fig5(); fig6(); figS1(); print("ok")
