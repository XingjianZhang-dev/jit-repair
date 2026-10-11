"""Shared figure style: widths for the 5p two-column layout (88 / 140 / 183 mm), text >= 7 pt at print size, one colour meaning per role."""
import matplotlib as mpl
MM = 1 / 25.4
W1, W15, W2 = 88 * MM, 140 * MM, 183 * MM  # 5p layout: column 88.9 mm, text 184 mm (Elsevier print 90 / 140 / 190 mm)
# One meaning per colour across every figure (palette validated for colour-vision deficiency, adjacent and all pairs)
GATED = "#2a78d6"     # compute spent where the gate fires (ours)
UNIFORM = "#eb6834"   # compute spent on every item (ungated voting, thinking mode)
REVISE = "#1baf7a"    # the revise spend
REF = "#6b6a66"       # references without extra compute (greedy, CoT, random allocation)
REF_LIGHT = "#b9b8b2"
CONF = "#6250d6"      # allocation by the model's own confidence (always with its own marker)
INK, INK2, GRID = "#1a1a19", "#52514e", "#e4e3df"


def apply():
    mpl.rcParams.update({
        "font.family": "TeX Gyre Heros", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
        "grid.linewidth": 0.5, "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 4,
    })


def save(fig, path):
    meta = {"Creator": None, "Producer": None, "CreationDate": None}
    fig.savefig(path + ".pdf", metadata=meta, bbox_inches=None)
    fig.savefig(path + ".png", dpi=300, bbox_inches=None)
