"""
mobasher_plot.py  -  House style for Mobasher-group engineering data plots.

Single source of truth for the LOOK of a results figure: moment-curvature,
load-deflection, stress-strain, R-curve, efficiency, parametric sweep. Every
figure imports this module and builds from these primitives so a whole paper
shares one pattern.

This is the data-plot companion to fbd_style.py. That one draws free bodies.
This one plots measured and simulated curves.

The style, distilled from the Grapher house figures of the group:
  - serif text, dejavuserif mathtext, axes linewidth 1.1
  - ticks INSIDE, on all four sides, length 5
  - saturated primary colours, one per series, never a colormap ramp
  - a thick line PLUS a sparse marker overlay, roughly 20 markers per curve,
    so the curve reads on a printed page and in greyscale
  - open and filled markers alternate so adjacent series stay distinct
  - NO legend box. Series are labelled directly on the curve with a short
    leader in the series colour
  - optional twin axes carrying the other unit system
  - 300-dpi PNG plus a vector PDF, always as a pair

Do not hard-code these values in individual figures. Change them here once and
every figure stays consistent. Copy this file into a project figures folder and
import it as  import mobasher_plot as M.

Author: Mobasher group / FEN.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

# ----------------------------------------------------------------------
# Locked house-style constants
# ----------------------------------------------------------------------
# Series colours, in order. Saturated, print-safe, and distinguishable in
# greyscale because each is paired with a distinct marker.
COLORS = ["#E01B1B",   # red
          "#1246C8",   # blue
          "#0A9A0A",   # green
          "#6E6E6E",   # grey
          "#8A2BE2",   # violet
          "#E08000",   # orange
          "#00868B",   # teal
          "#B03060"]   # maroon

# Marker cycle. "full" and "none" alternate so neighbours never blur together.
MARKERS = [("o", "full"), ("o", "none"), ("s", "none"), ("D", "full"),
           ("^", "none"), ("v", "full"), ("s", "full"), ("D", "none")]

LW_CURVE = 2.1        # the data line
LW_REF = 1.3          # reference / limit lines
LW_LEADER = 1.2       # annotation leader
MS = 6.2              # marker size
MEW = 1.4             # marker edge width, matters for the open markers
N_MARKERS = 20        # target markers per curve, subsampled from the data
STAR_MS = 13          # peak marker

GREY_GLYPH = "0.82"   # section glyph fill
DPI = 300

# ----------------------------------------------------------------------
# Unit conversions, so a figure never carries a magic number
# ----------------------------------------------------------------------
KNM_PER_KIPFT = 1.3558179      # kip-ft -> kN.m
KN_PER_KIP = 4.4482216
MM_PER_IN = 25.4
MPA_PER_KSI = 6.8947573
MPA_PER_PSI = 0.00689475729
FT_PER_M = 0.3048              # 1 m^-1 -> ft^-1  (inverse length)
NMM_PER_LBIN = 112.984829


def apply_style():
    """Apply the locked rcParams. Call once, before creating the figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.bbox": "tight",
    })


def new_figure(figsize=(7.6, 6.4)):
    """Create the standard single-panel figure. Returns (fig, ax)."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(direction="in", length=5, width=1.0)
    return fig, ax


def new_panels(n=2, figsize=None):
    """Create a 1-by-n panel row, e.g. capacity beside efficiency."""
    apply_style()
    if figsize is None:
        figsize = (5.5 * n, 4.4)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    for a in np.atleast_1d(axes):
        a.tick_params(direction="in", length=5, width=1.0)
    return fig, axes


def new_column(n=2, figsize=None, panel_h=3.1, width=7.6, sharex=True):
    """Create an n-by-1 panel COLUMN, one panel per row.

    The conventions file calls for a column whenever the panels show an
    elongated body, because a wide row squeezes each panel below the roughly
    250 px that stays legible at report width. There was no primitive for it,
    so every column figure was hand building `plt.subplots` and re-applying the
    tick parameters. Returns (fig, axes).
    """
    apply_style()
    if figsize is None:
        figsize = (width, panel_h * n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=sharex)
    for a in np.atleast_1d(axes):
        a.tick_params(direction="in", length=5, width=1.0)
    return fig, axes


# ----------------------------------------------------------------------
# The signature primitive: a line plus a sparse marker overlay
# ----------------------------------------------------------------------
def curve(ax, x, y, i=0, color=None, marker=None, fill=None,
          n_markers=N_MARKERS, lw=LW_CURVE, ls="-", zorder=4, label=None):
    """Plot one data series in house style.

    A thick line carries the shape and a sparse marker overlay carries the
    identity. `i` selects the colour and marker from the house cycle, so a
    caller that just increments `i` gets a consistent set for free.

    Pass `label` only when a legend is genuinely wanted. The house default is
    no legend, with `annotate_curve` used instead.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    color = COLORS[i % len(COLORS)] if color is None else color
    mk, fl = MARKERS[i % len(MARKERS)]
    marker = mk if marker is None else marker
    fill = fl if fill is None else fill

    ax.plot(x, y, ls, color=color, lw=lw, zorder=zorder, label=label)
    if n_markers and len(x) > 1:
        step = max(1, len(x) // n_markers)
        ax.plot(x[::step], y[::step], linestyle="none", marker=marker, ms=MS,
                color=color, mfc=(color if fill == "full" else "white"),
                mew=MEW, zorder=zorder + 1)
    return color


def peak_star(ax, x, y, color="k", ms=STAR_MS):
    """Mark a peak, a first-crack point, or any control point."""
    ax.plot([x], [y], marker="*", ms=ms, color=color, mec="k", mew=0.7, zorder=7)


def mark_peak(ax, x, y, color="k", xmax=None, tag=""):
    """Star the true maximum. Returns (x_peak, y_peak, starred).

    PASS THE FULL, UNCROPPED SERIES and give the plot limit as `xmax`.

    Truncation cannot be detected from cropped data. A curve cut before its
    maximum and a curve whose maximum is genuinely its last retained point look
    identical once the tail is gone. So the peak is always taken over the FULL
    series, and `xmax` decides only whether that peak is visible. When the true
    peak lies outside the frame the star is suppressed and `starred` returns
    False, so the caller labels the plateau instead of printing a boundary value
    that contradicts the caption.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    j = int(np.nanargmax(y))
    xp, yp = float(x[j]), float(y[j])
    if xmax is not None and xp > xmax:
        print(f"   {tag or 'series'}: true peak {yp:.4g} at x = {xp:.4g} lies "
              f"OUTSIDE the crop (xmax = {xmax:.4g}). Star suppressed, "
              f"label the plateau and state the crop in the caption.")
        return xp, yp, False
    peak_star(ax, xp, yp, color=color)
    return xp, yp, True


def ref_line(ax, value, axis="y", text=None, ls=(0, (5, 2)), color="0.35",
             tx=0.98, dy=0.6, fontsize=8.6, ha="right"):
    """Horizontal or vertical reference line, e.g. a steel capacity to beat.

    `ha` moves the label along the line. The default parks it at the right, which
    is the empty end of a rising envelope, but a figure whose data fills the right
    needs it at the left, and the rule that no label sits on data outranks the
    default. Pass ha="left" with a small `tx`.
    """
    if axis == "y":
        ax.axhline(value, ls=ls, color=color, lw=LW_REF, zorder=2)
        if text:
            ax.text(tx, value + dy, text, transform=ax.get_yaxis_transform(),
                    ha=ha, va="bottom", fontsize=fontsize, color=color)
    else:
        ax.axvline(value, ls=ls, color=color, lw=LW_REF, zorder=2)
        if text:
            ax.text(value, tx, text, transform=ax.get_xaxis_transform(),
                    ha="left", va="top", rotation=90, fontsize=fontsize, color=color)


# ----------------------------------------------------------------------
# Direct labelling, the house alternative to a legend box
# ----------------------------------------------------------------------
def annotate_curve(ax, text, xy, xytext, color="k", fontsize=10.5,
                   weight="bold", ha="left", va="center"):
    """Label a series ON the plot with a short leader in the series colour.

    `xy` is the anchor point on the curve. `xytext` is where the text sits.
    Choose `xytext` in clear space. The leader does the pointing, so the text
    never has to sit on top of the data.
    """
    ax.annotate(text, xy=xy, xytext=xytext, color=color, fontsize=fontsize,
                fontweight=weight, ha=ha, va=va,
                arrowprops=dict(arrowstyle="->", lw=LW_LEADER, color=color,
                                shrinkA=2, shrinkB=4), zorder=8)


# ----------------------------------------------------------------------
# Twin axes carrying the other unit system
# ----------------------------------------------------------------------
def dual_x(ax, factor, label, fontsize=12):
    """Add a top x-axis equal to the bottom one times `factor`.

    The house style draws ticks on all four sides, so the primary top ticks are
    switched off here. Without this the secondary axis lays its own ticks over
    them and the top edge shows a doubled, slightly offset tick set.
    """
    ax.tick_params(top=False)
    sec = ax.secondary_xaxis("top", functions=(lambda v: v * factor,
                                               lambda v: v / factor))
    sec.set_xlabel(label, fontsize=fontsize)
    sec.tick_params(direction="in", length=5, width=1.0)
    return sec


def dual_y(ax, factor, label, fontsize=12):
    """Add a right y-axis equal to the left one times `factor`.

    The house style draws ticks on all four sides, so the primary right ticks
    are switched off here. Without this the secondary axis lays its own ticks
    over them and the right edge shows a doubled, slightly offset tick set.
    """
    ax.tick_params(right=False)
    sec = ax.secondary_yaxis("right", functions=(lambda v: v * factor,
                                                 lambda v: v / factor))
    sec.set_ylabel(label, fontsize=fontsize)
    sec.tick_params(direction="in", length=5, width=1.0)
    return sec


# ----------------------------------------------------------------------
# Section glyphs, drawn in data coordinates
# ----------------------------------------------------------------------
def i_glyph(ax, x0, y0, w, h, tf, tw, fc=GREY_GLYPH, label=None,
            label_color="0.3", fontsize=8.2):
    """Draw an I or W section outline. Widths are in DATA units, so pick them
    against the axis ranges, not against the true section proportions."""
    xr, xl = (w + tw) / 2.0, (w - tw) / 2.0
    pts = [(0, 0), (w, 0), (w, tf), (xr, tf), (xr, h - tf), (w, h - tf),
           (w, h), (0, h), (0, h - tf), (xl, h - tf), (xl, tf), (0, tf)]
    ax.add_patch(Polygon([(x0 + px, y0 + py) for px, py in pts], closed=True,
                         fc=fc, ec="k", lw=1.0, zorder=3))
    if label:
        ax.text(x0 + w / 2.0, y0 - 0.06 * h, label, fontsize=fontsize,
                ha="center", va="top", color=label_color, fontweight="bold")


def rect_glyph(ax, x0, y0, w, h, fc="0.88", label=None,
               label_color="0.35", fontsize=8.2):
    """Draw a solid rectangular section outline."""
    ax.add_patch(Rectangle((x0, y0), w, h, fc=fc, ec="k", lw=1.0, zorder=3))
    if label:
        ax.text(x0 + w / 2.0, y0 - 0.06 * h, label, fontsize=fontsize,
                ha="center", va="top", color=label_color, fontweight="bold")


# ----------------------------------------------------------------------
# Checks and output
# ----------------------------------------------------------------------
BANNED = {"—": "em-dash", "–": "en-dash", ";": "semicolon",
          "“": "smart quote", "”": "smart quote",
          "‘": "smart quote", "’": "smart apostrophe",
          "≈": "approximately sign"}


def check_text(fig):
    """Warn on banned punctuation in any label outside mathtext.

    The group punctuation rule applies to figure text as well as to prose.
    Colons are allowed here because axis labels legitimately use them, but the
    em-dash, the semicolon, the smart quote, and the approximately sign are not.
    """
    bad = []
    for t in fig.findobj(match=plt.Text):
        s = t.get_text()
        if not s or "$" in s:
            continue
        for ch, why in BANNED.items():
            if ch in s:
                bad.append((why, s))
    if bad:
        print("TEXT CHECK FAILED")
        for why, s in bad:
            print(f"   {why} in: {s!r}")
    else:
        print("text check passed")
    return not bad


def save_fig(fig, path_no_ext, dpi=DPI, check=True, tight=True):
    """Write the 300-dpi PNG and the vector PDF as a pair, and run the check.

    `tight` controls both `fig.tight_layout()` and the tight bounding box. Leave it True
    for any figure built with `new_figure` or `new_panels`. Set it False when the panels
    were placed by hand with `fig.add_axes`, because tight_layout does not manage hand
    placed axes and the tight bounding box crops the reserved width of a 3D panel, which
    silently turns the intended figure aspect into a much taller one.
    """
    if check:
        check_text(fig)
    if tight:
        fig.tight_layout()
    # bbox_inches=None falls back to rcParams["savefig.bbox"], which apply_style sets to
    # "tight", so the crop has to be turned off through the rc context and not by None.
    ctx = {"savefig.bbox": "tight" if tight else "standard"}
    with plt.rc_context(ctx):
        for ext in ("png", "pdf"):
            fig.savefig(f"{path_no_ext}.{ext}", dpi=dpi)
    print(f"saved {path_no_ext}.png and .pdf")
