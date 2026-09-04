"""Shared loading and drawing for the produce scene elevations.

Two figures draw the same scene, one with every point in the file and one with
the segmented clusters only, so the loading, the colours and the axis limits live
here and each figure script stays small. The limits are deliberately IDENTICAL in
both figures, because the two are meant to be read against each other and a
reader cannot compare two elevations drawn at different scales.

UNITS. The scan file is metres. Everything drawn or printed here is millimetres.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

SCAN = os.path.join(ROOT, "scans", "real_produce.npy")
OUT_DIR = os.path.join(ROOT, "scans", "objects")

TABLE_BAND_MM = 6.0        # anything within this of the plane is drawn as table
ABOVE_PLANE_MM = 10.0      # where src.capture.segment starts its band
PT_SIZE = 3.4
TABLE_GREY = "0.86"

XLIM_MM = (-110.0, 530.0)
# The top is set by the tallest LABEL and not by the tallest body. The grapefruit
# crown is 86.5 mm, its two-line label adds about 32 mm at 8 pt on this axis, and
# a label that touches the frame reads as though it were cut off. 132 mm leaves a
# clear band above it.
YLIM_MM = (-14.0, 132.0)

XLABEL = r"Position along the table, $x$  (mm)"
YLABEL = r"$z$  (mm)"


def load():
    """Returns (points_m, colours_0_1, truth_dict, segmentation)."""
    from src.capture.segment import segment

    points = np.load(SCAN)
    colours = np.load(SCAN.replace(".npy", "_rgb.npy")) / 255.0
    with open(SCAN.replace(".npy", "_truth.json"), "r", encoding="utf-8") as fh:
        truth = json.load(fh)
    return points, colours, truth, segment(points)


def table_mask(points):
    return np.abs(points[:, 2]) < TABLE_BAND_MM / 1000.0


def cluster_mask(points, seg):
    mask = np.zeros(len(points), dtype=bool)
    for cluster in seg["clusters"]:
        mask[cluster] = True
    return mask


def draw_bodies(ax, points, colours, mask):
    """Scatter the item points in their measured colour, x against z, in mm."""
    ax.scatter(points[mask, 0] * 1000.0, points[mask, 2] * 1000.0,
               s=PT_SIZE, c=colours[mask], linewidths=0.0, zorder=4)


def draw_table(ax, points, mask, step=4):
    ax.scatter(points[mask, 0][::step] * 1000.0, points[mask, 2][::step] * 1000.0,
               s=0.5, c=TABLE_GREY, linewidths=0.0, zorder=2)


def frame(ax, title):
    """Apply the shared limits, labels and title. Equal aspect, always.

    Equal aspect is not optional on an elevation. Without it the bodies are
    stretched and a reader measuring the figure gets the wrong proportions.
    """
    ax.set_xlim(*XLIM_MM)
    ax.set_ylim(*YLIM_MM)
    ax.set_aspect("equal")
    ax.set_xlabel(XLABEL, fontsize=11)
    ax.set_ylabel(YLABEL, fontsize=11)
    ax.set_title(title, fontsize=10.5, loc="left", pad=5)


def label_bodies(ax, bodies, fontsize=8.0, dy=5.0):
    """Name and size each body in the empty band above it, never on the points.

    The source word rides only on the body that is not measured. Putting it on
    all four made the avocado and potato labels collide, because a 25 character
    label spans about 135 mm at this font against a 122 mm spacing between body
    centres.
    """
    for body in bodies:
        w, d = body["inplane_mm"]
        h = body["height_mm"]
        name = body["label"] + (" (nominal)" if body["scaled_to_nominal"] else "")
        ax.annotate("%s\n%.0f x %.0f x %.0f mm" % (name, w, d, h),
                    xy=(body["centre_xy_mm"][0], h + dy),
                    ha="center", va="bottom", fontsize=fontsize, linespacing=1.3)


def report(points, seg, bodies, table):
    print("%d points, %d item, %d table"
          % (len(points), int((~table).sum()), int(table.sum())))
    print("segment: %d clusters at eps = %.2f mm"
          % (len(seg["clusters"]), seg["eps_m_used"] * 1000.0))
    for body in bodies:
        print("  %-11s %6.1f x %5.1f x %5.1f mm  %s"
              % (body["label"], body["inplane_mm"][0], body["inplane_mm"][1],
                 body["height_mm"],
                 "nominal" if body["scaled_to_nominal"] else "measured"))
