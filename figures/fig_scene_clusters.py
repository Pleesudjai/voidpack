"""The segmented clusters of the produce scan, side elevation.

The companion to `fig_scene_all_points.py`. The flat bottoms sit exactly on the
10 mm line, which is where `points_above_plane` starts its band so that table
scatter never joins a body. Nothing is missing from the scan, the lowest 10 mm of
every body is simply not in a cluster.

Every number printed here is read in this session from the scan file and its
truth sidecar and from a live `segment` run.

Run from this folder, which holds mobasher_plot.py.
"""
import os

import mobasher_plot as M
import scene_common as S


def main():
    points, colours, truth, seg = S.load()
    table = S.table_mask(points)

    fig, ax = M.new_figure(figsize=(7.6, 2.6))
    S.draw_table(ax, points, table)
    S.draw_bodies(ax, points, colours, S.cluster_mask(points, seg))
    # Short label at the left, the one band with no data in it. Anything long
    # enough to explain the line would cross the egg and the grapefruit, so the
    # explanation lives in the title and only the value rides on the line.
    M.ref_line(ax, S.ABOVE_PLANE_MM, axis="y", tx=0.012, ha="left", dy=2.0,
               text=r"$z$ = 10 mm")
    S.frame(ax, "Segmented clusters only, the above-plane band starts at "
                r"$z$ = 10 mm")

    M.save_fig(fig, os.path.join(S.OUT_DIR, "scene_clusters"))
    S.report(points, seg, truth["bodies"], table)


if __name__ == "__main__":
    main()
