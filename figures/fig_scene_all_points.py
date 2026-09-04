"""Every point in the composed produce scan, side elevation.

The companion to `fig_scene_clusters.py`. Read together they answer why the four
bodies look as though they have flat bottoms. Here the bodies are closed and
rounded down to the table, which is what `web/app.js` draws, because the viewport
is given the whole cloud.

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
    S.draw_bodies(ax, points, colours, ~table)
    S.frame(ax, "Every point in the scan file, bodies closed to the table")
    S.label_bodies(ax, truth["bodies"])

    M.save_fig(fig, os.path.join(S.OUT_DIR, "scene_all_points"))
    S.report(points, seg, truth["bodies"], table)


if __name__ == "__main__":
    main()
