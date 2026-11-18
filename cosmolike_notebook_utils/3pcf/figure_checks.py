"""Check scientific figure labels against plotted lines before exporting.

The check uses the final display coordinates, after Matplotlib has placed
axes and legends. It includes the connecting segments and marker radius,
so a label between two sampled points cannot hide their connecting line.
This helper changes no plotting defaults and imports no compiled library.
"""

import numpy as np
from matplotlib.transforms import Bbox


def _touches_box(line, box, radius):
    """Check the actually drawn line/markers within a rectangular clip region.

    The text rectangle expands by the supplied display-space radius.
    Clipping then removes invisible parts of that rectangle before the
    path is tested. A marker-only series contributes vertices, not segments.

    Arguments:
      line = visible Matplotlib Line2D artist.
      box = text bounding box in display pixels.
      radius = extra clearance in pixels, including marker/line thickness.
    Returns:
      True when a drawn path or marker intersects the expanded visible box.
    Raises:
      ValueError for custom clipping or marker subsampling not checked here.
    """
    target = box.padded(radius)
    if line.get_clip_on():
        if line.get_clip_path() is not None:
            raise ValueError("This label check supports rectangular line clipping only.")
        clip = line.get_clip_box()
        if clip is not None:
            target = Bbox.intersection(target, clip)
            if target is None:
                return False
    path = line.get_path().transformed(line.get_transform())
    if line.get_linestyle() not in (None, "None", "", " "):
        if path.intersects_bbox(target, filled=False):
            return True
    if line.get_marker() not in (None, "None", "", " "):
        if line.get_markevery() is not None:
            raise ValueError("This label check requires markers at every supplied point.")
        for point in path.vertices:
            if np.all(np.isfinite(point)) and target.contains(*point):
                return True
    return False


def check_figure_labels(figure):
    """Refuse annotation or legend placement that touches a plotted line.

    Arguments:
      figure = Matplotlib figure with line plots and optional annotations.
    Returns:
      nothing after drawing and checking the final layout. Hidden artists
      are ignored. Axis tick labels are not annotations and are not moved.
      Supports ordinary connected or marker-only lines, rectangular clipping,
      and markers at every point; unsupported custom clipping/subsets fail.
    Raises:
      ValueError naming a label that intersects a data/reference line or
      a legend that extends beyond the figure. The caller can move the
      annotation, change its layout, or increase the available figure size.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    labels = list(figure.texts) + list(figure.legends)
    lines = []
    for axis in figure.axes:
        if not axis.get_visible():
            continue
        labels.extend(axis.texts)
        labels.extend([axis.title, axis.xaxis.label, axis.yaxis.label])
        legend = axis.get_legend()
        if legend is not None:
            labels.append(legend)
        for line in axis.lines:
            if line.get_visible() and line.get_alpha() != 0.0:
                lines.append(line)
    for label in labels:
        if not label.get_visible():
            continue
        if hasattr(label, "get_text") and not label.get_text():
            continue
        box = label.get_window_extent(renderer=renderer)
        for line in lines:
            # The geometric path contains line segments, while markers have
            # a finite size in points. Expand the text box by that radius
            # and a two-pixel gap before checking the full transformed path.
            radius = line.get_linewidth() / 2.0
            if line.get_marker() not in (None, "None", "", " "):
                radius = max(radius, line.get_markersize() / 2.0)
            padding = 2.0 + radius * figure.dpi / 72.0
            if _touches_box(line=line, box=box, radius=padding):
                name = label.get_text() if hasattr(label, "get_text") else "legend"
                raise ValueError(f"Move {name}: its label touches a plotted line.")
    for legend in figure.legends:
        box = legend.get_window_extent(renderer=renderer)
        if (box.x0 < figure.bbox.x0 or box.x1 > figure.bbox.x1
                or box.y0 < figure.bbox.y0 or box.y1 > figure.bbox.y1):
            raise ValueError("Increase figure size to contain the shared legend.")
