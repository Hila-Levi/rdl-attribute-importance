"""Figure style matching the ACM 'acmart' (sigconf) template used for SIGMOD.

Figures are produced at the exact width they will occupy in the paper, with the
template's own point sizes and body font, so LaTeX must insert them with no
scaling at all:

    \\includegraphics[width=\\textwidth]{figure.pdf}      % inside figure*
    \\includegraphics[width=\\columnwidth]{figure.pdf}    % single column

Scaling a figure inside LaTeX changes the size of its text, which is what makes
figure labels disagree with the surrounding body text.

The body font is Linux Libertine, the same face acmart uses. It is installed
under ~/.fonts; if it is missing, matplotlib falls back silently to DejaVu Serif,
so use_paper_style() checks for it and warns.
"""

import warnings
from contextlib import contextmanager

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# acmart sigconf geometry. 1 TeX pt = 1/72.27 in.
COLUMN_WIDTH_IN = 3.34   # \columnwidth = 241.15pt — single column
TEXT_WIDTH_IN = 7.01     # \textwidth   = 506.30pt — full width, figure*

# acmart body / caption / footnote sizes.
BODY_PT = 9
CAPTION_PT = 8
SMALL_PT = 7

PAPER_FONT = "Linux Libertine"

# Screen dpi for a figure shown at its real printed size in a notebook.
PREVIEW_DPI = 110


def _target_width(width):
    return TEXT_WIDTH_IN if width == "text" else COLUMN_WIDTH_IN


def paper_figsize(width_in, height_in, width="text"):
    """Scale an intended layout down to the paper's width, keeping its aspect.

    The grids in this project are laid out in inches per panel, which makes them
    considerably wider than the page. Scaling both dimensions by the same factor
    keeps the layout exactly as designed while making the figure the width it
    will really be printed at, so its text ends up at the template's sizes.
    """
    target = _target_width(width)
    return (target, height_in * (target / width_in))


def _apply_type_scale(scale):
    """Set every size-bearing rcParam to the template's size times `scale`."""
    plt.rcParams.update({
        "font.size": SMALL_PT * scale,
        "axes.titlesize": CAPTION_PT * scale,
        "axes.labelsize": CAPTION_PT * scale,
        "xtick.labelsize": SMALL_PT * scale,
        "ytick.labelsize": SMALL_PT * scale,
        "legend.fontsize": SMALL_PT * scale,
        "legend.title_fontsize": CAPTION_PT * scale,
        "figure.titlesize": BODY_PT * scale,
        # Lines must grow with the type, or they vanish once LaTeX scales down.
        "axes.linewidth": 0.6 * scale,
        "grid.linewidth": 0.4 * scale,
        "lines.linewidth": 1.0 * scale,
        "patch.linewidth": 0.5 * scale,
        "xtick.major.width": 0.6 * scale,
        "ytick.major.width": 0.6 * scale,
    })


def paper_scale(width_in, height_in, width="text", ns=None):
    """Keep a figure's designed layout, sized for the scaling LaTeX will apply.

    The dense grids here are laid out at a fixed size per panel, which makes them
    several times wider than the page, and LaTeX scales them down to fit. Point
    sizes and line widths are therefore multiplied by that same factor, so after
    the scaling the text lands exactly on the template's sizes while the layout
    itself is untouched. The figure must then be inserted at the full width it
    was scaled for:

        \\includegraphics[width=\\textwidth]{figure.pdf}

    Pass ns=globals() from a notebook whose plotting code passes its own size
    constants explicitly; they are rebound to the scaled values.
    """
    scale = max(width_in / _target_width(width), 1.0)
    _apply_type_scale(scale)
    if ns is not None:
        ns["TITLE_SIZE"] = BODY_PT * scale
        ns["LABEL_SIZE"] = CAPTION_PT * scale
        ns["TICK_SIZE"] = SMALL_PT * scale
        ns["ANNOT_SIZE"] = (SMALL_PT - 1) * scale
        if "ROW_LABEL_FONTSIZE" in ns:
            ns["ROW_LABEL_FONTSIZE"] = SMALL_PT * scale
    # A notebook renders the canvas at figure.dpi, so dividing it by the same
    # factor makes the inline preview appear at the size the figure will be
    # printed at — what you see in the output cell is then what the paper gets.
    # Saved PDFs are vector and unaffected: savefig passes its own dpi.
    plt.rcParams["figure.dpi"] = PREVIEW_DPI / scale
    return (width_in, height_in)


def use_paper_style(width="text", canvas_width_in=None, aspect=0.62):
    """Apply the paper's typography to every figure created from here on.

    width:           'text' or 'column' — the width the figure occupies in the
                     paper.
    canvas_width_in: only for a figure that is deliberately kept wider than the
                     page and will be scaled down by LaTeX. Point sizes are
                     multiplied by that scale factor so the text still lands on
                     the template's sizes after scaling. Leave as None when the
                     figure is created at its real width.
    aspect:          default height/width for figures that do not set figsize.
    """
    target = _target_width(width)
    scale = (canvas_width_in / target) if canvas_width_in else 1.0

    if not any(f.name == PAPER_FONT for f in fm.fontManager.ttflist):
        warnings.warn(
            f"{PAPER_FONT} is not available to matplotlib; text will fall back "
            "to another serif face and will not match the paper."
        )

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [PAPER_FONT],
        "font.size": SMALL_PT * scale,
        "font.weight": "normal",       # the template does not bolden body text
        "axes.titlesize": CAPTION_PT * scale,
        "axes.labelsize": CAPTION_PT * scale,
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "xtick.labelsize": SMALL_PT * scale,
        "ytick.labelsize": SMALL_PT * scale,
        "legend.fontsize": SMALL_PT * scale,
        "legend.title_fontsize": CAPTION_PT * scale,
        "figure.titlesize": BODY_PT * scale,
        "figure.figsize": (target, target * aspect),
        # Math set in the same face, so 2.7x10^-5 matches the text around it.
        "mathtext.fontset": "custom",
        "mathtext.rm": PAPER_FONT,
        "mathtext.it": f"{PAPER_FONT}:italic",
        "mathtext.bf": f"{PAPER_FONT}:bold",
        # Embed real fonts in the PDF rather than converting glyphs to outlines.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        # Thinner furniture, which suits small figures.
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
    return target


@contextmanager
def type_scale(ns, factor):
    """Temporarily scale a notebook's figure-text constants by `factor`.

    Each grid here ends up at a different width on the page, so the reduction
    LaTeX applies to it differs too. Scaling the four size constants per figure
    is what brings every one of them onto the template's 7-9pt, without touching
    any layout.
    """
    keys = ("TITLE_SIZE", "LABEL_SIZE", "TICK_SIZE", "ANNOT_SIZE", "ROW_LABEL_FONTSIZE")
    saved = {k: ns[k] for k in keys if k in ns}
    try:
        for k, v in saved.items():
            ns[k] = v * factor
        yield
    finally:
        ns.update(saved)
