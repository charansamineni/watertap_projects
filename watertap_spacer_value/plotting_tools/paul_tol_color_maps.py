import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# -------------------------------------------------------------------
# 1. PAUL TOL QUALITATIVE COLOR PALETTES
# -------------------------------------------------------------------

PAULTOL_QUAL = {
    "tol_bright": [
        (68, 119, 170), (102, 204, 238), (34, 136, 51),
        (204, 187, 68), (238, 102, 119), (170, 51, 119),
        (187, 187, 187)
    ],

    "tol_vibrant": [
        (0, 119, 187), (51, 187, 238), (0, 153, 136),
        (238, 119, 51), (204, 51, 17), (238, 51, 119),
        (187, 187, 187)
    ],

    "tol_muted": [
        (51, 34, 136), (136, 204, 238), (68, 170, 153),
        (17, 119, 51), (153, 153, 51), (221, 204, 119),
        (204, 102, 119), (136, 34, 85), (170, 68, 153)
    ],

    "tol_light": [
        (119, 170, 221), (153, 221, 255), (170, 255, 204),
        (255, 255, 153), (255, 204, 153), (255, 153, 153),
        (204, 153, 255)
    ],

    "tol_dark": [
        (34, 34, 85), (68, 85, 136), (102, 119, 170),
        (136, 153, 187), (170, 170, 170), (187, 187, 187),
        (204, 204, 204)
    ],
}

PAULTOL_QUAL = {k: np.array(v)/255 for k, v in PAULTOL_QUAL.items()}


# -------------------------------------------------------------------
# 2. PAUL TOL SEQUENTIAL & DIVERGING COLORMAPS (HEATMAPS)
# -------------------------------------------------------------------

PAULTOL_CONT = {
    "tol_sunset": [
        (255,255,204),(255,237,160),(254,217,118),(254,178,76),
        (253,141,60),(252,78,42),(227,26,28),(189,0,38),(128,0,38)
    ],

    "tol_sunset_dark": [
        (128,0,38),(189,0,38),(227,26,28),(252,78,42),
        (253,141,60),(254,178,76),(254,217,118),(255,237,160),(255,255,204)
    ],

    "tol_BuRd": [
        (33,102,172),(67,147,195),(146,197,222),(209,229,240),
        (253,219,199),(244,165,130),(214,96,77),(178,24,43),(103,0,31)
    ],

    "tol_PRGn": [
        (118,42,131),(153,112,171),(194,165,207),(231,212,232),
        (247,247,247),(217,240,211),(166,219,160),(90,174,97),(27,120,55)
    ],

    "tol_YlOrBr": [
        (255,255,212),(254,227,145),(254,196,79),(254,153,41),
        (236,112,20),(204,76,2),(153,52,4),(102,37,6),(51,20,10)
    ],

    "tol_YlOrRd": [
        (255,255,178),(254,217,118),(254,178,76),(253,141,60),
        (252,78,42),(227,26,28),(189,0,38),(128,0,38)
    ],

    "tol_Blues": [
        (247,251,255),(222,235,247),(198,219,239),(158,202,225),
        (107,174,214),(66,146,198),(33,113,181),(8,69,148)
    ],

    "tol_Greens": [
        (247,252,245),(229,245,224),(199,233,192),(161,217,155),
        (116,196,118),(65,171,93),(35,139,69),(0,90,50)
    ],
}

PAULTOL_CONT = {k: np.array(v)/255 for k, v in PAULTOL_CONT.items()}


# -------------------------------------------------------------------
# 3. REGISTER EVERYTHING WITH MATPLOTLIB
# -------------------------------------------------------------------

def register_paul_tol_colormaps():
    # qualitative (= categorical)
    for name, arr in PAULTOL_QUAL.items():
        cmap = mcolors.ListedColormap(arr, name=name)
        matplotlib.colormaps.register(cmap, name=name)

    # continuous (heatmaps)
    for name, arr in PAULTOL_CONT.items():
        cmap = mcolors.LinearSegmentedColormap.from_list(name, arr)
        matplotlib.colormaps.register(cmap, name=name)

register_paul_tol_colormaps()


# -------------------------------------------------------------------
# 4. UNIFIED FUNCTION API (compatible with psPlotKit)
# -------------------------------------------------------------------

def gen_paultol_colormap(
    map_name,
    num_samples=10,
    vmin=0,
    vmax=1,
    return_map=False
):
    """
    Works like your original `gen_colormap`.
    Supports both qualitative and continuous Paul Tol colormaps.
    """

    cmap = matplotlib.colormaps.get_cmap(map_name)

    # determine default number of samples from the cmap if not provided
    if num_samples is None:
        if hasattr(cmap, "colors"):
            num_samples = len(cmap.colors)
        else:
            num_samples = getattr(cmap, "N", 256)

    colors = cmap(np.linspace(0, 1, num_samples))

    if return_map:
        mappable = cm.ScalarMappable(
            norm=mcolors.Normalize(vmin, vmax),
            cmap=cmap
        )
        return colors, mappable
    else:
        return colors



def get_tol_cmap(name, *, discrete=None, reverse=False, as_hex=False):
    """
    Return a Paul Tol colormap (qualitative, sequential, diverging, rainbow, cyclic).

    Parameters
    ----------
    name : str
        Name of Tol color scheme. Accepts an optional `tol.` prefix.
    discrete : int, optional
        If given, returns a ListedColormap with N discrete colors.
    reverse : bool, optional
        Reverse color order.
    as_hex : bool, optional
        If True, return a list of hex colors instead of a cmap.
    """

    # -------------------------------------------------------
    # Normalize name
    # -------------------------------------------------------
    if isinstance(name, str) and name.startswith("tol."):
        name = name[4:]
    name = str(name).lower()

    # -------------------------------------------------------
    # Official Paul Tol color definitions (SRON 2021)
    # -------------------------------------------------------

    qualitative = {
        "bright": [
            "#4477AA", "#66CCEE", "#228833", "#CCBB44",
            "#EE6677", "#AA3377", "#BBBBBB"
        ],
        "vibrant": [
            "#0077BB", "#33BBEE", "#009988", "#EE7733",
            "#CC3311", "#EE3377", "#BBBBBB"
        ],
        "muted": [
            "#332288", "#88CCEE", "#44AA99", "#117733",
            "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"
        ],
        "pale": [
            "#BBCCEE", "#CCEEFF", "#CCDDAA", "#EEEEBB",
            "#FFCCCC", "#DDDDDD"
        ],
        "dark": [
            "#222255", "#225555", "#225522",
            "#666633", "#663333", "#552222"
        ],
        "light": [
            "#77AADD", "#99DDFF", "#44BB99",
            "#BBCC33", "#EE8866", "#EEDD88",
            "#FFAABB", "#AAAAAA"
        ],
    }

    # -------------------------------------------------------
    # Sequential color schemes (Tol 2021)
    # -------------------------------------------------------

    sequential = {
        "blue": [
            "#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1",
            "#6BAED6", "#4292C6", "#2171B5", "#084594"
        ],
        "green": [
            "#F7FCF5", "#E5F5E0", "#C7E9C0", "#A1D99B",
            "#74C476", "#41AB5D", "#238B45", "#005A32"
        ],
        "red": [
            "#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272",
            "#FB6A4A", "#EF3B2C", "#CB181D", "#99000D"
        ],
        "purple": [
            "#FCFBFD", "#EFEDF5", "#DADAEB", "#BCBDDC",
            "#9E9AC8", "#807DBA", "#6A51A3", "#4A1486"
        ],
        "orange": [
            "#FFF5EB", "#FEE6CE", "#FDD0A2", "#FDAE6B",
            "#FD8D3C", "#F16913", "#D94801", "#8C2D04"
        ],
        # Tol "Smooth Rainbow"
        "smoothrainbow": [
            "#E8ECFB", "#D9CCE3", "#D1BBD7", "#CAACCB", "#BA8DB4",
            "#AE76A3", "#AA559F", "#A24D99", "#9F4797", "#994091",
            "#8E3B91", "#7C358F", "#6D2C91", "#5E2592", "#4E1D92"
        ],
        # Tol "Discrete Rainbow"
        "rainbow_discrete": [
            "#781C81", "#3F37A2", "#3465A4", "#1A92C7",
            "#11B1D8", "#0ECAD8", "#35D5C4", "#66DEA3",
            "#98E482", "#CAE65D", "#F9E349"
        ],
        # Tol "Sunset" (sequential)
        "sunset": [
            "#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1",
            "#C2E4EF", "#EAECCC", "#FEDA8B", "#FDB366",
            "#F67E4B", "#DD3D2D", "#A50026"
        ],
    }

    # -------------------------------------------------------
    # Diverging color schemes
    # -------------------------------------------------------
    diverging = {
        "sunset-diverging": [
            "#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1",
            "#C2E4EF", "#EAECCC", "#FEDA8B", "#FDB366",
            "#F67E4B", "#DD3D2D", "#A50026"
        ],
        "burd": [
            "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0",
            "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D",
            "#B2182B"
        ],
        "prgn": [
            "#762A83", "#9970AB", "#C2A5CF",
            "#E7D4E8", "#F7F7F7", "#D9F0D3", "#ACD39E",
            "#5AAE61", "#1B7837", "#00441B"
        ],
    }

    # -------------------------------------------------------
    # Cyclic
    # -------------------------------------------------------
    cyclic = {
        "cyclic": [
            "#5E4FA2", "#3288BD", "#66C2A5", "#ABDDA4",
            "#E6F598", "#FEE08B", "#FDAE61", "#F46D43",
            "#D53E4F", "#9E0142", "#5E4FA2"
        ]
    }

    # -------------------------------------------------------
    # Combine all
    # -------------------------------------------------------
    all_maps = {
        **qualitative,
        **sequential,
        **diverging,
        **cyclic
    }

    # Aliases for convenience
    aliases = {
        "rainbow": "smoothrainbow",
        "rainbow_smooth": "smoothrainbow",
        "rainbow_dis": "rainbow_discrete",
        "diverging": "sunset-diverging",
        "seq": "blue",
    }

    if name in aliases:
        name = aliases[name]

    # -------------------------------------------------------
    # Validate and fetch
    # -------------------------------------------------------
    if name not in all_maps:
        raise ValueError(
            f"Unknown Tol colormap `{name}`. Valid names:\n{sorted(all_maps.keys())}"
        )

    colors = all_maps[name]

    # Reverse?
    if reverse:
        colors = list(reversed(colors))

    # -------------------------------------------------------
    # Hex output only
    # -------------------------------------------------------
    if as_hex:
        if discrete is not None:
            idx = np.linspace(0, len(colors) - 1, discrete).astype(int)
            return [colors[i] for i in idx]
        return list(colors)

    # -------------------------------------------------------
    # Discrete colormap
    # -------------------------------------------------------
    if discrete is not None:
        idx = np.linspace(0, len(colors) - 1, discrete).astype(int)
        sel = [colors[i] for i in idx]
        return ListedColormap(sel, name=f"{name}_{discrete}")

    # -------------------------------------------------------
    # Continuous colormap for sequential/diverging/cyclic
    # -------------------------------------------------------
    if name in sequential or name in diverging or name in cyclic:
        return LinearSegmentedColormap.from_list(name, colors)

    # Qualitative always discrete
    return ListedColormap(colors, name=name)


