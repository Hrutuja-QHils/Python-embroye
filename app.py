"""
Streamlit app: Egg Vitelline Vessel Order Counter (Fully Automatic)
====================================================================
Upload an egg photo -> the app automatically finds the embryo center and
counts primary / secondary / tertiary blood vessel branches. No clicking
required.

Install:
    pip install -r requirements.txt

Run:
    streamlit run app.py
"""

import io
import cv2
import numpy as np
import networkx as nx
from collections import Counter
from PIL import Image
from skimage.filters import frangi
from skimage.morphology import skeletonize, remove_small_objects
from skimage import img_as_float
from skan import Skeleton, summarize

import streamlit as st


# ----------------------------- core pipeline -----------------------------

def load_image(uploaded_file, max_width=1200):
    """Load an uploaded file (jpg/png) into an OpenCV BGR array, resized."""
    raw_bytes = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    scale = max_width / img.shape[1]
    img = cv2.resize(img, None, fx=scale, fy=scale)
    return img


def detect_embryo_center(img, blur_sigma=25):
    """Automatically find the embryo hub.

    The hub is a *wide* patch of elevated redness, while vessels are *thin*
    lines of elevated redness. Heavily blurring the redness channel washes
    out thin lines but preserves the wide hub, so its brightest point after
    blurring is a reliable estimate of the embryo center.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _, A, _ = cv2.split(lab)
    A_blur = cv2.GaussianBlur(A, (0, 0), sigmaX=blur_sigma)
    y, x = np.unravel_index(np.argmax(A_blur), A_blur.shape)
    return (int(y), int(x))  # (row, col)


def segment_vessels(img, frangi_threshold=6, close_kernel=5, close_iters=3, min_object_size=40):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _, A, _ = cv2.split(lab)

    A_f = img_as_float(A)
    vesselness = frangi(A_f, sigmas=range(1, 5), black_ridges=False)
    vesselness = (vesselness / (vesselness.max() + 1e-9) * 255).astype(np.uint8)

    _, mask = cv2.threshold(vesselness, frangi_threshold, 255, cv2.THRESH_BINARY)
    mask = remove_small_objects(mask > 0, min_size=10)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
    mask = remove_small_objects(mask > 0, min_size=min_object_size)
    return mask


def build_graph(skeleton_mask):
    skel_obj = Skeleton(skeleton_mask)
    branch_data = summarize(skel_obj, separator="-")

    G = nx.Graph()
    node_coords = {}
    for _, row in branch_data.iterrows():
        src, dst = int(row["node-id-src"]), int(row["node-id-dst"])
        G.add_edge(src, dst, length=row["branch-distance"])
        node_coords[src] = (row["image-coord-src-0"], row["image-coord-src-1"])
        node_coords[dst] = (row["image-coord-dst-0"], row["image-coord-dst-1"])

    return G, node_coords, skel_obj, branch_data


def classify_orders(G, node_coords, root_center, arm_search_radius=300, min_arm_edges=3):
    """Classify every vessel segment as primary/secondary/tertiary.

    The embryo hub is usually a solid blob rather than a thin line, which
    means skeletonizing it creates a mess right at the center and often
    breaks the network into separate disconnected "arms" (one per primary
    vessel). Rather than fight that, we treat each arm as an independent
    tree rooted at whichever of its nodes sits closest to the embryo
    center, and classify branch order *within* that arm:
        hop 1 from the arm's root -> primary
        hop 2                     -> secondary
        hop 3+                    -> tertiary (capped)
    """
    def dist_to_root(n):
        r, c = node_coords[n]
        return ((r - root_center[0]) ** 2 + (c - root_center[1]) ** 2) ** 0.5

    components = list(nx.connected_components(G))

    arms = []
    for comp in components:
        if len(comp) < min_arm_edges:
            continue
        if min(dist_to_root(n) for n in comp) < arm_search_radius:
            arms.append(comp)

    order_map = {}
    for comp in arms:
        Gc = G.subgraph(comp).copy()
        local_root = min(comp, key=dist_to_root)
        hop = nx.single_source_shortest_path_length(Gc, local_root)
        for u, v in Gc.edges():
            order = min(hop.get(u, 999), hop.get(v, 999)) + 1
            order_map[(u, v)] = min(order, 3)

    return order_map, arms


def draw_overlay(img, skel_obj, branch_data, order_map, root_center):
    colors = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}  # BGR: primary, secondary, tertiary
    overlay = img.copy()

    for i, row in branch_data.iterrows():
        src, dst = int(row["node-id-src"]), int(row["node-id-dst"])
        key = (src, dst) if (src, dst) in order_map else ((dst, src) if (dst, src) in order_map else None)
        if key is None:
            continue
        color = colors[order_map[key]]
        coords = skel_obj.path_coordinates(i).astype(int)
        for (r, c) in coords:
            cv2.circle(overlay, (c, r), 1, color, -1)

    cv2.circle(overlay, (root_center[1], root_center[0]), 8, (0, 255, 255), 3)
    return overlay


# ----------------------------- streamlit UI -----------------------------

st.set_page_config(page_title="Egg Vessel Counter", layout="wide")
st.title("🥚 Egg Vitelline Vessel Order Counter")
st.write(
    "Upload a candled/opened egg photo. The app automatically finds the "
    "embryo center and counts primary / secondary / tertiary vessel branches "
    "— no clicking needed."
)

with st.sidebar:
    st.header("Settings")
    st.caption("Defaults work well for most photos — only tweak if results look off.")
    frangi_threshold = st.slider("Frangi threshold", 1, 30, 6)
    close_kernel = st.slider("Gap-closing kernel size", 1, 15, 5, step=2)
    close_iters = st.slider("Gap-closing iterations", 1, 6, 3)
    min_object_size = st.slider("Minimum object size (px)", 5, 200, 40)
    arm_search_radius = st.slider("Arm search radius (px around embryo center)", 50, 600, 300)

uploaded_file = st.file_uploader("Upload egg photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_image(uploaded_file)

    with st.spinner("Detecting embryo center and analyzing vessels..."):
        root_center = detect_embryo_center(img)
        mask = segment_vessels(
            img,
            frangi_threshold=frangi_threshold,
            close_kernel=close_kernel,
            close_iters=close_iters,
            min_object_size=min_object_size,
        )
        skeleton = skeletonize(mask > 0)
        G, node_coords, skel_obj, branch_data = build_graph(skeleton)
        order_map, arms = classify_orders(
            G, node_coords, root_center, arm_search_radius=arm_search_radius
        )

    if not order_map:
        st.error(
            "No vessel network detected. Try lowering the Frangi threshold "
            "or increasing the arm search radius in the sidebar."
        )
    else:
        counts = Counter(order_map.values())
        overlay = draw_overlay(img, skel_obj, branch_data, order_map, root_center)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        col1, col2, col3 = st.columns(3)
        col1.metric("Primary", counts.get(1, 0))
        col2.metric("Secondary", counts.get(2, 0))
        col3.metric("Tertiary", counts.get(3, 0))

        st.subheader("Classified overlay")
        st.image(
            overlay_rgb,
            caption="Red = primary, Green = secondary, Blue = tertiary, Yellow circle = detected embryo center",
            use_column_width=True,
        )

        st.caption(
            f"Detected {len(arms)} vessel arm(s) radiating from the embryo. "
            "If the yellow circle isn't on the embryo, or an arm looks missing, "
            "try adjusting the sliders in the sidebar."
        )

        buf = io.BytesIO()
        Image.fromarray(overlay_rgb).save(buf, format="PNG")
        st.download_button(
            "Download overlay image",
            data=buf.getvalue(),
            file_name="vessel_classified_overlay.png",
            mime="image/png",
        )
else:
    st.info("Upload an image to get started.")
