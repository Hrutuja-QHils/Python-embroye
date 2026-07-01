"""
Streamlit app: Egg Vitelline Vessel Order Counter
==================================================
Upload an egg photo, click the embryo center, get primary/secondary/tertiary
vessel counts + a color-coded overlay.

Install:
    pip install streamlit streamlit-image-coordinates opencv-python-headless \
                scikit-image skan networkx pillow pillow-heif --break-system-packages

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
from streamlit_image_coordinates import streamlit_image_coordinates


# ----------------------------- core pipeline -----------------------------

def load_image(uploaded_file, max_width=1200):
    """Load an uploaded file (jpg/png/heic) into an OpenCV BGR array, resized."""
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if name.endswith(".heic"):
        import pillow_heif
        heif_file = pillow_heif.read_heif(raw_bytes)
        pil_img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
    else:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    scale = max_width / img.shape[1]
    img = cv2.resize(img, None, fx=scale, fy=scale)
    return img


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


def classify_orders(G, node_coords, root_center, hub_radius=90):
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)

    # Pick the component nearest the clicked root (not just the largest overall) —
    # this matters if the mask fragmented into multiple pieces.
    def min_dist_to_root(comp):
        return min(
            ((node_coords[n][0] - root_center[0]) ** 2 +
             (node_coords[n][1] - root_center[1]) ** 2) ** 0.5
            for n in comp
        )

    components.sort(key=min_dist_to_root)
    main_nodes = components[0]
    G_main = G.subgraph(main_nodes).copy()

    hub_nodes = [
        n for n in G_main.nodes()
        if ((node_coords[n][0] - root_center[0]) ** 2 +
            (node_coords[n][1] - root_center[1]) ** 2) ** 0.5 < hub_radius
    ]

    G2 = G_main.copy()
    root_id = "ROOT"
    G2.add_node(root_id)
    for n in hub_nodes:
        for nbr in G_main.neighbors(n):
            if nbr not in hub_nodes:
                G2.add_edge(root_id, nbr, length=G_main[n][nbr]["length"])
        if G2.has_node(n):
            G2.remove_node(n)

    if root_id not in G2 or G2.degree(root_id) == 0:
        return None, hub_nodes, main_nodes

    hop = nx.single_source_shortest_path_length(G2, root_id)

    order_map = {}
    for u, v in G2.edges():
        if u == root_id or v == root_id:
            order = 1
        else:
            order = min(hop.get(u, 999), hop.get(v, 999)) + 1
        order_map[(u, v)] = min(order, 3)

    return order_map, hub_nodes, main_nodes


def draw_overlay(img, skel_obj, branch_data, order_map, hub_nodes, main_nodes, root_center):
    colors = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}  # BGR
    overlay = img.copy()

    for i, row in branch_data.iterrows():
        src, dst = int(row["node-id-src"]), int(row["node-id-dst"])
        if src not in main_nodes or dst not in main_nodes:
            continue
        if src in hub_nodes or dst in hub_nodes:
            color = colors[1]
        elif (src, dst) in order_map:
            color = colors[order_map[(src, dst)]]
        elif (dst, src) in order_map:
            color = colors[order_map[(dst, src)]]
        else:
            continue
        coords = skel_obj.path_coordinates(i).astype(int)
        for (r, c) in coords:
            cv2.circle(overlay, (c, r), 1, color, -1)

    cv2.circle(overlay, (root_center[1], root_center[0]), 6, (0, 255, 255), 2)
    return overlay


# ----------------------------- streamlit UI -----------------------------

st.set_page_config(page_title="Egg Vessel Counter", layout="wide")
st.title("🥚 Egg Vitelline Vessel Order Counter")
st.write(
    "Upload a candled/opened egg photo, click the embryo center (the hub all "
    "vessels radiate from), and get primary / secondary / tertiary vessel counts."
)

with st.sidebar:
    st.header("Segmentation settings")
    frangi_threshold = st.slider("Frangi threshold", 1, 30, 6)
    close_kernel = st.slider("Gap-closing kernel size", 1, 15, 5, step=2)
    close_iters = st.slider("Gap-closing iterations", 1, 6, 3)
    min_object_size = st.slider("Minimum object size (px)", 5, 200, 40)
    hub_radius = st.slider("Hub radius (px around clicked center)", 10, 200, 90)

uploaded_file = st.file_uploader("Upload egg photo", type=["jpg", "jpeg", "png", "heic"])

if uploaded_file is not None:
    if "img" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
        st.session_state.img = load_image(uploaded_file)
        st.session_state.filename = uploaded_file.name
        st.session_state.root_center = None

    img = st.session_state.img
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_display = Image.fromarray(img_rgb)

    st.subheader("Step 1: Click the embryo center")
    coords = streamlit_image_coordinates(pil_display, key="click")

    if coords is not None:
        st.session_state.root_center = (coords["y"], coords["x"])  # (row, col)

    if st.session_state.root_center:
        st.success(f"Root point set at (row={st.session_state.root_center[0]}, col={st.session_state.root_center[1]})")

        if st.button("Run vessel analysis"):
            with st.spinner("Segmenting vessels..."):
                mask = segment_vessels(
                    img,
                    frangi_threshold=frangi_threshold,
                    close_kernel=close_kernel,
                    close_iters=close_iters,
                    min_object_size=min_object_size,
                )
                skeleton = skeletonize(mask > 0)
                G, node_coords, skel_obj, branch_data = build_graph(skeleton)
                order_map, hub_nodes, main_nodes = classify_orders(
                    G, node_coords, st.session_state.root_center, hub_radius=hub_radius
                )

            if order_map is None:
                st.error(
                    "No vessels found near the point you clicked. "
                    "Try clicking closer to the vessel hub, or increase the hub radius."
                )
            else:
                counts = Counter(order_map.values())
                overlay = draw_overlay(
                    img, skel_obj, branch_data, order_map, hub_nodes, main_nodes,
                    st.session_state.root_center
                )
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

                col1, col2, col3 = st.columns(3)
                col1.metric("Primary", counts.get(1, 0))
                col2.metric("Secondary", counts.get(2, 0))
                col3.metric("Tertiary", counts.get(3, 0))

                st.subheader("Classified overlay")
                st.image(
                    overlay_rgb,
                    caption="Red = primary, Green = secondary, Blue = tertiary, Yellow = root",
                    use_column_width=True,
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
        st.info("Click on the image above to set the embryo center.")
else:
    st.info("Upload an image to get started.")
