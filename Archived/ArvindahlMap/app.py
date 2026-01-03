#!/usr/bin/env python3
"""
Arvindahl Terrain Editor – Streamlit Web UI (stable)

- Visual heightmap editor for creating 3D-printable tiled STL terrain
- Sources: Procedural (Perlin), DEM (GeoTIFF), or grayscale image
- Tools: Raise / Lower / Smooth (applied once per stroke)
- Preview: 3D Surface (Plotly via HTML embed) or 2D Hillshade fallback
- Export: Watertight solid with base & skirt, sliced into tiles → ZIP of STLs

Recommended versions:
  streamlit==1.31.1
  streamlit-drawable-canvas==0.9.3

Run:
  streamlit run app.py
"""
from __future__ import annotations
import io, zipfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image

# Optional heavy deps
try:
    import rasterio
except Exception:
    rasterio = None

try:
    from noise import pnoise2
except Exception:
    pnoise2 = None

import plotly.graph_objects as go
import streamlit.components.v1 as components
import trimesh
from streamlit_drawable_canvas import st_canvas
from scipy.ndimage import gaussian_filter


# ---------- Core utilities ----------


def gen_perlin_heightmap(
    w: int,
    h: int,
    seed: int = 0,
    octaves: int = 6,
    roughness: float = 0.5,
    scale: float = 1 / 180.0,
    ridged: bool = False,
) -> np.ndarray:
    if pnoise2 is None:
        st.error("Install 'noise' for procedural terrain: pip install noise")
        st.stop()
    rng = np.random.RandomState(seed)
    ox, oy = rng.uniform(-1000, 1000, size=2)
    hm = np.zeros((h, w), dtype=np.float32)
    frequency = 1.0
    amplitude = 1.0
    max_amp = 0.0
    xs = np.arange(w)
    for _ in range(octaves):
        for y in range(h):
            yy = (y + oy) * scale * frequency
            row = np.array(
                [
                    pnoise2(
                        (x + ox) * scale * frequency,
                        yy,
                        repeatx=1024,
                        repeaty=1024,
                        base=seed,
                    )
                    for x in xs
                ],
                dtype=np.float32,
            )
            if ridged:
                row = 1.0 - np.abs(row)
            hm[y, :] += row * amplitude
        max_amp += amplitude
        amplitude *= roughness
        frequency *= 2.0
    hm = hm / max_amp
    hm -= hm.min()
    if hm.max() > 0:
        hm /= hm.max()
    return hm


def load_dem(path: Path, target_wh: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if rasterio is None:
        st.error("Install 'rasterio' for DEM import: pip install rasterio")
        st.stop()
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
    nodata = np.isnan(data) | np.isinf(data)
    if nodata.any():
        med = np.nanmedian(data)
        data[nodata] = med
    if target_wh is not None:
        data = resize_bilinear(data, target_wh)
    data -= data.min()
    if data.max() > 0:
        data /= data.max()
    return data.astype(np.float32)


def load_image_as_heightmap(
    file: Image.Image, target_wh: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    img = file.convert("L")
    if target_wh is not None:
        img = img.resize(target_wh, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def resize_bilinear(img: np.ndarray, target_wh: Tuple[int, int]) -> np.ndarray:
    tw, th = target_wh
    y = np.linspace(0, img.shape[0] - 1, th)
    x = np.linspace(0, img.shape[1] - 1, tw)
    xx, yy = np.meshgrid(x, y)
    x0 = np.floor(xx).astype(int)
    x1 = np.clip(x0 + 1, 0, img.shape[1] - 1)
    y0 = np.floor(yy).astype(int)
    y1 = np.clip(y0 + 1, 0, img.shape[0] - 1)
    wx = xx - x0
    wy = yy - y0
    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]
    out = (
        Ia * (1 - wx) * (1 - wy)
        + Ib * wx * (1 - wy)
        + Ic * (1 - wx) * wy
        + Id * wx * wy
    ).astype(np.float32)
    return out


def heightmap_to_solid(
    height: np.ndarray,
    map_mm: Tuple[float, float],
    z_mm: float,
    base_mm: float,
    skirt_mm: float = 0.0,
) -> trimesh.Trimesh:
    """
    Watertight solid:
      - top surface from heightfield (Z = base_mm + height*z_mm)
      - bottom slab at z=0
      - side walls stitched explicitly along all 4 edges (top→bottom)
    """
    H, W = height.shape
    Wmm, Hmm = map_mm

    xs = np.linspace(0.0, Wmm, W, dtype=np.float32)
    ys = np.linspace(0.0, Hmm, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    Z = (height.astype(np.float32) * z_mm) + float(base_mm)

    # Vertices (row-major)
    Vtop = np.column_stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)])
    Vbot = np.column_stack(
        [X.reshape(-1), Y.reshape(-1), np.zeros(H * W, dtype=np.float32)]
    )
    o_bot = len(Vtop)

    # Top faces (CCW from +Z)
    faces_top = []
    for y in range(H - 1):
        r0 = y * W
        r1 = (y + 1) * W
        for x in range(W - 1):
            i0 = r0 + x
            i1 = i0 + 1
            i2 = r1 + x
            i3 = i2 + 1
            faces_top.append([i0, i2, i1])
            faces_top.append([i1, i2, i3])

    # Bottom faces (flip so outward is −Z)
    faces_bot = []
    for y in range(H - 1):
        r0 = o_bot + y * W
        r1 = o_bot + (y + 1) * W
        for x in range(W - 1):
            i0 = r0 + x
            i1 = i0 + 1
            i2 = r1 + x
            i3 = i2 + 1
            faces_bot.append([i0, i1, i2])
            faces_bot.append([i1, i3, i2])

    # Side walls along the 4 edges (top→bottom)
    faces_side = []

    def stitch_strip(idx: np.ndarray, outward_ccw: bool = True):
        for a, b in zip(idx[:-1], idx[1:]):
            at, bt = int(a), int(b)
            ab, bb = o_bot + at, o_bot + bt
            if outward_ccw:
                faces_side.append([at, ab, bt])
                faces_side.append([bt, ab, bb])
            else:
                faces_side.append([at, bt, ab])
                faces_side.append([bt, bb, ab])

    # top edge (y=0, x:0→W-1) / right edge (x=W-1, y:0→H-1)
    stitch_strip(np.arange(0, W), True)
    stitch_strip((np.arange(0, H) * W + (W - 1)), True)
    # bottom edge (y=H-1, x:W-1→0) / left edge (x=0, y:H-1→0) with reversed winding
    stitch_strip((H - 1) * W + np.arange(W - 1, -1, -1), False)
    stitch_strip((np.arange(H - 1, -1, -1) * W + 0), False)

    V = np.vstack([Vtop, Vbot])
    F = np.vstack(
        [
            np.asarray(faces_top, dtype=np.int64),
            np.asarray(faces_bot, dtype=np.int64),
            np.asarray(faces_side, dtype=np.int64),
        ]
    )

    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

    # ---- Robust repair (API-compatible across trimesh versions)
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    try:
        import trimesh.repair as trr

        if hasattr(trr, "fix_winding"):
            trr.fix_winding(mesh)
    except Exception:
        pass
    try:
        mesh.compute_vertex_normals()
    except Exception:
        pass
    if not mesh.is_watertight:
        mesh.fill_holes()
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()

    return mesh


def export_tiles_zip(
    height: np.ndarray,
    map_mm: Tuple[float, float],
    tile_mm: Tuple[float, float],
    z_mm: float,
    base_mm: float,
    skirt_mm: float,
    prefix: str = "arvindahl",
) -> bytes:
    H, W = height.shape
    Wmm, Hmm = map_mm
    px_per_mm_x = W / Wmm
    px_per_mm_y = H / Hmm
    tile_px_x = int(round(tile_mm[0] * px_per_mm_x))
    tile_px_y = int(round(tile_mm[1] * px_per_mm_y))
    nx = int(np.ceil(W / tile_px_x))
    ny = int(np.ceil(H / tile_px_y))

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for j in range(ny):
            for i in range(nx):
                x0 = i * tile_px_x
                y0 = j * tile_px_y
                x1 = min(W, x0 + tile_px_x)
                y1 = min(H, y0 + tile_px_y)
                tile = height[y0:y1, x0:x1]
                tWmm = (x1 - x0) / px_per_mm_x
                tHmm = (y1 - y0) / px_per_mm_y

                # Build solid + light repair
                mesh = heightmap_to_solid(
                    tile, (tWmm, tHmm), z_mm=z_mm, base_mm=base_mm, skirt_mm=skirt_mm
                )
                mesh.remove_duplicate_faces()
                mesh.remove_degenerate_faces()
                mesh.remove_unreferenced_vertices()
                mesh.merge_vertices()
                if not mesh.is_watertight:
                    mesh.fill_holes()  # don't reassign

                # Export STL via BytesIO (version-compatible). This yields **binary STL** in most trimesh builds.
                buf = io.BytesIO()
                mesh.export(file_obj=buf, file_type="stl")
                stl_bytes = buf.getvalue()

                name = f"{prefix}_r{j:02d}c{i:02d}.stl"
                zf.writestr(name, stl_bytes)

    mem.seek(0)
    return mem.read()


def preview_3d(height: np.ndarray, title: str = "Terrain") -> go.Figure:
    """Plotly 3D surface with fixed camera; caller embeds via components.html."""
    H, W = height.shape
    x = np.arange(W)
    y = np.arange(H)
    z = height.astype(np.float32)
    z = (z - z.min()) / (z.max() - z.min() + 1e-8)

    surf = go.Surface(z=z, x=x, y=y, colorscale="Viridis", showscale=False)
    fig = go.Figure(data=[surf])
    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Elevation",
        ),
    )
    return fig


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Arvindahl Terrain Editor", layout="wide")
st.title("Arvindahl Terrain Editor")

with st.sidebar:
    st.header("Source")
    mode = st.selectbox(
        "Heightmap source", ["Procedural", "DEM GeoTIFF", "Image (grayscale)"]
    )

    target_w = st.number_input("Resolution width (px)", 400, 4096, 1200, step=100)
    target_h = st.number_input("Resolution height (px)", 400, 4096, 800, step=100)

    if mode == "Procedural":
        seed = st.number_input("Seed", 0, 999999, 11)
        octaves = st.slider("Octaves", 1, 10, 6)
        roughness = st.slider("Roughness (persistence)", 0.2, 0.9, 0.55)
        scale = st.number_input(
            "Feature scale (lower=bigger features)",
            0.0005,
            0.02,
            1 / 180.0,
            format="%f",
        )
        ridged = st.checkbox("Ridged style", value=False)
        if st.button("Generate terrain", use_container_width=True):
            st.session_state.height = gen_perlin_heightmap(
                target_w, target_h, seed, octaves, roughness, scale, ridged
            )
            st.session_state.last_obj_count = 0
            st.session_state.canvas_key = st.session_state.get("canvas_key", 0) + 1

    elif mode == "DEM GeoTIFF":
        dem_file = st.file_uploader("Upload GeoTIFF", type=["tif", "tiff"])
        if st.button("Load DEM", disabled=(dem_file is None), use_container_width=True):
            tmp = Path("/tmp/_upload.tif")
            tmp.write_bytes(dem_file.read())
            st.session_state.height = load_dem(tmp, (target_w, target_h))
            st.session_state.last_obj_count = 0
            st.session_state.canvas_key = st.session_state.get("canvas_key", 0) + 1

    else:
        img_file = st.file_uploader(
            "Upload grayscale heightmap (PNG/JPG)", type=["png", "jpg", "jpeg"]
        )
        if st.button(
            "Load image", disabled=(img_file is None), use_container_width=True
        ):
            img = Image.open(img_file)
            st.session_state.height = load_image_as_heightmap(img, (target_w, target_h))
            st.session_state.last_obj_count = 0
            st.session_state.canvas_key = st.session_state.get("canvas_key", 0) + 1

    st.header("Edit tools")
    brush_mode = st.selectbox("Brush", ["Raise", "Lower", "Smooth", "None"])
    brush_r = st.slider("Brush radius (px)", 1, 200, 40)
    brush_h = st.slider("Raise/Lower amount", 0.0, 0.5, 0.12)
    feather = st.slider("Feather", 0.0, 1.0, 0.3)  # reserved for future, cosmetic here
    smooth_sigma = st.slider("Smooth sigma", 0.0, 8.0, 2.0)

    # Reset canvas button (fixes rare blank/ghost cases)
    if st.button("Reset paint canvas"):
        st.session_state.last_obj_count = 0
        st.session_state.canvas_key = st.session_state.get("canvas_key", 0) + 1

    st.divider()
    st.header("Preview")
    preview_type = st.selectbox(
        "Preview type", ["3D Surface (WebGL)", "2D Hillshade (fast)"]
    )

    st.divider()
    st.header("Export")
    map_w_mm = st.number_input("Map width (mm)", 50.0, 3000.0, 914.4)
    map_h_mm = st.number_input("Map height (mm)", 50.0, 3000.0, 609.6)
    tile_w_mm = st.number_input("Tile width (mm)", 20.0, 200.0, 94.0)
    tile_h_mm = st.number_input("Tile height (mm)", 20.0, 200.0, 52.0)
    relief_mm = st.slider("Max relief height (mm)", 5.0, 50.0, 18.0)
    z_exag = st.slider("Z exaggeration", 1.0, 2.5, 1.5)
    base_mm = st.slider("Base thickness (mm)", 2.0, 10.0, 4.0)
    skirt_mm = st.slider("Edge skirt (mm)", 0.0, 5.0, 1.0)
    prefix = st.text_input("Filename prefix", "arvindahl")

    if st.button("Export STL tiles (ZIP)", use_container_width=True):
        if "height" not in st.session_state:
            st.warning("Create or load a heightmap first.")
        else:
            z_mm = relief_mm * z_exag
            data = export_tiles_zip(
                st.session_state.height.copy(),
                (map_w_mm, map_h_mm),
                (tile_w_mm, tile_h_mm),
                z_mm,
                base_mm,
                skirt_mm,
                prefix,
            )
            st.download_button(
                "Download STLs ZIP",
                data,
                file_name=f"{prefix}_tiles.zip",
                mime="application/zip",
                use_container_width=True,
            )

# ---------- App state defaults ----------
if "height" not in st.session_state:
    st.session_state.height = (
        gen_perlin_heightmap(
            800, 600, seed=11, octaves=6, roughness=0.55, scale=1 / 180.0, ridged=False
        )
        if pnoise2
        else np.zeros((600, 800), dtype=np.float32)
    )
if "last_obj_count" not in st.session_state:
    st.session_state.last_obj_count = 0
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

# ---------- Layout ----------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Paint (click/drag)")
    hm = st.session_state.height
    if hm is None or getattr(hm, "size", 0) == 0:
        st.session_state.height = gen_perlin_heightmap(
            800, 600, seed=11, octaves=6, roughness=0.55, scale=1 / 180.0, ridged=False
        )
        hm = st.session_state.height
    H, W = hm.shape

    # Display size (preserve aspect ratio)
    disp_w = min(800, int(800 * W / max(H, W)))
    disp_h = min(600, int(600 * H / max(H, W)))

    # Background image at the exact display size
    bg_img = (
        Image.fromarray((hm * 255).astype(np.uint8))
        .convert("RGB")
        .resize((disp_w, disp_h), Image.BILINEAR)
    )

    canvas_result = st_canvas(
        stroke_width=1,
        stroke_color="#ff0000",
        background_color="#000000",
        background_image=bg_img,
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        drawing_mode=(
            "freedraw" if (brush_mode in ["Raise", "Lower", "Smooth"]) else "transform"
        ),
        key=f"canvas-{st.session_state.canvas_key}",
    )

    # Apply brush ONCE per new stroke
    if canvas_result and brush_mode in ["Raise", "Lower", "Smooth"]:
        obj_count = 0
        try:
            if canvas_result.json_data and "objects" in canvas_result.json_data:
                obj_count = len(canvas_result.json_data["objects"]) or 0
        except Exception:
            obj_count = 0

        has_new_stroke = obj_count > st.session_state.last_obj_count

        if has_new_stroke and canvas_result.image_data is not None:
            ch, cw = canvas_result.image_data.shape[:2]
            drawn_rgb = canvas_result.image_data[:, :, :3].astype(np.float32)

            # Resize bg to actual canvas pixel size (handles HiDPI/Retina)
            bg_resized = bg_img.resize((cw, ch), Image.BILINEAR)
            bg_rgb = np.asarray(bg_resized).astype(np.float32)

            # Difference → mask where the user drew
            diff = np.mean(np.abs(drawn_rgb - bg_rgb), axis=2)  # 0..255
            mask_disp = (diff > 8.0).astype(np.float32)  # ignore toolbar pixels

            # Resize mask back to heightmap size
            mask_img = Image.fromarray((mask_disp * 255).astype(np.uint8)).resize(
                (W, H), Image.BILINEAR
            )
            mask = np.asarray(mask_img).astype(np.float32) / 255.0

            # Influence field (blurred mask) scaled to brush radius
            scale_x = W / cw
            scale_y = H / ch
            sigma = max(1.0, 0.5 * brush_r * 0.5 * (scale_x + scale_y))
            infl = gaussian_filter(mask, sigma=sigma)
            infl = infl - infl.min()
            if infl.max() > 0:
                infl = infl / infl.max()

            if brush_mode == "Smooth":
                blurred = gaussian_filter(hm, sigma=smooth_sigma)
                hm[:] = hm * (1 - infl) + blurred * infl
            else:
                sign = 1.0 if brush_mode == "Raise" else -1.0
                delta = sign * brush_h * (0.5 - 0.5 * np.cos(np.pi * infl))
                hm[:] = np.clip(hm + delta, 0.0, 1.0)

            st.session_state.height = hm
            st.session_state.last_obj_count = obj_count
            # Clear the drawn stroke for the next one
            st.session_state.canvas_key += 1

with col_right:
    st.subheader("3D Preview")
    if preview_type.startswith("3D"):
        fig = preview_3d(st.session_state.height, title="Terrain")
        # HTML embed sidesteps Firefox/Safari quirks
        components.html(
            fig.to_html(include_plotlyjs="cdn", full_html=False),
            height=520,
            scrolling=False,
        )
    else:
        # 2D hillshade fallback
        z = st.session_state.height
        z = (z - z.min()) / (z.max() - z.min() + 1e-8)
        az, alt = np.deg2rad(315), np.deg2rad(45)
        gy, gx = np.gradient(z)
        slope = np.pi / 2 - np.arctan(np.sqrt(gx * gx + gy * gy))
        aspect = np.arctan2(-gx, gy)
        shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(
            az - aspect
        )
        shade = (shade - shade.min()) / (shade.max() - shade.min() + 1e-8)
        st.image(
            (shade * 255).astype(np.uint8),
            clamp=True,
            caption="Hillshade preview",
            use_column_width=True,
        )

st.caption(
    "Tip: set Export sizes near your printer’s sweet spot and use Z exaggeration 1.3–1.6× for subtle-to-medium drama."
)
