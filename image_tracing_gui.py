import os
import math
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ExifTags, ImageOps

# Built-in GUI modules
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -------------------------------------------------------------------------
# Random Word Pool for Crazy Text Mode
# -------------------------------------------------------------------------
CRAZY_WORDS = [
    "NEBULA", "PRISM", "ECHO", "SOLAR", "KINETIC", "LUMEN", "AURA",
    "CHRONO", "VELOCITY", "VORTEX", "ZENITH", "COSMOS", "GLITCH",
    "ORBIT", "RADIANT", "PULSE", "HALO", "STATIC", "OBLIVION", "FLUX"
]

# -------------------------------------------------------------------------
# 1. Path Generators
# -------------------------------------------------------------------------

def generate_curve(
    curve_type: str = "archimedean_spiral",
    num_points: int = 400,
    seed: int = 42
) -> np.ndarray:
    np.random.seed(seed)
    random.seed(seed)
    t = np.linspace(0, 1, num_points)

    if curve_type == "archimedean_spiral":
        theta = t * 7 * np.pi
        r = t
        x = r * np.cos(theta)
        y = r * np.sin(theta)

    elif curve_type == "logarithmic_spiral":
        theta = t * 5 * np.pi
        r = 0.05 * np.exp(0.45 * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

    elif curve_type == "sunflower_fermat":
        indices = np.arange(num_points)
        golden_angle = np.pi * (3 - np.sqrt(5))
        theta = indices * golden_angle
        r = np.sqrt(indices)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

    elif curve_type == "infinity_lemniscate":
        angle = t * 2 * np.pi
        denom = 1 + np.sin(angle) ** 2
        x = np.cos(angle) / denom
        y = (np.sin(angle) * np.cos(angle)) / denom

    elif curve_type == "hawk_soaring":
        wind_drift = 3.5 * t
        radius = 1.0
        angle = t * 8 * np.pi
        x = wind_drift + radius * np.cos(angle)
        y = radius * np.sin(angle)

    elif curve_type == "lissajous":
        x = np.sin(3 * t * 2 * np.pi)
        y = np.sin(4 * t * 2 * np.pi)

    elif curve_type == "rose_curve":
        k = 4
        r = np.cos(k * t * 2 * np.pi)
        theta = t * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)

    elif curve_type == "bee_flight":
        steps = np.random.standard_cauchy(size=(num_points, 2))
        steps = np.clip(steps, -3.0, 3.0)
        pts = np.cumsum(steps, axis=0)
        x, y = pts[:, 0], pts[:, 1]

    else:
        raise ValueError(f"Unknown curve type: {curve_type}")

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
    return np.column_stack((x_norm, y_norm))


def extract_path_from_image(mask_path: str, num_points: int = 400) -> np.ndarray:
    img = Image.open(mask_path).convert("L")
    arr = np.array(img)

    if np.mean(arr) > 127:
        arr = 255 - arr

    y_indices, x_indices = np.where(arr > 128)
    if len(x_indices) == 0:
        raise ValueError("No path line detected in the provided mask image.")

    coords = np.column_stack((x_indices, y_indices))
    sort_order = np.lexsort((coords[:, 1], coords[:, 0]))
    sorted_coords = coords[sort_order]

    sample_idx = np.linspace(0, len(sorted_coords) - 1, num_points).astype(int)
    sampled = sorted_coords[sample_idx].astype(float)
    sampled[:, 0] /= arr.shape[1]
    sampled[:, 1] /= arr.shape[0]
    return sampled


# -------------------------------------------------------------------------
# 2. Metadata Extraction & Dominant Hue Calculation
# -------------------------------------------------------------------------

def get_image_datetime(file_path: Path) -> datetime:
    """
    Extracts camera capture timestamp from root EXIF or nested ExifOffset IFDs.
    Falls back to file birthtime (macOS) or modification time.
    """
    date_candidates = []

    try:
        with Image.open(file_path) as img:
            exif_data = img.getexif()
            if exif_data:
                # 1. Root EXIF tags
                for tag_id, val in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag_name in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                        date_candidates.append(str(val))

                # 2. Nested Sub-IFD (Tag 34665 / 0x8769) where cameras write capture time
                exif_ifd = exif_data.get_ifd(0x8769)
                if exif_ifd:
                    for tag_id, val in exif_ifd.items():
                        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                        if tag_name in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                            date_candidates.insert(0, str(val))

        formats_to_try = [
            "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y:%m:%d %H:%M",
        ]
        for candidate in date_candidates:
            clean_str = candidate.strip("\x00 \t\r\n")
            for fmt in formats_to_try:
                try:
                    return datetime.strptime(clean_str, fmt)
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass

    # 3. Native macOS creation time fallback
    try:
        stat = file_path.stat()
        birth = getattr(stat, "st_birthtime", None)
        if birth:
            return datetime.fromtimestamp(birth)
        return datetime.fromtimestamp(stat.st_mtime)
    except Exception:
        return datetime.now()


def get_dominant_hue(img: Image.Image) -> float:
    """Calculates dominant color hue (0.0 to 1.0) for Color Wheel Theory sorting."""
    thumb = img.convert("RGB").resize((40, 40), Image.Resampling.BOX)
    arr = np.array(thumb, dtype=float) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min + 1e-8

    h = np.zeros_like(c_max)
    mask_r = (c_max == r)
    mask_g = (c_max == g)
    mask_b = (c_max == b)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4
    h = h / 6.0

    sat = delta / (c_max + 1e-8)
    weights = sat.flatten()
    if weights.sum() > 0:
        return float(np.average(h.flatten(), weights=weights))
    return float(np.mean(h))


# -------------------------------------------------------------------------
# 3. Text Overlay Engine
# -------------------------------------------------------------------------

def render_relative_text(
    img: Image.Image,
    dt: datetime,
    text_option: str,
    text_ratio: float = 0.15
) -> Image.Image:
    if text_option == "none" or text_ratio <= 0.0:
        return img

    if text_option == "date_mm_dd_yy":
        label = dt.strftime("%m/%d/%y")
    elif text_option == "day_of_week":
        label = dt.strftime("%A")
    elif text_option == "month":
        label = dt.strftime("%B")
    elif text_option == "year":
        label = dt.strftime("%Y")
    elif text_option == "crazy_random_words":
        label = random.choice(CRAZY_WORDS)
    else:
        label = ""

    if not label:
        return img

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    font_size = max(10, int(annotated.height * text_ratio))
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    t_w = bbox[2] - bbox[0]
    t_h = bbox[3] - bbox[1]

    pad = 4
    x1 = (annotated.width - t_w) // 2 - pad
    y1 = annotated.height - t_h - pad * 2 - 4
    x2 = x1 + t_w + pad * 2
    y2 = y1 + t_h + pad * 2

    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    d_overlay = ImageDraw.Draw(overlay)
    d_overlay.rounded_rectangle([x1, y1, x2, y2], radius=4, fill=(0, 0, 0, 160))
    annotated = Image.alpha_composite(annotated, overlay)

    draw = ImageDraw.Draw(annotated)
    draw.text(((annotated.width - t_w) // 2, y1 + pad), label, font=font, fill=(255, 255, 255, 255))
    return annotated


# -------------------------------------------------------------------------
# 4. Image Ingestion, Recursive Traversal, & Normalization
# -------------------------------------------------------------------------

def load_and_normalize_inputs(
    folder_path: str,
    target_dim_size: int = 200,
    norm_mode: str = "equal_fixed_square_crop",
    render_mode: str = "images",
    text_option: str = "none",
    text_ratio: float = 0.15,
    recursive: bool = False
) -> List[Dict[str, Any]]:
    folder = Path(folder_path)

    if render_mode == "color_rectangles":
        colors = [
            (239, 68, 68), (59, 130, 246), (16, 185, 129),
            (245, 158, 11), (139, 92, 246), (236, 72, 153)
        ]
        ratios = [1.2, 0.8, 1.5, 1.0, 1.3, 0.7]
        items = []
        for i, col in enumerate(colors):
            ratio = ratios[i % len(ratios)]
            w = target_dim_size if "width" in norm_mode else int(target_dim_size * ratio)
            h = target_dim_size if "height" in norm_mode or "square" in norm_mode else int(w / ratio)
            img = Image.new("RGBA", (max(10, w), max(10, h)), col + (255,))
            items.append({
                "image": img,
                "datetime": datetime.now(),
                "hue": float(i) / len(colors),
                "name": f"color_{i}"
            })
        return items

    # Full collection of supported photo formats including WebP
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

    if folder.exists():
        if recursive:
            file_paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
        else:
            file_paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    else:
        file_paths = []

    if not file_paths:
        raise FileNotFoundError(f"No valid image files found in '{folder_path}' (Recursive={recursive})")

    records = []
    for p in file_paths:
        dt = get_image_datetime(p)
        try:
            raw_img = Image.open(p)
            # Transpose pixels to match camera sensor rotation before converting to RGBA
            raw_img = ImageOps.exif_transpose(raw_img)
            raw_img = raw_img.convert("RGBA")
            hue = get_dominant_hue(raw_img)
            records.append({"raw": raw_img, "datetime": dt, "hue": hue, "name": p.name})
        except Exception:
            continue

    if not records:
        raise ValueError("Could not decode any valid image files from the selected path.")

    normalized = []
    for item in records:
        img = item["raw"]
        orig_w, orig_h = img.size

        if norm_mode == "equal_fixed_square_crop":
            min_side = min(orig_w, orig_h)
            left = (orig_w - min_side) // 2
            top = (orig_h - min_side) // 2
            cropped = img.crop((left, top, left + min_side, top + min_side))
            scaled = cropped.resize((target_dim_size, target_dim_size), Image.Resampling.LANCZOS)

        elif norm_mode == "equal_fixed_square_pad":
            aspect = orig_w / orig_h
            if aspect >= 1.0:
                new_w = target_dim_size
                new_h = max(1, int(target_dim_size / aspect))
            else:
                new_h = target_dim_size
                new_w = max(1, int(target_dim_size * aspect))
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            scaled = Image.new("RGBA", (target_dim_size, target_dim_size), (0, 0, 0, 0))
            scaled.paste(resized, ((target_dim_size - new_w) // 2, (target_dim_size - new_h) // 2))

        elif norm_mode == "equal_height":
            aspect = orig_w / orig_h
            h = target_dim_size
            w = max(4, int(h * aspect))
            scaled = img.resize((w, h), Image.Resampling.LANCZOS)

        elif norm_mode == "equal_width":
            aspect = orig_h / orig_w
            w = target_dim_size
            h = max(4, int(w * aspect))
            scaled = img.resize((w, h), Image.Resampling.LANCZOS)

        elif norm_mode == "proportional_max_width":
            max_w = max(it["raw"].width for it in records)
            scale = target_dim_size / max_w
            scaled = img.resize((max(4, int(orig_w * scale)), max(4, int(orig_h * scale))), Image.Resampling.LANCZOS)

        elif norm_mode == "proportional_max_height":
            max_h = max(it["raw"].height for it in records)
            scale = target_dim_size / max_h
            scaled = img.resize((max(4, int(orig_w * scale)), max(4, int(orig_h * scale))), Image.Resampling.LANCZOS)

        else:
            scaled = img

        final_img = render_relative_text(scaled, item["datetime"], text_option, text_ratio)

        normalized.append({
            "image": final_img,
            "datetime": item["datetime"],
            "hue": item["hue"],
            "name": item["name"]
        })

    return normalized


# -------------------------------------------------------------------------
# 5. Metadata Clustering Criterion
# -------------------------------------------------------------------------

def assign_cluster_keys(items: List[Dict[str, Any]], criterion: str) -> Dict[str, List[Dict[str, Any]]]:
    clusters: Dict[str, List[Dict[str, Any]]] = {}

    for it in items:
        dt = it["datetime"]
        if criterion == "month":
            key = dt.strftime("%Y-%m (%B)")
            sort_val = (dt.year, dt.month)
        elif criterion == "year":
            key = dt.strftime("%Y")
            sort_val = dt.year
        elif criterion == "day_of_week":
            key = dt.strftime("%A")
            sort_val = dt.weekday()
        elif criterion == "exact_day":
            key = dt.strftime("%Y-%m-%d")
            sort_val = (dt.year, dt.month, dt.day)
        elif criterion == "hour":
            key = f"{dt.hour:02d}:00"
            sort_val = dt.hour
        elif criterion == "day_moment":
            h = dt.hour
            if 5 <= h < 12:
                key, sort_val = "Morning", 0
            elif 12 <= h < 17:
                key, sort_val = "Afternoon", 1
            elif 17 <= h < 21:
                key, sort_val = "Evening", 2
            else:
                key, sort_val = "Night", 3
        elif criterion == "season":
            m = dt.month
            if m in (3, 4, 5):
                key, sort_val = "Spring", 0
            elif m in (6, 7, 8):
                key, sort_val = "Summer", 1
            elif m in (9, 10, 11):
                key, sort_val = "Autumn/Fall", 2
            else:
                key, sort_val = "Winter", 3
        else:
            key, sort_val = "All", 0

        it["_cluster_sort"] = sort_val
        clusters.setdefault(key, []).append(it)

    sorted_keys = sorted(clusters.keys(), key=lambda k: clusters[k][0]["_cluster_sort"])
    return {k: clusters[k] for k in sorted_keys}


# -------------------------------------------------------------------------
# 6. Placement Planning & Color Sorting
# -------------------------------------------------------------------------

def plan_placements(
    items: List[Dict[str, Any]],
    num_points: int,
    strategy: str = "unique_spread",
    repetition_style: str = "golden_ratio"
) -> List[Tuple[int, int]]:
    num_items = len(items)

    # Color Gradient - Unique
    if strategy == "color_gradient_unique":
        sorted_indices = sorted(range(num_items), key=lambda i: items[i]["hue"])
        count = min(num_items, num_points)
        point_indices = np.linspace(0, num_points - 1, count).astype(int).tolist()
        return list(zip(point_indices, sorted_indices[:count]))

    # Color Gradient - Repeat
    elif strategy == "color_gradient_repeat":
        sorted_indices = sorted(range(num_items), key=lambda i: items[i]["hue"])
        img_seq = [sorted_indices[i % num_items] for i in range(num_points)]
        return list(zip(range(num_points), img_seq))

    # Unique Spread
    elif strategy == "unique_spread":
        count = min(num_items, num_points)
        point_indices = np.linspace(0, num_points - 1, count).astype(int).tolist()
        item_indices = list(range(count))
        if repetition_style in ("golden_ratio", "random"):
            random.shuffle(item_indices)
        return list(zip(point_indices, item_indices))

    # Unique Sequential
    elif strategy == "unique_sequential":
        count = min(num_items, num_points)
        item_indices = list(range(count))
        if repetition_style in ("golden_ratio", "random"):
            random.shuffle(item_indices)
        return list(zip(range(count), item_indices))

    # Repetition
    elif strategy == "repetition":
        if repetition_style == "round_robin":
            seq = [i % num_items for i in range(num_points)]
        elif repetition_style == "golden_ratio":
            phi = (1 + 5**0.5) / 2
            seq = [int(((i * phi) % 1.0) * num_items) for i in range(num_points)]
        elif repetition_style == "burst_clusters":
            seq = []
            cur = 0
            cnt = 0
            cl_len = random.randint(3, 6)
            for _ in range(num_points):
                seq.append(cur)
                cnt += 1
                if cnt >= cl_len:
                    cur = (cur + 1) % num_items
                    cnt = 0
                    cl_len = random.randint(3, 6)
        elif repetition_style == "random":
            seq = [random.randint(0, num_items - 1) for _ in range(num_points)]
        else:
            seq = [i % num_items for i in range(num_points)]
        return list(zip(range(num_points), seq))

    return list(zip(range(min(num_items, num_points)), range(min(num_items, num_points))))


def compute_scale_multiplier(progress: float, scale_variation: str, min_scale: float, max_scale: float) -> float:
    if scale_variation == "constant":
        return 1.0
    elif scale_variation == "sine_wave":
        wave = (np.sin(progress * 4 * np.pi - np.pi / 2) + 1.0) / 2.0
        return float(min_scale + wave * (max_scale - min_scale))
    elif scale_variation == "linear_taper":
        return float(max_scale - progress * (max_scale - min_scale))
    elif scale_variation == "bell_curve":
        bell = np.exp(-16 * (progress - 0.5) ** 2)
        return float(min_scale + bell * (max_scale - min_scale))
    return 1.0


# -------------------------------------------------------------------------
# 7. Master Tracing & Composition Engine
# -------------------------------------------------------------------------

def trace_images_on_path(
    folder_path: str = ".",
    output_path: str = "final_output.png",
    render_mode: str = "images",
    clustering_criterion: str = "none",
    cluster_distribution: str = "grid",
    allow_overlap: bool = False,
    recursive_search: bool = False,
    placement_strategy: str = "unique_spread",
    min_padding: int = 10,
    canvas_size: Tuple[int, int] = (2400, 2400),
    density: int = 350,
    curve_source: Union[str, np.ndarray] = "archimedean_spiral",
    mask_file: str = None,
    base_dim_size: int = 180,
    norm_mode: str = "equal_fixed_square_crop",
    repetition_style: str = "golden_ratio",
    scale_variation: str = "constant",
    scale_range: Tuple[float, float] = (0.5, 1.3),
    text_option: str = "none",
    text_ratio: float = 0.15,
    bg_color: Tuple[int, int, int, int] = (245, 247, 250, 0)
) -> Tuple[int, int]:
    # Ingest and normalize
    items = load_and_normalize_inputs(
        folder_path=folder_path,
        target_dim_size=base_dim_size,
        norm_mode=norm_mode,
        render_mode=render_mode,
        text_option=text_option,
        text_ratio=text_ratio,
        recursive=recursive_search
    )
    if not items:
        return 0, 0

    c_w, c_h = canvas_size
    canvas = Image.new("RGBA", canvas_size, bg_color)
    occupancy_mask = np.zeros((c_h, c_w), dtype=bool)

    if clustering_criterion != "none":
        cluster_map = assign_cluster_keys(items, clustering_criterion)
    else:
        cluster_map = {"All": items}

    num_clusters = len(cluster_map)

    # Compute cell positions
    cluster_centers = []
    if num_clusters == 1:
        cluster_centers = [(c_w / 2.0, c_h / 2.0, min(c_w, c_h) * 0.42)]
    elif cluster_distribution == "grid":
        cols = math.ceil(math.sqrt(num_clusters))
        rows = math.ceil(num_clusters / cols)
        dx = c_w / (cols + 1)
        dy = c_h / (rows + 1)
        radius = min(dx, dy) * 0.42
        for i in range(num_clusters):
            r = i // cols
            c = i % cols
            cluster_centers.append(((c + 1) * dx, (r + 1) * dy, radius))
    elif cluster_distribution == "circle":
        cx, cy = c_w / 2.0, c_h / 2.0
        orbit = min(c_w, c_h) * 0.32
        radius = (2 * math.pi * orbit / (num_clusters * 2.2))
        radius = min(radius, min(c_w, c_h) * 0.2)
        for i in range(num_clusters):
            angle = 2 * math.pi * i / num_clusters - math.pi / 2
            cluster_centers.append((cx + orbit * math.cos(angle), cy + orbit * math.sin(angle), radius))
    elif cluster_distribution == "random":
        np.random.seed(42)
        radius = min(c_w, c_h) / (math.sqrt(num_clusters) * 2.8)
        margin = radius + base_dim_size
        for _ in range(num_clusters):
            rx = random.uniform(margin, c_w - margin)
            ry = random.uniform(margin, c_h - margin)
            cluster_centers.append((rx, ry, radius))

    total_placed = 0
    total_attempted = 0

    # Trace each cluster
    for c_idx, (c_key, group_items) in enumerate(cluster_map.items()):
        cx, cy, c_radius = cluster_centers[c_idx]

        c_density = max(10, density // num_clusters) if clustering_criterion != "none" else density
        if mask_file and mask_file.strip():
            raw_pts = extract_path_from_image(mask_file.strip(), num_points=c_density)
        else:
            raw_pts = generate_curve(curve_source, num_points=c_density, seed=42 + c_idx)

        scheduled = plan_placements(
            items=group_items,
            num_points=c_density,
            strategy=placement_strategy,
            repetition_style=repetition_style
        )
        total_attempted += len(scheduled)

        for pt_idx, img_idx in scheduled:
            norm_x, norm_y = raw_pts[pt_idx]
            progress = pt_idx / max(1, len(raw_pts) - 1)
            src_img = group_items[img_idx]["image"]

            mult = compute_scale_multiplier(progress, scale_variation, scale_range[0], scale_range[1])
            curr_w = max(4, int(src_img.width * mult))
            curr_h = max(4, int(src_img.height * mult))
            scaled_img = src_img.resize((curr_w, curr_h), Image.Resampling.LANCZOS)

            px = int(cx + (norm_x - 0.5) * 2.0 * c_radius)
            py = int(cy + (norm_y - 0.5) * 2.0 * c_radius)

            x1 = px - curr_w // 2
            y1 = py - curr_h // 2
            x2 = x1 + curr_w
            y2 = y1 + curr_h

            if x1 < 0 or y1 < 0 or x2 >= c_w or y2 >= c_h:
                continue

            if not allow_overlap:
                cx1 = max(0, x1 - min_padding)
                cy1 = max(0, y1 - min_padding)
                cx2 = min(c_w, x2 + min_padding)
                cy2 = min(c_h, y2 + min_padding)

                if np.any(occupancy_mask[cy1:cy2, cx1:cx2]):
                    continue

                occupancy_mask[y1:y2, x1:x2] = True

            canvas.alpha_composite(scaled_img, dest=(x1, y1))
            total_placed += 1

    canvas.save(output_path)
    return total_placed, total_attempted


# -------------------------------------------------------------------------
# 8. Tkinter GUI Application
# -------------------------------------------------------------------------

class ImageTracerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Procedural Image Path & Cluster Studio")
        self.geometry("600x840")
        self.resizable(False, False)

        # File & Canvas
        self.var_folder = tk.StringVar(value=".")
        self.var_mask = tk.StringVar(value="")
        self.var_recursive = tk.BooleanVar(value=False)
        self.var_output = tk.StringVar(value="traced_result.png")
        self.var_canvas = tk.IntVar(value=2400)
        self.var_density = tk.IntVar(value=300)
        self.var_base_size = tk.IntVar(value=140)
        self.var_padding = tk.IntVar(value=10)

        # Trajectory & Rules
        self.var_curve = tk.StringVar(value="archimedean_spiral")
        self.var_render_mode = tk.StringVar(value="images")
        self.var_norm_mode = tk.StringVar(value="equal_fixed_square_crop")
        self.var_strategy = tk.StringVar(value="unique_spread")
        self.var_rep_style = tk.StringVar(value="golden_ratio")
        self.var_scale_var = tk.StringVar(value="constant")
        self.var_overlap = tk.BooleanVar(value=True)

        # Clustering
        self.var_cluster_crit = tk.StringVar(value="none")
        self.var_cluster_dist = tk.StringVar(value="grid")

        # Text Overlay
        self.var_text_opt = tk.StringVar(value="none")
        self.var_text_ratio = tk.IntVar(value=15)

        self._build_widgets()

    def _build_widgets(self):
        pad_opts = {"padx": 10, "pady": 3}

        # 1. Source Inputs
        frame_files = ttk.LabelFrame(self, text="Source Inputs")
        frame_files.pack(fill="x", **pad_opts)

        ttk.Label(frame_files, text="Images Folder:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(frame_files, textvariable=self.var_folder, width=28).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(frame_files, text="Browse...", command=self._browse_folder).grid(row=0, column=2, padx=4, pady=2)
        ttk.Checkbutton(frame_files, text="Recursive", variable=self.var_recursive).grid(row=0, column=3, padx=5, pady=2, sticky="w")

        ttk.Label(frame_files, text="B&W Mask File (Optional):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(frame_files, textvariable=self.var_mask, width=28).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(frame_files, text="Browse...", command=self._browse_mask).grid(row=1, column=2, padx=4, pady=2)

        # 2. Clustering Frame
        frame_cluster = ttk.LabelFrame(self, text="Metadata Clustering")
        frame_cluster.pack(fill="x", **pad_opts)

        c_criteria = ["none", "month", "year", "day_of_week", "exact_day", "hour", "day_moment", "season"]
        ttk.Label(frame_cluster, text="Cluster By:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(frame_cluster, textvariable=self.var_cluster_crit, values=c_criteria, state="readonly", width=20).grid(row=0, column=1, padx=5, pady=2)

        c_dists = ["grid", "circle", "random"]
        ttk.Label(frame_cluster, text="Clusters Layout:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        ttk.Combobox(frame_cluster, textvariable=self.var_cluster_dist, values=c_dists, state="readonly", width=14).grid(row=0, column=3, padx=5, pady=2)

        # 3. Trajectory & Placement
        frame_traj = ttk.LabelFrame(self, text="Trajectory & Sequencing")
        frame_traj.pack(fill="x", **pad_opts)

        curves = [
            "archimedean_spiral", "logarithmic_spiral", "sunflower_fermat",
            "infinity_lemniscate", "hawk_soaring", "rose_curve", "lissajous", "bee_flight"
        ]
        ttk.Label(frame_traj, text="Curve Shape:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(frame_traj, textvariable=self.var_curve, values=curves, state="readonly", width=20).grid(row=0, column=1, padx=5, pady=2)

        strategies = [
            "color_gradient_unique",
            "color_gradient_repeat",
            "unique_spread",
            "unique_sequential",
            "repetition"
        ]
        ttk.Label(frame_traj, text="Placement:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        ttk.Combobox(frame_traj, textvariable=self.var_strategy, values=strategies, state="readonly", width=18).grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(frame_traj, text="Repetition / Shuffle:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(frame_traj, textvariable=self.var_rep_style, values=["golden_ratio", "round_robin", "burst_clusters", "random"], state="readonly", width=20).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(frame_traj, text="Scale Pulse:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        ttk.Combobox(frame_traj, textvariable=self.var_scale_var, values=["constant", "sine_wave", "bell_curve", "linear_taper"], state="readonly", width=18).grid(row=1, column=3, padx=5, pady=2)

        # 4. Normalization & Text Overlay
        frame_style = ttk.LabelFrame(self, text="Image Sizing & Text Labels")
        frame_style.pack(fill="x", **pad_opts)

        norms = [
            "equal_fixed_square_crop", "equal_fixed_square_pad",
            "equal_height", "equal_width", "proportional_max_width", "proportional_max_height"
        ]
        ttk.Label(frame_style, text="Size Normalization:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(frame_style, textvariable=self.var_norm_mode, values=norms, state="readonly", width=20).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(frame_style, text="Render Mode:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        ttk.Combobox(frame_style, textvariable=self.var_render_mode, values=["images", "color_rectangles"], state="readonly", width=18).grid(row=0, column=3, padx=5, pady=2)

        txt_modes = ["none", "date_mm_dd_yy", "day_of_week", "month", "year", "crazy_random_words"]
        ttk.Label(frame_style, text="Text Label:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(frame_style, textvariable=self.var_text_opt, values=txt_modes, state="readonly", width=20).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(frame_style, text="Text Size (% img):").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        ttk.Spinbox(frame_style, from_=5, to=50, increment=5, textvariable=self.var_text_ratio, width=12).grid(row=1, column=3, padx=5, pady=2)

        # 5. Measurements & Flags
        frame_dims = ttk.LabelFrame(self, text="Measurements & Layout")
        frame_dims.pack(fill="x", **pad_opts)

        ttk.Label(frame_dims, text="Points Density:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(frame_dims, from_=5, to=4000, increment=50, textvariable=self.var_density, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(frame_dims, text="Base Size (px):").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        ttk.Spinbox(frame_dims, from_=10, to=1000, increment=10, textvariable=self.var_base_size, width=10).grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(frame_dims, text="Canvas Dim (px):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(frame_dims, from_=500, to=8000, increment=200, textvariable=self.var_canvas, width=10).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(frame_dims, text="Clearance (px):").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        ttk.Spinbox(frame_dims, from_=0, to=100, increment=2, textvariable=self.var_padding, width=10).grid(row=1, column=3, padx=5, pady=2)

        # 6. Overlap & Output
        frame_out = ttk.Frame(self)
        frame_out.pack(fill="x", padx=15, pady=4)
        ttk.Checkbutton(frame_out, text="Allow Overlap / Superposition", variable=self.var_overlap).pack(anchor="w")

        frame_fname = ttk.Frame(self)
        frame_fname.pack(fill="x", padx=15, pady=4)
        ttk.Label(frame_fname, text="Output Filename:").pack(side="left")
        ttk.Entry(frame_fname, textvariable=self.var_output, width=30).pack(side="left", padx=10)

        # 7. Action Button
        self.lbl_status = ttk.Label(self, text="Ready", foreground="gray")
        self.lbl_status.pack(pady=3)

        btn_run = tk.Button(
            self,
            text="Generate Image",
            bg="#007AFF",
            fg="white",
            font=("System", 13, "bold"),
            relief="flat",
            padx=10,
            pady=8,
            command=self._generate
        )
        btn_run.pack(fill="x", padx=20, pady=6)

    def _browse_folder(self):
        p = filedialog.askdirectory(title="Select Folder with Images")
        if p:
            self.var_folder.set(p)

    def _browse_mask(self):
        f = filedialog.askopenfilename(
            title="Select Black & White Mask",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif")]
        )
        if f:
            self.var_mask.set(f)

    def _generate(self):
        self.lbl_status.config(text="Processing and rendering...", foreground="blue")
        self.update_idletasks()

        folder = self.var_folder.get().strip() or "."
        mask = self.var_mask.get().strip()
        out_file = self.var_output.get().strip() or "output.png"
        c_dim = self.var_canvas.get()

        try:
            placed, attempted = trace_images_on_path(
                folder_path=folder,
                output_path=out_file,
                render_mode=self.var_render_mode.get(),
                clustering_criterion=self.var_cluster_crit.get(),
                cluster_distribution=self.var_cluster_dist.get(),
                allow_overlap=self.var_overlap.get(),
                recursive_search=self.var_recursive.get(),
                placement_strategy=self.var_strategy.get(),
                min_padding=self.var_padding.get(),
                canvas_size=(c_dim, c_dim),
                density=self.var_density.get(),
                curve_source=self.var_curve.get(),
                mask_file=mask if mask else None,
                base_dim_size=self.var_base_size.get(),
                norm_mode=self.var_norm_mode.get(),
                repetition_style=self.var_rep_style.get(),
                scale_variation=self.var_scale_var.get(),
                scale_range=(0.5, 1.3),
                text_option=self.var_text_opt.get(),
                text_ratio=self.var_text_ratio.get() / 100.0,
                bg_color=(245, 247, 250, 0)
            )

            self.lbl_status.config(text=f"Done! Placed {placed}/{attempted} elements.", foreground="green")

            # Launch Preview on macOS
            if sys.platform == "darwin":
                subprocess.run(["open", out_file])
            elif sys.platform == "win32":
                os.startfile(out_file)

        except Exception as e:
            self.lbl_status.config(text="Error occurred", foreground="red")
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    app = ImageTracerGUI()
    app.mainloop()