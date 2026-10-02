import os
import random
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw

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
        
    elif curve_type == "logarithmic_spiral":  # Moth around a bulb
        theta = t * 5 * np.pi
        r = 0.05 * np.exp(0.45 * theta)
        x, y = r * np.cos(theta), r * np.sin(theta)

    elif curve_type == "hawk_soaring":  # Looping glide drift
        wind_drift = 3.5 * t
        radius = 1.0
        angle = t * 8 * np.pi
        x = wind_drift + radius * np.cos(angle)
        y = radius * np.sin(angle)

    elif curve_type == "sunflower_fermat":  # Golden ratio plant packing
        indices = np.arange(num_points)
        golden_angle = np.pi * (3 - np.sqrt(5))  # ~137.5 degrees
        theta = indices * golden_angle
        r = np.sqrt(indices)
        x, y = r * np.cos(theta), r * np.sin(theta)

    elif curve_type == "infinity_lemniscate":  # Figure-eight
        angle = t * 2 * np.pi
        denom = 1 + np.sin(angle) ** 2
        x = np.cos(angle) / denom
        y = (np.sin(angle) * np.cos(angle)) / denom

    elif curve_type == "spirograph":  # Epicycloid loops
        R, r, d = 5.0, 3.0, 4.0
        theta = t * 6 * np.pi
        x = (R + r) * np.cos(theta) - d * np.cos((R + r) * theta / r)
        y = (R + r) * np.sin(theta) - d * np.sin((R + r) * theta / r)

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
        raise ValueError("No path points detected in the provided mask image.")

    coords = np.column_stack((x_indices, y_indices))
    sort_order = np.lexsort((coords[:, 1], coords[:, 0]))
    sorted_coords = coords[sort_order]

    sample_idx = np.linspace(0, len(sorted_coords) - 1, num_points).astype(int)
    sampled = sorted_coords[sample_idx].astype(float)
    sampled[:, 0] /= arr.shape[1]
    sampled[:, 1] /= arr.shape[0]
    return sampled


# -------------------------------------------------------------------------
# 2. Mock Photo Generator (Fallback when folder has no real photos)
# -------------------------------------------------------------------------

def create_rich_mock_photos(target_dir: Path):
    """Generates distinct procedural landscape-style photos for visual clarity."""
    target_dir.mkdir(parents=True, exist_ok=True)
    themes = [
        {"sky": (135, 206, 235), "ground": (34, 139, 34), "sun": (255, 215, 0), "label": "MEADOW"},
        {"sky": (255, 140, 0), "ground": (75, 0, 130), "sun": (255, 69, 0), "label": "SUNSET"},
        {"sky": (25, 25, 112), "ground": (10, 10, 50), "sun": (240, 248, 255), "label": "NIGHT"},
        {"sky": (240, 128, 128), "ground": (47, 79, 79), "sun": (255, 250, 205), "label": "DUSK"},
        {"sky": (70, 130, 180), "ground": (210, 180, 140), "sun": (255, 255, 224), "label": "BEACH"},
    ]

    for idx, theme in enumerate(themes):
        w, h = 300, 200 + (idx % 3) * 30
        img = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Sky and ground
        draw.rectangle([0, 0, w, int(h * 0.6)], fill=theme["sky"])
        draw.rectangle([0, int(h * 0.6), w, h], fill=theme["ground"])

        # Sun / Moon
        draw.ellipse([w // 2 - 25, int(h * 0.25), w // 2 + 25, int(h * 0.25) + 50], fill=theme["sun"])

        # Mountain peaks
        draw.polygon([(0, int(h * 0.6)), (w // 3, int(h * 0.35)), (w * 2 // 3, int(h * 0.6))], fill=(100, 100, 110))
        draw.polygon([(w // 3, int(h * 0.6)), (w * 3 // 4, int(h * 0.4)), (w, int(h * 0.6))], fill=(70, 70, 80))

        # Thin border to make edges crisp
        draw.rectangle([0, 0, w - 1, h - 1], outline=(30, 30, 30, 200), width=3)

        img.save(target_dir / f"photo_{idx + 1}.png")


# -------------------------------------------------------------------------
# 3. Image Ingestion and Normalization
# -------------------------------------------------------------------------

def load_and_normalize_inputs(
    folder_path: str,
    target_dim_size: int = 200,
    norm_mode: str = "equal_height",
    render_mode: str = "images"  # "images" or "color_rectangles"
) -> List[Image.Image]:
    folder = Path(folder_path)

    if render_mode == "color_rectangles":
        # Procedurally build pure colored rectangles
        colors = [
            (239, 68, 68), (59, 130, 246), (16, 185, 129),
            (245, 158, 11), (139, 92, 246), (236, 72, 153)
        ]
        ratios = [1.2, 0.8, 1.5, 1.0, 1.3, 0.7]
        rects = []
        for i, col in enumerate(colors):
            ratio = ratios[i % len(ratios)]
            if norm_mode in ("equal_height", "proportional_max_height"):
                h = target_dim_size
                w = int(h * ratio)
            else:
                w = target_dim_size
                h = int(w / ratio)
            img = Image.new("RGBA", (max(10, w), max(10, h)), col + (255,))
            rects.append(img)
        return rects

    # Render mode is "images"
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    file_paths = [p for p in folder.iterdir() if p.suffix.lower() in valid_exts] if folder.exists() else []

    # If folder is missing or has no files, generate distinct mock photos
    if not file_paths:
        print(f"No image files found in '{folder_path}'. Generating dummy landscape photos...")
        create_rich_mock_photos(folder)
        file_paths = [p for p in folder.iterdir() if p.suffix.lower() in valid_exts]

    raw_images = [Image.open(p).convert("RGBA") for p in file_paths]
    normalized = []

    if norm_mode == "proportional_max_width":
        max_orig_w = max(img.width for img in raw_images)
        scale = target_dim_size / max_orig_w
        for img in raw_images:
            w = max(4, int(img.width * scale))
            h = max(4, int(img.height * scale))
            normalized.append(img.resize((w, h), Image.Resampling.LANCZOS))

    elif norm_mode == "proportional_max_height":
        max_orig_h = max(img.height for img in raw_images)
        scale = target_dim_size / max_orig_h
        for img in raw_images:
            w = max(4, int(img.width * scale))
            h = max(4, int(img.height * scale))
            normalized.append(img.resize((w, h), Image.Resampling.LANCZOS))

    elif norm_mode == "equal_height":
        for img in raw_images:
            aspect = img.width / img.height
            h = target_dim_size
            w = max(4, int(h * aspect))
            normalized.append(img.resize((w, h), Image.Resampling.LANCZOS))

    elif norm_mode == "equal_width":
        for img in raw_images:
            aspect = img.height / img.width
            w = target_dim_size
            h = max(4, int(w * aspect))
            normalized.append(img.resize((w, h), Image.Resampling.LANCZOS))

    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    return normalized


# -------------------------------------------------------------------------
# 4. Sequencer & Dynamic Sizing
# -------------------------------------------------------------------------

def get_repetition_indices(num_images: int, num_points: int, distribution: str = "golden_ratio") -> List[int]:
    if distribution == "round_robin":
        return [i % num_images for i in range(num_points)]
    elif distribution == "golden_ratio":
        phi = (1 + 5**0.5) / 2
        return [int(((i * phi) % 1.0) * num_images) for i in range(num_points)]
    elif distribution == "burst_clusters":
        indices = []
        cur_img = 0
        counter = 0
        cl_len = random.randint(3, 6)
        for _ in range(num_points):
            indices.append(cur_img)
            counter += 1
            if counter >= cl_len:
                cur_img = (cur_img + 1) % num_images
                counter = 0
                cl_len = random.randint(3, 6)
        return indices
    elif distribution == "random":
        return [random.randint(0, num_images - 1) for _ in range(num_points)]
    return [i % num_images for i in range(num_points)]


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
# 5. Core Tracing Engine
# -------------------------------------------------------------------------

def trace_images_on_path(
    folder_path: str = "my_images",
    output_path: str = "final_output.png",
    render_mode: str = "images",           # TOGGLE: "images" to show photos, "color_rectangles" for flat blocks
    allow_overlap: bool = False,           # TOGGLE: True to allow stacking; False to prevent superposition
    min_padding: int = 10,                 # Separation distance in px between elements when overlap is False
    canvas_size: Tuple[int, int] = (2400, 2400),
    density: int = 350,
    curve_source: Union[str, np.ndarray] = "archimedean_spiral",
    mask_file: str = None,
    base_dim_size: int = 180,
    norm_mode: str = "equal_height",       # 'equal_height', 'equal_width', 'proportional_max_width', 'proportional_max_height'
    repetition_style: str = "golden_ratio",# 'golden_ratio', 'round_robin', 'burst_clusters', 'random'
    scale_variation: str = "sine_wave",    # 'sine_wave', 'bell_curve', 'linear_taper', 'constant'
    scale_range: Tuple[float, float] = (0.5, 1.3),
    bg_color: Tuple[int, int, int, int] = (245, 247, 250, 255)
):
    # 1. Curve Points
    if mask_file is not None:
        points = extract_path_from_image(mask_file, num_points=density)
    elif isinstance(curve_source, str):
        points = generate_curve(curve_source, num_points=density)
    else:
        points = curve_source

    # 2. Source Items
    items = load_and_normalize_inputs(
        folder_path=folder_path,
        target_dim_size=base_dim_size,
        norm_mode=norm_mode,
        render_mode=render_mode
    )
    num_items = len(items)

    # 3. Repetition Map
    sequence = get_repetition_indices(num_items, density, distribution=repetition_style)

    # 4. Canvas Setup & Overlap Buffer
    c_w, c_h = canvas_size
    canvas = Image.new("RGBA", canvas_size, bg_color)
    occupancy_mask = np.zeros((c_h, c_w), dtype=bool)

    margin = int(base_dim_size * max(scale_range) * 0.8)
    drawable_w = c_w - 2 * margin
    drawable_h = c_h - 2 * margin

    placed = 0

    for i, (norm_x, norm_y) in enumerate(points):
        progress = i / max(1, density - 1)
        src_item = items[sequence[i]]

        # Dynamic Scale
        mult = compute_scale_multiplier(progress, scale_variation, scale_range[0], scale_range[1])
        curr_w = max(4, int(src_item.width * mult))
        curr_h = max(4, int(src_item.height * mult))
        scaled_item = src_item.resize((curr_w, curr_h), Image.Resampling.LANCZOS)

        # Coordinate Mapping
        px = int(margin + norm_x * drawable_w)
        py = int(margin + norm_y * drawable_h)

        x1 = px - curr_w // 2
        y1 = py - curr_h // 2
        x2 = x1 + curr_w
        y2 = y1 + curr_h

        # Canvas Bounds
        if x1 < 0 or y1 < 0 or x2 >= c_w or y2 >= c_h:
            continue

        # Superposition Check
        if not allow_overlap:
            cx1 = max(0, x1 - min_padding)
            cy1 = max(0, y1 - min_padding)
            cx2 = min(c_w, x2 + min_padding)
            cy2 = min(c_h, y2 + min_padding)

            if np.any(occupancy_mask[cy1:cx2, cx1:cx2]):
                continue  # Skip image to avoid collision

            occupancy_mask[y1:y2, x1:x2] = True

        # Composite item
        canvas.alpha_composite(scaled_item, dest=(x1, y1))
        placed += 1

    canvas.save(output_path)
    print(f"Complete: Saved '{output_path}' with {placed}/{density} elements placed. (Mode: {render_mode}, Overlap: {allow_overlap})")


# -------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------
if __name__ == "__main__":
    trace_images_on_path(
        folder_path="sample inputs",             # Drop your real .png / .jpg files here
        output_path="spiral_result_2.png",
        render_mode="images",                # TOGGLE: "images" (shows photos) OR "color_rectangles" (shows blocks)
        allow_overlap=True,                 # TOGGLE: True allows superposition; False separates them cleanly
        min_padding=10,                      # Spacing between pictures
        canvas_size=(2400, 2400),
        density=100,
        mask_file='test.png',
        curve_source="sunflower_fermat",   # 'archimedean_spiral', 'lissajous', 'rose_curve', 'bee_flight'
        base_dim_size=100,
        norm_mode="equal_height",
        repetition_style="golden_ratio",  # constant, 'golden_ratio'
        scale_variation="constant",
        scale_range=(0.5, 1.3),
        bg_color=(245, 247, 250, 0)
    )