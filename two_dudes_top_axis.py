import trimesh
import numpy as np
import matplotlib.pyplot as plt

# # Load the mesh from GLB
# scene  = trimesh.load("./parks are beautiful and i am being honest.glb")

# # If it's a Scene, convert to a single Trimesh
# if isinstance(scene, trimesh.Scene):
#     mesh = trimesh.util.concatenate(
#         [g for g in scene.geometry.values()]
#     )
# else:
#     mesh = scene

# print(mesh)

# points, face_indices = trimesh.sample.sample_surface(mesh, 50000)
# print(points.shape)

# plt.scatter(points[:,0], points[:,1], points[:,2])

# plt.show()

import open3d as o3d

from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

def filter_points(points, axis='y', op='>', threshold=0.0):
    """
    Filter 3D points based on a condition applied to one axis.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 3) with 3D points.
    axis : str
        Which axis to filter on: 'x', 'y', or 'z'.
    op : str
        Comparison operator: one of '>', '>=', '<', '<=', '==', '!='.
    threshold : float
        Threshold value for comparison.

    Returns
    -------
    np.ndarray
        Filtered points.
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis not in axis_map:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
    idx = axis_map[axis]
    values = points[:, idx]

    if op == '>':
        mask = values > threshold
    elif op == '>=':
        mask = values >= threshold
    elif op == '<':
        mask = values < threshold
    elif op == '<=':
        mask = values <= threshold
    elif op == '==':
        mask = values == threshold
    elif op == '!=':
        mask = values != threshold
    else:
        raise ValueError("op must be one of '>', '>=', '<', '<=', '==', '!='")
    
    return points[mask]

# plt.rcParams['animation.ffmpeg_path'] = 'D:/Downloads/ffmpeg-2025-09-10-git-c1dc2e2b7c-full_build/bin'

color = 'g'
animate = False


# Load mesh from GLB
mesh = o3d.io.read_triangle_mesh("./parks are beautiful and i am being honest.glb")
mesh.compute_vertex_normals()

# Poisson-disk sampling (choose number of points you want)
pcd = mesh.sample_points_poisson_disk(number_of_points=4000)

# Convert to NumPy
import numpy as np
points = np.asarray(pcd.points)

print(points.shape)   # (50000, 3)

points = filter_points(points, threshold=0.06)


# === Parameters ===
zoom_factor = 0.8   # <1 = zoom in, >1 = zoom out
elev_angle = 30     # elevation of camera
frames = np.arange(1, 359, 2)  # rotation frames
fps = 20

# === Figure with two subplots: top = angle, bottom = 3D ===
fig = plt.figure(figsize=(6,7), constrained_layout=True)

gs = fig.add_gridspec(100, 1)  # 100 rows vertically

# Top subplot (angle indicator) = very short (5/100)
ax_angle = fig.add_subplot(gs[0:5, 0])
# Bottom subplot (3D points) = most of figure (95/100)
ax = fig.add_subplot(gs[5:, 0], projection='3d')

# --- 3D scatter ---
scatter = ax.scatter(points[:,0], points[:,1], points[:,2], s=3, c=color, marker='*')

# Black background
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.set_axis_off()

# === Equal aspect + zoom ===
x_mid, y_mid, z_mid = points.mean(axis=0)
x_range = (points[:,0].max() - points[:,0].min()) * zoom_factor / 2
y_range = (points[:,1].max() - points[:,1].min()) * zoom_factor / 2
z_range = (points[:,2].max() - points[:,2].min()) * zoom_factor / 2
max_range = max(x_range, y_range, z_range)

ax.set_xlim(x_mid - max_range, x_mid + max_range)
# ax.set_ylim(y_mid - max_range, y_mid + max_range)
ax.set_ylim(0, y_mid + max_range)
ax.set_zlim(z_mid - max_range, z_mid + max_range)
# ax.set_zlim(0, z_mid + max_range)
ax.text(0.2,1,0,'tano', color=color)
ax.text(-0.6,1.2,0.5,'marcelo', color=color)
ax.set_aspect('equal')
ax.view_init(elev=47, azim=0, roll=90) # elevation=30°, azimuth=45°

# === Angle subplot (horizontal line indicator) ===
ax_angle.set_xlim(0, 360)
ax_angle.set_ylim(0, 1)
ax_angle.get_yaxis().set_visible(False)
angle_line, = ax_angle.plot([0,0], [0,0.2], color=color, lw=3)
# ax_angle.set_xlabel("Azimuth angle (deg)", color='white')
ax_angle.set_facecolor('black')
ax_angle.tick_params(colors=color)
plt.tight_layout()


# plt.subplots_adjust(top=0.75, bottom=0.05)


plt.show()



import umap

# UMAP: 3D → 2D

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=75,   # balances local + global structure
    min_dist=0.2,     # keeps humans spread out, not collapsed
    metric='euclidean',
    random_state=42
)
embedding = reducer.fit_transform(points)

# Plot 2D embedding
plt.figure(figsize=(6,6))
plt.scatter(embedding[:,0], embedding[:,1], s=10, c='steelblue', alpha=0.7)
plt.title("UMAP projection (3D → 2D)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.axis('equal')
plt.show()


def animate_3d_to_2d(points3d, points2d, n_frames=100, interval=50, save_path=None):
    # Ensure numpy arrays
    points3d = np.array(points3d)
    points2d = np.array(points2d)

    # Lift 2D into 3D space
    points2d_3d = np.hstack([points2d, np.zeros((points2d.shape[0], 1))])

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    scat = ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2], c=points3d[:,2], cmap="viridis", s=5)

    # Fix axis limits for coherence
    all_pts = np.vstack([points3d, points2d_3d])
    ax.set_xlim(all_pts[:,0].min(), all_pts[:,0].max())
    ax.set_ylim(all_pts[:,1].min(), all_pts[:,1].max())
    ax.set_zlim(all_pts[:,2].min(), all_pts[:,2].max())

    def update(frame):
        t = frame / (n_frames-1)
        interp = (1-t)*points3d + t*points2d_3d
        scat._offsets3d = (interp[:,0], interp[:,1], interp[:,2])
        scat.set_array(interp[:,2])  # color by z
        return scat,

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    if save_path:
        anim.save(save_path, writer="ffmpeg", fps=30)
    else:
        plt.show()
        
animate_3d_to_2d(points, embedding, n_frames=80, save_path="transition.mp4")

if animate:
    # === Animation function ===
    def update(frame):
        # Rotate 3D view 
        ax.view_init(elev=frame, azim=0, roll=90)
        # Update angle line
        angle_line.set_data([frame, frame], [0,1])
        return scatter, angle_line

    # === Animate and save GIF ===
    ani = FuncAnimation(fig, update, frames=frames, interval=50)
    ani.save(f"rotating_pointcloud_{color}.gif", writer=PillowWriter(fps=fps))


    # Create an FFmpeg writer
    writer = FFMpegWriter(fps=20, codec='h264', metadata=dict(artist='Me'))


    writer = FFMpegWriter(
        fps=20,                 # higher FPS for smooth animation
        codec='libx264',         # H.264 codec (widely compatible)
        bitrate=8000,            # kilobits per second, higher = better quality
        extra_args=['-pix_fmt', 'yuv420p']  # ensures broad compatibility
    )

    # Save to MP4
    ani.save(f"rotating_pointcloud_{color}.mp4", writer=writer)