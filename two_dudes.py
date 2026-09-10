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


# Load mesh from GLB
mesh = o3d.io.read_triangle_mesh("./parks are beautiful and i am being honest.glb")
mesh.compute_vertex_normals()

# Poisson-disk sampling (choose number of points you want)
pcd = mesh.sample_points_poisson_disk(number_of_points=4000)

# Convert to NumPy
import numpy as np
points = np.asarray(pcd.points)

print(points.shape)   # (50000, 3)

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:,0], points[:,1], points[:,2], s=1, color='whitesmoke')


# Rotate the view:
# ax.view_init(elev=47, azim=0, roll=90) # elevation=30°, azimuth=45°

zoom_factor = 0.6  # smaller = zoom in, bigger = zoom out

x_mid, y_mid, z_mid = points.mean(axis=0)
x_range = (points[:,0].max() - points[:,0].min()) * zoom_factor / 2
y_range = (points[:,1].max() - points[:,1].min()) * zoom_factor / 2
z_range = (points[:,2].max() - points[:,2].min()) * zoom_factor / 2
max_range = max(x_range, y_range, z_range)

ax.set_xlim(x_mid - max_range, x_mid + max_range)
ax.set_ylim(y_mid - max_range, y_mid + max_range)
ax.set_zlim(z_mid - max_range, z_mid + max_range)
# --- Remove axes ---
ax.set_axis_off()
ax.set_aspect('equal')
ax.dist = 3  # default ~10, smaller = zoom in
# --- Black background ---
fig.patch.set_facecolor('black')   # figure background
ax.set_facecolor('black')          # plot background

plt.subplots_adjust(top=0.75, bottom=0.05)

plt.show()


from matplotlib.animation import FuncAnimation, PillowWriter

# Animation function
def update(frame):
    ax.view_init(elev=frame, azim=0, roll=90)
    return fig,

ani = FuncAnimation(fig, update, frames=np.arange(1, 359, 2), interval=50)
ani.save("rotating_pointcloud.gif", writer=PillowWriter(fps=20))