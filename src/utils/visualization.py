from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import art3d
import matplotlib
from environment.world import World
from environment.drone import Drone

matplotlib.use("TkAgg")


class Visualizer:
    def __init__(self):
        plt.style.use("dark_background")

        # Create main figure with larger size
        self.fig = plt.figure(figsize=(24, 18))

        # Create main 3D axis
        self.ax = self.fig.add_subplot(
            111, projection="3d", position=[0.2, 0.1, 0.6, 0.8]
        )

        # Create legend on left and status on right
        self.legend_ax = self.fig.add_axes([0.02, 0.2, 0.15, 0.6])  # Left side
        self.status_ax = self.fig.add_axes([0.82, 0.7, 0.17, 0.25])  # Upper right

        self.legend_ax.axis("off")
        self.status_ax.axis("off")

        self.trail_points = {}
        self.fig.patch.set_facecolor("#1C1C1C")
        self.ax.set_facecolor("#1C1C1C")

        # Define colors
        self.drone_colors = {
            "active": "#00FF00",  # Bright green
            "failed": "#FF0000",  # Bright red
            "charging": "#FFFF00",  # Yellow
        }

        self.building_colors = ["#1E90FF", "#4682B4", "#4169E1"]

    def _plot_drone_body(self, drone):
        """Enhanced drone visualization"""
        color = self.drone_colors.get(drone.state.status, "#FFFFFF")
        position = drone.state.position
        velocity = drone.state.velocity

        # Large central sphere for drone body
        self.ax.scatter(
            *position,
            c=color,
            s=300,  # Increased size
            marker="o",
            edgecolor="white",
            linewidth=1,
            alpha=1.0,
        )  # Full opacity

        # Add propeller visualization
        radius = 1.0
        theta = np.linspace(0, 2 * np.pi, 20)
        for offset in [
            (radius, 0, 0),
            (-radius, 0, 0),
            (0, radius, 0),
            (0, -radius, 0),
        ]:
            x = position[0] + offset[0] + 0.5 * np.cos(theta)
            y = position[1] + offset[1] + 0.5 * np.sin(theta)
            z = position[2] + offset[2] + np.zeros_like(theta)
            self.ax.plot(x, y, z, color=color, alpha=0.5)

        # Direction indicator with increased visibility
        if np.any(velocity):
            norm_vel = velocity / np.linalg.norm(velocity)
            self.ax.quiver(
                *position,
                *norm_vel,
                length=3.0,  # Increased length
                color=color,
                arrow_length_ratio=0.3,
                linewidth=2,
            )  # Thicker arrow

    def _plot_trail(self, drone_id):
        """Enhanced trail visualization"""
        trail = self.trail_points[drone_id]
        if len(trail) > 1:
            points = np.array([p[0] for p in trail])
            statuses = [p[1] for p in trail]
            colors = [self.drone_colors[s] for s in statuses]

            # Plot trail segments with enhanced visibility
            for i in range(len(points) - 1):
                alpha = 0.3 + 0.7 * (i + 1) / len(points)  # Brighter trails
                self.ax.plot(
                    points[i : i + 2, 0],
                    points[i : i + 2, 1],
                    points[i : i + 2, 2],
                    color=colors[i],
                    alpha=alpha,
                    linewidth=3,
                )  # Thicker lines

    def _plot_drones(self, drones):
        """Plot drones with trails"""
        for drone in drones:
            # Initialize trail if not exists
            if drone.state.id not in self.trail_points:
                self.trail_points[drone.state.id] = []

            # Update trail points
            self.trail_points[drone.state.id].append(
                (drone.state.position.copy(), drone.state.status)
            )

            # Limit trail length but keep enough points for visibility
            if len(self.trail_points[drone.state.id]) > 50:  # Longer trails
                self.trail_points[drone.state.id].pop(0)

            # Plot trails first (so they appear behind drones)
            self._plot_trail(drone.state.id)

            # Plot drone body
            self._plot_drone_body(drone)

    def _update_legend(self):
        """Updated legend with clearer symbols"""
        self.legend_ax.clear()
        self.legend_ax.axis("off")

        # Title with background
        title_bbox = dict(
            facecolor="#333333", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.5"
        )

        self.legend_ax.text(
            0.5,
            0.95,
            "LEGEND",
            fontsize=16,
            fontweight="bold",
            ha="center",
            color="white",
            bbox=title_bbox,
        )

        # Legend items with actual colored samples
        items = [
            ("Target Point", "★", "gold"),
            ("Active Drone", "●", self.drone_colors["active"]),
            ("Failed Drone", "●", self.drone_colors["failed"]),
            ("Building", "■", self.building_colors[0]),
            ("Coverage Area", "∙", "cyan"),
            ("Drone Path", "➔", "white"),
        ]

        for i, (label, symbol, color) in enumerate(items):
            y = 0.8 - i * 0.12  # More spacing between items

            # Add colored box
            rect = FancyBboxPatch(
                (0.1, y - 0.02),
                0.8,
                0.08,
                facecolor=color,
                alpha=0.3,
                boxstyle="round,pad=0.02",
            )
            self.legend_ax.add_patch(rect)

            # Add symbol and label
            self.legend_ax.text(
                0.2,
                y,
                f"{symbol} {label}",
                fontsize=12,
                color="white",
                fontweight="bold",
            )

    def plot_world(self, world: World, show: bool = True):
        """Main plotting function with all components"""
        self.ax.clear()

        # Enhanced world boundaries
        world_size = max(world.dimensions)
        self.ax.set_xlim([-world_size * 0.1, world_size * 1.2])
        self.ax.set_ylim([-world_size * 0.1, world_size * 1.2])
        self.ax.set_zlim([0, world_size])

        # Clear labels and grid
        self.ax.set_xlabel("X (meters)", fontsize=12, labelpad=20)
        self.ax.set_ylabel("Y (meters)", fontsize=12, labelpad=20)
        self.ax.set_zlabel("Z (meters)", fontsize=12, labelpad=20)

        # Enhanced grid
        self.ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        self.ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        self.ax.yaxis.set_major_locator(plt.MultipleLocator(10))
        self.ax.zaxis.set_major_locator(plt.MultipleLocator(10))

        # Plot all components
        self._plot_ground(world.dimensions)
        self._plot_buildings(world.obstacles)
        self._plot_coverage(world.coverage_grid)
        self._plot_targets(world.target_points)
        self._plot_drones(world.drones)  # Plot drones last so they're most visible

        # Update information panels
        self._update_status_display(world)
        self._update_legend()

        # Optimize view angle
        self.ax.view_init(elev=35, azim=45)

        if show:
            plt.pause(0.05)

    # [Previous methods remain the same]

    def _plot_ground(self, dimensions):
        """Plot ground with grid texture"""
        x = np.linspace(0, dimensions[0], 50)
        y = np.linspace(0, dimensions[1], 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # Plot base ground
        self.ax.plot_surface(
            X, Y, Z, alpha=0.3, color="#2F4F4F", linewidth=1, antialiased=True
        )

    def _plot_buildings(self, buildings):
        """Plot all buildings"""
        for building in buildings:
            x, y, z = building.position
            w, l, h = building.size
            self._create_building(x, y, z, w, l, h)

    def _plot_targets(self, targets):
        """Plot target points with beacons"""
        if targets:
            points = np.array(targets)
            # Plot target stars
            self.ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c="gold",
                s=200,
                marker="*",
                edgecolor="white",
                linewidth=1,
                alpha=1.0,
                label="Target",
            )

            # Add light beams
            for point in points:
                z_line = np.linspace(0, point[2], 20)
                x_line = np.ones_like(z_line) * point[0]
                y_line = np.ones_like(z_line) * point[1]
                self.ax.plot(
                    x_line,
                    y_line,
                    z_line,
                    color="gold",
                    alpha=0.2,
                    linestyle="--",
                    linewidth=2,
                )

    def _plot_coverage(self, coverage_grid):
        """Plot coverage area"""
        x, y, z = np.where(coverage_grid > 0)
        if len(x) > 0:
            # Create scattered points for coverage
            self.ax.scatter(x, y, z, c="cyan", alpha=0.2, s=10, label="Coverage")

            # Add coverage volume visualization
            if len(x) > 3:
                try:
                    hull_points = np.vstack((x, y, z)).T
                    self.ax.plot_trisurf(x, y, z, alpha=0.05, color="cyan", shade=True)
                except:
                    pass

    def _update_status_display(self, world):
        """Update status panel"""
        self.status_ax.clear()
        self.status_ax.axis("off")

        # Title
        title_bbox = dict(
            facecolor="#333333", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.5"
        )
        self.status_ax.text(
            0.5,
            0.95,
            "SIMULATION STATUS",
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="top",
            bbox=title_bbox,
            color="white",
        )

        # Metrics
        metrics = [
            ("Coverage", f"{world.get_coverage_percentage():.1%}", "#4CAF50"),
            (
                "Active Drones",
                str(sum(1 for d in world.drones if d.state.status == "active")),
                "#2196F3",
            ),
            ("Total Drones", str(len(world.drones)), "#9C27B0"),
            ("Buildings", str(len(world.obstacles)), "#FF9800"),
            ("Targets", str(len(world.target_points)), "#FFC107"),
        ]

        for i, (label, value, color) in enumerate(metrics):
            y = 0.75 - i * 0.15
            rect = FancyBboxPatch(
                (0.1, y),
                0.8,
                0.12,
                facecolor=color,
                alpha=0.2,
                boxstyle="round,pad=0.02",
            )
            self.status_ax.add_patch(rect)
            self.status_ax.text(0.15, y + 0.06, f"{label}:", fontsize=12, color="white")
            self.status_ax.text(
                0.65, y + 0.06, value, fontsize=12, color=color, fontweight="bold"
            )

    def _create_building(self, x, y, z, w, l, h):
        """Create a modern-looking building"""
        # Create vertices
        vertices = np.array(
            [
                [x, y, z],
                [x + w, y, z],
                [x + w, y + l, z],
                [x, y + l, z],  # Bottom face
                [x, y, z + h],
                [x + w, y, z + h],
                [x + w, y + l, z + h],
                [x, y + l, z + h],  # Top face
            ]
        )

        # Define faces
        faces = [
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],  # Side faces
            [0, 1, 2, 3],
            [4, 5, 6, 7],  # Bottom and top faces
        ]

        # Create glass-like modern building effect
        colors = ["#1E90FF", "#4682B4", "#4169E1", "#4169E1", "#1E90FF", "#4682B4"]
        alphas = [0.6, 0.6, 0.6, 0.6, 0.4, 0.8]  # Different transparencies for faces

        # Add each face with modern building effect
        for i, face in enumerate(faces):
            collection = Poly3DCollection([vertices[face]])
            collection.set_alpha(alphas[i])
            collection.set_facecolor(colors[i])
            collection.set_edgecolor("white")
            collection.set_linewidth(0.5)
            self.ax.add_collection3d(collection)

        # Add window effects for more realism
        window_spacing = 2
        window_size = 0.3

        # Add windows on each face
        for level in range(int(h / 3)):
            z_pos = z + level * 3 + 1.5
            # Front face windows
            for wx in np.arange(x + 2, x + w - 1, window_spacing):
                self.ax.scatter(wx, y + 0.1, z_pos, color="yellow", alpha=0.5, s=50)
            # Right face windows
            for wy in np.arange(y + 2, y + l - 1, window_spacing):
                self.ax.scatter(x + w - 0.1, wy, z_pos, color="yellow", alpha=0.5, s=50)
            # Left face windows
            for wy in np.arange(y + 2, y + l - 1, window_spacing):
                self.ax.scatter(x + 0.1, wy, z_pos, color="yellow", alpha=0.5, s=50)

        # Add roof detail
        roof_height = 1.0
        roof_vertices = np.array(
            [
                [x + w / 4, y + l / 4, z + h],
                [x + 3 * w / 4, y + l / 4, z + h],
                [x + 3 * w / 4, y + 3 * l / 4, z + h],
                [x + w / 4, y + 3 * l / 4, z + h],
                [x + w / 2, y + l / 2, z + h + roof_height],  # Peak point
            ]
        )

        roof_faces = [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],  # Side triangles
            [0, 1, 2, 3],  # Base
        ]

        # Add roof structure
        roof = Poly3DCollection([roof_vertices[f] for f in roof_faces])
        roof.set_facecolor("#4682B4")
        roof.set_alpha(0.7)
        roof.set_edgecolor("white")
        self.ax.add_collection3d(roof)
