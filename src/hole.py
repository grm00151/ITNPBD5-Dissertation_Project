import numpy as np
import pyvista as pv
from PIL import Image

class Club:

    def __init__(self, name, launch_angle, ball_speed, spin_rate, colour):

        self.name = name
        self.launch_angle = launch_angle
        self.ball_speed = ball_speed
        self.spin_rate = spin_rate
        self.colour = colour

class Player:

    def __init__(self, handicap):

        self.handicap = handicap
        self.clubs = [
            Club("Driver", 12.6, 133, 3275, "red"),
            Club("3 Wood", 11.5, 125, 3700, "orangered"),
            Club("5 Wood", 13.0, 120, 4200, "orange"),
            Club("4 Hybrid", 15.0, 112, 5000, "limegreen"),
            Club("5 Iron", 14.0, 104, 5300, "blue"),
            Club("6 Iron", 15.5, 100, 5800, "mediumblue"),
            Club("7 Iron", 17.0, 96, 6400, "royalblue"),
            Club("8 Iron", 19.0, 92, 7100, "navy"),
            Club("9 Iron", 21.0, 88, 8000, "skyblue"),
            Club("Pitching Wedge", 24.0, 83, 9000, "magenta"),
            Club("Gap Wedge", 27.0, 79, 10000, "violet"),
            Club("Sand Wedge", 30.0, 74, 10800, "orchid"),
            Club("Lob Wedge", 33.0, 69, 11500, "purple"),
            Club("Putter", 3.0, 15, 0, "white")
        ]

class Hole:

    def __init__(self, player, heightmap_path, surfacemap_path):

        self.player = player

        self.heightmap = self.load_heightmap(heightmap_path)
        self.surfacemap = self.load_surfacemap(surfacemap_path)

        self.rows, self.cols = self.heightmap.shape

        self.create_mesh()

        # Hole 1 - Flat
        #self.tee_position = np.array([320,570, 37])
        #self.hole_position = np.array([320, 111, 21])

        # Hole 2 - Towers
        self.tee_position = np.array([316,570, 37])
        self.hole_position = np.array([384, 117, 21])

    def load_heightmap(self, path):

        img = Image.open(path).convert("L")

        height = np.array(img).astype(np.float32)

        height /= 255.0
        height *= 65.0

        return height
    
    def load_surfacemap(self, path):
        
        img = Image.open(path).convert("RGB")

        surface = np.array(img)
        
        return surface
    
    def get_surface(self, x, y):

        x = int(np.clip(x, 0, self.cols - 1))
        y = int(np.clip(y, 0, self.rows - 1))

        colour = tuple(self.surfacemap[y, x])

        surfaces = {
            (173,209,158): "fairway",
            (149,184,136): "rough",
            (205,235,176): "green",
            (245,234,198): "bunker",
            (113,149,101): "out"
        }

        return surfaces.get(colour, "unknown")

    def create_mesh(self):

        x = np.arange(self.cols, dtype=np.float32)
        y = np.arange(self.rows, dtype=np.float32)

        xx, yy = np.meshgrid(x, y)

        self.grid = pv.StructuredGrid(
            xx,
            yy,
            self.heightmap
        )

    def get_height(self, x, y):
        
        x = round(np.clip(x, 0, self.cols - 1))
        y = round(np.clip(y, 0, self.rows - 1))
        
        return self.heightmap[y, x]
    
    def get_allowed_club(self, position, club_index):
        
        surface = self.get_surface(position[0], position[1])

        if surface == "bunker":
            return next(i for i, club in enumerate(self.player.clubs) if club.name == "Sand Wedge")
        
        if surface == "green":
            return next(i for i, club in enumerate(self.player.clubs) if club.name == "Putter")
        
        return club_index
    
    def simulate_flight(self, start_position, club, power, direction):

        gravity = 10.73
        dt = 0.05

        speed = club.ball_speed * 0.48889 * (power / 100.0)

        launch_angle = np.radians(club.launch_angle)
        direction_angle = np.radians(direction)

        x = start_position[0]
        y = start_position[1]
        z = self.get_height(x, y)

        vx = speed * np.cos(launch_angle) * np.cos(direction_angle)
        vy = speed * np.cos(launch_angle) * np.sin(direction_angle)
        vz = speed * np.sin(launch_angle)

        trajectory = []

        while True:

            drag = 0.95 ** dt

            vx *= drag
            vy *= drag
            vz *= drag

            x += vx * dt
            y += vy * dt
            z += vz * dt

            vz -= gravity * dt

            trajectory.append([x, y, z])

            if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
                break

            terrain = self.get_height(x, y)

            if vz <= 0 and z <= terrain:
                z = terrain
                trajectory.append([x, y, z])
                break

        return x, y, z, vx, vy, vz, trajectory
    
    def get_surface_geometry(self, x, y):

        hx1 = self.get_height(max(x - 1, 0), y)
        hx2 = self.get_height(min(x + 1, self.cols - 1), y)

        hy1 = self.get_height(x, max(y - 1, 0))
        hy2 = self.get_height(x, min(y + 1, self.rows - 1))

        slope_x = (hx2 - hx1) / 2.0
        slope_y = (hy2 - hy1) / 2.0

        normal = np.array([-slope_x, -slope_y, 1.0])
        normal /= np.linalg.norm(normal)

        return normal, slope_x, slope_y
    
    def calculate_roll_speed(self, x, y, vx, vy, vz):

        horizontal_speed = np.hypot(vx, vy)

        if horizontal_speed == 0:
            return 0.0

        normal, _, _ = self.get_surface_geometry(x, y)

        velocity = np.array([vx, vy, vz])
        impact_speed = np.linalg.norm(velocity)

        if impact_speed == 0:
            return 0.0

        velocity /= impact_speed

        impact_cos = np.clip(np.dot(-velocity, normal), 0.0, 1.0)

        tangent_factor = np.sqrt(1.0 - impact_cos)

        return horizontal_speed * 0.15 * tangent_factor
    
    def simulate_roll(self, x, y, z, vx, vy, roll_speed, trajectory):

        dt = 0.05

        horizontal_speed = np.hypot(vx, vy)

        if horizontal_speed == 0:
            return True, x, y, z

        while roll_speed > 0.1:

            _, slope_x, slope_y = self.get_surface_geometry(x, y)

            x += (vx / horizontal_speed) * roll_speed * dt
            y += (vy / horizontal_speed) * roll_speed * dt

            x -= slope_x * 0.5
            y -= slope_y * 0.5

            if (x < 0 or x >= self.cols or y < 0 or y >= self.rows):
                break

            surface = self.get_surface(x, y)

            if surface == "out":
                return False, x, y, z

            z = self.get_height(x, y)

            trajectory.append([x, y, z])

            roll_speed *= 0.95

        return True, x, y, z
    
    def simulate_shot(self, start_position, power, direction, club_index):
        
        club_index = self.get_allowed_club(start_position, club_index)
        club = self.player.clubs[int(club_index)]

        x, y, z, vx, vy, vz, trajectory = self.simulate_flight(start_position, club, power, direction)

        roll_speed = self.calculate_roll_speed(x, y, vx, vy, vz)

        out_of_bounds, x, y, z = self.simulate_roll(x, y, z, vx, vy, roll_speed, trajectory)

        if not out_of_bounds:
            return np.array([x, y, z]), trajectory, False

        return np.array([x, y, z]), trajectory, True

    def show(self):

        plotter = pv.Plotter()

        plotter.add_mesh(self.grid, cmap="terrain")

        def pick_point(point):
            print("\nSelected point:")
            print("X =", int(point[0]))
            print("Y =", int(point[1]))
            print("Z =", point[2])

        plotter.enable_point_picking(callback=pick_point, show_message=True)

        plotter.show()

    def show_shot(self, start_position, power, direction, club_index):
        
        position, trajectory, _ = self.simulate_shot(start_position, power, direction, club_index) 
        
        print(self.get_surface(position[0], position[1]))
        
        plotter = pv.Plotter() 

        rgb = np.flipud(np.rot90(self.surfacemap))
        rgb = rgb.reshape(-1, 3)

        self.grid["SurfaceColours"] = rgb
        
        plotter.add_mesh(self.grid, scalars="SurfaceColours", rgb=True)

        club = self.player.clubs[club_index]
        
        path = pv.lines_from_points(np.array(trajectory)) 
        
        plotter.add_mesh(path, color=club.colour, line_width=5, label=club.name) 

        plotter.add_legend()
        plotter.show()

    def show_strategy(self, variables):

        position = np.copy(self.tee_position)

        plotter = pv.Plotter()

        rgb = np.flipud(np.rot90(self.surfacemap))
        rgb = rgb.reshape(-1, 3)

        self.grid["SurfaceColours"] = rgb

        plotter.add_mesh(self.grid, scalars="SurfaceColours", rgb=True)

        for i in range(0, len(variables), 3):

            power = variables[i]
            direction = variables[i + 1]

            club_index = int(round(variables[i + 2]))
            club_index = np.clip(club_index, 0, len(self.player.clubs) - 1)

            position, trajectory, _ = self.simulate_shot(position, power, direction, club_index)

            club = self.player.clubs[club_index]

            path = pv.lines_from_points(np.array(trajectory))

            plotter.add_mesh(path, color=club.colour, line_width=5, label=club.name)
        
        plotter.add_legend()
        plotter.show()