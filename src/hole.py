import numpy as np
import pyvista as pv
import random
from PIL import Image

class Club:

    def __init__(self, name, launch_angle, ball_speed, spin_rate, colour, max_axis):

        self.name = name
        self.launch_angle = launch_angle
        self.ball_speed = ball_speed
        self.spin_rate = spin_rate
        self.colour = colour
        self.max_axis = max_axis

class Player:

    def __init__(self, handicap):

        self.handicap = handicap
        self.clubs = [
            Club("Driver", 12.0, 175.0, 2500, "red", 90),
            Club("3 Wood", 11.0, 166.5, 3200, "orange", 80),
            Club("5 Wood", 12.5, 153.0, 3800, "chocolate", 80),
            Club("4 Hybrid", 14.0, 141.0, 4500, "limegreen", 80),
            Club("5 Iron", 15.0, 129.0, 5200, "blue", 67),
            Club("6 Iron", 16.5, 121.0, 5800, "mediumblue", 67),
            Club("7 Iron", 18.0, 114.0, 6500, "royalblue", 67),
            Club("8 Iron", 20.0, 104.5, 7300, "navy", 67),
            Club("9 Iron", 22.5, 96.0, 8300, "skyblue", 67),
            Club("Pitching Wedge", 25.5, 86.5, 9400, "magenta", 50),
            Club("Gap Wedge", 28.5, 79.5, 10300, "violet", 42),
            Club("Sand Wedge", 31.5, 72.0, 11100, "orchid", 35),
            Club("Lob Wedge", 35.0, 66.0, 11900, "purple", 30),
            Club("Putter", 3.0, 12.0, 0, "grey", 0)
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
        self.tee_position = np.array([316.0, 570.0, 59.39])
        self.hole_position = np.array([384.0, 117.0, 46.65])

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

        self.grid = pv.StructuredGrid(xx, yy, self.heightmap)

        rgb = np.flipud(np.rot90(self.surfacemap))
        rgb = rgb.reshape(-1, 3)

        self.grid["SurfaceColours"] = rgb

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

    def get_friction(self, surface):
        friction = {
            "fairway": 0.95,
            "rough": 0.90,
            "green": 0.97
        }

        return friction.get(surface, 0.88)

    def get_handicap_norm(self):
        return 1 - (self.player.handicap + 54) / 62

    def calculate_spin_axis(self, club):
        std_dev = (club.max_axis * self.get_handicap_norm()) / 3

        while True:
            spin_axis = random.gauss(0, std_dev)

            if -club.max_axis <= spin_axis <= club.max_axis:
                return spin_axis

    def calculate_shot_speed(self, club, power, power_scale=100.0):
        max_speed = club.ball_speed
        min_speed = max_speed * 0.2
        speed_range = max_speed - min_speed

        speed = (min_speed + (speed_range - random.random() * speed_range * self.get_handicap_norm()))

        return speed * 0.48889 * (power / power_scale)

    def calculate_launch_angle(self, club):
        max_launch = club.launch_angle
        min_launch = max_launch * 0.2
        launch_range = max_launch - min_launch

        launch_angle = (min_launch + (launch_range - random.random() * launch_range * self.get_handicap_norm()))

        return np.radians(launch_angle)

    def calculate_initial_velocity(self, speed, launch_angle, direction):
        direction_angle = np.radians(direction)

        vx = speed * np.cos(launch_angle) * np.cos(direction_angle)
        vy = speed * np.cos(launch_angle) * np.sin(direction_angle)
        vz = speed * np.sin(launch_angle)

        return vx, vy, vz
    
    def simulate_flight(self, start_position, club, power, direction):
        gravity = 10.73
        dt = 0.01

        spin_axis = self.calculate_spin_axis(club)
        speed = self.calculate_shot_speed(club, power)
        launch_angle = self.calculate_launch_angle(club)

        vx, vy, vz = self.calculate_initial_velocity(speed, launch_angle, direction)

        x = start_position[0]
        y = start_position[1]
        z = self.get_height(x, y)

        trajectory = []

        while True:
            drag = 0.95 ** dt

            vx *= drag
            vy *= drag
            vz *= drag

            speed_mag = np.sqrt(vx**2 + vy**2 + vz**2)

            backspin_factor = club.spin_rate / 10000.0

            curve = np.sin(np.radians(spin_axis))
            side_accel = curve * backspin_factor * speed_mag

            horizontal_speed = np.hypot(vx, vy)

            if horizontal_speed > 0:
                old_vx = vx
                old_vy = vy

                vx += (-old_vy / horizontal_speed) * side_accel * dt
                vy += (old_vx / horizontal_speed) * side_accel * dt

            lift_accel = 0.03 * backspin_factor * speed_mag

            vz += (lift_accel - gravity) * dt

            x += vx * dt
            y += vy * dt
            z += vz * dt

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
    
    def simulate_roll(self, x, y, z, vx, vy, vz, roll_speed, trajectory):

        gravity = 10.73
        dt = 0.05

        horizontal_speed = np.hypot(vx, vy)

        if horizontal_speed == 0:
            if surface == "out":
                return True, x, y, z
            return False, x, y, z

        while roll_speed > 0.1:

            _, slope_x, slope_y = self.get_surface_geometry(x, y)

            x += (vx / horizontal_speed) * roll_speed * dt
            y += (vy / horizontal_speed) * roll_speed * dt

            x -= slope_x * dt
            y -= slope_y * dt

            if (x < 0 or x >= self.cols or y < 0 or y >= self.rows):
                break

            surface = self.get_surface(x, y)

            vz -= gravity * dt
            z += vz * dt

            terrain = self.get_height(x, y)

            if z <= terrain:
                z = terrain
                vz = 0.0

            trajectory.append([x, y, z])

            if surface == "out":
                return True, x, y, z

            friction = self.get_friction(surface)

            roll_speed *= friction

        return False, x, y, z
    
    def simulate_putt(self, start_position, club, power, direction):
        
        gravity = 10.73
        dt = 0.05

        direction_angle = np.radians(direction)

        x = start_position[0]
        y = start_position[1]
        z = self.get_height(x, y)

        speed = self.calculate_shot_speed(club, power, power_scale=20.0)

        vx = speed * np.cos(direction_angle)
        vy = speed * np.sin(direction_angle)
        vz = 0.0

        trajectory = [[x, y, z]]

        while speed > 0.05:
            
            _, slope_x, slope_y = self.get_surface_geometry(x, y)

            vx -= slope_x * dt
            vy -= slope_y * dt

            x += vx * dt
            y += vy * dt

            if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
                break

            surface = self.get_surface(x, y)

            speed = np.hypot(vx, vy)

            vz -= gravity * dt
            z += vz * dt

            terrain = self.get_height(x, y)

            if z <= terrain:
                z = terrain
                vz = 0.0

            trajectory.append([x, y, z])

            if surface == "out":
                return np.array([x, y, z]), trajectory, True

            friction = self.get_friction(surface)

            vx *= friction
            vy *= friction

        return np.array([x, y, z]), trajectory, False
    
    def simulate_shot(self, start_position, power, direction, club_index):

        club = self.player.clubs[int(club_index)]

        if club.name == "Putter":

            position, trajectory, out_of_bounds = self.simulate_putt(start_position, club, power, direction)

            return position, trajectory, out_of_bounds, club_index

        x, y, z, vx, vy, vz, trajectory = self.simulate_flight(start_position, club, power, direction)

        roll_speed = self.calculate_roll_speed(x, y, vx, vy, vz)

        out_of_bounds, x, y, z = self.simulate_roll(x, y, z, vx, vy, vz, roll_speed, trajectory)

        if out_of_bounds:
            return np.array([x, y, z]), trajectory, True, club_index

        return np.array([x, y, z]), trajectory, False, club_index

    def show(self):

        plotter = pv.Plotter()

        plotter.add_mesh(self.grid, scalars="SurfaceColours", rgb=True)

        def pick_point(point):
            print("\nSelected point:")
            print("X =", int(point[0]))
            print("Y =", int(point[1]))
            print("Z =", point[2])

        plotter.enable_point_picking(callback=pick_point, show_message=True)

        plotter.show()

    def show_shot(self, start_position, power, direction, club_index):
        
        position, trajectory, _, _ = self.simulate_shot(start_position, power, direction, club_index) 
        
        print(self.get_surface(position[0], position[1]))
        
        plotter = pv.Plotter() 
        
        plotter.add_mesh(self.grid, scalars="SurfaceColours", rgb=True)

        club = self.player.clubs[club_index]
        
        path = pv.lines_from_points(np.array(trajectory)) 
        
        plotter.add_mesh(path, color=club.colour, line_width=5, label=club.name)

        plotter.enable_fly_to_right_click()
        plotter.add_legend()
        plotter.show()

    def show_strategy(self, solution, screenshot=None):

        plotter = pv.Plotter(off_screen=screenshot is not None)

        plotter.add_mesh(self.grid, scalars="SurfaceColours", rgb=True)

        legend_clubs = set()

        for trajectory, club_index in solution.trajectories:

            club = self.player.clubs[club_index]

            path = pv.lines_from_points(np.array(trajectory))

            if club.name not in legend_clubs:
                plotter.add_mesh(path, color=club.colour, line_width=5, label=club.name)
                legend_clubs.add(club.name)
            else:
                plotter.add_mesh(path, color=club.colour, line_width=5)
        
        plotter.enable_fly_to_right_click()
        plotter.add_legend()
        if screenshot is not None:
            plotter.camera_position = [(1200, 320, 550), (320, 320, 53), (0, 0, 1)]
            plotter.render()
            plotter.screenshot(screenshot, scale=4)
        else:
            plotter.show()
            
        plotter.close()