import numpy as np
import pyvista as pv
from PIL import Image

class Club:

    def __init__(self, name, launch_angle, ball_speed, colour):

        self.name = name
        self.launch_angle = launch_angle
        self.ball_speed = ball_speed
        self.colour = colour

class Hole:

    def __init__(self, heightmap_path, surfacemap_path):

        self.heightmap = self.load_heightmap(heightmap_path)

        self.surfacemap = self.load_surfacemap(surfacemap_path)

        self.rows, self.cols = self.heightmap.shape

        self.create_mesh()

        #Hole 3 - Towers
        self.tee_position = np.array([316,570, 37])
        self.hole_position = np.array([384, 117, 21])

        #Flat
        #self.tee_position = np.array([320,570, 37])
        #self.hole_position = np.array([320, 111, 21])

        #Typical carry distance for an average male amateur golfer
        self.clubs = [
            Club("Driver", 12.6, 133, "red"),
            Club("3 Wood", 11.5, 125, "orangered"),
            Club("5 Wood", 13.0, 120, "orange"),
            Club("3 Hybrid", 14.0, 116, "lime"),
            Club("4 Hybrid", 15.0, 112, "limegreen"),
            Club("4 Iron", 13.0, 108, "dodgerblue"),
            Club("5 Iron", 14.0, 104, "blue"),
            Club("6 Iron", 15.5, 100, "mediumblue"),
            Club("7 Iron", 17.0, 96, "royalblue"),
            Club("8 Iron", 19.0, 92, "navy"),
            Club("9 Iron", 21.0, 88, "skyblue"),
            Club("Pitching Wedge", 24.0, 83, "magenta"),
            Club("Gap Wedge", 27.0, 79, "violet"),
            Club("Sand Wedge", 30.0, 74, "orchid"),
            Club("Lob Wedge", 33.0, 69, "purple"),
            Club("Putter", 3.0, 15, "white")
        ]

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
            (102,205,102): "fairway",
            (34,139,34): "rough",
            (50,205,50): "green",
            (237,201,175): "bunker",
            (254,255,255): "out"
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
    
    def simulate_shot(self, start_position, power, direction, club_index):

        gravity = 10.73
        dt = 0.05

        club = self.clubs[int(club_index)]

        speed = club.ball_speed * 0.48889 * (power / 100.0)

        launch_angle = np.radians(club.launch_angle)

        direction_angle = np.radians(direction)

        x = start_position[0]
        y = start_position[1]
        z = self.get_height(x, y)

        trajectory = []

        vx = speed * np.cos(launch_angle) * np.cos(direction_angle)
        vy = speed * np.cos(launch_angle) * np.sin(direction_angle)
        vz = speed * np.sin(launch_angle)

        landed = False

        while not landed:

            drag = 0.95 ** dt

            vx *= drag
            vy *= drag
            vz *= drag

            x += vx * dt
            y += vy * dt
            z += vz * dt

            vz -= gravity * dt

            trajectory.append([x, y, z])

            if (x < 0 or x >= self.cols or y < 0 or y >= self.rows):
                break

            terrain = self.get_height(x, y)

            if vz <= -0 and z <= terrain:
                z = terrain
                surface = self.get_surface(x, y)
                trajectory.append([x, y, z])
                landed = True
        
        horizontal_speed = np.sqrt(vx**2 + vy**2)

        h = self.get_height(x, y)
        hx1 = self.get_height(max(x - 1, 0), y)
        hx2 = self.get_height(min(x + 1, self.cols - 1), y)
        hy1 = self.get_height(x, max(y - 1, 0))
        hy2 = self.get_height(x, min(y + 1, self.rows - 1))

        slope_x = (hx2 - hx1) / 2.0
        slope_y = (hy2 - hy1) / 2.0

        normal = np.array([-slope_x, -slope_y, 1.0])
        normal /= np.linalg.norm(normal)


        velocity = np.array([vx, vy, vz])
        impact_speed = np.linalg.norm(velocity)

        if impact_speed > 0:
            
            velocity /= impact_speed

            impact_cos = np.clip(np.dot(-velocity, normal), 0.0, 1.0)

            tangent_factor = np.sqrt(1.0 - impact_cos)

            roll_speed = horizontal_speed * 0.15 * tangent_factor
        else:
            roll_speed = 0.0

        if horizontal_speed > 0:

            while roll_speed > 0.1:

                h = self.get_height(x, y)
                hx1 = self.get_height(max(x - 1, 0), y)
                hx2 = self.get_height(min(x + 1, self.cols - 1), y)
                hy1 = self.get_height(x, max(y - 1, 0))
                hy2 = self.get_height(x, min(y + 1, self.rows - 1))

                slope_x = (hx2 - hx1) / 2
                slope_y = (hy2 - hy1) / 2

                x += (vx / horizontal_speed) * roll_speed * dt
                y += (vy / horizontal_speed) * roll_speed * dt

                x -= slope_x * 0.5
                y -= slope_y * 0.5

                surface = self.get_surface(x, y)

                if surface == "out":
                    return np.array([x, y, z]), trajectory, False

                if (x < 0 or x >= self.cols or y < 0 or y >= self.rows):
                    break
            
                z = self.get_height(x, y)

                trajectory.append([x, y, z])

                roll_speed *= 0.95

        surface = self.get_surface(x, y)

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
        
        _, trajectory, _ = self.simulate_shot(start_position, power, direction, club_index) 
        
        plotter = pv.Plotter() 

        rgb = np.flipud(np.rot90(self.surfacemap))
        rgb = rgb.reshape(-1, 3)

        self.grid["SurfaceColours"] = rgb
        
        plotter.add_mesh(self.grid, scalars="SurfaceColours", rgb=True)

        club = self.clubs[club_index]
        
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
            club_index = np.clip(club_index, 0, len(self.clubs) - 1)

            position, trajectory, _ = self.simulate_shot(position, power, direction, club_index)

            club = self.clubs[club_index]

            path = pv.lines_from_points(np.array(trajectory))

            plotter.add_mesh(path, color=club.colour, line_width=5, label=club.name)
        
        plotter.add_legend()
        plotter.show()