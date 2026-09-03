import numpy as np
import pyvista as pv
import random
from PIL import Image

class Club:
    # store the physical and visual properties used to model a golf club. 

    def __init__(self, name, launch_angle, ball_speed, spin_rate, colour, max_axis):
        # initalise the club parameters used by the shot simulation.

        self.name = name # name of the club.
        self.launch_angle = launch_angle # launch angle in degrees.
        self.ball_speed = ball_speed # maximum ball speed.
        self.spin_rate = spin_rate # backspin rate.
        self.colour = colour # colour used when displaying the club's trajectory. 
        self.max_axis = max_axis # maximum possible spin axis in degrees.

class Player:
    # represent a player and the club available to them. 

    def __init__(self, handicap):
        # initialise the player with a handicap and standard club set. 

        # store the player's handicap so it can affect shot accuracy and perofrmance.
        self.handicap = handicap

        # list of clubs available to the player
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
    # load a golf hole and provide terrain, physics and visualisation methods.

    def __init__(self, player, heightmap_path, surfacemap_path):
        # initialise the hole from its terrain and surface map images. 

        self.player = player # player object whose clubs and handicap affect simulation/

        self.heightmap = self.load_heightmap(heightmap_path) # path to the greyscale terrain heightmap.
        self.surfacemap = self.load_surfacemap(surfacemap_path) # path to the RGB surface classification map.

        self.rows, self.cols = self.heightmap.shape 

        self.create_mesh() # build the 3D terrain mesh used by PyVista for visualisation.

        self.tee_position = np.array([316.0, 570.0, 59.39])
        self.hole_position = np.array([384.0, 117.0, 46.65])

    def load_heightmap(self, path):
        # convert the greyscale heightmap into scaled floating point elevations.

        # open the image, convert it to greyscale
        img = Image.open(path).convert("L")

        # conver the image into a numpy float point array for calculations
        height = np.array(img).astype(np.float32)

        # convert 0-255 pixel brightness into the elevation ranged used by the hole.
        # normalise pixel values to [0, 1], then scale to course elevation.
        height /= 255.0
        height *= 65.0

        return height
    
    def load_surfacemap(self, path):
        # load the RBG image used to identify fairway, rough, green and hazards.
        
        # keep all three RGB colour channels because each colour represents a surface.
        img = Image.open(path).convert("RGB")

        # convert the image into a NumPy array.
        surface = np.array(img)
        
        return surface
    
    def get_surface(self, x, y):
        # return the terrain type at the supplied map coordinates.

        # make sure x and y are valid pixel positions within the map.
        x = int(np.clip(x, 0, self.cols - 1))
        y = int(np.clip(y, 0, self.rows - 1))

        # read the RGB colour stored at this map location.
        colour = tuple(self.surfacemap[y, x])

        # match known RGB cp;purs to the golf course surfaces.
        # RGB values correspond to the colour coded surface map.
        surfaces = {
            (173,209,158): "fairway",
            (149,184,136): "rough",
            (205,235,176): "green",
            (245,234,198): "bunker",
            (113,149,101): "out"
        }

        # return the matching surface or unknown if the colour is not recognied
        return surfaces.get(colour, "unknown")

    def create_mesh(self):
        # create a PyVista structured grid and attach the surface colours.

        # create x and y coordinates for every column and row of the heightmap.
        x = np.arange(self.cols, dtype=np.float32)
        y = np.arange(self.rows, dtype=np.float32)

        # create 2d coordinate grids so every heightmap has an x and y position.
        xx, yy = np.meshgrid(x, y)

        # the heightmap supplies the z elevation value for every x, y position.
        # use the heightmap as the Z coordinate of the structured terrain grid. 
        self.grid = pv.StructuredGrid(xx, yy, self.heightmap)

        # rotate and flip the surface image so its colours line up with the grid.
        rgb = np.flipud(np.rot90(self.surfacemap))
        rgb = rgb.reshape(-1, 3)

        # attach the RGB values to the grid so PyVista can display the correct surface colours.
        self.grid["SurfaceColours"] = rgb

    def get_height(self, x, y):
        # return the terrain elevation at the supplied map coordinate. 

        # make sure x and y are valid pixel positions within the map.
        x = round(np.clip(x, 0, self.cols - 1))
        y = round(np.clip(y, 0, self.rows - 1))

        # return the height stored at the selected row and column. 
        return self.heightmap[y, x]
    
    def get_allowed_club(self, position, club_index):
        # select a surface specific club where course rules require one.
        
        # find out whether the ball is on the fairway, rough, green etc
        surface = self.get_surface(position[0], position[1])

        # force the player to use a club based on surface
        if surface == "bunker":
            return next(i for i, club in enumerate(self.player.clubs) if club.name == "Sand Wedge")
        
        if surface == "green":
            return next(i for i, club in enumerate(self.player.clubs) if club.name == "Putter")

        # otherwise allow the club originally selected by the player
        return club_index

    def get_friction(self, surface):
        # return the multiplier for a given terrain surface. 

        # higher valuess mean less speed is lost on each rolling timestep.
        friction = {
            "fairway": 0.95,
            "rough": 0.90,
            "green": 0.97
        }

        # use a defualt friction value for surfaces not listed
        return friction.get(surface, 0.88)

    def get_handicap_norm(self):
        # convert the player's handicap to the simulations normalised skill factor. 
        # handicpa 8 = 0 and -54 = 1
        return 1 - (self.player.handicap + 54) / 62

    def calculate_spin_axis(self, club):
        # generate a random spin axis error based on player skill.

        # better player has a smaller standard deviation
        std_dev = (club.max_axis * self.get_handicap_norm()) / 3

        # generate unti the result stays within the clubs allowed range.
        while True:
            # a gaussian distribution models typical shot variation around zero. 
            spin_axis = random.gauss(0, std_dev)

            # reject values outside the clubs maximum possible range 
            if -club.max_axis <= spin_axis <= club.max_axis:
                return spin_axis

    def calculate_shot_speed(self, club, power, power_scale=100.0):
        # calculate launch speed from club performance, player skill and power. 
        max_speed = club.ball_speed # clubs listed max speed.
        min_speed = max_speed * 0.2 # allow the minimum shot speed to be 20% of max speed.
        speed_range = max_speed - min_speed # work out the amount of speed between min and max.

        # random vary the speed accoring to players skill level
        speed = (min_speed + (speed_range - random.random() * speed_range * self.get_handicap_norm()))

        # convert from mph to yard/s 
        # scale it according to the selected shot power. 
        return speed * 0.48889 * (power / power_scale)

    def calculate_launch_angle(self, club):
        # generate a lauch angle influeneced by the player's skill level. 
        max_launch = club.launch_angle # clubs listed max launch angle.
        min_launch = max_launch * 0.2 # allow the minimim launch angle to be 20% of max launch.
        launch_range = max_launch - min_launch # calculate the range between the min and max launch angles.

        # randomly vary launch angle accoring to player's skill. 
        launch_angle = (min_launch + (launch_range - random.random() * launch_range * self.get_handicap_norm()))

        # convert degress to radians so it works with np.
        return np.radians(launch_angle)

    def calculate_initial_velocity(self, speed, launch_angle, direction):
        # resolve launch speed into horizontal and vertical velocity.
        direction_angle = np.radians(direction)

        # calculate the velocity along x and y-axis as well as upward.
        vx = speed * np.cos(launch_angle) * np.cos(direction_angle)
        vy = speed * np.cos(launch_angle) * np.sin(direction_angle)
        vz = speed * np.sin(launch_angle)

        return vx, vy, vz
    
    def simulate_flight(self, start_position, club, power, direction):
        # simulate the airborne phase of a shot until it lands or leaves the map.
        gravity = 10.73 # gravitational acceleration in yard/s.
        dt = 0.01 # integration timestep. 

        spin_axis = self.calculate_spin_axis(club) # add random spin-axis to determine sideways curvature.
        speed = self.calculate_shot_speed(club, power) # calculate the intial launch speed. 
        launch_angle = self.calculate_launch_angle(club) # calculate the inital launch angle. 

        # convert speed, launch angle and direction into x, y and z velocities.
        vx, vy, vz = self.calculate_initial_velocity(speed, launch_angle, direction)

        # start the ball at the supplied x, y.
        x = start_position[0]
        y = start_position[1]
        # stat at the terrain height beneath the ball.
        z = self.get_height(x, y)

        # store each simulated postion so the full flight path can be drawn later.
        trajectory = []

        # continue calculating flight until a stop condition is met.
        while True:
            # apply aerodynamic drag to graducally reduce all velocity components.
            drag = 0.95 ** dt

            vx *= drag
            vy *= drag
            vz *= drag

            # calculate the ball's total 3D speed.
            speed_mag = np.sqrt(vx**2 + vy**2 + vz**2)

            # scale club's spin rate to a manageable acceleration factor.
            backspin_factor = club.spin_rate / 10000.0

            # convert spin axis into a sideways curve value. 
            curve = np.sin(np.radians(spin_axis))
            side_accel = curve * backspin_factor * speed_mag

            # calculates speed across the horizontal z-y plane only.
            horizontal_speed = np.hypot(vx, vy)

            # only apply sideways curve to horizontal movement.
            if horizontal_speed > 0:
                old_vx = vx
                old_vy = vy

                # rotate part of the veolcity sideways to create curveatue in x and y.
                vx += (-old_vy / horizontal_speed) * side_accel * dt
                vy += (old_vx / horizontal_speed) * side_accel * dt

            # backspin produces a small upward lift.
            lift_accel = 0.03 * backspin_factor * speed_mag

            # update the vertical velocity using lift minus the gravity.
            vz += (lift_accel - gravity) * dt

            # move the ball along x, y-axis and vertically for one timestep. 
            x += vx * dt
            y += vy * dt
            z += vz * dt

            # svae this new position so the trajectory can be tracked and displayed. 
            trajectory.append([x, y, z])

            # stop the simulation if the trajectory leaves the course map. 
            if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
                break

            # find the terrain elevation at the ball's current x,y positon.
            terrain = self.get_height(x, y)

            # a landing occurs when the ball is descending and has reached the terrain.
            if vz <= 0 and z <= terrain:
                z = terrain
                trajectory.append([x, y, z])
                break

        # return the final position, velocity and trajectory.
        return x, y, z, vx, vy, vz, trajectory

    # AI was used to help create this method.
    def get_surface_geometry(self, x, y):
        # estimate local terrain slope and surface normal from neighbouring heights. 

        # height one map unit to the left and right of current point.
        hx1 = self.get_height(max(x - 1, 0), y)
        hx2 = self.get_height(min(x + 1, self.cols - 1), y)

        # height one map unit below and above of current point.
        hy1 = self.get_height(x, max(y - 1, 0))
        hy2 = self.get_height(x, min(y + 1, self.rows - 1))

        # the terrain slope in the x and y direction.
        slope_x = (hx2 - hx1) / 2.0 # 2 map units
        slope_y = (hy2 - hy1) / 2.0

        # construct a vector perpendicular to the sloping surface.
        normal = np.array([-slope_x, -slope_y, 1.0])
        # normalise the vector
        normal /= np.linalg.norm(normal)

        return normal, slope_x, slope_y

    # AI was used to help create this method.
    def calculate_roll_speed(self, x, y, vx, vy, vz):
        # estimate the inital ground roll speed after the ball lands.

        # find the speed of the ball across the ground.
        horizontal_speed = np.hypot(vx, vy)

        # no horizontal movement means the balls not rolling.
        if horizontal_speed == 0:
            return 0.0

        # calulates the direction of the terrain surface at impact.
        normal, _, _ = self.get_surface_geometry(x, y)

        # store the full incoming velocity as a 3D vector. 
        velocity = np.array([vx, vy, vz])

        # calculate how fast the ball is moving at the monment of impact. 
        impact_speed = np.linalg.norm(velocity)

        # a zero impact speed gives no roll direction.
        if impact_speed == 0:
            return 0.0

        # convert velocity into a unit direction vector.
        velocity /= impact_speed

        # measure how directly the ball is approaching the surface normal. 
        impact_cos = np.clip(np.dot(-velocity, normal), 0.0, 1.0)

        # convert the impact angle into a factor affecting roll amount. 
        tangent_factor = np.sqrt(1.0 - impact_cos)

        # return an estimated inital ground roll speed.
        return horizontal_speed * 0.15 * tangent_factor

    def simulate_roll(self, x, y, z, vx, vy, vz, roll_speed, trajectory):
        # simulate post impact rolling until the ball stops or goes out of bounds.

        gravity = 10.73 # gravity is used while the ball follows the terrain.
        dt = 0.05 # rolling uses a larger timestep due to bottleneck reasons.

        # calculate the horizonal speed of the velocity.
        horizontal_speed = np.hypot(vx, vy)

        # identify the surface the ball's on.
        surface = self.get_surface(x, y)

        # if there is no horizontal sped the ball doesnt roll.
        if horizontal_speed == 0:
            if surface == "out":
                return True, x, y, z
            return False, x, y, z

        # continue while the ball has enough speed to roll.
        while roll_speed > 0.1:

            # calculate the terrain slope as the ball moves.
            _, slope_x, slope_y = self.get_surface_geometry(x, y)

            # move in the current direction using the roll speed.
            x += (vx / horizontal_speed) * roll_speed * dt
            y += (vy / horizontal_speed) * roll_speed * dt

            # let the terrain slope push the ball downholl in x and y.
            x -= slope_x * dt
            y -= slope_y * dt

            # stop if rolling takes the ball outside the map.
            if (x < 0 or x >= self.cols or y < 0 or y >= self.rows):
                break

            # update the surface after moving.
            surface = self.get_surface(x, y)

            # apply gravity to the vertical rolling velocity.
            vz -= gravity * dt
            # update the balls vertical position.
            z += vz * dt

            # find the terrain height below the ball.
            terrain = self.get_height(x, y)

            # if the ball has gone below the terrain, place it back on the surface.
            if z <= terrain:
                z = terrain
                vz = 0.0

            # save the current roll speed.
            trajectory.append([x, y, z])

            # stop if the ball rolls out of bounds.
            if surface == "out":
                return True, x, y, z

            # reduce rolling speed accoridng to the currecnt terrain surface. 
            friction = self.get_friction(surface)

            # apply friction so the ball gradually slows down. 
            roll_speed *= friction

        return False, x, y, z

    def simulate_putt(self, start_position, club, power, direction):
        # simulate putt while accounting for slope and green friction.

        gravity = 10.73 
        dt = 0.05 # rolling uses a larger timestep due to bottleneck reasons.

        # convert the putting direction from degrees to radians.
        direction_angle = np.radians(direction)

        # start at supplied x, y and z coordinate.
        x = start_position[0]
        y = start_position[1]
        z = self.get_height(x, y)

        # putting uses a smaller power scale than full golf shots. 
        speed = self.calculate_shot_speed(club, power, power_scale=20.0)

        # resolve putting speed into the x and y direction.
        vx = speed * np.cos(direction_angle)
        vy = speed * np.sin(direction_angle)
        vz = 0.0

        # store starting point 
        trajectory = [[x, y, z]]

        # continue while the ball is moving faster than threshhold.
        while speed > 0.05:

            # calculate the current local slope of the green.
            _, slope_x, slope_y = self.get_surface_geometry(x, y)

            # adjust horizontal velocity according to the local green slope.
            vx -= slope_x * dt
            vy -= slope_y * dt

            # move the ball using its current velocity 
            x += vx * dt
            y += vy * dt

            # stop the simulation if the trajectory leaves the course map. 
            if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
                break

            # check if the ball has changed surface.
            surface = self.get_surface(x, y)

            # recalculate the current speed. 
            speed = np.hypot(vx, vy)

            # apply gravity to vertical and update postion.
            vz -= gravity * dt
            z += vz * dt

            # find the terrain elevation below the ball.
            terrain = self.get_height(x, y)

            # keep the ball on top of the terrain. 
            if z <= terrain:
                z = terrain
                vz = 0.0

            # store the point for displaying putting trajectory.
            trajectory.append([x, y, z])

            # a putt that reached out of bound ends right away.
            if surface == "out":
                return np.array([x, y, z]), trajectory, True

            # get the friction multiplyer for the current surface
            friction = self.get_friction(surface)

            # apply the friction to the x and y velocity 
            vx *= friction
            vy *= friction

        return np.array([x, y, z]), trajectory, False


    def simulate_shot(self, start_position, power, direction, club_index):
        # run the appropriate simulation for a selected club.

        # convert selected club into an int and retriecve that club.
        club = self.player.clubs[int(club_index)]

        # putts use a ground based where other clubs use flight + roll.
        if club.name == "Putter":

            # run putting simulation.
            position, trajectory, out_of_bounds = self.simulate_putt(start_position, club, power, direction)

            return position, trajectory, out_of_bounds, club_index

        # simulate the airborne part of a golf shot.
        x, y, z, vx, vy, vz, trajectory = self.simulate_flight(start_position, club, power, direction)

        # estimate how fast the ball start to roll after landing.
        roll_speed = self.calculate_roll_speed(x, y, vx, vy, vz)

        # simulate the post impact ground roll
        out_of_bounds, x, y, z = self.simulate_roll(x, y, z, vx, vy, vz, roll_speed, trajectory)

        # if roll leaves the course, report the shot as out of bounds. 
        if out_of_bounds:
            return np.array([x, y, z]), trajectory, True, club_index

        # reutrn final resting position and trajectory 
        return np.array([x, y, z]), trajectory, False, club_index

    def show(self):

        # display the course mesh and allow the user to select terrain points.
        plotter = pv.Plotter()

        # draw the course using the stored RGB surface colours 
        plotter.add_mesh(self.grid, scalars="SurfaceColours", rgb=True)

        # runs whenever a user picks a point on the course
        def pick_point(point):
            # Display x,y and z coordinates
            print("\nSelected point:")
            print("X =", int(point[0]))
            print("Y =", int(point[1]))
            print("Z =", point[2])

        # allow mouse point picking and connect it to callback.
        plotter.enable_point_picking(callback=pick_point, show_message=True)

        # open the intective 3D course window 
        plotter.show()

    def show_shot(self, start_position, power, direction, club_index):
        # display a single simulated shot over the 3D course. 

        # run the shot simulations and keep its final positions and trajectory
        position, trajectory, _, _ = self.simulate_shot(start_position, power, direction, club_index) 

        # print the surface where the shot finsihes 
        print(self.get_surface(position[0], position[1]))

        # create a new plotting window
        plotter = pv.Plotter() 

        # draw the hole terrain and surface colours
        plotter.add_mesh(self.grid, scalars="SurfaceColours", rgb=True)

        # retrive the selected club so it name and colour can be used.
        club = self.player.clubs[club_index]

        # convert the trajectory points into a pyvista line.
        path = pv.lines_from_points(np.array(trajectory)) 

        # draw the shot path using the club's assigned colour. 
        plotter.add_mesh(path, color=club.colour, line_width=5, label=club.name)

        # allow the camera to fly to the location using right click.
        plotter.enable_fly_to_right_click()
        # add a legend showing the club used for the shot.
        plotter.add_legend()
        # display the interactive plot.
        plotter.show()

    def show_strategy(self, solution, screenshot=None):
        #Render all trajctories in a strategy solution.

        # use off screen rendering when a screenshot filename has been supplied.
        plotter = pv.Plotter(off_screen=screenshot is not None)

        # draw the course terrain with its surface colours.
        plotter.add_mesh(self.grid, scalars="SurfaceColours", rgb=True)

        # track clubs already represented in legend ti avoid duplicate entries.
        legend_clubs = set()

        # draw each planned trajectory using the colour assigned to it club. 
        for trajectory, club_index in solution.trajectories:

            club = self.player.clubs[club_index]

            # convert the trajectory points into a pyvista line.
            path = pv.lines_from_points(np.array(trajectory))

            # add the club to the legend only the first time it appears.
            if club.name not in legend_clubs:
                # draw the trajectory and label it with the club name.
                plotter.add_mesh(path, color=club.colour, line_width=5, label=club.name)
                legend_clubs.add(club.name)
            else:
                plotter.add_mesh(path, color=club.colour, line_width=5)

        # enable camera movement with right click
        plotter.enable_fly_to_right_click()
        # display the legend for the different clubs 
        plotter.add_legend()

        # render to a file when a screenshot path is supplied: otherwise open UI. 
        if screenshot is not None:
            # set a fixed camera postion so screenshots are consistent 
            plotter.camera_position = [(1200, 320, 550), (320, 320, 53), (0, 0, 1)]
            # render the scene before taking the screenshot
            plotter.render()
            # save the scene to the screenshot file
            plotter.screenshot(screenshot, scale=4)
        else:
            # open the normal interacctive window
            plotter.show()

        # close the plotter
        plotter.close()