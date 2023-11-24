# TODO: Idea – what if you have these satisfying "ticks" that play as the wheel rotates? Eh, nah...

import numpy as np
import time
import os
from collections import deque
import pygame
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)
from playsound import playsound

# Pygame + gameloop setup
width = 800
height = 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Lorentz Waterwheel")
pygame.init()

rho, theta, phi = 150., 0, np.pi/2  # Rho is the distance from world origin to near clipping plane
v_rho, v_theta, v_phi = 0, 0, 0
focus = 10000.  # Distance from near clipping plane to eye

# Save the animation? TODO: Make sure you're saving to correct destination!!
save_anim = True

# Animation saving setup
path_to_save = '/Users/adityaabhyankar/Desktop/Programming/LorenzWaterCycle/output'
if save_anim:
    for filename in os.listdir(path_to_save):
        # Check if the file name follows the required format
        b1 = filename.startswith("frame") and filename.endswith(".png")
        b2 = filename.startswith("output.mp4")
        if b1 or b2:
            os.remove(os.path.join(path_to_save, filename))
            print('Deleted frame ' + filename)

# Coordinate Shift Functions
def A(val):
    return np.array([val[0] + width / 2, -val[1] + height / 2])


def A_inv(val):
    global width, height
    return np.array([val[0] - width / 2, -(val[1] - height / 2)])


def A_many(vals):
    return [A(v) for v in vals]   # Galaxy brain function


def lerp(u, u0, u1):
    return ((1. - u) * u0) + (u * u1)


# Normal perspective project
def world_to_plane(v):
    '''
        Converts from point in 3D to its 2D perspective projection, based on location of camera.

        v: vector in R^3 to convert.
        NOTE: Here, we do NOT "A" the final output.
    '''
    # Camera params
    global rho, theta, phi, focus

    # Radial distance to eye from world's origin.
    eye_rho = rho + focus

    # Vector math from geometric computation (worked out on white board, check iCloud for possible picture)
    eye_to_origin = -np.array([eye_rho * np.sin(phi) * np.cos(theta),
                               eye_rho * np.sin(phi) * np.sin(theta), eye_rho * np.cos(phi)])

    eye_to_ei = eye_to_origin + v
    origin_to_P = np.array(
        [rho * np.sin(phi) * np.cos(theta), rho * np.sin(phi) * np.sin(theta), rho * np.cos(phi)])

    # Formula for intersecting t: t = (n•(a-b)) / (n•v)
    tau = np.dot(eye_to_origin, origin_to_P - v) / np.dot(eye_to_origin, eye_to_ei)
    r_t = v + (tau * eye_to_ei)

    # Location of image coords in terms of world coordinates.
    tile_center_world = -origin_to_P + r_t

    # Spherical basis vectors
    theta_hat = np.array([-np.sin(theta), np.cos(theta), 0])
    phi_hat = -np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), -np.sin(phi)])

    # Actual transformed 2D coords
    tile_center = np.array([np.dot(tile_center_world, theta_hat), np.dot(tile_center_world, phi_hat)])

    return tile_center


def world_to_plane_many(pts):
    return np.array([world_to_plane(v) for v in pts])  # galaxy brain function


# Keyhandling
def handle_keys(keys_pressed):
    global rho, theta, phi, focus
    m = 300
    drho, dphi, dtheta, dfocus = 10, np.pi/m, np.pi/m, 10

    if keys_pressed[pygame.K_w]:
        phi -= dphi
    if keys_pressed[pygame.K_a]:
        theta -= dtheta
    if keys_pressed[pygame.K_s]:
        phi += dphi
    if keys_pressed[pygame.K_d]:
        theta += dtheta
    if keys_pressed[pygame.K_p]:
        rho -= drho
    if keys_pressed[pygame.K_o]:
        rho += drho
    if keys_pressed[pygame.K_k]:
        focus -= dfocus
    if keys_pressed[pygame.K_l]:
        focus += dfocus


# Map (x, y) to (0, *A_inv(x, y)), e.g. map (width / 2 + x, height / 2 + y) to (x, y), which maps in turn to (0, x, y),
# i.e. the screen maps to the yz plane.
def make3d(pt, z=0):
    return np.array([z, *A_inv(pt)])


def make_many_3d(pts, z_arr=None):
    assert z_arr is None or len(pts) == len(z_arr)
    if z_arr is None:
        return [make3d(pt) for pt in pts]

    return [make3d(pt, z=z_arr[k]) for k, pt in enumerate(pts)]


# Colors
colors = {
    'white': np.array([255., 255., 255.]),
    'black': np.array([0., 0., 0.]),
    'red': np.array([255, 71, 0]),
    'lightblue': np.array([0, 200, 255]),
    'blue': np.array([0, 30, 255]),
    'indigo': np.array([71, 0, 218]),
    'fullred': np.array([255, 0, 0]),
    'fullblue': np.array([0, 0, 255]),
    'darkblue': np.array([20, 0, 120]),
    'START': np.array([255, 255, 255])
}


# WaterWheel Class
class WaterWheel:

    def __init__(self, x, y, radius, n_cups, theta_0):
        """
        :param x: x coordinate of center of wheel
        :param y: y coordinate of center of wheel
        :param radius: radius of the circum-circle of the polygon
        :param n_cups: number of cups on the wheel; number of corners of wheel
        :param theta_0: initial angle from horizontal
        """
        self.x = x
        self.y = y
        self.radius = radius
        self.theta = theta_0
        self.n_cups = n_cups

        self.omega = 0
        self.cup_masses = np.zeros(n_cups)

        # TEMPORARY
        # self.cup_masses[0] = 5

        self.max_mass_per_cup = 10
        self.g = 0.0000005   # from old TikTok: 0.0000005

        self.point_locs = self.get_point_locs()

    def get_point_locs(self):
        points = []
        for i in range(self.n_cups):
            points.append(self.x + (self.radius * np.cos((2 * np.pi / self.n_cups * i) + self.theta)))
            points.append(self.y - (self.radius * np.sin((2 * np.pi / self.n_cups * i) + self.theta)))

        return points

    def get_point_locs_2(self):
        '''
            Same as above method, but returns n x 2 matrix with [x,y] coordinate pair for the n cups.
            The first method instead returns a flattened out 1 x 2n vector instead.
        '''

        flattened = np.array(self.get_point_locs())
        n = len(flattened)
        return flattened.reshape((int(n/2), 2))

    def get_center_of_mass(self):
        '''Returns tuple (x,y) location of center of mass of water'''

        locs = self.get_point_locs_2()
        return np.dot(locs.T, self.cup_masses) / sum(self.cup_masses)

    def update_step(self):
        # Rotation
        v = 0.04  # Ideal: 0.03
        damping = -v*self.omega
        x = np.cos((2*np.pi/self.n_cups * np.arange(self.n_cups)) + self.theta)

        torque = - self.g * self.radius * np.dot(self.cup_masses, x)
        torque += damping

        self.omega += torque
        self.theta += self.omega
        self.point_locs = self.get_point_locs()

        # Water leaking (Ideal: 0.006)
        leak_rate = 0.006

        for i in range(len(self.cup_masses)):
            self.cup_masses[i] = max(self.cup_masses[i] - leak_rate, 0)

        # Water adding (Ideal: 0.08)
        fill_rate = 0.08

        for i in range(len(self.cup_masses)):
            if self.get_cup_points(i)[1] <= self.y - self.radius - 27:
                self.cup_masses[i] = min(self.cup_masses[i] + fill_rate, self.max_mass_per_cup)

    def get_cup_points(self, cup_num, pairs=False):
        x = self.point_locs[2*cup_num]
        y = self.point_locs[(2*cup_num)+1]

        base_ratio = 1.5  # top base length / bottom base length
        altitude = 60
        bot_base_len = 40
        top_base_len = base_ratio * bot_base_len

        pts = [x - (top_base_len/2), y - (altitude/2), x + (top_base_len/2), y - (altitude/2),
               x + (bot_base_len/2), y + (altitude/2), x - (bot_base_len/2), y + (altitude/2)]

        if not pairs:
            return pts

        flattened = np.array(pts)
        n = len(flattened)
        return flattened.reshape((int(n / 2), 2))

    def get_water_points(self, cup_num, pairs=False):
        x = self.point_locs[2 * cup_num]
        y = self.point_locs[(2 * cup_num) + 1]

        mass_ratio = self.cup_masses[cup_num] / self.max_mass_per_cup

        base_ratio = 1.5  # top base length / bottom base length
        altitude = 60
        bot_base_len = 40
        top_base_len = bot_base_len * (1 + (mass_ratio * (base_ratio - 1)))

        h = mass_ratio * altitude

        pts = [x - (top_base_len / 2), y + (altitude / 2) - h, x + (top_base_len / 2), y + (altitude / 2) - h,
               x + (bot_base_len / 2), y + (altitude / 2), x - (bot_base_len / 2), y + (altitude / 2)]

        if not pairs:
            return pts

        flattened = np.array(pts)
        n = len(flattened)
        return flattened.reshape((int(n / 2), 2))


# Create wheel + center of mass positions array
wheel = WaterWheel(width/2, height/2, 200, 20, np.radians(-1 - 0.0001 + 0.001))
com_positions = deque([width/2, height/2 - 200])
omegas = deque([0.])
count = 0

# Angular velocity rescale (for displaying)
omega_mult = 10000.0 / 3.0

# Saving to file setup
f = open('CenterOfMassLocations.txt', 'wb')
np.savetxt(f, [], header="x, y, omega\n")

# System Params
animate = True
com_tracking = True  # Do we even wanna track at all
anti_resolution = 5  # The smaller the more points will be plotted
n_points_allowed = 1000

while count < 20000 or not com_tracking:
    # Reset frame
    count += 1
    window.fill(colors['black'])

    # Update wheel + center of mass array
    wheel.update_step()
    com = wheel.get_center_of_mass()
    prev_com = [com_positions[len(com_positions)-2], com_positions[len(com_positions)-1]]

    # Add more points to path + save
    if np.linalg.norm(com - prev_com) >= anti_resolution and com_tracking:
        com_positions.append(com[0])
        com_positions.append(com[1])
        omegas.append(wheel.omega)

        # Save each data point to file
        np.savetxt(f, np.column_stack((com[0], com[1], wheel.omega)), fmt='%f')

    # Cull oldest points if exceeds limit
    if len(com_positions) > n_points_allowed:
        com_positions.popleft()
        com_positions.popleft()  # pop last x and y values
        omegas.popleft()         # pop last omega

    if animate:
        # Display Path
        if len(com_positions) >= 4:
            positions = np.array(com_positions)
            positions = positions.reshape((int(len(positions) / 2), 2))
            positions = A_many(world_to_plane_many(make_many_3d(positions, z_arr=np.array(omegas) * omega_mult)))
            pygame.draw.lines(window, (110, 0, 255), False, positions, width=10)
            pygame.draw.lines(window, colors['white'], False, positions, width=2)

        # Display COM point
        point_rad = 10
        z = wheel.omega * omega_mult
        pygame.draw.circle(window, (110, 0, 255), A(world_to_plane(make3d(com, z))), point_rad, 0)
        pygame.draw.circle(window, colors['white'], A(world_to_plane(make3d(com, z))), point_rad, 2)

        # Display wheel
        pts = A_many(world_to_plane_many(make_many_3d(wheel.get_point_locs_2())))
        pygame.draw.polygon(window, colors['darkblue'], pts, 3)
        for i in range(wheel.n_cups):
            pts = A_many(world_to_plane_many(make_many_3d(wheel.get_water_points(i, pairs=True))))
            blue = lerp(wheel.cup_masses[i] / wheel.max_mass_per_cup, colors['lightblue'], colors['fullblue'])
            pygame.draw.polygon(window, blue, pts, 0)
            pts = A_many(world_to_plane_many(make_many_3d(wheel.get_cup_points(i, pairs=True))))
            pygame.draw.polygon(window, colors['red'], pts, 2)

    # # Display wheel center dot
    # center_rad = 10
    # pygame.draw.circle(window, colors['white'], (wheel.x, wheel.y), center_rad)

    # Handle 3D keys (for debugging)
    keys_pressed = pygame.key.get_pressed()
    handle_keys(keys_pressed)

    # Save frame
    if save_anim:
        pygame.image.save(window, path_to_save + '/frame' + str(count) + '.png')
        print('Saved frame ' + str(count))

    pygame.display.update()
    # time.sleep(0.001)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            pygame.quit()

            # Use ffmpeg to combine the PNG images into a video
            if save_anim:
                input_files = path_to_save + '/frame%d.png'
                output_file = path_to_save + '/output.mp4'
                ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
                os.system(
                    f'{ffmpeg_path} -r 60 -i {input_files} -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf "eq=brightness=0.00:saturation=1.3" {output_file} > /dev/null 2>&1')
                print('Saved video to ' + output_file)





# # Full 2D Plot of Projection
# x = []
# y = []
# omegas = []
# with open('CenterOfMassLocations.txt', 'r') as f:
#     line = f.readline()
#     header = []
#     while line:
#         if line.startswith('#'):
#             header.append(line)
#         else:
#             data = line.split(' ')
#             x.append(float(data[0]))
#             y.append(float(data[1]))
#             omegas.append(float(data[2][:len(data[1])-2]))
#
#         line = f.readline()
#
# print(omegas)

# plt.plot(x, y, linewidth=0.5, color='red')
# plt.xlabel('x center')
# plt.ylabel('y center')
# plt.title('Center of Mass of System')
#
# if com_tracking:
#     plt.show()

# 3D full graph with angular velocity
# from mpl_toolkits.mplot3d import axes3d
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, omegas, linewidth=0.5, color='red')
# plt.xlabel('x center')
# plt.ylabel('y center')
# ax.set_zlabel('Angular Velocity')
# ax.grid(False)
# plt.title('Full Trajectory of System in Phase Space')
# plt.show()


# TODO: Make the tracing of the path look cooler (like a color gradient or trail)