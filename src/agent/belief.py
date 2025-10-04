import random

import pomdp_py

from src.domain.maze_state import MazeState


def initialize_default_belief_state(grid_height, grid_width, walls=None):
    belief_dict = {}
    for x in range(grid_width):
        for y in range(grid_height):
            belief_dict[MazeState(x, y, grid_width, grid_height)] = 1 / (grid_width * grid_height)
    return pomdp_py.Histogram(belief_dict)

def initialize_particle_belief(grid_height, grid_width, walls, n_particles=100):
    particles = []
    for _ in range(n_particles):
        x = random.randint(0, grid_width-1)
        y = random.randint(0, grid_height-1)
        particles.append(MazeState(x, y, grid_height, grid_width))
    return pomdp_py.Particles(particles)