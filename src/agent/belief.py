import random

import pomdp_py

from src.domain.maze_state import MazeState

available_planners = {'pomcp', 'pouct', 'po-rollout'}

def initialize_uniform_belief(planner_name:str, grid_height,grid_width):
    _planner_name = planner_name.strip().lower()
    if _planner_name == "pomcp":
        return initialize_uniform_particle_belief(grid_height,grid_width)
    elif _planner_name == "pouct":
        return initialize_uniform_histogram_belief(grid_height,grid_width)
    else:
        raise ValueError(f"Unknown planner: {planner_name}")

def initialize_uniform_histogram_belief(grid_height, grid_width, walls=None):
    belief_dict = {}
    for x in range(grid_width):
        for y in range(grid_height):
            belief_dict[MazeState(x, y, grid_width, grid_height)] = 1 / (grid_width * grid_height)
    return pomdp_py.Histogram(belief_dict)

def initialize_uniform_particle_belief(grid_height, grid_width,walls=None, n_particles=1000):
    particles = []
    for _ in range(n_particles):
        x = random.randint(0, grid_width-1)
        y = random.randint(0, grid_height-1)
        particles.append(MazeState(x, y, grid_height, grid_width))
    return pomdp_py.Particles(particles)