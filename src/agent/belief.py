import pomdp_py

from src.domain.maze_state import MazeState


def inititalize_default_belief_state(grid_height, grid_width):
    belief_dict = {}
    for i in range(0,grid_height):
        for j in range(0,grid_width):
            maze_state = MazeState(i,j)
            belief_dict[maze_state] = 1/(grid_height * grid_width)

    return pomdp_py.Histogram(belief_dict)
