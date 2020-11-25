import numpy as np
from random import random


def rand_loc(count, dimension):
    return np.random.rand(count, 2) * dimension


def distance(co1, co2):
    return np.linalg.norm(co1 - co2)


def rand_offset(locations, prev_offsets, dimension, acceleration, speed):
    offset_offsets = np.random.random((len(locations), 2)) * (acceleration * 2) - acceleration
    offsets = np.clip(prev_offsets + offset_offsets, -speed, speed)
    if len(locations):
        new_locations = offsets + locations
        return np.clip(new_locations, 0, dimension - 1), offsets
    else:
        return locations, offsets


def create_grid(positions, sectors, box_size):
    _grid = [[[] for _ in range(sectors)] for __ in range(sectors)]
    for position in positions:
        new_pos = grid_pos(position, box_size)
        _grid[new_pos[0]][new_pos[1]].append(position)
    return _grid


def grid_pos(position, box_size):
    return (position // box_size).astype(int)


def radius_count(center, grid, sectors, box_size, infection_range):
    new_pos = grid_pos(center, box_size)
    grid_x, grid_y = new_pos
    cnt = 0
    for x in range(grid_x - 1, grid_x + 2):
        for y in range(grid_y - 1, grid_y + 2):
            if x in range(sectors) and y in range(sectors):
                for loc in grid[x][y]:
                    if distance(center, loc) <= infection_range:
                        cnt += 1
    return cnt


def spread(cnt, probability):
    prob = 1 - (1 - probability) ** cnt
    if random() < prob:
        return True
    else:
        return False
