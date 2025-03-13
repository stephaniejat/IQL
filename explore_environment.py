import gymnasium as gym
import numpy as np
import curses
import time
from itertools import cycle
from utils import generate_random_map


# CONSTANTS
GRID_SIZE = 4
P = 0.5
ROTATE = 0
DESC = generate_random_map(GRID_SIZE, P, ROTATE)
EMPTY_BOARD = np.full((GRID_SIZE, GRID_SIZE), '.')
EMPTY_BOARD[0][0] = 'X'
EMPTY_BOARD[-1][-1] = "G"
start_position_cycle = cycle([(0, 0), (0,-1), (-1,-1), (-1,0)])
START_POSITION = next(start_position_cycle)
for i in range(ROTATE):
    EMPTY_BOARD = np.array(list(zip(*EMPTY_BOARD[::-1])))
    START_POSITION = next(start_position_cycle)


def print_grid(stdscr, grid, message=""):
    stdscr.clear()
    for row in grid:
        stdscr.addstr(' '.join(row) + '\n')

    stdscr.addstr("\n")
    stdscr.addstr("ENV desc:\n")
    stdscr.addstr(str(DESC))

    stdscr.addstr("\n")
    
    stdscr.addstr(message + "\n")
    stdscr.refresh()
    if message != "":
        time.sleep(1)


def refresh_starting_position(stdscr, current_state):
    grid = current_state
    grid[START_POSITION] = "X"
    print_grid(stdscr, grid, "")
    return grid, START_POSITION

def main(stdscr):
    curses.curs_set(0)  # Hide the cursor
    stdscr.nodelay(1)   # Make getch() non-blocking
    stdscr.timeout(100) # Refresh every 100ms

    env = gym.make('FrozenLake-v1', desc=DESC, map_name=None, is_slippery=False)
    env.reset()

    grid, position = refresh_starting_position(stdscr, EMPTY_BOARD)

    # RUN
    while True:
        key = stdscr.getch()
        if key == curses.KEY_LEFT:
            action = 0
        elif key == curses.KEY_DOWN:
            action = 1
        elif key == curses.KEY_RIGHT:
            action = 2
        elif key == curses.KEY_UP:
            action = 3
        else:
            continue

        obs, reward, terminated, truncated, info = env.step(action)
        
        new_position = divmod(obs, GRID_SIZE)

        if terminated:
            if reward == 0:
                grid[new_position] = 'H'
                message = "You fell through the ice!"
            else:
                # grid[new_position] = 'G'
                message = "Goal reached!"
        else:
            grid[position] = '.'
            grid[new_position] = 'X'
            position = new_position
            message = ""

        print_grid(stdscr, grid, message)

        if terminated or truncated:
            stdscr.addstr("Resetting environment...\n")
            stdscr.refresh()
            time.sleep(1)

            grid[position] = '.'

            env.reset()
            grid, position = refresh_starting_position(stdscr, grid)

if __name__ == "__main__":
    curses.wrapper(main)