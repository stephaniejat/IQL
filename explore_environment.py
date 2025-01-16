import gymnasium as gym
import numpy as np
import curses
import time
from utils import generate_random_map

def print_grid(stdscr, grid, env, message):
    stdscr.clear()
    for row in grid:
        stdscr.addstr(' '.join(row) + '\n')

    stdscr.addstr("\n")
    stdscr.addstr("ENV desc:\n")
    stdscr.addstr(str(env.env.env.env.desc))

    stdscr.addstr("\n")
    
    stdscr.addstr(message + "\n")
    stdscr.refresh()
    if message != "":
        time.sleep(1)


def refresh_starting_position(stdscr, env, current_state = np.full((4, 4), '0')):
    env.reset()
    grid = current_state
    start_position = (0, 0)
    grid[start_position] = 'X'
    grid[-1][-1] = "G"
    print_grid(stdscr, grid, env, "")
    return grid, start_position

def main(stdscr, grid_size = 4, p_hole= 0.5):
    curses.curs_set(0)  # Hide the cursor
    stdscr.nodelay(1)   # Make getch() non-blocking
    stdscr.timeout(100) # Refresh every 100ms
    desc = generate_random_map(grid_size, p_hole, 1)
    env = gym.make('FrozenLake-v1', desc=desc, map_name=None, is_slippery=False)
    grid, position = refresh_starting_position(stdscr, env)

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
        
        new_position = divmod(obs, 4)

        if terminated:
            if reward == 0:
                grid[new_position] = 'L'
                message = "Truncated state reached!"
            else:
                # grid[new_position] = 'G'
                message = "Goal reached!"
        else:
            grid[position] = '0'
            grid[new_position] = 'X'
            position = new_position
            message = ""

        print_grid(stdscr, grid, env, message)

        if terminated or truncated:
            stdscr.addstr("Resetting environment...\n")
            stdscr.refresh()
            time.sleep(1)

            grid[position] = '0'
            grid, position = refresh_starting_position(stdscr, env, grid)

if __name__ == "__main__":
    curses.wrapper(main)