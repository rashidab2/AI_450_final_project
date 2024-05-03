import numpy as np
import time
import random
from collections import deque

# Backtracking Search for Sudoku
def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num or board[row//3*3 + i//3][col//3*3 + i%3] == num:
            return False
    return True

def backtracking_search(board):
    empty = find_empty_location(board)
    if not empty:
        return True
    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if backtracking_search(board):
                return True
            board[row][col] = 0
    return False

def find_empty_location(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

# Tabu Search for Sudoku
def initial_solution(board):
    """Generates an initial board configuration, trying to minimize row and column conflicts."""
    n = 9
    solution = np.copy(board)  # Start with the given board
    for row in range(n):
        for col in range(n):
            if solution[row][col] == 0:  # If the cell is empty
                # Find which numbers are not in the current row and column
                numbers_not_in_row = set(range(1, n + 1)) - set(solution[row])
                numbers_not_in_col = set(range(1, n + 1)) - set(solution[:, col])
                
                # Use the intersection of those sets to avoid conflicts
                valid_numbers = list(numbers_not_in_row & numbers_not_in_col)
                
                # If there are no valid numbers, we can just continue and fill it with a random choice,
                # acknowledging that it may create a conflict
                if not valid_numbers:
                    valid_numbers = list(set(range(1, n + 1)) - set(solution[row]))
                    if not valid_numbers:
                        valid_numbers = list(range(1, n + 1))  # Last resort: any number
                
                # Randomly choose one of the valid numbers for this cell
                chosen_number = random.choice(valid_numbers)
                solution[row][col] = chosen_number
    
    return solution

def is_valid_board(board):
    n = 9

    # Check each row and column
    for i in range(n):
        row = [num for num in board[i, :] if num != 0]
        col = [num for num in board[:, i] if num != 0]
        if len(row) != len(set(row)) or len(col) != len(set(col)):
            return False
    
    # Check each 3x3 subgrid
    for i in range(0, n, 3):
        for j in range(0, n, 3):
            subgrid = [num for num in board[i:i+3, j:j+3].flatten() if num != 0]
            if len(subgrid) != len(set(subgrid)):
                return False
    
    return True

def calculate_fitness(board):
    # Revised fitness function
    n = 9
    fitness = 0
    for i in range(n):
        fitness += (9 - len(set(board[i, :])))  # Conflicts in row
        fitness += (9 - len(set(board[:, i])))  # Conflicts in column
    for i in range(0, n, 3):
        for j in range(0, n, 3):
            subgrid = board[i:i+3, j:j+3].flatten()
            fitness += (9 - len(set(subgrid)))  # Conflicts in subgrid
    return 243 - fitness

def get_neighbors(board, original_board, tabu_list):
    neighbors = []
    n = 9

    # Instead of checking all possible swaps, perform a smaller number of swaps
    num_swaps = 5  # You can tune this parameter
    for _ in range(num_swaps):
        i, j = random.choice([(x, y) for x in range(n) for y in range(n) if original_board[x][y] == 0])
        k, l = random.choice([(x, y) for x in range(n) for y in range(n) if original_board[x][y] == 0])
        if i != k or j != l:
            new_board = np.copy(board)
            new_board[i, j], new_board[k, l] = new_board[k, l], new_board[i, j]
            new_configuration = tuple(map(tuple, new_board))
            if new_configuration not in tabu_list:
                neighbors.append(new_board)
    return neighbors

def tabu_search(board, tabu_size=50, time_limit=60):
    original_board = np.array(board, dtype=int)
    current_solution = initial_solution(original_board)
    
    if is_valid_board(current_solution):
        return current_solution  # If the initial board is already valid
    
    tabu_list = [tuple(map(tuple, current_solution))]  # Initialize tabu list with the initial solution

    while True:
        neighbors = get_neighbors(current_solution, original_board, tabu_list)
        if not neighbors:
            current_solution = initial_solution(original_board)  # Restart if no neighbors
            continue

        for neighbor in neighbors:
            if is_valid_board(neighbor):
                return neighbor  # Return the first valid board found

        # Update the current solution to a random neighbor (not necessarily a better one)
        current_solution = random.choice(neighbors)

        # Convert the current solution to a tuple of tuples and add it to the tabu list
        current_config = tuple(map(tuple, current_solution))
        tabu_list.append(current_config)

        # Ensure the tabu list doesn't exceed its size limit
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)     

'''def tabu_search(board, tabu_size=50, time_limit=60):
    start_time = time.time()  # Record the start time
    original_board = np.array(board, dtype=int)
    current_solution = initial_solution(original_board)
    
    if is_valid_board(current_solution):
        return current_solution  # If the initial board is already valid
    
    tabu_list = [tuple(map(tuple, current_solution))]  # Initialize tabu list with the initial solution

    while time.time() - start_time < time_limit:  # Continue until the time limit is reached
        neighbors = get_neighbors(current_solution, original_board, tabu_list)
        if not neighbors:
            # No valid neighbors, restart with a new initial solution
            current_solution = initial_solution(original_board)
            tabu_list = [tuple(map(tuple, current_solution))]
            continue

        # Evaluate the neighbors and choose the best one based on some criteria (e.g., least conflicts)
        neighbors.sort(key=lambda x: calculate_fitness(x), reverse=True)
        best_neighbor = neighbors[0]
        
        if is_valid_board(best_neighbor):
            return best_neighbor  # Return the first valid board found
        
        # Update the current solution to the best neighbor
        current_solution = best_neighbor

        # Convert the current solution to a tuple of tuples and add it to the tabu list
        current_config = tuple(map(tuple, current_solution))
        tabu_list.append(current_config)
        
        # Ensure the tabu list doesn't exceed its size limit
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)  # Remove the oldest element in the tabu list

    # If the time limit is reached, return the current solution even if it's not valid
    return current_solution        
# This solution was implemented because the time it take to reach a solution is far to much, 
#I stop it right at the time_limit for testing purpose, it was runned without the time limit
It takes awhile.
'''
# Example Sudoku board
board1 = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
board2 = [
        [8, 5, 0, 0, 0, 2, 4, 0, 0],
        [7, 2, 0, 0, 0, 0, 0, 0, 9],
        [0, 0, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 7, 0, 0, 2],
        [3, 0, 5, 0, 0, 0, 9, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 0, 0, 7, 0],
        [0, 1, 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 6, 0, 4, 0]
    ]
board3 = [
        [0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 6, 0, 0, 0, 0, 3],
        [0, 7, 4, 0, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 0, 2],
        [0, 8, 0, 0, 4, 0, 0, 1, 0],
        [6, 0, 0, 5, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 7, 8, 0],
        [5, 0, 0, 0, 0, 9, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 0]
    ]
def test_backtracking_sudoku_solver(board, solver):
    start_time = time.time()
    solution = solver(board)
    end_time = time.time()
    
    # Assuming the solver returns a boolean indicating success or failure
    if np.all(solution):
        print(f"Solution found by {solver.__name__}:")
        # Assuming the solver modifies the board in-place, otherwise print the returned solution
        for row in board:
            print(row)
    else:
        print(f"No solution found by {solver.__name__}.")

    print(f"Execution time: {end_time - start_time:.4f} seconds\n")

# Example Sudoku boards
boards = [board1, board2, board3]

# Test each board with each solver
solvers = [backtracking_search]

for board in boards:
    for solver in solvers:
        # Make a deep copy of the board to avoid modifying the original
        board_copy = [row[:] for row in board]
        test_backtracking_sudoku_solver(board_copy, solver)

def test_tabu_search(boards, tabu_size):
    for i, board in enumerate(boards):
        print(f"Testing board {i + 1}")
        solution = tabu_search(board, tabu_size)  # Pass each individual board to the function
        for row in solution:
            print(row)
        if is_valid_board(solution):
            print(f"Board {i + 1} solved successfully:\n{np.matrix(solution)}\n")
        else:
            print(f"Board {i + 1} was not solved successfully.\n")

# Now you can call test_tabu_search with the list of boards
test_tabu_search(boards, tabu_size=50)
