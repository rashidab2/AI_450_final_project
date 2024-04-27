import numpy as np
import time
import random

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

# Genetic Algorithm Functions
def fixed_positions(board):
    return [(i, j) for i in range(9) for j in range(9) if board[i][j] != 0]

def create_individual(board, fixed):
    individual = np.copy(board)
    for i in range(9):
        missing = list(set(range(1, 10)) - set(individual[i, :]))
        random.shuffle(missing)
        for j in range(9):
            if individual[i, j] == 0:
                individual[i, j] = missing.pop()
    return individual

def calculate_fitness(individual):
    score = 0
    for i in range(9):
        score += len(set(individual[i, :]))  # Rows
        score += len(set(individual[:, i]))  # Columns
        block = individual[i//3*3:(i//3+1)*3, i%3*3:(i%3+1)*3]
        score += len(set(block.flatten()))  # Blocks
    return score

def selection(population, fitness_scores):
    # Create a cumulative sum of fitness scores
    cumulative_fitness = np.cumsum(fitness_scores)
    total_fitness = cumulative_fitness[-1]
    
    # Select individuals based on their fitness probability
    new_population = []
    for _ in range(len(population)):
        # Generate a random number in the range of the total fitness
        r = random.uniform(0, total_fitness)
        # Find the individual that corresponds to the random number
        for i, individual_fitness in enumerate(cumulative_fitness):
            if r <= individual_fitness:
                new_population.append(population[i])
                break
    return new_population

def crossover(parent1, parent2, fixed):
    child = np.copy(parent1)
    for i in range(9):
        if random.random() > 0.5:
            for j in range(9):
                if (i, j) not in fixed:
                    child[i, j] = parent2[i, j]
    return child

def mutate(individual, fixed, mutation_rate=0.05):
    for i in range(9):
        if random.random() < mutation_rate:
            j1, j2 = random.sample([j for j in range(9) if (i, j) not in fixed], 2)
            individual[i, j1], individual[i, j2] = individual[i, j2], individual[i, j1]
    return individual

# Genetic Algorithm for Sudoku
def genetic_algorithm_sudoku(board, population_size=100, generations=1000):
    fixed = fixed_positions(board)
    population = [create_individual(board, fixed) for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = [calculate_fitness(individual) for individual in population]
        if max(fitness_scores) == 243:  # 243 is the maximum fitness score for a solved Sudoku
            return population[fitness_scores.index(max(fitness_scores))]

        # Selection
        population = selection(population, fitness_scores)
        
        # Create the next generation
        new_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = population[i], population[i+1]
            child1, child2 = crossover(parent1, parent2, fixed), crossover(parent2, parent1, fixed)
            child1, child2 = mutate(child1, fixed), mutate(child2, fixed)
            new_population.extend([child1, child2])
        population = new_population

    # If no solution found, return the best individual
    best_fitness = max(fitness_scores)
    best_individual = population[fitness_scores.index(best_fitness)]
    return best_individual


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

def tabu_search(board, tabu_size=50):
    original_board = np.array(board, dtype=int)
    current_solution = initial_solution(original_board)
    
    if is_valid_board(current_solution):
        return current_solution  # If the initial board is already valid
    
    tabu_list = set()
    tabu_list.add(tuple(map(tuple, current_solution)))

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
        # Update the tabu list
        tabu_list.add(tuple(map(tuple, current_solution)))
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)  # Ensure the tabu list doesn't exceed its size limit
        


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

def genetic_sudoku_solver(board):
    start_time = time.time()
    solution = genetic_algorithm_sudoku(board)
    end_time = time.time()
    if calculate_fitness(solution) == 243:
        print("Found a solution with genetic algorithm:")
        print(solution)
    else:
        print("No perfect solution found. Best attempt:")
        print(solution)
    print(f"Fitness score: {calculate_fitness(solution)}\n")
    print(f"Execution time: {end_time - start_time:.4f} seconds\n")
boards = [board1, board2, board3]
for board in boards:
        genetic_sudoku_solver(board)

def test_tabu_search(boards,tabu_size):
    for i, board in enumerate(boards):
        print(f"Testing board {i + 1}")
        solution = tabu_search(board,tabu_size)
        for row in solution:
            print(row)
        if is_valid_board(solution):
            print(f"Board {i + 1} solved successfully:\n{np.matrix(solution)}\n")
        else:
            print(f"Board {i + 1} was not solved successfully.\n")
tabu_size = 50
boards = [board1, board2, board3]
# Test the Tabu Search on the defined boards
test_tabu_search(boards,tabu_size)
