import time

def solve_cryptarithmetic_backtracking(puzzle):
    letters = set(''.join(puzzle))  # Collect all unique letters
    digits = '0123456789'
    letter_permutations = generate_permutations(digits, len(letters))

    for perm in letter_permutations:
        solution = dict(zip(letters, perm))
        if check_solution(puzzle, solution):
            return solution
    return None

def generate_permutations(elements, length):
    # Base case: if the length is 1, yield each element
    if length == 1:
        for element in elements:
            yield (element,)
    else:
        # Recursive case: for each permutation of length - 1,
        # prepend each element
        for perm in generate_permutations(elements, length - 1):
            for element in elements:
                if element not in perm:
                    yield (element,) + perm

def check_solution(puzzle, solution):
    def apply_solution(word):
        return int(''.join(solution.get(letter, letter) for letter in word))
    
    s1, s2, s3 = puzzle
    return apply_solution(s1) + apply_solution(s2) == apply_solution(s3)

# Define the puzzle: SEND + MORE = MONEY
puzzle = ('SEND', 'MORE', 'MONEY')

# Measure the time taken to solve the puzzle
start_time = time.time()
solution = solve_cryptarithmetic_backtracking(puzzle)
end_time = time.time()

if solution:
    print("Backtracking solution:")
    for word in puzzle:
        print(' '.join(solution.get(letter, letter) for letter in word))
    print()
else:
    print("No solution found.")

print(f"Time taken for backtracking: {end_time - start_time} seconds")
import random

def initial_solution(letters):
    digits = list(range(10))
    random.shuffle(digits)
    return dict(zip(letters, digits))

def evaluate(solution, s1, s2, s3):
    def word_to_number(word, mapping):
        return int(''.join(str(mapping[letter]) for letter in word))
    
    num1 = word_to_number(s1, solution)
    num2 = word_to_number(s2, solution)
    num3 = word_to_number(s3, solution)
    
    if num1 + num2 == num3:
        return 0  # Correct solution
    else:
        return abs(num3 - (num1 + num2))  # Difference from correct solution

def get_neighbors(solution):
    neighbors = []
    letters = list(solution.keys())
    for i in range(len(letters)):
        for j in range(i + 1, len(letters)):
            neighbor = solution.copy()
            # Swap two values
            neighbor[letters[i]], neighbor[letters[j]] = neighbor[letters[j]], neighbor[letters[i]]
            neighbors.append(neighbor)
    return neighbors

def tabu_search(s1, s2, s3, tabu_tenure):
    letters = set(s1 + s2 + s3)
    current_solution = initial_solution(letters)
    # Initialize Tabu list
    tabu_list = []
    while True:
        current_evaluation = evaluate(current_solution, s1, s2, s3)
        # If a valid solution is found, return it
        if current_evaluation == 0:
            return current_solution
        neighbors = get_neighbors(current_solution)
        # Filter out neighbors that are in the tabu list
        neighbors = [neighbor for neighbor in neighbors if neighbor not in tabu_list]
        # If no non-tabu neighbors, get a new random solution
        if not neighbors:
            current_solution = initial_solution(letters)
            continue
        # Evaluate neighbors and select one randomly
        # Since we are not necessarily looking for the best, but just a valid one
        # we can pick any neighbor that improves upon the current solution or is not worse.
        neighbors.sort(key=lambda sol: evaluate(sol, s1, s2, s3))
        for neighbor in neighbors:
            if evaluate(neighbor, s1, s2, s3) <= current_evaluation:
                current_solution = neighbor
                break
        # Update the Tabu list
        tabu_list.append(current_solution)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        return tabu_list
#Note that tabu search sometimes doesn't come up with a valid solution 
# Other function definitions (initial_solution, evaluate, get_neighbors) remain unchanged.
# Define the puzzle: SEND + MORE = MONEY
s1 = "SEND"
s2 = "MORE"
s3 = "MONEY"
# Parameters for Tabu Search
tabu_tenure = 50

# Solve the puzzle
start_time = time.time()  # Start a timer for the search operation
solution = tabu_search(s1, s2, s3, tabu_tenure)
end_time = time.time()  # End the timer after finding the solution

# Print the solution and the time taken to find it
print(f"Solution: {solution}")
print(f"Time taken: {end_time - start_time} seconds")
