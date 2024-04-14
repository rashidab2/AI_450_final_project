def solve_cryptarithmetic(puzzle):
    words = puzzle.split()
    unique_chars = set(''.join(words))
    if len(unique_chars) > 10:
        return "Invalid puzzle: More than 10 unique characters"
    
    chars = ''.join(words)
    leading_chars = set([word[0] for word in words])
    if len(chars) != len(set(chars)):
        return "Invalid puzzle: Repeated characters"
    
    if len(leading_chars) > 2:
        return "Invalid puzzle: More than 2 leading characters"
    
    def assign_digit(char_index, used_digits):
        if char_index == len(chars):
            return True
        
        char = chars[char_index]
        for digit in range(10):
            if (char in leading_chars and digit == 0) or (char in used_digits):
                continue
            mapping[char] = digit
            if assign_digit(char_index + 1, used_digits | {char}):
                return True
        del mapping[char]
        return False
    
    mapping = {}
    if assign_digit(0, set()):
        result = ''
        for word in words:
            num = ''.join(str(mapping[char]) for char in word)
            result += f'{word} = {int(num)}\n'
        return result
    else:
        return "No solution found"

# Example usage
puzzle = "SEND + MORE = MONEY"
print(solve_cryptarithmetic(puzzle))
