import sys
import os
from src.lexer import Lexer, LexerError


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <source_file>")
        return

    filename = sys.argv[1]
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    try:
        with open(filename, "r") as f:
            source_code = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    lexer = Lexer(source_code)
    try:
        tokens = lexer.tokenize()
        
        if not lexer.has_error:
            for token in tokens:
                print(token.to_string())

    except LexerError as e:
        # This catch block might not be reached if Lexer handles errors internally
        pass
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
