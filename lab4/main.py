import sys
import os
import argparse
from src.lexer import Lexer, LexerError
from src.parser import Parser
from src.ast_nodes import ASTPrinter


def main():
    # 解析命令行参数
    arg_parser = argparse.ArgumentParser(description="SysY Compiler Frontend")
    arg_parser.add_argument("source_file", help="Source file to compile (.sy)")
    arg_parser.add_argument("--lexer", "-l", action="store_true", help="Only run lexer and output tokens")
    arg_parser.add_argument("--parser", "-p", action="store_true", help="Run parser and output AST (default)")
    arg_parser.add_argument("--semantic", "-s", action="store_true", help="Run full semantic analysis")

    args = arg_parser.parse_args()

    filename = args.source_file
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return 1

    try:
        with open(filename, "r") as f:
            source_code = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return 1

    # 词法分析
    lexer = Lexer(source_code)
    try:
        tokens = lexer.tokenize()

        if lexer.has_error:
            # 有词法错误，停止处理
            return 1

        # 如果只需要词法分析输出
        if args.lexer:
            for token in tokens:
                print(token.to_string())
            return 0

        # 语法分析
        parser = Parser(tokens)
        ast = parser.parse()

        if parser.has_error:
            # 有语法错误，停止处理
            return 1

        # 如果不需要语义分析，输出 AST
        if not args.semantic:
            printer = ASTPrinter()
            output = printer.print_ast(ast)
            print(output)
            return 0

        # TODO: 语义分析
        # semantic_analyzer = SemanticAnalyzer()
        # semantic_analyzer.analyze(ast)
        # if semantic_analyzer.has_error:
        #     return 1

        print("success")
        return 0

    except LexerError as e:
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
