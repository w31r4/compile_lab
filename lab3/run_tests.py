import os
import sys
from pathlib import Path

# 添加 src 目录到 sys.path 以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.parser import parse_grammar_file
from src.left_recursion import eliminate_left_recursion
from src.left_factoring import LeftFactoringExtractor
from src.first_follow import FirstFollowCalculator


def run_test(grammar_file):
    print(f"\n{'='*20} Testing {grammar_file.name} {'='*20}")

    try:
        # 1. 解析文法
        print("\n[1] Parsing Grammar...")
        grammar = parse_grammar_file(str(grammar_file))
        print("Success!")
        print(grammar)

        # 2. 消除左递归
        print("\n[2] Eliminating Left Recursion...")
        grammar_no_lr = eliminate_left_recursion(grammar)
        print("Success!")
        print(grammar_no_lr)

        # 3. 提取左公因子
        print("\n[3] Extracting Left Factors...")
        extractor = LeftFactoringExtractor(grammar_no_lr)
        grammar_no_lf = extractor.extract()
        print("Success!")
        print(grammar_no_lf)

        # 4. 计算 FIRST 和 FOLLOW 集
        print("\n[4] Computing FIRST & FOLLOW Sets...")
        calculator = FirstFollowCalculator(grammar_no_lf)
        calculator.compute_first_sets()
        calculator.compute_follow_sets()

        print("\nFIRST Sets:")
        print(calculator.get_first_sets_str())

        print("\nFOLLOW Sets:")
        print(calculator.get_follow_sets_str())

        print(f"\n✅ {grammar_file.name} Passed!")

    except Exception as e:
        print(f"\n❌ {grammar_file.name} Failed!")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


def main():
    test_dir = Path("test_grammars")
    if not test_dir.exists():
        print(f"Test directory {test_dir} not found!")
        return

    grammar_files = sorted(list(test_dir.glob("*.txt")))

    if not grammar_files:
        print("No grammar files found in test_grammars/")
        return

    print(f"Found {len(grammar_files)} test grammars.")

    for grammar_file in grammar_files:
        run_test(grammar_file)


if __name__ == "__main__":
    main()
