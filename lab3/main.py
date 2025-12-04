#!/usr/bin/env python3
"""
语法分析算法实现主程序。

本程序提供了一个统一的命令行接口，用于执行以下任务：
- 任务3.1: 消除左递归
- 任务3.2: 提取左公因子
- 任务3.3: 计算FIRST集和FOLLOW集

使用方法:
    python main.py <grammar_file> [options]

参数:
    grammar_file: 文法文件路径（.txt格式）

选项:
    --eliminate-left-recursion, -e: 消除左递归
    --extract-left-factoring, -f: 提取左公因子
    --compute-first, -first: 计算FIRST集
    --compute-follow, -follow: 计算FOLLOW集
    --all, -a: 执行所有操作
    --verbose, -v: 显示详细处理过程
    --output <file>, -o <file>: 将结果输出到文件
"""

import argparse
import sys
from pathlib import Path

from src.parser import parse_grammar_file
from src.left_recursion import eliminate_left_recursion, LeftRecursionEliminator
from src.left_factoring import extract_left_factoring, LeftFactoringExtractor
from src.first_follow import FirstFollowCalculator


def print_header(title: str) -> None:
    """打印标题"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title: str) -> None:
    """打印章节标题"""
    print("\n" + "-" * 50)
    print(f" {title}")
    print("-" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="语法分析算法实现 - 任务3.1, 3.2, 3.3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 消除左递归
  python main.py test_grammars/expr_grammar.txt -e -v
  
  # 提取左公因子
  python main.py test_grammars/expr_grammar.txt -f -v
  
  # 计算FIRST集和FOLLOW集
  python main.py test_grammars/expr_grammar.txt -first -follow -v
  
  # 执行所有操作
  python main.py test_grammars/expr_grammar.txt -a -v
        """,
    )

    parser.add_argument("grammar_file", help="文法文件路径（.txt格式）")
    parser.add_argument("--eliminate-left-recursion", "-e", action="store_true", help="消除左递归（任务3.1）")
    parser.add_argument("--extract-left-factoring", "-f", action="store_true", help="提取左公因子（任务3.2）")
    parser.add_argument("--compute-first", "-first", action="store_true", help="计算FIRST集（任务3.3）")
    parser.add_argument("--compute-follow", "-follow", action="store_true", help="计算FOLLOW集（任务3.3）")
    parser.add_argument("--all", "-a", action="store_true", help="执行所有操作（任务3.1, 3.2, 3.3）")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细处理过程")
    parser.add_argument("--output", "-o", metavar="FILE", help="将结果输出到文件")

    args = parser.parse_args()

    # 检查是否至少指定了一个操作
    if not (
        args.eliminate_left_recursion
        or args.extract_left_factoring
        or args.compute_first
        or args.compute_follow
        or args.all
    ):
        parser.error("请至少指定一个操作（-e, -f, -first, -follow, -a）")

    # 如果指定了--all，则执行所有操作
    if args.all:
        args.eliminate_left_recursion = True
        args.extract_left_factoring = True
        args.compute_first = True
        args.compute_follow = True

    # 检查文法文件是否存在
    grammar_path = Path(args.grammar_file)
    if not grammar_path.exists():
        print(f"错误: 文法文件不存在: {grammar_path}", file=sys.stderr)
        sys.exit(1)

    # 解析文法
    try:
        print_header("解析文法文件")
        print(f"文件: {grammar_path}")

        grammar = parse_grammar_file(str(grammar_path))
        print("\n文法解析成功!")
        print_section("原始文法")
        print(grammar)

    except Exception as e:
        print(f"错误: 文法文件解析失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 用于存储所有输出
    all_output = []
    all_output.append("=" * 70)
    all_output.append(" 语法分析算法实现 - 实验报告")
    all_output.append("=" * 70)
    all_output.append(f"\n文法文件: {grammar_path}")
    all_output.append("\n" + str(grammar))

    current_grammar = grammar

    # 任务3.1: 消除左递归
    if args.eliminate_left_recursion:
        print_header("任务3.1: 消除左递归")
        all_output.append("\n" + "=" * 70)
        all_output.append(" 任务3.1: 消除左递归")
        all_output.append("=" * 70)

        try:
            eliminator = LeftRecursionEliminator(current_grammar)
            result_grammar = eliminator.eliminate()

            if args.verbose:
                print(eliminator.get_processing_log())
                all_output.append("\n详细处理过程:")
                all_output.append(eliminator.get_processing_log())

            print_section("消除左递归后的文法")
            print(result_grammar)
            all_output.append("\n结果文法:")
            all_output.append(str(result_grammar))

            current_grammar = result_grammar

        except Exception as e:
            print(f"错误: 消除左递归失败: {e}", file=sys.stderr)
            if args.verbose:
                import traceback

                traceback.print_exc()

    # 任务3.2: 提取左公因子
    if args.extract_left_factoring:
        print_header("任务3.2: 提取左公因子")
        all_output.append("\n" + "=" * 70)
        all_output.append(" 任务3.2: 提取左公因子")
        all_output.append("=" * 70)

        try:
            extractor = LeftFactoringExtractor(current_grammar)
            result_grammar = extractor.extract()

            if args.verbose:
                print(extractor.get_processing_log())
                all_output.append("\n详细处理过程:")
                all_output.append(extractor.get_processing_log())

            print_section("提取左公因子后的文法")
            print(result_grammar)
            all_output.append("\n结果文法:")
            all_output.append(str(result_grammar))

            current_grammar = result_grammar

        except Exception as e:
            print(f"错误: 提取左公因子失败: {e}", file=sys.stderr)
            if args.verbose:
                import traceback

                traceback.print_exc()

    # 任务3.3: 计算FIRST集和FOLLOW集
    if args.compute_first or args.compute_follow:
        print_header("任务3.3: 计算FIRST集和FOLLOW集")
        all_output.append("\n" + "=" * 70)
        all_output.append(" 任务3.3: 计算FIRST集和FOLLOW集")
        all_output.append("=" * 70)

        try:
            calculator = FirstFollowCalculator(current_grammar)

            # 计算FIRST集
            if args.compute_first:
                print_section("计算FIRST集")
                all_output.append("\n计算FIRST集:")

                first_sets = calculator.compute_first_sets()

                if args.verbose:
                    print(calculator.get_processing_log())
                    all_output.append("\n详细计算过程:")
                    all_output.append(calculator.get_processing_log())

                print(calculator.get_first_sets_str())
                all_output.append("\n结果:")
                all_output.append(calculator.get_first_sets_str())

            # 计算FOLLOW集
            if args.compute_follow:
                print_section("计算FOLLOW集")
                all_output.append("\n计算FOLLOW集:")

                follow_sets = calculator.compute_follow_sets()

                if args.verbose:
                    print(calculator.get_processing_log())
                    all_output.append("\n详细计算过程:")
                    all_output.append(calculator.get_processing_log())

                print(calculator.get_follow_sets_str())
                all_output.append("\n结果:")
                all_output.append(calculator.get_follow_sets_str())

        except Exception as e:
            print(f"错误: 计算FIRST/FOLLOW集失败: {e}", file=sys.stderr)
            if args.verbose:
                import traceback

                traceback.print_exc()

    # 如果指定了输出文件，将结果写入文件
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("\n".join(all_output))
            print(f"\n结果已保存到: {args.output}")
        except Exception as e:
            print(f"错误: 无法写入输出文件: {e}", file=sys.stderr)
            sys.exit(1)

    print_header("所有任务执行完成")

    # 如果未指定输出文件，询问是否保存结果
    if not args.output:
        print("\n是否将结果保存到文件? (y/n): ", end="")
        try:
            response = input().strip().lower()
            if response == "y":
                output_file = input("请输入输出文件名: ").strip()
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(all_output))
                print(f"结果已保存到: {output_file}")
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
