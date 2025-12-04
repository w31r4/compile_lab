import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 添加 src 目录到 sys.path 以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.parser import parse_grammar_file
from src.left_recursion import eliminate_left_recursion
from src.left_factoring import LeftFactoringExtractor
from src.first_follow import FirstFollowCalculator
from src.ll1_parser import LL1Parser
from src.grammar import Grammar, EPSILON


ProductionSet = Set[Tuple[str, Tuple[str, ...]]]
SymbolSet = Dict[str, Set[str]]


def _prod_to_set(grammar: Grammar) -> ProductionSet:
    """将文法的产生式转换为便于比较的集合表示。"""
    result: ProductionSet = set()
    for prod in grammar.productions:
        if not prod.right or prod.right == [EPSILON]:
            right = ("@",)
        else:
            right = tuple(sym.value for sym in prod.right)
        result.add((prod.left.value, right))
    return result


def _calc_symbol_sets(calculator: FirstFollowCalculator) -> Tuple[SymbolSet, SymbolSet]:
    """提取 FIRST/FOLLOW 集（仅非终结符）为便于比较的字符串集合。"""
    first_sets: SymbolSet = {}
    for nt in calculator.grammar.non_terminals:
        first_sets[nt.value] = {sym.value for sym in calculator.first_sets[nt]}

    follow_sets: SymbolSet = {}
    for nt in calculator.grammar.non_terminals:
        follow_sets[nt.value] = {sym.value for sym in calculator.follow_sets[nt]}

    return first_sets, follow_sets


def _table_snapshot(ll1: LL1Parser) -> Set[Tuple[str, str, Tuple[str, ...]]]:
    """将 LL(1) 表转换为集合快照 (A, a, rhs)。"""
    snapshot: Set[Tuple[str, str, Tuple[str, ...]]] = set()
    for nt, row in ll1.table.items():
        for terminal, prod in row.items():
            rhs = ("@",) if (not prod.right or prod.right == [EPSILON]) else tuple(sym.value for sym in prod.right)
            snapshot.add((nt.value, terminal, rhs))
    return snapshot


def _assert_table(actual: Set[Tuple[str, str, Tuple[str, ...]]], expected: Set[Tuple[str, str, Tuple[str, ...]]], name: str) -> None:
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        msg = [
            f"{name}: LL(1) 表不匹配",
            f"  missing ({len(missing)}): {sorted(missing)}",
            f"  extra ({len(extra)}): {sorted(extra)}",
        ]
        raise AssertionError("\n".join(msg))


def _assert_prod_set(actual: ProductionSet, expected: ProductionSet, stage: str, name: str) -> None:
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        msg = [
            f"{name}: {stage} productions differ",
            f"  missing ({len(missing)}): {sorted(missing)}",
            f"  extra ({len(extra)}): {sorted(extra)}",
        ]
        raise AssertionError("\n".join(msg))


def _assert_symbol_sets(actual: SymbolSet, expected: SymbolSet, stage: str, name: str) -> None:
    if actual.keys() != expected.keys():
        missing = expected.keys() - actual.keys()
        extra = actual.keys() - expected.keys()
        raise AssertionError(f"{name}: {stage} keys differ (missing={missing}, extra={extra})")

    for nt, exp_set in expected.items():
        act_set = actual.get(nt, set())
        if act_set != exp_set:
            missing = exp_set - act_set
            extra = act_set - exp_set
            msg = [
                f"{name}: {stage} set differ for {nt}",
                f"  expected: {sorted(exp_set)}",
                f"  actual:   {sorted(act_set)}",
                f"  missing:  {sorted(missing)}",
                f"  extra:    {sorted(extra)}",
            ]
            raise AssertionError("\n".join(msg))


# 预期结果快照（基于当前算法输出）
EXPECTED = {
    "complex_grammar.txt": {
        "after_left_recursion": {
            ("A", ("a", "b", "c")),
            ("A", ("a", "b", "d")),
            ("B", ("a", "b", "e")),
            ("S", ("a", "b", "c")),
            ("S", ("a", "b", "d")),
            ("S", ("a", "b", "e")),
        },
        "after_left_factoring": {
            ("A", ("a", "A'")),
            ("A'", ("b", "A''")),
            ("A''", ("c",)),
            ("A''", ("d",)),
            ("B", ("a", "b", "e")),
            ("S", ("a", "S'")),
            ("S'", ("b", "S''")),
            ("S''", ("c",)),
            ("S''", ("d",)),
            ("S''", ("e",)),
        },
        "first_sets": {
            "A": {"a"},
            "A'": {"b"},
            "A''": {"c", "d"},
            "B": {"a"},
            "S": {"a"},
            "S'": {"b"},
            "S''": {"c", "d", "e"},
        },
        "follow_sets": {
            "A": set(),
            "A'": set(),
            "A''": set(),
            "B": set(),
            "S": {"$"},
            "S'": {"$"},
            "S''": {"$"},
        },
        "ll1_table": {
            ("A", "a", ("a", "A'")),
            ("A'", "b", ("b", "A''")),
            ("A''", "c", ("c",)),
            ("A''", "d", ("d",)),
            ("B", "a", ("a", "b", "e")),
            ("S", "a", ("a", "S'")),
            ("S'", "b", ("b", "S''")),
            ("S''", "c", ("c",)),
            ("S''", "d", ("d",)),
            ("S''", "e", ("e",)),
        },
        "is_ll1": True,
    },
    "expr_grammar.txt": {
        "after_left_recursion": {
            ("E", ("T", "E'")),
            ("E'", ("+", "T", "E'")),
            ("E'", ("@",)),
            ("T", ("(", "E", ")", "T'")),
            ("T", ("id", "T'")),
            ("T'", ("*", "F", "T'")),
            ("T'", ("@",)),
            ("F", ("(", "E", ")")),
            ("F", ("id",)),
        },
        "after_left_factoring": {
            ("E", ("T", "E'")),
            ("E'", ("+", "T", "E'")),
            ("E'", ("@",)),
            ("T", ("(", "E", ")", "T'")),
            ("T", ("id", "T'")),
            ("T'", ("*", "F", "T'")),
            ("T'", ("@",)),
            ("F", ("(", "E", ")")),
            ("F", ("id",)),
        },
        "first_sets": {
            "E": {"(", "id"},
            "E'": {"+", "@"},
            "F": {"(", "id"},
            "T": {"(", "id"},
            "T'": {"*", "@"},
        },
        "follow_sets": {
            "E": {"$", ")"},
            "E'": {"$", ")"},
            "F": {"$", ")", "*", "+"},
            "T": {"$", ")", "+"},
            "T'": {"$", ")", "+"},
        },
        "ll1_table": {
            ("E", "(", ("T", "E'")),
            ("E", "id", ("T", "E'")),
            ("E'", "$", ("@",)),
            ("E'", ")", ("@",)),
            ("E'", "+", ("+", "T", "E'")),
            ("F", "(", ("(", "E", ")")),
            ("F", "id", ("id",)),
            ("T", "(", ("(", "E", ")", "T'")),
            ("T", "id", ("id", "T'")),
            ("T'", "$", ("@",)),
            ("T'", ")", ("@",)),
            ("T'", "*", ("*", "F", "T'")),
            ("T'", "+", ("@",)),
        },
        "is_ll1": True,
    },
    "expr_grammar_raw.txt": {
        "after_left_recursion": {
            ("E", ("T", "E'")),
            ("E'", ("+", "T", "E'")),
            ("E'", ("@",)),
            ("T", ("(", "E", ")", "T'")),
            ("T", ("id", "T'")),
            ("T'", ("*", "F", "T'")),
            ("T'", ("@",)),
            ("F", ("(", "E", ")")),
            ("F", ("id",)),
        },
        "after_left_factoring": {
            ("E", ("T", "E'")),
            ("E'", ("+", "T", "E'")),
            ("E'", ("@",)),
            ("T", ("(", "E", ")", "T'")),
            ("T", ("id", "T'")),
            ("T'", ("*", "F", "T'")),
            ("T'", ("@",)),
            ("F", ("(", "E", ")")),
            ("F", ("id",)),
        },
        "first_sets": {
            "E": {"(", "id"},
            "E'": {"+", "@"},
            "F": {"(", "id"},
            "T": {"(", "id"},
            "T'": {"*", "@"},
        },
        "follow_sets": {
            "E": {"$", ")"},
            "E'": {"$", ")"},
            "F": {"$", ")", "*", "+"},
            "T": {"$", ")", "+"},
            "T'": {"$", ")", "+"},
        },
        "ll1_table": {
            ("E", "(", ("T", "E'")),
            ("E", "id", ("T", "E'")),
            ("E'", "$", ("@",)),
            ("E'", ")", ("@",)),
            ("E'", "+", ("+", "T", "E'")),
            ("F", "(", ("(", "E", ")")),
            ("F", "id", ("id",)),
            ("T", "(", ("(", "E", ")", "T'")),
            ("T", "id", ("id", "T'")),
            ("T'", "$", ("@",)),
            ("T'", ")", ("@",)),
            ("T'", "*", ("*", "F", "T'")),
            ("T'", "+", ("@",)),
        },
        "is_ll1": True,
    },
    "ll1_grammar.txt": {
        "after_left_recursion": {
            ("S", ("a", "L", ";")),
            ("L", ("S",)),
            ("L", ("c", "b")),
        },
        "after_left_factoring": {
            ("S", ("a", "L", ";")),
            ("L", ("S",)),
            ("L", ("c", "b")),
        },
        "first_sets": {
            "L": {"a", "c"},
            "S": {"a"},
        },
        "follow_sets": {
            "L": {";"},
            "S": {"$", ";"},
        },
        "ll1_table": {
            ("L", "a", ("S",)),
            ("L", "c", ("c", "b")),
            ("S", "a", ("a", "L", ";")),
        },
        "is_ll1": True,
    },
    "mixed_grammar.txt": {
        "after_left_recursion": {
            ("Q", ("w", "R")),
            ("Q", ("w", "y")),
            ("R", ("z",)),
            ("R", ("y",)),
            ("P", ("Q", "z", "P'")),
            ("P'", ("x", "y", "P'")),
            ("P'", ("@",)),
        },
        "after_left_factoring": {
            ("Q", ("w", "Q'")),
            ("Q'", ("R",)),
            ("Q'", ("y",)),
            ("R", ("z",)),
            ("R", ("y",)),
            ("P", ("Q", "z", "P'")),
            ("P'", ("x", "y", "P'")),
            ("P'", ("@",)),
        },
        "first_sets": {
            "P": {"w"},
            "P'": {"@", "x"},
            "Q": {"w"},
            "Q'": {"y", "z"},
            "R": {"y", "z"},
        },
        "follow_sets": {
            "P": {"$"},
            "P'": {"$"},
            "Q": {"z"},
            "Q'": {"z"},
            "R": {"z"},
        },
        "ll1_table": {
            ("P", "w", ("Q", "z", "P'")),
            ("P'", "$", ("@",)),
            ("P'", "x", ("x", "y", "P'")),
            ("Q", "w", ("w", "Q'")),
            ("Q'", "y", ("R",)),
            ("Q'", "z", ("R",)),
            ("R", "y", ("y",)),
            ("R", "z", ("z",)),
        },
        "is_ll1": False,
    },
}


def run_test(grammar_file: Path) -> None:
    name = grammar_file.name
    print(f"\n{'='*20} Testing {name} {'='*20}")

    if name not in EXPECTED:
        raise ValueError(f"No expected snapshot configured for {name}")

    expected = EXPECTED[name]

    # 1. 解析文法
    grammar = parse_grammar_file(str(grammar_file))
    print("[1] Parsed grammar.")

    # 2. 消除左递归
    grammar_no_lr = eliminate_left_recursion(grammar)
    actual_lr = _prod_to_set(grammar_no_lr)
    _assert_prod_set(actual_lr, expected["after_left_recursion"], "after left recursion elimination", name)
    print("[2] Left recursion eliminated and validated.")

    # 3. 提取左公因子
    extractor = LeftFactoringExtractor(grammar_no_lr)
    grammar_no_lf = extractor.extract()
    actual_lf = _prod_to_set(grammar_no_lf)
    _assert_prod_set(actual_lf, expected["after_left_factoring"], "after left factoring", name)
    print("[3] Left factoring completed and validated.")

    # 4. 计算 FIRST 和 FOLLOW 集
    calculator = FirstFollowCalculator(grammar_no_lf)
    calculator.compute_first_sets()
    calculator.compute_follow_sets()
    first_sets, follow_sets = _calc_symbol_sets(calculator)

    _assert_symbol_sets(first_sets, expected["first_sets"], "FIRST sets", name)
    _assert_symbol_sets(follow_sets, expected["follow_sets"], "FOLLOW sets", name)
    print("[4] FIRST/FOLLOW sets validated.")

    # 5. 构造 LL(1) 预测分析表
    ll1 = LL1Parser(grammar_no_lf, calculator.first_sets, calculator.follow_sets)
    ll1.build_table()
    table_snapshot = _table_snapshot(ll1)
    _assert_table(table_snapshot, expected["ll1_table"], name)
    if expected.get("is_ll1", True) != ll1.is_ll1:
        raise AssertionError(f"{name}: LL(1) 判定不符，期望 {expected.get('is_ll1', True)}, 实际 {ll1.is_ll1}")
    print("[5] LL(1) table validated.")

    print(f"✅ {name} Passed!")


def main():
    test_dir = Path("test_grammars")
    if not test_dir.exists():
        print(f"Test directory {test_dir} not found!")
        sys.exit(1)

    grammar_files = sorted(test_dir.glob("*.txt"))
    if not grammar_files:
        print("No grammar files found in test_grammars/")
        sys.exit(1)

    print(f"Found {len(grammar_files)} test grammars.")

    failed: List[str] = []
    for grammar_file in grammar_files:
        try:
            run_test(grammar_file)
        except Exception as exc:
            failed.append(grammar_file.name)
            print(f"❌ {grammar_file.name} Failed: {exc}")

    if failed:
        print("\nSome tests failed:", ", ".join(failed))
        sys.exit(1)

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
