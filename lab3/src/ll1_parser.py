"""
LL(1) 分析表构造与预测分析（任务3.4）。

基于已消除左递归、完成左公因子提取的文法，以及 FIRST/FOLLOW 集，
构造 LL(1) 预测分析表并判断文法是否为 LL(1)。
"""

from __future__ import annotations

from typing import Dict, List, Set

from .grammar import Grammar, Symbol, SymbolType, Production, EPSILON
from .first_follow import FirstFollowCalculator


TableType = Dict[Symbol, Dict[str, Production]]


class LL1Parser:
    """LL(1) 分析表构造器。"""

    def __init__(
        self,
        grammar: Grammar,
        first_sets: Dict[Symbol, Set[Symbol]] | None = None,
        follow_sets: Dict[Symbol, Set[Symbol]] | None = None,
    ):
        self.grammar = grammar
        self.first_sets: Dict[Symbol, Set[Symbol]]
        self.follow_sets: Dict[Symbol, Set[Symbol]]

        if first_sets is None or follow_sets is None:
            calculator = FirstFollowCalculator(grammar)
            calculator.compute_first_sets()
            calculator.compute_follow_sets()
            self.first_sets = calculator.first_sets
            self.follow_sets = calculator.follow_sets
        else:
            self.first_sets = first_sets
            self.follow_sets = follow_sets

        self.table: TableType = {}
        self.conflicts: List[str] = []
        self.is_ll1: bool = True

    def build_table(self) -> TableType:
        """构造预测分析表，并记录冲突信息。"""
        self.table = {}
        self.conflicts = []
        self.is_ll1 = True

        for production in self.grammar.productions:
            first_alpha = self._first_of_string(production.right)

            for terminal in (sym for sym in first_alpha if sym != EPSILON):
                self._add_entry(production.left, terminal.value, production)

            if EPSILON in first_alpha:
                for follow_sym in self.follow_sets.get(production.left, set()):
                    if follow_sym.symbol_type != SymbolType.TERMINAL:
                        continue
                    self._add_entry(production.left, follow_sym.value, production)

        return self.table

    def _add_entry(self, non_terminal: Symbol, terminal_value: str, production: Production) -> None:
        """在表中添加一项，若发生冲突则记录并标记为非 LL(1)。"""
        row = self.table.setdefault(non_terminal, {})
        if terminal_value in row and row[terminal_value] != production:
            self.is_ll1 = False
            conflict_msg = (
                f"冲突: M[{non_terminal}, {terminal_value}] 已有 {row[terminal_value]}, "
                f"再次填入 {production}"
            )
            self.conflicts.append(conflict_msg)
        else:
            row[terminal_value] = production

    def _first_of_string(self, symbols: List[Symbol]) -> Set[Symbol]:
        """计算符号串的 FIRST 集（使用已知 FIRST 集）。"""
        if not symbols:
            return {EPSILON}

        result: Set[Symbol] = set()

        for i, symbol in enumerate(symbols):
            if symbol == EPSILON:
                result.add(EPSILON)
                break

            symbol_first = self.first_sets.get(symbol, {symbol})
            result.update(symbol_first - {EPSILON})

            if EPSILON not in symbol_first:
                break

            if i == len(symbols) - 1:
                result.add(EPSILON)

        return result

    def format_table(self) -> str:
        """将表格格式化为字符串，便于展示。"""
        terminals = self._collect_terminals()
        header = ["NT"] + terminals
        lines = ["\t".join(header)]

        for nt in sorted(self.grammar.non_terminals, key=lambda x: x.value):
            row_cells = [nt.value]
            table_row = self.table.get(nt, {})
            for t in terminals:
                prod = table_row.get(t)
                if prod:
                    rhs = " ".join(str(sym) for sym in prod.right) if prod.right else str(EPSILON)
                    row_cells.append(rhs)
                else:
                    row_cells.append("")
            lines.append("\t".join(row_cells))

        status_line = "该文法是 LL(1)" if self.is_ll1 else "该文法不是 LL(1)"
        if self.conflicts:
            lines.append("\n冲突: " + "; ".join(self.conflicts))
        lines.append(status_line)
        return "\n".join(lines)

    def _collect_terminals(self) -> List[str]:
        """收集表头所需的终结符（包括 $）。"""
        terms: Set[str] = {t.value for t in self.grammar.terminals}
        for follows in self.follow_sets.values():
            for sym in follows:
                if sym.symbol_type == SymbolType.TERMINAL:
                    terms.add(sym.value)
        terms.discard(EPSILON.value)
        return sorted(terms)


def build_ll1_parsing_table(
    grammar: Grammar,
    first_sets: Dict[Symbol, Set[Symbol]] | None = None,
    follow_sets: Dict[Symbol, Set[Symbol]] | None = None,
) -> LL1Parser:
    """便捷函数：构造 LL(1) 预测分析表。"""
    parser = LL1Parser(grammar, first_sets=first_sets, follow_sets=follow_sets)
    parser.build_table()
    return parser
