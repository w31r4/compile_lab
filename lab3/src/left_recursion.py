"""
消除左递归算法实现（任务3.1）。

本模块提供消除上下文无关文法中直接和间接左递归的功能。
算法步骤：
1. 对非终结符进行排序
2. 按顺序处理每个非终结符，通过代换将间接左递归转换为直接左递归
3. 消除直接左递归
"""

from typing import List

from .grammar import Grammar, Symbol, SymbolType, Production, EPSILON


class LeftRecursionEliminator:
    """
    左递归消除器类。

    提供消除文法中直接和间接左递归的功能，并包含可视化输出。
    """

    def __init__(self, grammar: Grammar):
        """
        初始化左递归消除器。

        参数:
            grammar: 输入文法
        """
        self.original_grammar = grammar
        self.result_grammar = Grammar()
        self.processing_log: List[str] = []

    def eliminate(self) -> Grammar:
        """
        消除文法中的所有左递归。

        返回:
            消除左递归后的等价文法
        """
        self.processing_log = []
        self.processing_log.append("=" * 60)
        self.processing_log.append("开始消除左递归")
        self.processing_log.append("=" * 60)
        self.processing_log.append("\n原始文法:")
        self.processing_log.append(str(self.original_grammar))

        # 步骤1: 复制原始文法
        self._copy_grammar()

        # 步骤2: 对非终结符排序
        sorted_non_terminals = sorted(list(self.result_grammar.non_terminals), key=lambda x: x.value)
        self.processing_log.append(f"\n非终结符排序: {[str(nt) for nt in sorted_non_terminals]}")

        # 步骤3: 消除间接左递归
        self._eliminate_indirect_left_recursion(sorted_non_terminals)

        # 步骤4: 消除直接左递归
        self._eliminate_direct_left_recursion(sorted_non_terminals)

        self.processing_log.append("\n" + "=" * 60)
        self.processing_log.append("消除左递归完成")
        self.processing_log.append("=" * 60)
        self.processing_log.append("\n结果文法:")
        self.processing_log.append(str(self.result_grammar))

        return self.result_grammar

    def _copy_grammar(self) -> None:
        """复制原始文法到结果文法"""
        self.result_grammar = Grammar()

        for symbol in self.original_grammar.symbol_map.values():
            self.result_grammar.add_symbol(symbol.value, symbol.symbol_type)

        for production in self.original_grammar.productions:
            self.result_grammar.add_production(production.left, production.right.copy())

        if self.original_grammar.start_symbol:
            start_value = self.original_grammar.start_symbol.value
            self.result_grammar.start_symbol = self.result_grammar.symbol_map[start_value]

    def _eliminate_indirect_left_recursion(self, sorted_non_terminals: List[Symbol]) -> None:
        """
        消除间接左递归。

        算法：对于每个非终结符Ai，检查其产生式是否以排在前面的非终结符Aj开头，
        如果是，则进行代换。
        """
        self.processing_log.append("\n" + "-" * 50)
        self.processing_log.append("步骤3: 消除间接左递归")
        self.processing_log.append("-" * 50)

        for i, ai in enumerate(sorted_non_terminals):
            self.processing_log.append(f"\n处理非终结符 {ai}:")
            ai_productions = self.result_grammar.get_productions_for(ai)

            new_productions: List[Production] = []
            productions_to_remove: List[Production] = []

            for prod in ai_productions:
                if not prod.right or prod.right[0] == EPSILON:
                    continue

                first_symbol = prod.right[0]
                if first_symbol.symbol_type != SymbolType.NON_TERMINAL:
                    continue

                try:
                    j = sorted_non_terminals.index(first_symbol)
                except ValueError:
                    continue

                if j < i:
                    self.processing_log.append(f"  发现间接左递归: {prod}")
                    self.processing_log.append(f"  用 {first_symbol} 的产生式进行代换:")

                    productions_to_remove.append(prod)
                    aj_productions = self.result_grammar.get_productions_for(first_symbol)

                    for aj_prod in aj_productions:
                        new_right = aj_prod.right + prod.right[1:]
                        new_prod = Production(ai, new_right)
                        new_productions.append(new_prod)
                        self.processing_log.append(f"    -> {new_prod}")

            for prod in productions_to_remove:
                self.result_grammar.productions.remove(prod)

            for prod in new_productions:
                self.result_grammar.add_production(prod.left, prod.right)

    def _eliminate_direct_left_recursion(self, sorted_non_terminals: List[Symbol]) -> None:
        """
        消除直接左递归。

        对每个Ai:
        - 非左递归产生式 Ai -> β 转换为 Ai -> βAi'
        - 左递归产生式 Ai -> Aiα 转换为 Ai' -> αAi'
        - 添加 Ai' -> ε
        - 若 Ai 仅有左递归产生式，额外添加 Ai -> Ai'
        """
        self.processing_log.append("\n" + "-" * 50)
        self.processing_log.append("步骤4: 消除直接左递归")
        self.processing_log.append("-" * 50)

        for ai in sorted_non_terminals:
            ai_productions = self.result_grammar.get_productions_for(ai)

            left_recursive_prods = [p for p in ai_productions if p.right and p.right[0] == ai]
            non_left_recursive_prods = [p for p in ai_productions if not (p.right and p.right[0] == ai)]

            if not left_recursive_prods:
                continue

            self.processing_log.append(f"\n处理非终结符 {ai}:")
            self.processing_log.append(f"  发现 {len(left_recursive_prods)} 个左递归产生式")

            new_nt_name = f"{ai.value}'"
            while new_nt_name in self.result_grammar.symbol_map:
                new_nt_name += "'"
            new_nt = self.result_grammar.add_symbol(new_nt_name, SymbolType.NON_TERMINAL)
            self.processing_log.append(f"  创建新非终结符: {new_nt}")

            for prod in ai_productions:
                self.result_grammar.productions.remove(prod)

            if non_left_recursive_prods:
                for prod in non_left_recursive_prods:
                    beta = prod.right
                    if not beta or (len(beta) == 1 and beta[0] == EPSILON):
                        new_right = [new_nt]
                    else:
                        new_right = beta + [new_nt]
                    self.result_grammar.add_production(ai, new_right)
                    self.processing_log.append(f"  转换: {prod} -> {ai} -> {' '.join(str(s) for s in new_right)}")
            else:
                self.result_grammar.add_production(ai, [new_nt])
                self.processing_log.append(f"  添加: {ai} -> {new_nt}")

            for prod in left_recursive_prods:
                alpha = prod.right[1:]
                if len(alpha) == 1 and alpha[0] == EPSILON:
                    alpha = []
                new_right = alpha + [new_nt]
                self.result_grammar.add_production(new_nt, new_right)
                self.processing_log.append(f"  添加: {new_nt} -> {' '.join(str(s) for s in new_right)}")

            self.result_grammar.add_production(new_nt, [EPSILON])
            self.processing_log.append(f"  添加: {new_nt} -> {EPSILON}")

    def get_processing_log(self) -> str:
        """获取处理日志，用于可视化输出。"""
        return "\n".join(self.processing_log)


def eliminate_left_recursion(grammar: Grammar) -> Grammar:
    """
    消除文法左递归的便捷函数。

    参数:
        grammar: 输入文法

    返回:
        消除左递归后的文法
    """
    eliminator = LeftRecursionEliminator(grammar)
    return eliminator.eliminate()
