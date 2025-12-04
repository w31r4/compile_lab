"""
提取左公因子算法实现（任务3.2）。

本模块使用基于首符号分组的递归算法来提取左公因子，避免原有 Trie 实现
在重建剩余部分时的复杂性和潜在错误。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .grammar import Grammar, Symbol, SymbolType, Production, EPSILON


class LeftFactoringExtractor:
    """
    左公因子提取器。

    对每个非终结符按产生式右部的首符号分组，若某组含有多个产生式则提取该首符号
    为公因子并引入新的非终结符，递归处理剩余部分。
    """

    def __init__(self, grammar: Grammar):
        self.original_grammar = grammar
        self.result_grammar = Grammar()
        self.processing_log: List[str] = []

    def extract(self) -> Grammar:
        """提取文法中的所有左公因子。"""
        self.processing_log = []
        self.processing_log.append("=" * 60)
        self.processing_log.append("开始提取左公因子")
        self.processing_log.append("=" * 60)
        self.processing_log.append("\n原始文法:")
        self.processing_log.append(str(self.original_grammar))

        self._copy_grammar()
        self._extract_left_factoring()

        self.processing_log.append("\n" + "=" * 60)
        self.processing_log.append("提取左公因子完成")
        self.processing_log.append("=" * 60)
        self.processing_log.append("\n结果文法:")
        self.processing_log.append(str(self.result_grammar))

        return self.result_grammar

    def _copy_grammar(self) -> None:
        """复制原始文法到结果文法。"""
        self.result_grammar = Grammar()

        for symbol in self.original_grammar.symbol_map.values():
            self.result_grammar.add_symbol(symbol.value, symbol.symbol_type)

        for production in self.original_grammar.productions:
            self.result_grammar.add_production(production.left, production.right.copy())

        if self.original_grammar.start_symbol:
            start_value = self.original_grammar.start_symbol.value
            self.result_grammar.start_symbol = self.result_grammar.symbol_map[start_value]

    def _extract_left_factoring(self) -> None:
        """遍历所有非终结符并进行左公因子提取。"""
        self.processing_log.append("\n" + "-" * 50)
        self.processing_log.append("提取左公因子")
        self.processing_log.append("-" * 50)

        queue = list(self.result_grammar.non_terminals)
        processed = set()

        while queue:
            nt = queue.pop(0)
            if nt in processed:
                continue

            new_non_terminals = self._process_non_terminal(nt)
            processed.add(nt)
            queue.extend(new_non_terminals)

    def _process_non_terminal(self, non_terminal: Symbol) -> List[Symbol]:
        """
        对单个非终结符进行左公因子提取。

        返回新创建的非终结符列表，供后续递归处理。
        """
        productions = self.result_grammar.get_productions_for(non_terminal)
        if len(productions) < 2:
            return []

        self.processing_log.append(f"\n处理非终结符 {non_terminal}:")
        self.processing_log.append("  原始产生式:")
        for prod in productions:
            self.processing_log.append(f"    {prod}")

        groups = self._group_by_first_symbol(productions)
        factoring_groups = {k: v for k, v in groups.items() if k is not None and len(v) > 1}

        if not factoring_groups:
            self.processing_log.append("  未发现可提取的左公因子")
            return []

        new_non_terminals: List[Symbol] = []
        productions_to_remove: List[Production] = []
        productions_to_add: List[tuple[Symbol, List[Symbol]]] = []

        for first_symbol, group in factoring_groups.items():
            new_nt = self._create_new_non_terminal(non_terminal)
            new_non_terminals.append(new_nt)
            productions_to_remove.extend(group)

            new_right = [first_symbol, new_nt]
            productions_to_add.append((non_terminal, new_right))
            self.processing_log.append(f"  发现左公因子: {first_symbol}")
            self.processing_log.append(f"  创建新非终结符: {new_nt}")
            self.processing_log.append(f"  替换为: {non_terminal} -> {' '.join(str(s) for s in new_right)}")

            for prod in group:
                remainder = prod.right[1:]
                if not remainder or remainder == [EPSILON]:
                    remainder = [EPSILON]
                productions_to_add.append((new_nt, remainder))
                self.processing_log.append(
                    f"    {non_terminal} 的产生式 {prod} 贡献: {new_nt} -> {' '.join(str(s) for s in remainder)}"
                )

        for prod in productions_to_remove:
            if prod in self.result_grammar.productions:
                self.result_grammar.productions.remove(prod)

        for left, right in productions_to_add:
            self.result_grammar.add_production(left, right)

        return new_non_terminals

    @staticmethod
    def _group_by_first_symbol(
        productions: List[Production],
    ) -> Dict[Optional[Symbol], List[Production]]:
        """按右部首符号分组产生式，空产生式归为 None 组。"""
        groups: Dict[Optional[Symbol], List[Production]] = {}
        for prod in productions:
            if not prod.right or prod.right == [EPSILON]:
                key = None
            else:
                key = prod.right[0]
            groups.setdefault(key, []).append(prod)
        return groups

    def _create_new_non_terminal(self, base: Symbol) -> Symbol:
        """基于原非终结符生成唯一的新非终结符。"""
        new_name = f"{base.value}'"
        while new_name in self.result_grammar.symbol_map:
            new_name += "'"
        return self.result_grammar.add_symbol(new_name, SymbolType.NON_TERMINAL)

    def get_processing_log(self) -> str:
        """获取处理日志，用于可视化输出。"""
        return "\n".join(self.processing_log)


def extract_left_factoring(grammar: Grammar) -> Grammar:
    """
    提取文法左公因子的便捷函数。

    参数:
        grammar: 输入文法

    返回:
        提取左公因子后的文法
    """
    extractor = LeftFactoringExtractor(grammar)
    return extractor.extract()
