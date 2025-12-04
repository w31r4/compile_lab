"""
提取左公因子算法实现（任务3.2）。

本模块提供从上下文无关文法中提取左公因子的功能。
使用Trie树（前缀树）来辅助识别最长公共前缀，确保输出文法无二义性且与输入文法等价。
"""

from __future__ import annotations

from typing import List, Dict, Set, Optional, Tuple
from .grammar import Grammar, Symbol, SymbolType, Production, EPSILON


class TrieNode:
    """
    Trie树节点类，用于构建前缀树。

    属性:
        children: 子节点字典（符号值 -> TrieNode）
        productions: 到达该节点的产生式列表（用于叶子节点）
        is_end: 是否为单词结尾
    """

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.productions: List[Production] = []
        self.is_end: bool = False

    def insert(self, symbols: List[Symbol], production: Production) -> None:
        """
        将符号序列插入Trie树。

        参数:
            symbols: 符号序列
            production: 对应的产生式
        """
        if not symbols:
            self.is_end = True
            self.productions.append(production)
            return

        first_symbol = symbols[0]
        if first_symbol.value not in self.children:
            self.children[first_symbol.value] = TrieNode()

        self.children[first_symbol.value].insert(symbols[1:], production)

    def get_longest_common_prefix(self) -> Tuple[List[Symbol], List[TrieNode]]:
        """
        获取最长公共前缀。

        返回:
            (prefix_symbols, child_nodes)
            prefix_symbols: 公共前缀的符号列表
            child_nodes: 前缀之后的子节点列表
        """
        prefix_symbols = []
        current = self

        while len(current.children) == 1 and not current.is_end:
            symbol_value, child_node = next(iter(current.children.items()))
            prefix_symbols.append(Symbol(symbol_value, SymbolType.TERMINAL))
            current = child_node

        # 收集所有子节点
        child_nodes = list(current.children.values())
        if current.is_end and current.productions:
            # 如果有结束节点，也作为一个子节点
            end_node = TrieNode()
            end_node.productions = current.productions
            end_node.is_end = True
            child_nodes.append(end_node)

        return prefix_symbols, child_nodes


class LeftFactoringExtractor:
    """
    左公因子提取器类。

    提供从文法中提取左公因子的功能，使用Trie树辅助识别最长公共前缀，
    并包含可视化输出。
    """

    def __init__(self, grammar: Grammar):
        """
        初始化左公因子提取器。

        参数:
            grammar: 输入文法
        """
        self.original_grammar = grammar
        self.result_grammar = Grammar()
        self.processing_log = []  # 处理日志，用于可视化输出

    def extract(self) -> Grammar:
        """
        提取文法中的所有左公因子。

        返回:
            提取左公因子后的等价文法
        """
        self.processing_log = []
        self.processing_log.append("=" * 60)
        self.processing_log.append("开始提取左公因子")
        self.processing_log.append("=" * 60)
        self.processing_log.append(f"\n原始文法:")
        self.processing_log.append(str(self.original_grammar))

        # 复制原始文法
        self._copy_grammar()

        # 对每个非终结符提取左公因子
        self._extract_left_factoring()

        self.processing_log.append("\n" + "=" * 60)
        self.processing_log.append("提取左公因子完成")
        self.processing_log.append("=" * 60)
        self.processing_log.append(f"\n结果文法:")
        self.processing_log.append(str(self.result_grammar))

        return self.result_grammar

    def _copy_grammar(self) -> None:
        """复制原始文法到结果文法"""
        self.result_grammar = Grammar()

        # 复制所有符号
        for symbol in self.original_grammar.symbol_map.values():
            self.result_grammar.add_symbol(symbol.value, symbol.symbol_type)

        # 复制所有产生式
        for production in self.original_grammar.productions:
            self.result_grammar.add_production(production.left, production.right.copy())

        # 复制开始符号
        if self.original_grammar.start_symbol:
            start_value = self.original_grammar.start_symbol.value
            self.result_grammar.start_symbol = self.result_grammar.symbol_map[start_value]

    def _extract_left_factoring(self) -> None:
        """
        提取左公因子的主算法。

        对每个非终结符：
        1. 构建其所有产生式右部的Trie树
        2. 查找最长公共前缀
        3. 如果存在公共前缀，提取并创建新的非终结符
        """
        self.processing_log.append("\n" + "-" * 50)
        self.processing_log.append("提取左公因子")
        self.processing_log.append("-" * 50)

        # 获取所有非终结符
        non_terminals = list(self.result_grammar.non_terminals)

        for nt in non_terminals:
            self._process_non_terminal(nt)

    def _process_non_terminal(self, non_terminal: Symbol) -> None:
        """
        处理单个非终结符，提取其产生式的左公因子。

        参数:
            non_terminal: 非终结符
        """
        # 获取该非终结符的所有产生式
        productions = self.result_grammar.get_productions_for(non_terminal)

        if len(productions) < 2:
            return  # 只有一个产生式，无需提取

        self.processing_log.append(f"\n处理非终结符 {non_terminal}:")
        self.processing_log.append(f"  原始产生式:")
        for prod in productions:
            self.processing_log.append(f"    {prod}")

        # 构建Trie树
        trie_root = TrieNode()
        for prod in productions:
            trie_root.insert(prod.right, prod)

        # 查找最长公共前缀
        prefix_symbols, child_nodes = trie_root.get_longest_common_prefix()

        if not prefix_symbols or len(child_nodes) < 2:
            self.processing_log.append(f"  未发现可提取的左公因子")
            return

        # 有公共前缀，需要提取
        self.processing_log.append(f"  发现左公因子: {' '.join(str(s) for s in prefix_symbols)}")

        # 创建新的非终结符
        new_nt_name = f"{non_terminal.value}''"
        # 确保新名称唯一
        while new_nt_name in self.result_grammar.symbol_map:
            new_nt_name += "'"

        new_nt = self.result_grammar.add_symbol(new_nt_name, SymbolType.NON_TERMINAL)
        self.processing_log.append(f"  创建新非终结符: {new_nt}")

        # 移除所有旧产生式
        for prod in productions:
            self.result_grammar.productions.remove(prod)

        # 添加新的产生式：A -> prefix B
        new_right = prefix_symbols + [new_nt]
        self.result_grammar.add_production(non_terminal, new_right)
        self.processing_log.append(f"  添加: {non_terminal} -> {' '.join(str(s) for s in new_right)}")

        # 为每个子节点添加产生式：B -> suffix
        for i, child_node in enumerate(child_nodes):
            if child_node.is_end and child_node.productions:
                # 这是结束节点，对应空串情况
                self.result_grammar.add_production(new_nt, [EPSILON])
                self.processing_log.append(f"  添加: {new_nt} -> {EPSILON}")
            else:
                # 递归处理子节点
                self._process_child_node(new_nt, child_node, i)

    def _process_child_node(self, new_nt: Symbol, node: TrieNode, index: int) -> None:
        """
        处理Trie树的子节点，生成新的产生式。

        参数:
            new_nt: 新创建的非终结符
            node: Trie节点
            index: 子节点索引（用于调试）
        """
        # 收集从该节点开始的所有路径
        paths = self._collect_all_paths(node)

        for path in paths:
            if path:  # 非空路径
                self.result_grammar.add_production(new_nt, path)
                self.processing_log.append(f"  添加: {new_nt} -> {' '.join(str(s) for s in path)}")

    def _collect_all_paths(self, node: TrieNode) -> List[List[Symbol]]:
        """
        收集从Trie节点开始的所有路径。

        参数:
            node: Trie节点

        返回:
            所有路径的列表，每条路径是一个符号列表
        """
        paths = []

        if node.is_end and node.productions:
            # 到达叶子节点
            for prod in node.productions:
                if not prod.right or prod.right == [EPSILON]:
                    paths.append([EPSILON])
                else:
                    paths.append(prod.right)

        for symbol_value, child_node in node.children.items():
            symbol = Symbol(symbol_value, SymbolType.TERMINAL)
            child_paths = self._collect_all_paths(child_node)

            for child_path in child_paths:
                path = [symbol] + child_path
                paths.append(path)

        return paths

    def get_processing_log(self) -> str:
        """
        获取处理日志，用于可视化输出。

        返回:
            处理日志字符串
        """
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
