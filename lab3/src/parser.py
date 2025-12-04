"""
文法文件解析器。

本模块提供从文本文件解析上下文无关文法的功能。
支持我们约定的文法文件格式，包括注释、非终结符、终结符、开始符号和产生式的解析。
"""

import re
from typing import List, Tuple
from .grammar import Grammar, Symbol, SymbolType, EPSILON


class GrammarParser:
    """
    文法文件解析器类。

    负责解析特定格式的文法文件，并将其转换为 Grammar 对象。
    """

    @staticmethod
    def parse_file(file_path: str) -> Grammar:
        """
        从文件解析文法。

        参数:
            file_path: 文法文件的路径

        返回:
            Grammar对象

        异常:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return GrammarParser.parse_string(content)
        except FileNotFoundError:
            raise FileNotFoundError(f"文法文件不存在: {file_path}")

    @staticmethod
    def parse_string(content: str) -> Grammar:
        """
        从字符串解析文法。

        参数:
            content: 文法文件的字符串内容

        返回:
            Grammar对象

        异常:
            ValueError: 文件格式错误
        """
        # 移除注释（# 后面的内容）
        lines = []
        for line in content.split("\n"):
            # 移除行内注释
            comment_pos = line.find("#")
            if comment_pos != -1:
                line = line[:comment_pos]
            line = line.strip()
            if line:  # 只保留非空行
                lines.append(line)

        if len(lines) < 4:
            raise ValueError("文法文件格式错误：至少需要4行（非终结符、终结符、开始符号、至少一个产生式）")

        grammar = Grammar()

        # 解析非终结符（第1行）
        non_terminal_values = [nt.strip() for nt in lines[0].split(",")]
        for nt_val in non_terminal_values:
            if nt_val:
                grammar.add_symbol(nt_val, SymbolType.NON_TERMINAL)

        # 解析终结符（第2行）
        terminal_values = [t.strip() for t in lines[1].split(",")]
        for t_val in terminal_values:
            if t_val:
                grammar.add_symbol(t_val, SymbolType.TERMINAL)

        # 解析开始符号（第3行）
        start_value = lines[2].strip()
        if start_value not in grammar.symbol_map:
            raise ValueError(f"开始符号 '{start_value}' 未在非终结符中定义")
        start_symbol = grammar.symbol_map[start_value]
        if start_symbol.symbol_type != SymbolType.NON_TERMINAL:
            raise ValueError(f"开始符号 '{start_value}' 必须是非终结符")
        grammar.start_symbol = start_symbol

        # 解析产生式（剩余行）
        for line in lines[3:]:
            GrammarParser._parse_production_line(line, grammar)

        # 验证文法
        GrammarParser._validate_grammar(grammar)

        return grammar

    @staticmethod
    def _parse_production_line(line: str, grammar: Grammar) -> None:
        """
        解析一行产生式。

        支持格式：
        - A -> B C
        - A -> B C | D E
        - A -> @  （@表示空串）

        参数:
            line: 产生式行
            grammar: 文法对象
        """
        # 检查是否有 -> 符号
        if "->" not in line:
            raise ValueError(f"产生式格式错误（缺少 '->'）: {line}")

        # 分割左部和右部
        left_part, right_part = line.split("->", 1)
        left_part = left_part.strip()
        right_part = right_part.strip()

        # 解析左部
        if left_part not in grammar.symbol_map:
            raise ValueError(f"产生式左部 '{left_part}' 未定义")
        left_symbol = grammar.symbol_map[left_part]
        if left_symbol.symbol_type != SymbolType.NON_TERMINAL:
            raise ValueError(f"产生式左部 '{left_part}' 必须是非终结符")

        # 解析右部（支持 | 分隔多个候选式）
        candidates = [c.strip() for c in right_part.split("|")]

        for candidate in candidates:
            if not candidate:
                continue

            # 处理空串
            if candidate == "@":
                grammar.add_production(left_symbol, [EPSILON])
                continue

            # 解析符号序列
            symbols = []
            # 使用正则表达式匹配符号（支持多字符符号）
            # 符号可以是：非终结符（已在符号表中）、终结符（已在符号表中）
            tokens = re.findall(r"\S+", candidate)

            for token in tokens:
                if token in grammar.symbol_map:
                    symbols.append(grammar.symbol_map[token])
                else:
                    # 如果符号未定义，假设它是终结符
                    symbol = grammar.add_symbol(token, SymbolType.TERMINAL)
                    symbols.append(symbol)

            grammar.add_production(left_symbol, symbols)

    @staticmethod
    def _validate_grammar(grammar: Grammar) -> None:
        """
        验证文法的有效性。

        参数:
            grammar: 文法对象

        异常:
            ValueError: 文法无效
        """
        if not grammar.non_terminals:
            raise ValueError("文法必须至少包含一个非终结符")

        if not grammar.terminals:
            raise ValueError("文法必须至少包含一个终结符")

        if grammar.start_symbol is None:
            raise ValueError("文法必须指定开始符号")

        if not grammar.productions:
            raise ValueError("文法必须至少包含一个产生式")

        # 检查所有产生式的符号是否都已定义
        for production in grammar.productions:
            if production.left not in grammar.non_terminals:
                raise ValueError(f"产生式左部未定义: {production.left}")

            for symbol in production.right:
                if symbol != EPSILON and symbol not in grammar.symbol_map.values():
                    raise ValueError(f"产生式中包含未定义的符号: {symbol}")


# 辅助函数，方便直接使用
def parse_grammar_file(file_path: str) -> Grammar:
    """
    从文件解析文法的便捷函数。

    参数:
        file_path: 文法文件的路径

    返回:
        Grammar对象
    """
    return GrammarParser.parse_file(file_path)


def parse_grammar_string(content: str) -> Grammar:
    """
    从字符串解析文法的便捷函数。

    参数:
        content: 文法文件的字符串内容

    返回:
        Grammar对象
    """
    return GrammarParser.parse_string(content)
