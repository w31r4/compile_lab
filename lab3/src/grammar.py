"""
上下文无关文法的核心数据结构定义。

本模块定义了用于表示上下文无关文法（CFG）的核心类，包括文法符号、产生式和文法本身。
这些类为后续的语法分析算法（消除左递归、提取左公因子、计算FIRST/FOLLOW集等）提供了统一的数据接口。
"""

from dataclasses import dataclass, field
from typing import Set, List, Dict, Optional, Union
from enum import Enum


class SymbolType(Enum):
    """文法符号类型枚举"""

    TERMINAL = "terminal"  # 终结符
    NON_TERMINAL = "non_terminal"  # 非终结符
    EPSILON = "epsilon"  # 空串


@dataclass
class Symbol:
    """
    文法符号类，表示终结符、非终结符或空串。

    属性:
        value: 符号的值（如 'E', '+', 'id'）
        symbol_type: 符号的类型（终结符、非终结符或空串）
    """

    value: str
    symbol_type: SymbolType

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash((self.value, self.symbol_type))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Symbol):
            return False
        return self.value == other.value and self.symbol_type == other.symbol_type


# 预定义空串符号
EPSILON = Symbol("@", SymbolType.EPSILON)


@dataclass
class Production:
    """
    产生式类，表示文法的一条规则。

    属性:
        left: 产生式左部（非终结符）
        right: 产生式右部（符号列表）

    示例:
        E -> T E'  表示为 Production(left=E, right=[T, E'])
    """

    left: Symbol
    right: List[Symbol]

    def __str__(self) -> str:
        right_str = " ".join(str(sym) for sym in self.right) if self.right else str(EPSILON)
        return f"{self.left} -> {right_str}"

    def __hash__(self) -> int:
        return hash((self.left, tuple(self.right)))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Production):
            return False
        return self.left == other.left and self.right == other.right


@dataclass
class Grammar:
    """
    上下文无关文法类，包含文法的所有组成部分。

    属性:
        non_terminals: 非终结符集合（Symbol集合）
        terminals: 终结符集合（Symbol集合）
        productions: 产生式列表
        start_symbol: 开始符号（非终结符）
        symbol_map: 符号映射，用于快速查找符号（值 -> Symbol对象）

    注意:
        - 所有符号（包括终结符和非终结符）都存储在 symbol_map 中，便于快速查找和去重
        - 空串 '@' 被视为特殊的终结符
    """

    non_terminals: Set[Symbol] = field(default_factory=set)
    terminals: Set[Symbol] = field(default_factory=set)
    productions: List[Production] = field(default_factory=list)
    start_symbol: Optional[Symbol] = None
    symbol_map: Dict[str, Symbol] = field(default_factory=dict)

    def add_symbol(self, value: str, symbol_type: SymbolType) -> Symbol:
        """
        添加一个符号到文法中。

        如果符号已存在，则返回已存在的符号对象；
        否则创建新符号并添加到相应的集合中。

        参数:
            value: 符号的值
            symbol_type: 符号的类型

        返回:
            Symbol对象
        """
        if value in self.symbol_map:
            return self.symbol_map[value]

        symbol = Symbol(value, symbol_type)
        self.symbol_map[value] = symbol

        if symbol_type == SymbolType.NON_TERMINAL:
            self.non_terminals.add(symbol)
        elif symbol_type == SymbolType.TERMINAL:
            self.terminals.add(symbol)

        return symbol

    def add_production(self, left: Symbol, right: List[Symbol]) -> None:
        """
        添加一个产生式到文法中。

        参数:
            left: 产生式左部（必须是非终结符）
            right: 产生式右部（符号列表）
        """
        if left.symbol_type != SymbolType.NON_TERMINAL:
            raise ValueError(f"产生式左部必须是终结符: {left}")

        production = Production(left, right)
        self.productions.append(production)

    def get_productions_for(self, non_terminal: Symbol) -> List[Production]:
        """
        获取指定非终结符的所有产生式。

        参数:
            non_terminal: 非终结符

        返回:
            该非终结符的所有产生式列表
        """
        return [p for p in self.productions if p.left == non_terminal]

    def __str__(self) -> str:
        """返回文法的字符串表示"""
        lines = []
        lines.append("上下文无关文法:")
        lines.append(f"  非终结符: {{{', '.join(sorted(str(nt) for nt in self.non_terminals))}}}")
        lines.append(f"  终结符: {{{', '.join(sorted(str(t) for t in self.terminals))}}}")
        lines.append(f"  开始符号: {self.start_symbol}")
        lines.append("\n  产生式:")
        for production in self.productions:
            lines.append(f"    {production}")

        return "\n".join(lines)
