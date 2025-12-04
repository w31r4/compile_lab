"""
FIRST集和FOLLOW集计算算法实现（任务3.3）。

本模块提供计算上下文无关文法中FIRST集和FOLLOW集的功能。
FIRST集：一个非终结符可以推导出的所有终结符的首符号集合。
FOLLOW集：在某个句型中可以紧跟在某个非终结符后面的终结符集合。
"""

from typing import Dict, Set, List
from .grammar import Grammar, Symbol, SymbolType, EPSILON


class FirstFollowCalculator:
    """
    FIRST集和FOLLOW集计算器类。

    提供计算文法中所有非终结符的FIRST集和FOLLOW集的功能，
    并包含可视化输出。
    """

    def __init__(self, grammar: Grammar):
        """
        初始化计算器。

        参数:
            grammar: 输入文法
        """
        self.grammar = grammar
        self.first_sets: Dict[Symbol, Set[Symbol]] = {}
        self.follow_sets: Dict[Symbol, Set[Symbol]] = {}
        self.first_log = []
        self.follow_log = []

    def clear_log(self):
        """清除日志"""
        self.first_log = []
        self.follow_log = []

    def compute_first_sets(self) -> Dict[Symbol, Set[Symbol]]:
        """
        计算所有非终结符的FIRST集。

        算法：
        1. 初始化：对每个终结符a，FIRST(a) = {a}；对ε，FIRST(ε) = {ε}
        2. 迭代计算非终结符的FIRST集，直到不再变化

        返回:
            FIRST集字典（非终结符 -> 终结符集合）
        """
        self.first_log = []
        self.first_log.append("=" * 60)
        self.first_log.append("开始计算FIRST集")
        self.first_log.append("=" * 60)

        # 步骤1: 初始化
        self._initialize_first_sets()

        # 步骤2: 迭代计算
        self._iterate_first_sets()

        self.first_log.append("\n" + "=" * 60)
        self.first_log.append("FIRST集计算完成")
        self.first_log.append("=" * 60)

        return self.first_sets

    def _initialize_first_sets(self) -> None:
        """初始化FIRST集"""
        self.first_log.append("\n步骤1: 初始化FIRST集")
        self.first_log.append("-" * 40)

        # 初始化终结符的FIRST集
        for terminal in self.grammar.terminals:
            self.first_sets[terminal] = {terminal}
            self.first_log.append(f"  FIRST({terminal}) = {{{terminal}}}")

        # 初始化空串的FIRST集
        self.first_sets[EPSILON] = {EPSILON}
        self.first_log.append(f"  FIRST({EPSILON}) = {{{EPSILON}}}")

        # 初始化非终结符的FIRST集为空集
        for non_terminal in self.grammar.non_terminals:
            self.first_sets[non_terminal] = set()
            self.first_log.append(f"  FIRST({non_terminal}) = {{}}")

    def _iterate_first_sets(self) -> None:
        """迭代计算非终结符的FIRST集"""
        self.first_log.append("\n步骤2: 迭代计算非终结符的FIRST集")
        self.first_log.append("-" * 40)

        iteration = 0
        changed = True

        while changed:
            changed = False
            iteration += 1

            self.first_log.append(f"\n  第 {iteration} 次迭代:")

            for non_terminal in self.grammar.non_terminals:
                old_size = len(self.first_sets[non_terminal])

                # 处理该非终结符的所有产生式
                for production in self.grammar.get_productions_for(non_terminal):
                    self._add_first_from_production(production)

                new_size = len(self.first_sets[non_terminal])
                if new_size > old_size:
                    changed = True
                    self.first_log.append(
                        f"    FIRST({non_terminal}) = {{{self._set_to_str(self.first_sets[non_terminal])}}}"
                    )

        self.first_log.append(f"\n  共迭代 {iteration} 次，FIRST集不再变化")

    def _add_first_from_production(self, production) -> None:
        """
        从产生式添加FIRST集元素。

        对于产生式 A -> X1 X2 ... Xk:
        - 将FIRST(X1)中的所有非ε元素加入FIRST(A)
        - 如果ε ∈ FIRST(X1)，则将FIRST(X2)中的所有非ε元素加入FIRST(A)
        - 如果ε ∈ FIRST(X1), FIRST(X2), ..., FIRST(Xk)，则将ε加入FIRST(A)
        """
        left_nt = production.left
        right_symbols = production.right

        if not right_symbols or right_symbols == [EPSILON]:
            # 产生式右部为空或只有ε
            self.first_sets[left_nt].add(EPSILON)
            return

        # 遍历右部的每个符号
        for i, symbol in enumerate(right_symbols):
            # 添加当前符号的FIRST集（除ε外）
            if symbol in self.first_sets:
                symbol_first = self.first_sets[symbol]
                self.first_sets[left_nt].update(symbol_first - {EPSILON})

            # 如果当前符号的FIRST集不包含ε，停止
            if symbol not in self.first_sets or EPSILON not in self.first_sets[symbol]:
                break

            # 如果所有符号的FIRST集都包含ε，添加ε
            if i == len(right_symbols) - 1:
                self.first_sets[left_nt].add(EPSILON)

    def compute_follow_sets(self) -> Dict[Symbol, Set[Symbol]]:
        """
        计算所有非终结符的FOLLOW集。

        算法：
        1. 初始化：FOLLOW(开始符号) = {$}，其他非终结符的FOLLOW集为空
        2. 迭代应用规则，直到不再变化

        返回:
            FOLLOW集字典（非终结符 -> 终结符集合）
        """
        self.follow_log.append("\n" + "=" * 60)
        self.follow_log.append("开始计算FOLLOW集")
        self.follow_log.append("=" * 60)

        # 步骤1: 初始化
        self._initialize_follow_sets()

        # 步骤2: 迭代计算
        self._iterate_follow_sets()

        self.follow_log.append("\n" + "=" * 60)
        self.follow_log.append("FOLLOW集计算完成")
        self.follow_log.append("=" * 60)

        return self.follow_sets

    def _initialize_follow_sets(self) -> None:
        """初始化FOLLOW集"""
        self.follow_log.append("\n步骤1: 初始化FOLLOW集")
        self.follow_log.append("-" * 40)

        # 创建结束符号
        self.end_symbol = Symbol("$", SymbolType.TERMINAL)
        self.first_sets[self.end_symbol] = {self.end_symbol}

        # 初始化所有非终结符的FOLLOW集为空集
        for non_terminal in self.grammar.non_terminals:
            self.follow_sets[non_terminal] = set()

        # 开始符号的FOLLOW集包含结束符号
        if self.grammar.start_symbol:
            self.follow_sets[self.grammar.start_symbol].add(self.end_symbol)
            self.follow_log.append(f"  FOLLOW({self.grammar.start_symbol}) = {{{self.end_symbol}}} (开始符号)")

        for non_terminal in self.grammar.non_terminals:
            if non_terminal != self.grammar.start_symbol:
                self.follow_log.append(f"  FOLLOW({non_terminal}) = {{}}")

    def _iterate_follow_sets(self) -> None:
        """迭代计算FOLLOW集"""
        self.follow_log.append("\n步骤2: 迭代计算FOLLOW集")
        self.follow_log.append("-" * 40)

        iteration = 0
        changed = True

        while changed:
            changed = False
            iteration += 1

            self.follow_log.append(f"\n  第 {iteration} 次迭代:")

            for production in self.grammar.productions:
                right_symbols = production.right

                # 遍历产生式右部的每个符号
                for i, symbol in enumerate(right_symbols):
                    if symbol.symbol_type != SymbolType.NON_TERMINAL:
                        continue  # 只处理非终结符

                    old_size = len(self.follow_sets[symbol])

                    # 规则1: A -> αBβ
                    # 将FIRST(β)中的所有非ε元素加入FOLLOW(B)
                    if i + 1 < len(right_symbols):
                        beta = right_symbols[i + 1 :]
                        first_of_beta = self._compute_first_of_string(beta)
                        self.follow_sets[symbol].update(first_of_beta - {EPSILON})

                        # 规则2: 如果ε ∈ FIRST(β)，则将FOLLOW(A)加入FOLLOW(B)
                        if EPSILON in first_of_beta:
                            self.follow_sets[symbol].update(self.follow_sets[production.left])
                    else:
                        # 规则3: A -> αB
                        # 将FOLLOW(A)加入FOLLOW(B)
                        self.follow_sets[symbol].update(self.follow_sets[production.left])

                    new_size = len(self.follow_sets[symbol])
                    if new_size > old_size:
                        changed = True
                        self.follow_log.append(
                            f"    FOLLOW({symbol}) = {{{self._set_to_str(self.follow_sets[symbol])}}}"
                        )

        self.follow_log.append(f"\n  共迭代 {iteration} 次，FOLLOW集不再变化")

    def _compute_first_of_string(self, symbols: List[Symbol]) -> Set[Symbol]:
        """
        计算符号串的FIRST集。

        参数:
            symbols: 符号列表

        返回:
            符号串的FIRST集
        """
        if not symbols:
            return {EPSILON}

        result = set()

        for i, symbol in enumerate(symbols):
            if symbol not in self.first_sets:
                # 如果符号未定义，假设它是终结符
                return {symbol}

            symbol_first = self.first_sets[symbol]
            result.update(symbol_first - {EPSILON})

            if EPSILON not in symbol_first:
                break

            if i == len(symbols) - 1:
                result.add(EPSILON)

        return result

    def get_first_sets_str(self) -> str:
        """
        获取FIRST集的字符串表示。

        返回:
            FIRST集的格式化字符串
        """
        lines = ["FIRST集:"]
        lines.append("-" * 40)

        for non_terminal in sorted(self.grammar.non_terminals, key=lambda x: x.value):
            first_set = self.first_sets[non_terminal]
            lines.append(f"  FIRST({non_terminal}) = {{{self._set_to_str(first_set)}}}")

        return "\n".join(lines)

    def get_follow_sets_str(self) -> str:
        """
        获取FOLLOW集的字符串表示。

        返回:
            FOLLOW集的格式化字符串
        """
        lines = ["\nFOLLOW集:"]
        lines.append("-" * 40)

        for non_terminal in sorted(self.grammar.non_terminals, key=lambda x: x.value):
            follow_set = self.follow_sets[non_terminal]
            lines.append(f"  FOLLOW({non_terminal}) = {{{self._set_to_str(follow_set)}}}")

        return "\n".join(lines)

    def _set_to_str(self, symbol_set: Set[Symbol]) -> str:
        """
        将符号集合转换为字符串。

        参数:
            symbol_set: 符号集合

        返回:
            排序后的符号字符串
        """
        return ", ".join(sorted(str(sym) for sym in symbol_set))

    def get_processing_log(self, include_first: bool = True, include_follow: bool = True) -> str:
        """
        获取处理日志，用于可视化输出。

        参数:
            include_first: 是否包含FIRST集日志
            include_follow: 是否包含FOLLOW集日志

        返回:
            处理日志字符串
        """
        log_parts = []
        if include_first:
            log_parts.extend(self.first_log)
        if include_follow:
            log_parts.extend(self.follow_log)
        return "\n".join(log_parts)


def compute_first_sets(grammar: Grammar) -> Dict[Symbol, Set[Symbol]]:
    """
    计算FIRST集的便捷函数。

    参数:
        grammar: 输入文法

    返回:
        FIRST集字典
    """
    calculator = FirstFollowCalculator(grammar)
    return calculator.compute_first_sets()


def compute_follow_sets(grammar: Grammar) -> Dict[Symbol, Set[Symbol]]:
    """
    计算FOLLOW集的便捷函数。

    参数:
        grammar: 输入文法

    返回:
        FOLLOW集字典
    """
    calculator = FirstFollowCalculator(grammar)
    # 先计算FIRST集
    calculator.compute_first_sets()
    return calculator.compute_follow_sets()
