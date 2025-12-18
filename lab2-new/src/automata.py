# -*- coding: utf-8 -*-
"""
Regex -> NFA -> DFA -> MinDFA

算法：
  (1) Thompson 构造：正则表达式 -> NFA
  (2) 子集构造：NFA -> DFA
  (3) 表填充算法：DFA 最小化
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


EPSILON: Optional[str] = None  # 用 None 表示 ε 转移


@dataclass
class NFA:
    start: int
    accept: int
    transitions: Dict[int, Dict[Optional[str], Set[int]]]


@dataclass
class DFA:
    start: int
    accepts: Set[int]
    transitions: Dict[int, Dict[str, int]]
    alphabet: Set[str]
    state_sets: Dict[int, FrozenSet[int]] | None = None  # DFA 状态 -> NFA 状态集合（可选）


def add_edge(transitions: Dict[int, Dict[Optional[str], Set[int]]], src: int, symbol: Optional[str], dst: int) -> None:
    if src not in transitions:
        transitions[src] = {}
    if symbol not in transitions[src]:
        transitions[src][symbol] = set()
    transitions[src][symbol].add(dst)


def merge_transitions(
    t1: Dict[int, Dict[Optional[str], Set[int]]],
    t2: Dict[int, Dict[Optional[str], Set[int]]],
) -> Dict[int, Dict[Optional[str], Set[int]]]:
    transitions: Dict[int, Dict[Optional[str], Set[int]]] = {}
    for src, mapping in t1.items():
        transitions[src] = {sym: set(dsts) for sym, dsts in mapping.items()}
    for src, mapping in t2.items():
        if src not in transitions:
            transitions[src] = {}
        for sym, dsts in mapping.items():
            transitions[src].setdefault(sym, set()).update(dsts)
    return transitions


def insert_concat(regex: str) -> str:
    """
    在正则中插入显式连接符 '.'，例如：
      a(b|c)*d  ->  a.(b|c)*.d
    """
    result: List[str] = []
    for i, c in enumerate(regex):
        result.append(c)
        if c == " ":
            continue

        # 当前字符如果是：普通字符、右括号、星号 -> 之后可能需要接连接符
        if c not in ["|", "("]:
            if i + 1 < len(regex):
                d = regex[i + 1]
                if d == " ":
                    continue
                # 后一个字符如果是：普通字符 或 '(' -> 中间插入 '.'
                if d not in ["|", ")", "*"]:
                    result.append(".")
    return "".join(result)


def regex_to_postfix(regex: str) -> str:
    """中缀正则 -> 后缀表达式（* 最高，. 其次，| 最低）"""
    precedence = {"*": 3, ".": 2, "|": 1}
    output: List[str] = []
    stack: List[str] = []
    regex = insert_concat(regex)

    for c in regex:
        if c == " ":
            continue
        if c == "(":
            stack.append(c)
        elif c == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if stack and stack[-1] == "(":
                stack.pop()
        elif c in precedence:
            while stack and stack[-1] != "(" and precedence.get(stack[-1], 0) >= precedence[c]:
                output.append(stack.pop())
            stack.append(c)
        else:
            output.append(c)

    while stack:
        output.append(stack.pop())
    return "".join(output)


def postfix_to_nfa(postfix: str) -> NFA:
    """根据后缀正则，用 Thompson 构造 NFA。"""
    state_id = 0

    def new_state() -> int:
        nonlocal state_id
        s = state_id
        state_id += 1
        return s

    stack: List[NFA] = []
    ops = {"*", ".", "|"}

    for c in postfix:
        if c not in ops:
            start = new_state()
            accept = new_state()
            trans: Dict[int, Dict[Optional[str], Set[int]]] = {}
            add_edge(trans, start, c, accept)
            stack.append(NFA(start, accept, trans))
            continue

        if c == "*":
            nfa1 = stack.pop()
            start = new_state()
            accept = new_state()
            trans = merge_transitions(nfa1.transitions, {})
            add_edge(trans, start, EPSILON, nfa1.start)
            add_edge(trans, start, EPSILON, accept)
            add_edge(trans, nfa1.accept, EPSILON, nfa1.start)
            add_edge(trans, nfa1.accept, EPSILON, accept)
            stack.append(NFA(start, accept, trans))
            continue

        if c == ".":
            nfa2 = stack.pop()
            nfa1 = stack.pop()
            start = nfa1.start
            accept = nfa2.accept
            trans = merge_transitions(nfa1.transitions, nfa2.transitions)
            add_edge(trans, nfa1.accept, EPSILON, nfa2.start)
            stack.append(NFA(start, accept, trans))
            continue

        if c == "|":
            nfa2 = stack.pop()
            nfa1 = stack.pop()
            start = new_state()
            accept = new_state()
            trans = merge_transitions(nfa1.transitions, nfa2.transitions)
            add_edge(trans, start, EPSILON, nfa1.start)
            add_edge(trans, start, EPSILON, nfa2.start)
            add_edge(trans, nfa1.accept, EPSILON, accept)
            add_edge(trans, nfa2.accept, EPSILON, accept)
            stack.append(NFA(start, accept, trans))
            continue

    if len(stack) != 1:
        raise ValueError("非法正则表达式：无法构造 NFA")
    return stack[0]


def regex_to_nfa(regex: str) -> NFA:
    postfix = regex_to_postfix(regex)
    return postfix_to_nfa(postfix)


def epsilon_closure(transitions: Dict[int, Dict[Optional[str], Set[int]]], states: Iterable[int]) -> FrozenSet[int]:
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for tgt in transitions.get(s, {}).get(EPSILON, set()):
            if tgt not in closure:
                closure.add(tgt)
                stack.append(tgt)
    return frozenset(closure)


def move(transitions: Dict[int, Dict[Optional[str], Set[int]]], states: Iterable[int], symbol: str) -> Set[int]:
    result: Set[int] = set()
    for s in states:
        for tgt in transitions.get(s, {}).get(symbol, set()):
            result.add(tgt)
    return result


def nfa_to_dfa(nfa: NFA) -> DFA:
    transitions = nfa.transitions

    alphabet: Set[str] = set()
    for mapping in transitions.values():
        for sym in mapping:
            if sym is not EPSILON:
                alphabet.add(sym)  # type: ignore[arg-type]

    start_closure = epsilon_closure(transitions, {nfa.start})
    dfa_states: Dict[FrozenSet[int], int] = {start_closure: 0}
    dfa_start = 0
    dfa_accepts: Set[int] = set()
    dfa_trans: Dict[int, Dict[str, int]] = {}
    queue: deque[FrozenSet[int]] = deque([start_closure])

    if nfa.accept in start_closure:
        dfa_accepts.add(dfa_start)

    while queue:
        S = queue.popleft()
        s_id = dfa_states[S]
        dfa_trans.setdefault(s_id, {})

        for sym in alphabet:
            U = epsilon_closure(transitions, move(transitions, S, sym))
            if not U:
                continue
            if U not in dfa_states:
                new_id = len(dfa_states)
                dfa_states[U] = new_id
                queue.append(U)
                if nfa.accept in U:
                    dfa_accepts.add(new_id)
            dfa_trans[s_id][sym] = dfa_states[U]

    dfa = DFA(start=dfa_start, accepts=dfa_accepts, transitions=dfa_trans, alphabet=alphabet)
    dfa.state_sets = {id_: state_set for state_set, id_ in dfa_states.items()}
    return dfa


def reachable_states(dfa: DFA) -> Set[int]:
    visited: Set[int] = set()
    stack = [dfa.start]
    while stack:
        s = stack.pop()
        if s in visited:
            continue
        visited.add(s)
        for nxt in dfa.transitions.get(s, {}).values():
            stack.append(nxt)
    return visited


def make_total_dfa(dfa: DFA) -> Set[int]:
    states = reachable_states(dfa)
    for s in states:
        dfa.transitions.setdefault(s, {})

    sink: Optional[int] = None
    if dfa.alphabet:
        for s in list(states):
            for a in dfa.alphabet:
                if a not in dfa.transitions[s]:
                    if sink is None:
                        sink = max(states) + 1 if states else 0
                        dfa.transitions[sink] = {}
                        for a2 in dfa.alphabet:
                            dfa.transitions[sink][a2] = sink
                    dfa.transitions[s][a] = sink
        if sink is not None:
            states.add(sink)
    return states


def minimize_dfa(dfa: DFA) -> DFA:
    states = make_total_dfa(dfa)
    states_sorted = sorted(states)
    idx = {s: i for i, s in enumerate(states_sorted)}
    n = len(states_sorted)

    table = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            if (states_sorted[i] in dfa.accepts) != (states_sorted[j] in dfa.accepts):
                table[i][j] = True

    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i):
                if table[i][j]:
                    continue
                si, sj = states_sorted[i], states_sorted[j]
                for a in dfa.alphabet:
                    p = dfa.transitions[si][a]
                    q = dfa.transitions[sj][a]
                    if p == q:
                        continue
                    ii = idx[max(p, q)]
                    jj = idx[min(p, q)]
                    if table[ii][jj]:
                        table[i][j] = True
                        changed = True
                        break

    parent = {s: s for s in states_sorted}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i):
            if not table[i][j]:
                union(states_sorted[i], states_sorted[j])

    reps: Dict[int, int] = {}
    new_id = 0
    for s in states_sorted:
        r = find(s)
        if r not in reps:
            reps[r] = new_id
            new_id += 1

    state_map = {s: reps[find(s)] for s in states_sorted}

    new_trans: Dict[int, Dict[str, int]] = {}
    for s in states_sorted:
        ns = state_map[s]
        new_trans.setdefault(ns, {})
        for a, t in dfa.transitions.get(s, {}).items():
            new_trans[ns][a] = state_map[t]

    new_accepts = {state_map[s] for s in dfa.accepts if s in state_map}
    new_start = state_map[dfa.start]
    return DFA(start=new_start, accepts=new_accepts, transitions=new_trans, alphabet=set(dfa.alphabet))


def dfa_match(dfa: DFA, s: str) -> bool:
    state: Optional[int] = dfa.start
    for ch in s:
        if ch not in dfa.alphabet:
            return False
        if state is None:
            return False
        state = dfa.transitions.get(state, {}).get(ch)
        if state is None:
            return False
    return state in dfa.accepts


def trace_dfa(dfa: DFA, s: str) -> Tuple[List[int], bool, str]:
    path = [dfa.start]
    state = dfa.start
    for ch in s:
        if ch not in dfa.alphabet:
            return path, False, f"字符 {ch!r} 不在字母表中"
        nxt = dfa.transitions.get(state, {}).get(ch)
        if nxt is None:
            return path, False, f"在状态 {state} 上遇到 {ch!r} 没有转移"
        path.append(nxt)
        state = nxt
    return path, state in dfa.accepts, "匹配完成"


def collect_nfa_alphabet(nfa: NFA) -> Set[str]:
    return {sym for m in nfa.transitions.values() for sym in m if sym is not EPSILON}  # type: ignore[misc]


def trace_nfa(nfa: NFA, s: str) -> Tuple[List[int], bool, str]:
    alphabet = collect_nfa_alphabet(nfa)
    for ch in s:
        if ch not in alphabet:
            return [nfa.start], False, f"字符 {ch!r} 不在字母表中"

    start = (nfa.start, 0)
    queue: deque[Tuple[int, int]] = deque([start])
    visited = {start}
    prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
    last = start

    def build_path(node: Tuple[int, int]) -> List[int]:
        seq: List[int] = []
        cur: Optional[Tuple[int, int]] = node
        while cur is not None:
            seq.append(cur[0])
            cur = prev.get(cur)
        return list(reversed(seq))

    while queue:
        state, pos = queue.popleft()
        last = (state, pos)
        if state == nfa.accept and pos == len(s):
            return build_path((state, pos)), True, "找到接受路径"

        for nxt in nfa.transitions.get(state, {}).get(EPSILON, set()):
            node = (nxt, pos)
            if node not in visited:
                visited.add(node)
                prev[node] = (state, pos)
                queue.append(node)

        if pos < len(s):
            ch = s[pos]
            for nxt in nfa.transitions.get(state, {}).get(ch, set()):
                node = (nxt, pos + 1)
                if node not in visited:
                    visited.add(node)
                    prev[node] = (state, pos)
                    queue.append(node)

    return build_path(last), False, "未找到接受路径"


def nfa_match(nfa: NFA, s: str) -> bool:
    _, ok, _ = trace_nfa(nfa, s)
    return ok


def build_automata_from_regex(regex: str) -> Tuple[NFA, DFA, DFA]:
    nfa = regex_to_nfa(regex)
    dfa = nfa_to_dfa(nfa)
    mdfa = minimize_dfa(DFA(dfa.start, set(dfa.accepts), {k: dict(v) for k, v in dfa.transitions.items()}, set(dfa.alphabet)))
    return nfa, dfa, mdfa


def prepare_test_results(nfa: NFA, dfa: DFA, mdfa: DFA, strings: Iterable[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for text in strings:
        nfa_path, nfa_ok, nfa_reason = trace_nfa(nfa, text)
        dfa_path, dfa_ok, dfa_reason = trace_dfa(dfa, text)
        mdfa_path, mdfa_ok, mdfa_reason = trace_dfa(mdfa, text)
        results.append(
            {
                "input": text,
                "nfa_accept": nfa_ok,
                "dfa_accept": dfa_ok,
                "mdfa_accept": mdfa_ok,
                "nfa_path": nfa_path,
                "dfa_path": dfa_path,
                "mdfa_path": mdfa_path,
                "nfa_reason": nfa_reason,
                "dfa_reason": dfa_reason,
                "mdfa_reason": mdfa_reason,
            }
        )
    return results
