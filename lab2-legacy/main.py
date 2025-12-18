# -*- coding: utf-8 -*-
"""
词法分析实验：正规表达式 → NFA → DFA → 最小 DFA 的参考实现
算法：
  (1) Thompson 构造：正则表达式 -> NFA
  (2) 子集构造：NFA -> DFA
  (3) 表填充算法：DFA 最小化（选做）
"""

from collections import deque
import argparse
import json
import os
import shutil
import subprocess

EPSILON = None  # 用 None 表示 ε 转移
DEFAULT_REGEX = "(a|b)*abb"


# ========= 一、NFA / DFA 的数据结构 =========


class NFA:
    def __init__(self, start, accept, transitions):
        """
        start      : 起始状态编号
        accept     : 接受状态编号
        transitions: {state: {symbol: set(next_states)}}
        """
        self.start = start
        self.accept = accept
        self.transitions = transitions


class DFA:
    def __init__(self, start, accepts, transitions, alphabet):
        """
        start      : 起始状态编号
        accepts    : 接受状态集合
        transitions: {state: {symbol: next_state}}
        alphabet   : 输入字母表集合（不含 EPSILON）
        """
        self.start = start
        self.accepts = set(accepts)
        self.transitions = transitions
        self.alphabet = set(alphabet)


# ========= 二、一些通用小工具 =========


def add_edge(transitions, src, symbol, dst):
    """在 NFA 转移表里加一条边：src --symbol--> dst"""
    if src not in transitions:
        transitions[src] = {}
    if symbol not in transitions[src]:
        transitions[src][symbol] = set()
    transitions[src][symbol].add(dst)


def merge_transitions(t1, t2):
    """合并两份 {state: {symbol: set(next_states)}} 转移表"""
    transitions = {}
    for src, mapping in t1.items():
        transitions[src] = {sym: set(dsts) for sym, dsts in mapping.items()}
    for src, mapping in t2.items():
        if src not in transitions:
            transitions[src] = {}
        for sym, dsts in mapping.items():
            transitions[src].setdefault(sym, set()).update(dsts)
    return transitions


# ========= 三、正则表达式 -> 后缀 + NFA (Thompson) =========


def insert_concat(regex):
    """
    在正则中插入显式连接符 '.'，
    例如：a(b|c)*d  ->  a.(b|c)*.d
    """
    result = []
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


def regex_to_postfix(regex):
    """
    中缀正则 -> 后缀表达式
    运算符优先级： * 最高，. 其次，| 最低
    """
    precedence = {"*": 3, ".": 2, "|": 1}
    output = []
    stack = []
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
        elif c in precedence:  # 运算符
            while stack and stack[-1] != "(" and precedence.get(stack[-1], 0) >= precedence[c]:
                output.append(stack.pop())
            stack.append(c)
        else:
            # 普通字符，直接输出
            output.append(c)

    while stack:
        output.append(stack.pop())

    return "".join(output)


def postfix_to_nfa(postfix):
    """
    根据后缀正则，用 Thompson 构造 NFA。
    """
    state_id = 0

    def new_state():
        nonlocal state_id
        s = state_id
        state_id += 1
        return s

    stack = []
    ops = set(["*", ".", "|"])

    for c in postfix:
        if c not in ops:
            # 单字符 NFA
            start = new_state()
            accept = new_state()
            trans = {}
            add_edge(trans, start, c, accept)
            stack.append(NFA(start, accept, trans))

        elif c == "*":
            # 星号闭包
            nfa1 = stack.pop()
            start = new_state()
            accept = new_state()
            trans = merge_transitions(nfa1.transitions, {})
            add_edge(trans, start, EPSILON, nfa1.start)
            add_edge(trans, start, EPSILON, accept)
            add_edge(trans, nfa1.accept, EPSILON, nfa1.start)
            add_edge(trans, nfa1.accept, EPSILON, accept)
            stack.append(NFA(start, accept, trans))

        elif c == ".":
            # 连接
            nfa2 = stack.pop()  # 右
            nfa1 = stack.pop()  # 左
            start = nfa1.start
            accept = nfa2.accept
            trans = merge_transitions(nfa1.transitions, nfa2.transitions)
            add_edge(trans, nfa1.accept, EPSILON, nfa2.start)
            stack.append(NFA(start, accept, trans))

        elif c == "|":
            # 并
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

    assert len(stack) == 1
    return stack[0]


def regex_to_nfa(regex):
    """封装：正则 -> NFA"""
    postfix = regex_to_postfix(regex)
    # print("Postfix:", postfix)
    return postfix_to_nfa(postfix)


# ========= 四、NFA -> DFA (子集构造) =========


def epsilon_closure(transitions, states):
    """给定状态集合，求其 ε-闭包，返回 frozenset"""
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for tgt in transitions.get(s, {}).get(EPSILON, set()):
            if tgt not in closure:
                closure.add(tgt)
                stack.append(tgt)
    return frozenset(closure)


def move(transitions, states, symbol):
    """从 states 出发，沿着 symbol 转移能到达的 NFA 状态集合"""
    result = set()
    for s in states:
        for tgt in transitions.get(s, {}).get(symbol, set()):
            result.add(tgt)
    return result


def nfa_to_dfa(nfa):
    """子集构造：NFA -> DFA"""
    transitions = nfa.transitions

    # 提取字母表（去掉 EPSILON）
    alphabet = set()
    for _, mapping in transitions.items():
        for sym in mapping:
            if sym is not EPSILON:
                alphabet.add(sym)

    # DFA 起始状态 = ε-闭包({NFA.start})
    start_closure = epsilon_closure(transitions, {nfa.start})

    dfa_states = {start_closure: 0}  # NFA 状态集合 -> DFA 状态编号
    dfa_start = 0
    dfa_accepts = set()
    dfa_trans = {}
    queue = deque([start_closure])

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
    # 记录“DFA 状态 -> 对应 NFA 状态集合”，方便调试 / 写报告画表
    dfa.state_sets = {id_: state_set for state_set, id_ in dfa_states.items()}
    return dfa


# ========= 五、DFA 最小化（表填充算法，选做） =========


def reachable_states(dfa):
    """从 DFA 起始状态出发，求可达状态集合"""
    visited = set()
    stack = [dfa.start]
    while stack:
        s = stack.pop()
        if s in visited:
            continue
        visited.add(s)
        for nxt in dfa.transitions.get(s, {}).values():
            stack.append(nxt)
    return visited


def make_total_dfa(dfa):
    """
    确保 DFA 对每个状态和每个输入符号都有定义（必要时添加“死状态”）。
    返回处理后的可达状态集合。
    """
    states = reachable_states(dfa)
    for s in states:
        dfa.transitions.setdefault(s, {})

    sink = None
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


def minimize_dfa(dfa):
    """用表填充算法最小化 DFA，返回新的 DFA"""
    states = make_total_dfa(dfa)
    states = sorted(states)
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)

    # table[i][j] 只用 i>j 的部分；True 表示这对状态可区分
    table = [[False] * n for _ in range(n)]

    # 第一步：一个是终态、一个不是终态 -> 可区分
    for i in range(n):
        for j in range(i):
            if (states[i] in dfa.accepts) != (states[j] in dfa.accepts):
                table[i][j] = True

    # 第二步：迭代标记
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i):
                if table[i][j]:
                    continue
                si, sj = states[i], states[j]
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

    # 第三步：未被标记的状态对属于同一等价类，用并查集合并
    parent = {s: s for s in states}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i):
            if not table[i][j]:
                union(states[i], states[j])

    # 为每个等价类分配新的状态编号
    reps = {}
    new_id = 0
    for s in states:
        r = find(s)
        if r not in reps:
            reps[r] = new_id
            new_id += 1

    state_map = {s: reps[find(s)] for s in states}

    # 构造新的转移函数
    new_trans = {}
    for s in states:
        ns = state_map[s]
        new_trans.setdefault(ns, {})
        for a, t in dfa.transitions.get(s, {}).items():
            new_trans[ns][a] = state_map[t]

    new_accepts = {state_map[s] for s in dfa.accepts if s in state_map}
    new_start = state_map[dfa.start]

    return DFA(start=new_start, accepts=new_accepts, transitions=new_trans, alphabet=set(dfa.alphabet))


# ========= 六、DFA 识别函数 + 简单测试 =========


def dfa_match(dfa, s):
    """用 DFA 判断字符串 s 是否被接受"""
    state = dfa.start
    for ch in s:
        if ch not in dfa.alphabet:
            return False
        state = dfa.transitions.get(state, {}).get(ch)
        if state is None:
            return False
    return state in dfa.accepts


def trace_dfa(dfa, s):
    """
    返回 DFA 运行时经过的状态序列，方便交互演示。
    (path, accepted, reason)
    """
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


def collect_nfa_alphabet(nfa):
    return {sym for m in nfa.transitions.values() for sym in m if sym is not EPSILON}


def trace_nfa(nfa, s):
    """
    返回 NFA 上的一个接受路径（若存在），使用 BFS 遍历 (state, pos)。
    (path, accepted, reason)
    """
    alphabet = collect_nfa_alphabet(nfa)
    for ch in s:
        if ch not in alphabet:
            return [nfa.start], False, f"字符 {ch!r} 不在字母表中"

    start = (nfa.start, 0)
    queue = deque([start])
    visited = {start}
    prev = {}  # (state,pos) -> (prev_state,pos)
    last = start

    def build_path(node):
        seq = []
        cur = node
        while cur is not None:
            seq.append(cur[0])
            cur = prev.get(cur)
        return list(reversed(seq))

    while queue:
        state, pos = queue.popleft()
        last = (state, pos)
        if state == nfa.accept and pos == len(s):
            return build_path((state, pos)), True, "找到接受路径"

        # ε 转移
        for nxt in nfa.transitions.get(state, {}).get(EPSILON, set()):
            node = (nxt, pos)
            if node not in visited:
                visited.add(node)
                prev[node] = (state, pos)
                queue.append(node)

        # 消耗一个字符的转移
        if pos < len(s):
            ch = s[pos]
            for nxt in nfa.transitions.get(state, {}).get(ch, set()):
                node = (nxt, pos + 1)
                if node not in visited:
                    visited.add(node)
                    prev[node] = (state, pos)
                    queue.append(node)

    return build_path(last), False, "未找到接受路径"


def nfa_match(nfa, s):
    path, ok, _ = trace_nfa(nfa, s)
    return ok


# ========= 七、简单可视化与交互 =========


def collect_states_from_transitions(transitions, start, accepts):
    states = {start, *accepts} if isinstance(accepts, set) else {start, accepts}
    for src, mapping in transitions.items():
        states.add(src)
        for dsts in mapping.values():
            if isinstance(dsts, set):
                states.update(dsts)
            else:
                states.add(dsts)
    return sorted(states)


def print_nfa_table(nfa):
    symbols = sorted({sym for m in nfa.transitions.values() for sym in m if sym is not EPSILON})
    states = collect_states_from_transitions(nfa.transitions, nfa.start, {nfa.accept})
    header_syms = symbols + (["ε"] if any(EPSILON in m for m in nfa.transitions.values()) else [])
    print("\nNFA 转移表（状态数 {}）".format(len(states)))
    print("状态 | " + " | ".join(f"{sym:^5}" for sym in header_syms))
    print("-" * (7 + 8 * len(header_syms)))
    for s in states:
        cells = []
        for sym in symbols:
            dsts = nfa.transitions.get(s, {}).get(sym, set())
            cells.append("{" + ",".join(map(str, sorted(dsts))) + "}")
        if "ε" in header_syms:
            dsts = nfa.transitions.get(s, {}).get(EPSILON, set())
            cells.append("{" + ",".join(map(str, sorted(dsts))) + "}")
        marker = "*" if s == nfa.accept else " "
        prefix = "->" if s == nfa.start else "  "
        print(f"{prefix}{s:>3}{marker}| " + " | ".join(f"{c:>5}" for c in cells))


def print_dfa_table(dfa, title="DFA 转移表"):
    symbols = sorted(dfa.alphabet)
    states = collect_states_from_transitions(dfa.transitions, dfa.start, dfa.accepts)
    print(f"\n{title}（状态数 {len(states)}）")
    print("状态 | " + " | ".join(f"{sym:^5}" for sym in symbols))
    print("-" * (7 + 8 * len(symbols)))
    for s in states:
        cells = []
        for sym in symbols:
            dst = dfa.transitions.get(s, {}).get(sym)
            cells.append("" if dst is None else str(dst))
        marker = "*" if s in dfa.accepts else " "
        prefix = "->" if s == dfa.start else "  "
        print(f"{prefix}{s:>3}{marker}| " + " | ".join(f"{c:>5}" for c in cells))


def automaton_to_dot(transitions, start, accepts, is_nfa, name="automaton"):
    """
    生成 Graphviz DOT 字符串，可用 dot -Tpng 渲染。
    accepts 可以是集合或单个状态。
    """
    accept_set = set(accepts) if not isinstance(accepts, set) else accepts
    lines = [f"digraph {name} {{", "  rankdir=LR;", "  node [shape=circle];", "  __start [shape=point];"]
    lines.append(f"  __start -> {start};")
    for acc in accept_set:
        lines.append(f"  {acc} [shape=doublecircle];")

    for src, mapping in transitions.items():
        for sym, dsts in mapping.items():
            label = "ε" if sym is EPSILON else sym
            if is_nfa:
                for dst in dsts:
                    lines.append(f'  {src} -> {dst} [label="{label}"];')
            else:
                lines.append(f'  {src} -> {dsts} [label="{label}"];')

    lines.append("}")
    return "\n".join(lines)


def get_dot_strings(nfa, dfa, mdfa):
    return {
        "nfa": automaton_to_dot(nfa.transitions, nfa.start, {nfa.accept}, True, "NFA"),
        "dfa": automaton_to_dot(dfa.transitions, dfa.start, dfa.accepts, False, "DFA"),
        "mdfa": automaton_to_dot(mdfa.transitions, mdfa.start, mdfa.accepts, False, "MinDFA"),
    }


def export_graphs(nfa, dfa, mdfa, base_name="automaton", auto_png=False, quiet=False, ask_user=True):
    dots = get_dot_strings(nfa, dfa, mdfa)
    folder = os.getcwd()
    outputs = [
        (dots["nfa"], f"{base_name}_nfa.dot"),
        (dots["dfa"], f"{base_name}_dfa.dot"),
        (dots["mdfa"], f"{base_name}_mdfa.dot"),
    ]
    for dot_str, filename in outputs:
        path = os.path.join(folder, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(dot_str)
        if not quiet:
            print(f"已导出 {filename}")
    dot_bin = shutil.which("dot")
    if dot_bin:
        should_render = auto_png
        if not auto_png and ask_user:
            choice = input("检测到 Graphviz，可直接导出 PNG（y/N）? ").strip().lower()
            should_render = choice == "y"
        if should_render:
            for _, filename in outputs:
                png_path = filename.replace(".dot", ".png")
                cmd = [dot_bin, "-Tpng", filename, "-o", png_path]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if not quiet:
                    if res.returncode == 0:
                        print(f"生成 {png_path}")
                    else:
                        print(f"生成 {png_path} 失败：{res.stderr.strip()}")
    else:
        if not quiet:
            print("未找到 dot 命令，若需 PNG 可自行安装 Graphviz 后运行：dot -Tpng <dot文件> -o <png文件>")


def build_automata_from_regex(regex):
    nfa = regex_to_nfa(regex)
    dfa = nfa_to_dfa(nfa)
    mdfa = minimize_dfa(dfa)
    return nfa, dfa, mdfa


def sanitize_filename(text):
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    trimmed = cleaned.strip("_")
    return trimmed[:30] or "automaton"


def dfa_to_dict(dfa):
    return {
        "start": dfa.start,
        "accepts": sorted(dfa.accepts),
        "alphabet": sorted(dfa.alphabet),
        "transitions": {
            str(state): {sym: tgt for sym, tgt in trans.items()} for state, trans in dfa.transitions.items()
        },
    }


def prepare_test_results(nfa, dfa, mdfa, strings):
    results = []
    for text in strings:
        nfa_path, nfa_ok, _ = trace_nfa(nfa, text)
        path, ok, _ = trace_dfa(dfa, text)
        results.append(
            {
                "input": text,
                "nfa_accept": nfa_ok,
                "dfa_accept": dfa_match(dfa, text),
                "mdfa_accept": dfa_match(mdfa, text),
                "nfa_path": nfa_path,
                "path": path,
            }
        )
    return results


def generate_frontend_page(regex, dots_unused, dfa_dict_unused, mdfa_dict_unused, test_results, output_path, meta=None):
    """
    输出纯前端版页面：包含 JS 版正规式->NFA->DFA->最小DFA 构造，
    支持在页面内修改正则、批量输入字符串检测，并在浏览器直接渲染/下载图。
    """
    data = {"regex": regex or DEFAULT_REGEX, "defaultRegex": DEFAULT_REGEX, "defaultTests": test_results or []}
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>自动机可视化</title>
  <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/viz.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/full.render.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
    header {{ padding: 20px; background: linear-gradient(120deg, #0ea5e9, #6366f1); color: #0b1220; }}
    h1 {{ margin: 0 0 6px 0; font-weight: 700; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); }}
    .panel {{ background: rgba(15,23,42,0.8); border: 1px solid rgba(148,163,184,0.2); border-radius: 12px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }}
    .panel h2 {{ margin-top: 0; color: #38bdf8; font-size: 18px; }}
    .chips span {{ display: inline-block; margin: 4px 6px 4px 0; padding: 4px 10px; border-radius: 16px; background: rgba(148,163,184,0.15); border: 1px solid rgba(148,163,184,0.3); font-size: 12px; }}
    #graph-container {{ background: #0b1220; min-height: 320px; display: flex; align-items: center; justify-content: center; border-radius: 10px; border: 1px dashed rgba(148,163,184,0.3); padding: 12px; }}
    select, button, input, textarea {{ background: #0b1220; color: #e2e8f0; border: 1px solid rgba(148,163,184,0.4); border-radius: 8px; padding: 8px 10px; }}
    button {{ cursor: pointer; transition: background 0.2s, transform 0.1s; }}
    button:hover {{ background: #1e293b; }}
    button:active {{ transform: translateY(1px); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 8px; border-bottom: 1px solid rgba(148,163,184,0.2); text-align: left; }}
    th {{ color: #94a3b8; font-weight: 600; }}
    .badge-true {{ color: #22c55e; }}
    .badge-false {{ color: #f87171; }}
    code {{ background: rgba(148,163,184,0.15); padding: 2px 6px; border-radius: 6px; }}
    .flex {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .muted {{ color: #94a3b8; font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>自动机可视化</h1>
      <div class="flex" style="align-items:center; gap:12px;">
        <div class="muted">正则：<code id="regex"></code></div>
        <input id="regex-input" placeholder="输入新的正则表达式（默认：{DEFAULT_REGEX}）" style="flex:1; min-width:200px;">
        <button id="regex-apply">应用正则</button>
      </div>
  </header>
  <div class="container">
    <div class="panel" style="grid-column: span 2;">
      <h2>图形展示</h2>
      <div class="flex">
        <label>查看：
          <select id="graph-select">
            <option value="nfa">NFA</option>
            <option value="dfa">DFA</option>
            <option value="mdfa">最小 DFA</option>
          </select>
        </label>
        <button id="rerender-btn">重新渲染</button>
        <button id="download-btn">下载当前图为 PNG</button>
        <span class="muted">在浏览器渲染 DOT，无需额外工具</span>
      </div>
      <div id="graph-container"></div>
    </div>
    <div class="panel" style="grid-column: span 2;">
      <h2>批量检测（多行输入，点击检测后表格会刷新）</h2>
      <div class="flex" style="align-items:center;">
        <textarea id="batch-input" rows="4" style="flex:1;" placeholder="每行一个输入串"></textarea>
        <button id="batch-btn">批量检测</button>
      </div>
      <table style="margin-top:8px;">
        <thead><tr><th>输入串</th><th>NFA</th><th>DFA</th><th>最小 DFA</th><th>路径(NFA)</th><th>路径(最小 DFA)</th></tr></thead>
        <tbody id="batch-body"></tbody>
      </table>
      <div class="muted">用于批量字符串统一检测；没有预设结果，完全按当前正则和输入刷新。</div>
    </div>
    <div class="panel">
      <h2>单个串检测</h2>
      <div class="flex" style="align-items:center;">
        <input id="manual-input" placeholder="输入要检测的字符串" style="flex:1; min-width:180px;">
        <button id="manual-btn">检测</button>
      </div>
      <div id="manual-result" style="margin-top:8px;"></div>
      <div class="muted" id="manual-path"></div>
    </div>
    <div class="panel">
      <h2>当前自动机信息</h2>
      <div class="chips">
        <span id="chip-abc"></span>
        <span id="chip-dfa"></span>
        <span id="chip-mdfa"></span>
      </div>
      <div class="muted">修改正则后自动重建 NFA/DFA/最小 DFA 并刷新图与检测逻辑。</div>
    </div>
  </div>

  <script>
    const APP_DATA = {json.dumps(data, ensure_ascii=False)};
    const DEFAULT_REGEX = APP_DATA.defaultRegex || "(a|b)*abb";
    const EPS = null;
    const EPS_KEY = String(EPS);

    // ==== 正则 -> 后缀 ====
    function insertConcat(regex) {{
      const out = [];
      for (let i = 0; i < regex.length; i++) {{
        const c = regex[i];
        if (c === " ") continue;
        out.push(c);
        const next = regex[i + 1];
        if (!next || next === " " || c === "|" || c === "(") continue;
        if (c !== "|" && c !== "(") {{
          if (next !== "|" && next !== ")" && next !== "*") out.push(".");
        }}
      }}
      return out.join("");
    }}

    function regexToPostfix(regex) {{
      const prec = {{ "*": 3, ".": 2, "|": 1 }};
      const out = [];
      const stack = [];
      const src = insertConcat(regex);
      for (const c of src) {{
        if (c === "(") stack.push(c);
        else if (c === ")") {{
          while (stack.length && stack[stack.length - 1] !== "(") out.push(stack.pop());
          if (stack.length && stack[stack.length - 1] === "(") stack.pop();
        }} else if (prec[c]) {{
          while (stack.length && stack[stack.length - 1] !== "(" && prec[stack[stack.length - 1]] >= prec[c]) {{
            out.push(stack.pop());
          }}
          stack.push(c);
        }} else {{
          out.push(c);
        }}
      }}
      while (stack.length) out.push(stack.pop());
      return out.join("");
    }}

    // ==== NFA 构造 ====
    function addEdge(trans, src, sym, dst) {{
      if (!trans[src]) trans[src] = {{}};
      if (!trans[src][sym]) trans[src][sym] = new Set();
      trans[src][sym].add(dst);
    }}

    function postfixToNfa(postfix) {{
      let sid = 0;
      const newState = () => sid++;
      const stack = [];
      for (const c of postfix) {{
        if (c === "*") {{
          const nfa1 = stack.pop();
          const start = newState();
          const accept = newState();
          const trans = JSON.parse(JSON.stringify(nfa1.transitions, (_, v) => v instanceof Set ? Array.from(v) : v));
          // restore sets
          for (const k of Object.keys(trans)) {{
            for (const s of Object.keys(trans[k])) trans[k][s] = new Set(trans[k][s]);
          }}
          addEdge(trans, start, EPS, nfa1.start);
          addEdge(trans, start, EPS, accept);
          addEdge(trans, nfa1.accept, EPS, nfa1.start);
          addEdge(trans, nfa1.accept, EPS, accept);
          stack.push({{ start, accept, transitions: trans }});
        }} else if (c === ".") {{
          const nfa2 = stack.pop();
          const nfa1 = stack.pop();
          const trans = JSON.parse(JSON.stringify({{...nfa1.transitions, ...nfa2.transitions}}, (_, v) => v instanceof Set ? Array.from(v) : v));
          for (const k of Object.keys(trans)) {{
            for (const s of Object.keys(trans[k])) trans[k][s] = new Set(trans[k][s]);
          }}
          addEdge(trans, nfa1.accept, EPS, nfa2.start);
          stack.push({{ start: nfa1.start, accept: nfa2.accept, transitions: trans }});
        }} else if (c === "|") {{
          const nfa2 = stack.pop();
          const nfa1 = stack.pop();
          const start = newState();
          const accept = newState();
          const trans = JSON.parse(JSON.stringify({{...nfa1.transitions, ...nfa2.transitions}}, (_, v) => v instanceof Set ? Array.from(v) : v));
          for (const k of Object.keys(trans)) {{
            for (const s of Object.keys(trans[k])) trans[k][s] = new Set(trans[k][s]);
          }}
          addEdge(trans, start, EPS, nfa1.start);
          addEdge(trans, start, EPS, nfa2.start);
          addEdge(trans, nfa1.accept, EPS, accept);
          addEdge(trans, nfa2.accept, EPS, accept);
          stack.push({{ start, accept, transitions: trans }});
        }} else {{
          const start = newState();
          const accept = newState();
          const trans = {{}};
          addEdge(trans, start, c, accept);
          stack.push({{ start, accept, transitions: trans }});
        }}
      }}
      return stack.pop();
    }}

    const regexToNfa = (re) => postfixToNfa(regexToPostfix(re));

    // ==== NFA -> DFA ====
    const setKey = (set) => Array.from(set).sort((a, b) => a - b).join(",");
    const union = (a, b) => new Set([...a, ...b]);

    function epsilonClosure(trans, states) {{
      const stack = [...states];
      const closure = new Set(states);
      while (stack.length) {{
        const s = stack.pop();
        const eps = trans[s]?.[EPS] || new Set();
        for (const t of eps) if (!closure.has(t)) {{ closure.add(t); stack.push(t); }}
      }}
      return closure;
    }}

    function move(trans, states, sym) {{
      const res = new Set();
      for (const s of states) {{
        const dsts = trans[s]?.[sym] || new Set();
        dsts.forEach((d) => res.add(d));
      }}
      return res;
    }}

    function nfaToDfa(nfa) {{
      const alphabet = new Set();
      for (const mapping of Object.values(nfa.transitions)) {{
        for (const sym of Object.keys(mapping)) if (sym !== EPS_KEY) alphabet.add(sym);
      }}
      const startSet = epsilonClosure(nfa.transitions, new Set([nfa.start]));
      const dfaStates = new Map();
      const queue = [];
      const trans = {{}};
      dfaStates.set(setKey(startSet), 0);
      queue.push(startSet);
      const accepts = new Set();
      if (startSet.has(nfa.accept)) accepts.add(0);
      while (queue.length) {{
        const S = queue.shift();
        const sid = dfaStates.get(setKey(S));
        trans[sid] = trans[sid] || {{}};
        for (const sym of alphabet) {{
          const U = epsilonClosure(nfa.transitions, move(nfa.transitions, S, sym));
          if (U.size === 0) continue;
          const key = setKey(U);
          if (!dfaStates.has(key)) {{
            dfaStates.set(key, dfaStates.size);
            queue.push(U);
            if (U.has(nfa.accept)) accepts.add(dfaStates.size - 1);
          }}
          trans[sid][sym] = dfaStates.get(key);
        }}
      }}
      return {{ start: 0, accepts, transitions: trans, alphabet }};
    }}

    // ==== DFA 最小化 ====
    function reachableStates(dfa) {{
      const vis = new Set();
      const stack = [dfa.start];
      while (stack.length) {{
        const s = stack.pop();
        if (vis.has(s)) continue;
        vis.add(s);
        const m = dfa.transitions[s] || {{}};
        for (const nxt of Object.values(m)) stack.push(nxt);
      }}
      return vis;
    }}

    function makeTotalDfa(dfa) {{
      const states = reachableStates(dfa);
      const alpha = Array.from(dfa.alphabet);
      let sink = null;
      for (const s of states) {{
        dfa.transitions[s] = dfa.transitions[s] || {{}};
        for (const a of alpha) {{
          if (dfa.transitions[s][a] === undefined) {{
            if (sink === null) {{
              sink = Math.max(...states) + 1;
              dfa.transitions[sink] = {{}};
              for (const a2 of alpha) dfa.transitions[sink][a2] = sink;
            }}
            dfa.transitions[s][a] = sink;
          }}
        }}
      }}
      if (sink !== null) states.add(sink);
      return new Set(states);
    }}

    function minimizeDfa(dfa) {{
      const states = Array.from(makeTotalDfa(dfa)).sort((a, b) => a - b);
      const idx = new Map(states.map((s, i) => [s, i]));
      const n = states.length;
      const table = Array.from({{ length: n }}, () => Array(n).fill(false));
      for (let i = 0; i < n; i++) {{
        for (let j = 0; j < i; j++) {{
          if (dfa.accepts.has(states[i]) !== dfa.accepts.has(states[j])) table[i][j] = true;
        }}
      }}
      let changed = true;
      const alpha = Array.from(dfa.alphabet);
      while (changed) {{
        changed = false;
        for (let i = 0; i < n; i++) {{
          for (let j = 0; j < i; j++) {{
            if (table[i][j]) continue;
            for (const a of alpha) {{
              const p = dfa.transitions[states[i]][a];
              const q = dfa.transitions[states[j]][a];
              if (p === q) continue;
              const ii = idx.get(Math.max(p, q));
              const jj = idx.get(Math.min(p, q));
              if (table[ii][jj]) {{ table[i][j] = true; changed = true; break; }}
            }}
          }}
        }}
      }}
      const parent = new Map(states.map((s) => [s, s]));
      const find = (x) => {{ while (parent.get(x) !== x) {{ parent.set(x, parent.get(parent.get(x))); x = parent.get(x); }} return x; }};
      const unite = (x, y) => {{ const rx = find(x), ry = find(y); if (rx !== ry) parent.set(ry, rx); }};
      for (let i = 0; i < n; i++) {{
        for (let j = 0; j < i; j++) if (!table[i][j]) unite(states[i], states[j]);
      }}
      const reps = new Map();
      let newId = 0;
      for (const s of states) {{
        const r = find(s);
        if (!reps.has(r)) reps.set(r, newId++);
      }}
      const mapState = (s) => reps.get(find(s));
      const newTrans = {{}};
      for (const s of states) {{
        const ns = mapState(s);
        newTrans[ns] = newTrans[ns] || {{}};
        for (const [a, t] of Object.entries(dfa.transitions[s] || {{}})) newTrans[ns][a] = mapState(t);
      }}
      const newAccepts = new Set(Array.from(dfa.accepts).map(mapState));
      const newStart = mapState(dfa.start);
      return {{ start: newStart, accepts: newAccepts, transitions: newTrans, alphabet: new Set(dfa.alphabet) }};
    }}

    // ==== 识别与 DOT ====
    function traceNfa(nfa, str) {{
      const alphabet = collectAlphabet(nfa);
      for (const ch of str) {{
        if (!alphabet.has(ch)) return {{ path: [nfa.start], ok: false, reason: `字符 '${{ch}}' 不在字母表中` }};
      }}
      const key = (s, i) => `${{s}}@${{i}}`;
      const prev = new Map();
      const visited = new Set();
      const queue = [];
      const push = (state, pos, fromKey, via) => {{
        const k = key(state, pos);
        if (visited.has(k)) return;
        visited.add(k);
        if (fromKey) prev.set(k, {{ from: fromKey, via }});
        queue.push({{ state, pos }});
      }};
      push(nfa.start, 0, null, null);
      let lastKey = key(nfa.start, 0);

      const buildPath = (endKey) => {{
        if (!endKey) return [nfa.start];
        const seq = [];
        let cur = endKey;
        while (cur) {{
          const [stateStr] = cur.split("@");
          seq.push(Number(stateStr));
          const info = prev.get(cur);
          cur = info?.from || null;
        }}
        return seq.reverse();
      }};

      while (queue.length) {{
        const {{ state, pos }} = queue.shift();
        const currKey = key(state, pos);
        lastKey = currKey;
        if (state === nfa.accept && pos === str.length) {{
          return {{ ok: true, path: buildPath(currKey), reason: "找到接受路径" }};
        }}
        const eps = nfa.transitions[state]?.[EPS] || new Set();
        for (const nxt of eps) push(nxt, pos, currKey, "ε");
        if (pos < str.length) {{
          const ch = str[pos];
          const dsts = nfa.transitions[state]?.[ch] || new Set();
          for (const nxt of dsts) push(nxt, pos + 1, currKey, ch);
        }}
      }}
      return {{ ok: false, path: buildPath(lastKey), reason: "未找到接受路径" }};
    }}

    function nfaMatch(nfa, str) {{
      return traceNfa(nfa, str).ok;
    }}

    function dfaMatch(dfa, str) {{
      let state = dfa.start;
      for (const ch of str) {{
        if (!dfa.alphabet.has(ch)) return false;
        state = dfa.transitions[state]?.[ch];
        if (state === undefined) return false;
      }}
      return dfa.accepts.has(state);
    }}

    function traceDfa(dfa, str) {{
      let state = dfa.start;
      const path = [state];
      for (const ch of str) {{
        if (!dfa.alphabet.has(ch)) return {{ path, ok: false, reason: `字符 '${{ch}}' 不在字母表中` }};
        const nxt = dfa.transitions[state]?.[ch];
        if (nxt === undefined) return {{ path, ok: false, reason: `状态 ${{state}} 遇到 '${{ch}}' 无转移` }};
        path.push((state = nxt));
      }}
      return {{ path, ok: dfa.accepts.has(state), reason: "完成匹配" }};
    }}

    function automatonToDot(trans, start, accepts, isNfa, name) {{
      const lines = [`digraph ${{name}} {{`, "  rankdir=LR;", '  node [shape=circle];', '  __start [shape=point];', `  __start -> ${{start}};`];
      accepts.forEach((a) => lines.push(`  ${{a}} [shape=doublecircle];`));
      for (const [src, mapping] of Object.entries(trans)) {{
        for (const [sym, dsts] of Object.entries(mapping)) {{
          const label = sym === EPS_KEY ? "ε" : sym;
          if (isNfa) {{
            (Array.from(dsts)).forEach((d) => lines.push(`  ${{src}} -> ${{d}} [label="${{label}}"];`));
          }} else {{
            lines.push(`  ${{src}} -> ${{dsts}} [label="${{label}}"];`);
          }}
        }}
      }}
      lines.push("}}");
      return lines.join("\\n");
    }}

    function collectAlphabet(nfa) {{
      const alpha = new Set();
      for (const mapping of Object.values(nfa.transitions)) {{
        for (const sym of Object.keys(mapping)) if (sym !== EPS_KEY) alpha.add(sym);
      }}
      return alpha;
    }}

    function cloneDfa(dfa) {{
      const copy = {{
        start: dfa.start,
        accepts: new Set(dfa.accepts),
        alphabet: new Set(dfa.alphabet),
        transitions: {{}},
      }};
      for (const [s, mapping] of Object.entries(dfa.transitions)) {{
        copy.transitions[s] = {{}};
        for (const [a, t] of Object.entries(mapping)) copy.transitions[s][a] = t;
      }}
      return copy;
    }}

    function buildAll(regex) {{
      const nfa = regexToNfa(regex || "");
      const dfa = nfaToDfa(nfa);
      const mdfa = minimizeDfa(cloneDfa(dfa));
      const dots = {{
        nfa: automatonToDot(
          Object.fromEntries(Object.entries(nfa.transitions).map(([k, v]) => [k, Object.fromEntries(Object.entries(v).map(([a, s]) => [a, Array.from(s)]))])),
          nfa.start,
          new Set([nfa.accept]),
          true,
          "NFA"
        ),
        dfa: automatonToDot(
          dfa.transitions,
          dfa.start,
          dfa.accepts,
          false,
          "DFA"
        ),
        mdfa: automatonToDot(
          mdfa.transitions,
          mdfa.start,
          mdfa.accepts,
          false,
          "MinDFA"
        ),
      }};
      return {{
        regex,
        nfa,
        dfa,
        mdfa,
        dots,
        meta: {{
          alphabet: Array.from(collectAlphabet(nfa)).sort(),
          dfa_states: Object.keys(dfa.transitions).length,
          mdfa_states: Object.keys(mdfa.transitions).length,
        }},
      }};
    }}

    function updateChips(meta) {{
      document.getElementById("chip-abc").textContent = "字母表: " + meta.alphabet.join(", ");
      document.getElementById("chip-dfa").textContent = "DFA 状态数: " + meta.dfa_states;
      document.getElementById("chip-mdfa").textContent = "最小 DFA 状态数: " + meta.mdfa_states;
    }}

    function updateRegexLabel(regex) {{
      document.getElementById("regex").textContent = regex;
      document.getElementById("regex-input").value = regex;
    }}

    let CURRENT = buildAll(APP_DATA.regex || DEFAULT_REGEX);

    function renderGraph(kind) {{
      const dot = CURRENT.dots[kind];
      const container = document.getElementById("graph-container");
      container.innerHTML = "渲染中...";
      const viz = new Viz();
      viz.renderSVGElement(dot).then(el => {{
        container.innerHTML = "";
        container.appendChild(el);
        container.dataset.currentKind = kind;
        container.dataset.currentSvg = new XMLSerializer().serializeToString(el);
      }}).catch(err => {{
        container.innerHTML = "渲染失败: " + err;
      }});
    }}

    function bindGraphSwitcher() {{
      const sel = document.getElementById("graph-select");
      sel.addEventListener("change", () => renderGraph(sel.value));
      renderGraph(sel.value);
    }}

    function bindGraphActions() {{
      document.getElementById("rerender-btn").addEventListener("click", () => renderGraph(document.getElementById("graph-select").value));
      document.getElementById("download-btn").addEventListener("click", () => {{
        const container = document.getElementById("graph-container");
        const svgString = container.dataset.currentSvg;
        const kind = container.dataset.currentKind || document.getElementById("graph-select").value;
        if (!svgString) {{ alert("请先渲染图形"); return; }}
        const img = new Image();
        const svgBlob = new Blob([svgString], {{type: "image/svg+xml;charset=utf-8"}});
        const url = URL.createObjectURL(svgBlob);
        img.onload = () => {{
          const canvas = document.createElement("canvas");
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(img, 0, 0);
          canvas.toBlob(blob => {{
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = `${{kind}}.png`;
            a.click();
            URL.revokeObjectURL(a.href);
          }});
          URL.revokeObjectURL(url);
        }};
        img.src = url;
      }});
    }}

    function bindManual() {{
      document.getElementById("manual-btn").addEventListener("click", () => {{
        const text = document.getElementById("manual-input").value || "";
        const res = {{
          nfa: traceNfa(CURRENT.nfa, text),
          dfa: traceDfa(CURRENT.dfa, text),
          mdfa: traceDfa(CURRENT.mdfa, text),
        }};
        const out = document.getElementById("manual-result");
        out.innerHTML = `
          <span class="${{res.nfa.ok ? "badge-true" : "badge-false"}}">NFA: ${{res.nfa.ok}}</span>
          <span class="${{res.dfa.ok ? "badge-true" : "badge-false"}}">DFA: ${{res.dfa.ok}}</span>
          <span class="${{res.mdfa.ok ? "badge-true" : "badge-false"}}">最小 DFA: ${{res.mdfa.ok}}</span>
        `;
        out.className = "";
        const nfaPath = res.nfa.path.length ? res.nfa.path.join(" -> ") : "无可行路径";
        const mdfaPath = res.mdfa.path.length ? res.mdfa.path.join(" -> ") : "无可行路径";
        const nfaReason = res.nfa.ok ? "" : `（${{res.nfa.reason}}）`;
        const mdfaReason = res.mdfa.ok ? "" : `（${{res.mdfa.reason}}）`;
        document.getElementById("manual-path").innerHTML = `
          NFA 路径: ${{nfaPath}}{{nfaReason}}<br>
          最小 DFA 路径: ${{mdfaPath}}{{mdfaReason}}
        `;
      }});
    }}

    function bindBatch() {{
      document.getElementById("batch-btn").addEventListener("click", () => {{
        const lines = document.getElementById("batch-input").value.split("\\n").map(s => s.trim()).filter(Boolean);
        const body = document.getElementById("batch-body");
        body.innerHTML = "";
        lines.forEach(line => {{
          const res = {{
            nfa: traceNfa(CURRENT.nfa, line),
            dfa: traceDfa(CURRENT.dfa, line),
            mdfa: traceDfa(CURRENT.mdfa, line),
          }};
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><code>${{line}}</code></td>
            <td class="${{res.nfa.ok ? "badge-true" : "badge-false"}}">${{res.nfa.ok}}</td>
            <td class="${{res.dfa.ok ? "badge-true" : "badge-false"}}">${{res.dfa.ok}}</td>
            <td class="${{res.mdfa.ok ? "badge-true" : "badge-false"}}">${{res.mdfa.ok}}</td>
            <td class="muted">${{res.nfa.path.join(" -> ")}}</td>
            <td class="muted">${{res.mdfa.path.join(" -> ")}}</td>
          `;
          body.appendChild(tr);
        }});
      }});
    }}

    function bindRegexApply() {{
      document.getElementById("regex-apply").addEventListener("click", () => {{
        const val = document.getElementById("regex-input").value.trim() || DEFAULT_REGEX;
        try {{
          CURRENT = buildAll(val);
          updateRegexLabel(val);
          updateChips(CURRENT.meta);
          document.getElementById("batch-body").innerHTML = "";
          document.getElementById("manual-result").textContent = "";
          document.getElementById("manual-path").textContent = "";
          renderGraph(document.getElementById("graph-select").value);
        }} catch (e) {{
          alert("正则解析失败，请检查输入。");
          console.error(e);
        }}
      }});
    }}

    function init() {{
      updateRegexLabel(CURRENT.regex);
      updateChips(CURRENT.meta);
      bindGraphSwitcher();
      bindGraphActions();
      bindManual();
      bindBatch();
      bindRegexApply();
    }}

    init();
  </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"前端可视化已生成：{output_path}")


def read_strings_multi():
    print("输入待测试字符串（支持多行），连续回车结束：")
    strings = []
    while True:
        line = input()
        if line == "":
            break
        strings.append(line)
    if not strings:
        print("未输入内容。")
    return strings


def run_batch_mode(args):
    regex = args.regex or DEFAULT_REGEX
    strings = args.strings or []
    nfa, dfa, mdfa = build_automata_from_regex(regex)
    base_name = f"automaton_{sanitize_filename(regex)}"
    dots = get_dot_strings(nfa, dfa, mdfa)
    meta = {
        "alphabet": sorted(dfa.alphabet),
        "dfa_states": len(dfa.transitions),
        "mdfa_states": len(mdfa.transitions),
    }

    print(f"正则：{regex}")
    print(f"字母表：{meta['alphabet']}")
    print(f"DFA 状态数：{meta['dfa_states']}，最小 DFA 状态数：{meta['mdfa_states']}")
    export_graphs(
        nfa,
        dfa,
        mdfa,
        base_name=base_name,
        auto_png=not args.no_png,
        quiet=False,
        ask_user=not args.no_interactive,
    )
    results = prepare_test_results(nfa, dfa, mdfa, strings)

    if strings:
        print("批量检测结果：")
        for item in results:
            print(
                f"  {item['input']!r} -> NFA: {item['nfa_accept']}，DFA: {item['dfa_accept']}，最小 DFA: {item['mdfa_accept']}"
            )

    if args.output_html:
        output_path = args.output_html
        generate_frontend_page(regex, dots, dfa_to_dict(dfa), dfa_to_dict(mdfa), results, output_path, meta=meta)
    return results


def interactive_cli():
    current_regex = None
    nfa = dfa = mdfa = None

    def load_regex(regex):
        nonlocal current_regex, nfa, dfa, mdfa
        current_regex = regex or DEFAULT_REGEX
        nfa, dfa, mdfa = build_automata_from_regex(current_regex)
        base_name = f"automaton_{sanitize_filename(current_regex)}"
        print("\n==============================")
        print(f"当前正则：{current_regex}")
        print(f"字母表：{sorted(dfa.alphabet)}")
        print(f"DFA 状态数：{len(dfa.transitions)}，最小 DFA 状态数：{len(mdfa.transitions)}")
        print(f"已自动导出 DOT 文件前缀：{base_name}_*.dot")
        export_graphs(nfa, dfa, mdfa, base_name=base_name, auto_png=True, quiet=True)
        print("若安装了 Graphviz 已自动生成 PNG；否则可在菜单 4 里手动导出。")

    load_regex(input(f"请输入正规表达式（回车使用默认 {DEFAULT_REGEX}）：").strip())

    while True:
        print("\n【操作菜单】")
        print("1. 批量测试字符串是否被接受")
        print("2. 查看 NFA/DFA 转移表")
        print("3. 显示匹配路径（逐步演示）")
        print("4. 导出 Graphviz (.dot/.png)")
        print("5. 换一个正则表达式")
        print("0. 退出")
        choice = input("请选择：").strip()

        if choice == "0":
            print("已退出。")
            break
        elif choice == "1":
            texts = read_strings_multi()
            for text in texts:
                res_nfa = nfa_match(nfa, text)
                res_dfa = dfa_match(dfa, text)
                res_mdfa = dfa_match(mdfa, text)
                print(f"{text!r} -> NFA: {res_nfa}，DFA: {res_dfa}，最小 DFA: {res_mdfa}")
        elif choice == "2":
            print_nfa_table(nfa)
            print_dfa_table(dfa, "DFA 转移表")
            print_dfa_table(mdfa, "最小 DFA 转移表")
        elif choice == "3":
            text = input("输入需要演示的字符串：")
            nfa_path, nfa_ok, nfa_reason = trace_nfa(nfa, text)
            dfa_path, dfa_ok, dfa_reason = trace_dfa(dfa, text)
            mdfa_path, mdfa_ok, mdfa_reason = trace_dfa(mdfa, text)
            print(f"NFA 路径：{' -> '.join(map(str, nfa_path))}")
            print(f"  接受：{nfa_ok}，原因：{nfa_reason}")
            print(f"DFA 路径：{' -> '.join(map(str, dfa_path))}")
            print(f"  接受：{dfa_ok}，原因：{dfa_reason}")
            print(f"最小 DFA 路径：{' -> '.join(map(str, mdfa_path))}")
            print(f"  接受：{mdfa_ok}，原因：{mdfa_reason}")
        elif choice == "4":
            base = input("导出文件名前缀（默认为 automaton）：").strip() or "automaton"
            export_graphs(nfa, dfa, mdfa, base)
        elif choice == "5":
            load_regex(input("新的正规表达式：").strip())
        else:
            print("无效选项，请重新选择。")


def parse_args():
    parser = argparse.ArgumentParser(description="正则 -> NFA -> DFA -> 最小 DFA 的演示工具")
    parser.add_argument("--regex", help="指定正则表达式，默认 (a|b)*abb")
    parser.add_argument("--strings", nargs="*", help="待检测的字符串列表（空格分隔）")
    parser.add_argument("--output-html", dest="output_html", help="生成可视化 HTML 文件路径")
    parser.add_argument("--no-png", action="store_true", help="导出 DOT 时不尝试生成 PNG")
    parser.add_argument("--no-interactive", action="store_true", help="仅执行导出/检测，不进入交互菜单")
    return parser.parse_args()


def main():
    args = parse_args()
    has_batch_inputs = bool(args.regex or args.strings or args.output_html)
    if has_batch_inputs:
        run_batch_mode(args)
        if args.no_interactive:
            return
    else:
        if args.no_interactive:
            return
    interactive_cli()


if __name__ == "__main__":
    main()
