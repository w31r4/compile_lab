# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, Optional

from .automata import DFA, EPSILON, NFA


def automaton_to_dot(
    transitions,
    start: int,
    accepts,
    is_nfa: bool,
    name: str = "automaton",
) -> str:
    accept_set = set(accepts) if not isinstance(accepts, set) else accepts
    lines = [
        f"digraph {name} {{",
        "  rankdir=LR;",
        "  splines=false;",
        "  node [shape=circle];",
        "  __start [shape=point];",
        f"  __start -> {start};",
    ]
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


def get_dot_strings(nfa: NFA, dfa: DFA, mdfa: DFA) -> Dict[str, str]:
    return {
        "nfa": automaton_to_dot(nfa.transitions, nfa.start, {nfa.accept}, True, "NFA"),
        "dfa": automaton_to_dot(dfa.transitions, dfa.start, dfa.accepts, False, "DFA"),
        "mdfa": automaton_to_dot(mdfa.transitions, mdfa.start, mdfa.accepts, False, "MinDFA"),
    }


def render_dot_to_png_bytes(dot_str: str) -> Optional[bytes]:
    dot_bin = shutil.which("dot")
    if not dot_bin:
        return None
    res = subprocess.run([dot_bin, "-Tpng"], input=dot_str.encode("utf-8"), capture_output=True)
    if res.stdout:
        return res.stdout
    return None


def export_graphs(
    nfa: NFA,
    dfa: DFA,
    mdfa: DFA,
    base_name: str = "automaton",
    auto_png: bool = False,
    quiet: bool = False,
    ask_user: bool = True,
) -> None:
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
    if not dot_bin:
        if not quiet:
            print("未找到 dot 命令，若需 PNG 可自行安装 Graphviz 后运行：dot -Tpng <dot文件> -o <png文件>")
        return

    should_render = auto_png
    if not auto_png and ask_user:
        choice = input("检测到 Graphviz，可直接导出 PNG（y/N）? ").strip().lower()
        should_render = choice == "y"

    if not should_render:
        return

    for _, filename in outputs:
        png_path = filename.replace(".dot", ".png")
        cmd = [dot_bin, "-Tpng", filename, "-o", png_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if not quiet:
            ok = os.path.exists(png_path) and os.path.getsize(png_path) > 0
            if ok:
                if res.returncode == 0:
                    print(f"生成 {png_path}")
                else:
                    print(f"生成 {png_path}（dot 返回码 {res.returncode}，但已输出文件）")
            else:
                print(f"生成 {png_path} 失败：{res.stderr.strip()}")
