#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

from src.automata import (
    build_automata_from_regex,
    dfa_match,
    nfa_match,
    prepare_test_results,
    trace_dfa,
    trace_nfa,
)
from src.exporting import export_graphs, get_dot_strings
from src.html_export import write_frontend_html


DEFAULT_REGEX = "(a|b)*abb"


def sanitize_filename(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    trimmed = cleaned.strip("_")
    return trimmed[:30] or "automaton"


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


def print_nfa_table(nfa) -> None:
    from src.automata import EPSILON

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


def print_dfa_table(dfa, title: str = "DFA 转移表") -> None:
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


def read_strings_multi() -> list[str]:
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


def run_batch_mode(args) -> list[dict]:
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
        write_frontend_html(regex=regex, test_results=results, output_path=args.output_html, meta=meta)
        print(f"前端可视化已生成：{args.output_html}")

    if args.dump_dot:
        for name, dot in dots.items():
            path = f"{base_name}_{name}.dot"
            with open(path, "w", encoding="utf-8") as f:
                f.write(dot)
            print(f"已写入 {path}")

    return results


def interactive_cli() -> None:
    current_regex = None
    nfa = dfa = mdfa = None

    def load_regex(regex: str) -> None:
        nonlocal current_regex, nfa, dfa, mdfa
        current_regex = regex or DEFAULT_REGEX
        nfa, dfa, mdfa = build_automata_from_regex(current_regex)
        base_name = f"automaton_{sanitize_filename(current_regex)}"
        print("\n==============================")
        print(f"当前正则：{current_regex}")
        print(f"字母表：{sorted(dfa.alphabet)}")
        print(f"DFA 状态数：{len(dfa.transitions)}，最小 DFA 状态数：{len(mdfa.transitions)}")
        print(f"已自动导出 DOT 文件前缀：{base_name}_*.dot")
        export_graphs(nfa, dfa, mdfa, base_name=base_name, auto_png=True, quiet=True, ask_user=False)
        print("若安装了 Graphviz 已自动生成 PNG；否则可在菜单 4 里手动导出。")

    load_regex(input(f"请输入正规表达式（回车使用默认 {DEFAULT_REGEX}）：").strip())

    while True:
        print("\n【操作菜单】")
        print("1. 批量测试字符串是否被接受")
        print("2. 查看 NFA/DFA 转移表")
        print("3. 显示匹配路径（逐步演示）")
        print("4. 导出 Graphviz (.dot/.png)")
        print("5. 生成离线可视化 HTML")
        print("6. 换一个正则表达式")
        print("0. 退出")
        choice = input("请选择：").strip()

        if choice == "0":
            print("已退出。")
            break
        if choice == "1":
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
            export_graphs(nfa, dfa, mdfa, base, auto_png=True, quiet=False, ask_user=False)
        elif choice == "5":
            out = input("输出 HTML 文件名（默认为 visualization.html）：").strip() or "visualization.html"
            meta = {
                "alphabet": sorted(dfa.alphabet),
                "dfa_states": len(dfa.transitions),
                "mdfa_states": len(mdfa.transitions),
            }
            # 默认把上一次 batch 的结果留空：页面内再自行输入
            write_frontend_html(regex=current_regex or DEFAULT_REGEX, test_results=[], output_path=out, meta=meta)
            print(f"已生成：{os.path.abspath(out)}")
        elif choice == "6":
            load_regex(input("新的正规表达式：").strip())
        else:
            print("无效选项，请重新选择。")


def parse_args():
    parser = argparse.ArgumentParser(description="正则 -> NFA -> DFA -> 最小 DFA（lab2-new）")
    parser.add_argument("--regex", help="指定正则表达式，默认 (a|b)*abb")
    parser.add_argument("--strings", nargs="*", help="待检测的字符串列表（空格分隔）")
    parser.add_argument("--output-html", dest="output_html", help="生成离线可视化 HTML 文件路径")
    parser.add_argument("--no-png", action="store_true", help="导出 DOT 时不尝试生成 PNG")
    parser.add_argument("--no-interactive", action="store_true", help="仅执行导出/检测，不进入交互菜单")
    parser.add_argument("--dump-dot", action="store_true", help="额外写出 dot（nfa/dfa/mdfa）到当前目录")
    return parser.parse_args()


def main():
    args = parse_args()
    has_batch_inputs = bool(args.regex or args.strings or args.output_html or args.dump_dot)
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

