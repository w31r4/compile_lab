import streamlit as st
import pandas as pd

from src.automata import EPSILON, build_automata_from_regex, prepare_test_results, trace_dfa, trace_nfa
from src.exporting import get_dot_strings, render_dot_to_png_bytes
from src.html_export import generate_frontend_html


DEFAULT_REGEX = "(a|b)*abb"
DEFAULT_STRINGS = "abb\naabb\nab\n"

def _sanitize_filename(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    trimmed = cleaned.strip("_")
    return trimmed[:30] or "automaton"


def _collect_states_from_nfa(nfa) -> list[int]:
    states = {nfa.start, nfa.accept}
    for src, mapping in nfa.transitions.items():
        states.add(src)
        for dsts in mapping.values():
            states.update(dsts)
    return sorted(states)


def _collect_states_from_dfa(dfa) -> list[int]:
    states = {dfa.start, *dfa.accepts}
    for src, mapping in dfa.transitions.items():
        states.add(src)
        states.update(mapping.values())
    return sorted(states)


def _nfa_table_df(nfa) -> pd.DataFrame:
    symbols = sorted({sym for m in nfa.transitions.values() for sym in m if sym is not EPSILON})
    has_epsilon = any(EPSILON in m for m in nfa.transitions.values())
    cols = symbols + (["ε"] if has_epsilon else [])
    rows = []
    for s in _collect_states_from_nfa(nfa):
        row = {"state": s, "start": (s == nfa.start), "accept": (s == nfa.accept)}
        for sym in symbols:
            dsts = nfa.transitions.get(s, {}).get(sym, set())
            row[sym] = "{" + ",".join(map(str, sorted(dsts))) + "}" if dsts else ""
        if has_epsilon:
            dsts = nfa.transitions.get(s, {}).get(EPSILON, set())
            row["ε"] = "{" + ",".join(map(str, sorted(dsts))) + "}" if dsts else ""
        rows.append(row)
    df = pd.DataFrame(rows).set_index("state")
    return df[["start", "accept"] + cols]


def _dfa_table_df(dfa) -> pd.DataFrame:
    symbols = sorted(dfa.alphabet)
    rows = []
    for s in _collect_states_from_dfa(dfa):
        row = {"state": s, "start": (s == dfa.start), "accept": (s in dfa.accepts)}
        for sym in symbols:
            nxt = dfa.transitions.get(s, {}).get(sym)
            row[sym] = "" if nxt is None else str(nxt)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("state")
    return df[["start", "accept"] + symbols]


def main() -> None:
    st.set_page_config(page_title="Lab2-new 自动机可视化", page_icon="🔁", layout="wide")

    st.title("🔁 Lab2-new：正则 → NFA → DFA → 最小 DFA")
    st.caption("后端算法在 `lab2-new/src/`，前端使用 Streamlit 展示与导出。")

    with st.sidebar:
        st.header("输入")
        regex = st.text_input("正则表达式", value=DEFAULT_REGEX, help="支持：括号()、并|、闭包*、隐式连接。")
        st.divider()
        st.header("检测")
        single = st.text_input("单串检测", value="abb")
        multi = st.text_area("批量检测（每行一个）", value=DEFAULT_STRINGS, height=160)

    if not regex.strip():
        st.info("请输入正则表达式。")
        return

    try:
        nfa, dfa, mdfa = build_automata_from_regex(regex.strip())
    except Exception as exc:
        st.error(f"构造自动机失败：{exc}")
        return

    meta = {
        "alphabet": sorted(dfa.alphabet),
        "dfa_states": len(dfa.transitions),
        "mdfa_states": len(mdfa.transitions),
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("字母表", ", ".join(meta["alphabet"]) if meta["alphabet"] else "(空)")
    c2.metric("DFA 状态数", meta["dfa_states"])
    c3.metric("最小 DFA 状态数", meta["mdfa_states"])

    tabs = st.tabs(["图形", "转移表", "检测结果", "导出"])
    dots = get_dot_strings(nfa, dfa, mdfa)

    with tabs[0]:
        t1, t2, t3 = st.tabs(["NFA", "DFA", "MinDFA"])
        for tab, key, title in [(t1, "nfa", "NFA"), (t2, "dfa", "DFA"), (t3, "mdfa", "MinDFA")]:
            with tab:
                st.subheader(title)
                st.graphviz_chart(dots[key], use_container_width=True)
                with st.expander("DOT 源码", expanded=False):
                    st.code(dots[key], language="dot")

    with tabs[1]:
        st.subheader("NFA 转移表")
        st.dataframe(_nfa_table_df(nfa), use_container_width=True)
        st.subheader("DFA 转移表")
        st.dataframe(_dfa_table_df(dfa), use_container_width=True)
        st.subheader("最小 DFA 转移表")
        st.dataframe(_dfa_table_df(mdfa), use_container_width=True)

    with tabs[2]:
        st.subheader("单串")
        nfa_path, nfa_ok, nfa_reason = trace_nfa(nfa, single)
        dfa_path, dfa_ok, dfa_reason = trace_dfa(dfa, single)
        mdfa_path, mdfa_ok, mdfa_reason = trace_dfa(mdfa, single)
        st.write(f"NFA：`{nfa_ok}`，原因：{nfa_reason}；路径：`{' -> '.join(map(str, nfa_path))}`")
        st.write(f"DFA：`{dfa_ok}`，原因：{dfa_reason}；路径：`{' -> '.join(map(str, dfa_path))}`")
        st.write(f"MinDFA：`{mdfa_ok}`，原因：{mdfa_reason}；路径：`{' -> '.join(map(str, mdfa_path))}`")

        st.divider()
        st.subheader("批量")
        inputs = [line.strip() for line in multi.splitlines() if line.strip()]
        results = prepare_test_results(nfa, dfa, mdfa, inputs)
        if results:
            df = pd.DataFrame(
                [
                    {
                        "input": r["input"],
                        "NFA": r["nfa_accept"],
                        "DFA": r["dfa_accept"],
                        "MinDFA": r["mdfa_accept"],
                        "MinDFA path": " -> ".join(map(str, r["mdfa_path"])),
                    }
                    for r in results
                ]
            )
            st.dataframe(df, use_container_width=True)
        else:
            st.info("未输入批量字符串。")

    with tabs[3]:
        st.subheader("Graphviz 导出")
        base = f"automaton_{_sanitize_filename(regex)}"
        st.download_button("下载 NFA dot", data=dots["nfa"], file_name=f"{base}_nfa.dot", mime="text/vnd.graphviz")
        st.download_button("下载 DFA dot", data=dots["dfa"], file_name=f"{base}_dfa.dot", mime="text/vnd.graphviz")
        st.download_button("下载 MinDFA dot", data=dots["mdfa"], file_name=f"{base}_mdfa.dot", mime="text/vnd.graphviz")

        png_nfa = render_dot_to_png_bytes(dots["nfa"])
        png_dfa = render_dot_to_png_bytes(dots["dfa"])
        png_mdfa = render_dot_to_png_bytes(dots["mdfa"])
        if png_nfa and png_dfa and png_mdfa:
            st.download_button("下载 NFA png", data=png_nfa, file_name=f"{base}_nfa.png", mime="image/png")
            st.download_button("下载 DFA png", data=png_dfa, file_name=f"{base}_dfa.png", mime="image/png")
            st.download_button("下载 MinDFA png", data=png_mdfa, file_name=f"{base}_mdfa.png", mime="image/png")
        else:
            st.info("未检测到可用的 `dot` 渲染（或渲染失败），只能下载 dot。")

        st.divider()
        st.subheader("离线 HTML 导出（lab2 功能继承）")
        batch_inputs = [line.strip() for line in multi.splitlines() if line.strip()]
        results = prepare_test_results(nfa, dfa, mdfa, batch_inputs)
        html = generate_frontend_html(regex=regex.strip(), test_results=results, meta=meta)
        st.download_button("下载 visualization.html", data=html, file_name="visualization.html", mime="text/html")


if __name__ == "__main__":
    main()
