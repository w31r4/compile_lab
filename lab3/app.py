import streamlit as st

from src.parser import parse_grammar_string
from src.left_recursion import LeftRecursionEliminator
from src.left_factoring import LeftFactoringExtractor
from src.first_follow import FirstFollowCalculator

DEFAULT_GRAMMAR = """# 表达式文法（带左递归和左公因子）
# 非终结符
E, T, F
# 终结符
+, *, (, ), id
# 开始符号
E
# 产生式
E -> E + T | T
T -> T * F | F
F -> ( E ) | id
"""


def render_results(grammar_content: str) -> None:
    """Parse grammar and run all algorithms in order; render results inline."""
    try:
        grammar = parse_grammar_string(grammar_content)
    except Exception as exc:
        st.error(f"文法解析失败：{exc}")
        return

    st.success("✅ 文法解析成功")
    st.code(str(grammar), language="text")

    st.divider()
    st.subheader("任务3.1: 消除左递归")
    eliminator = LeftRecursionEliminator(grammar)
    grammar_no_lr = eliminator.eliminate()
    with st.expander("处理日志", expanded=False):
        st.text(eliminator.get_processing_log())
    st.code(str(grammar_no_lr), language="text")

    st.divider()
    st.subheader("任务3.2: 提取左公因子")
    extractor = LeftFactoringExtractor(grammar_no_lr)
    grammar_no_lf = extractor.extract()
    with st.expander("处理日志", expanded=False):
        st.text(extractor.get_processing_log())
    st.code(str(grammar_no_lf), language="text")

    st.divider()
    st.subheader("任务3.3: FIRST 集")
    calculator = FirstFollowCalculator(grammar_no_lf)
    calculator.compute_first_sets()
    with st.expander("计算过程", expanded=False):
        st.text(calculator.get_processing_log(include_follow=False))
    st.code(calculator.get_first_sets_str(), language="text")

    st.divider()
    st.subheader("任务3.3: FOLLOW 集")
    calculator.clear_log()
    calculator.compute_follow_sets()
    with st.expander("计算过程", expanded=False):
        st.text(calculator.get_processing_log(include_first=False))
    st.code(calculator.get_follow_sets_str(), language="text")


def main() -> None:
    st.set_page_config(page_title="语法分析算法可视化", page_icon="🎯", layout="wide")

    st.title("🎯 语法分析算法可视化")
    st.caption("输入文法后自动解析并依次执行左递归消除、左公因子提取、FIRST/FOLLOW 计算。")

    grammar_content = st.text_area(
        "文法输入",
        value=DEFAULT_GRAMMAR,
        height=260,
        help="遵循四段式：非终结符、终结符、开始符号、产生式。内容变更后将自动处理。",
    )

    if grammar_content.strip():
        render_results(grammar_content)
    else:
        st.info("请在上方输入文法以开始。")


if __name__ == "__main__":
    main()
