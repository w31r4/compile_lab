"""
SysY 编译器前端可视化演示

使用 Streamlit 展示词法分析、语法分析、语义分析的完整流程。
"""

import streamlit as st
import os
from io import StringIO
import sys

from src.lexer import Lexer, LexerError
from src.parser import Parser
from src.ast_nodes import ASTPrinter
from src.semantic_analyzer import SemanticAnalyzer


# 默认示例代码
DEFAULT_CODE = """// SysY 示例程序
int main() {
    int student_id = 20220001;
    int a = 10;
    int b = 20;
    int sum = a + b;
    
    if (sum > 25) {
        return 1;
    } else {
        return 0;
    }
}
"""

# 预设测试用例
TEST_CASES = {
    "基础测试 (test_01)": "test_cases/test_01_basic.sy",
    "算术运算 (test_02)": "test_cases/test_02_arithmetic.sy",
    "控制流 (test_03)": "test_cases/test_03_control.sy",
    "函数定义 (test_04)": "test_cases/test_04_func.sy",
    "数组操作 (test_05)": "test_cases/test_05_array.sy",
    "常量与全局 (test_06)": "test_cases/test_06_const_global.sy",
    "浮点数 (test_07)": "test_cases/test_07_float.sy",
    "复杂程序 (test_08)": "test_cases/test_08_complex.sy",
    "词法错误 (test_09)": "test_cases/test_09_lex_error.sy",
    "语法错误 (test_10)": "test_cases/test_10_syntax_error.sy",
    "语义错误 (test_11)": "test_cases/test_11_semantic_error.sy",
    "八进制测试": "test_cases/test_octal.sy",
}


def load_test_file(filepath: str) -> str:
    """加载测试文件内容"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"// 无法加载文件: {e}"


def run_lexer(source_code: str) -> tuple:
    """运行词法分析器，返回 (tokens, errors, error_output)"""
    # 捕获错误输出
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    lexer = Lexer(source_code)
    tokens = []
    errors = []

    try:
        tokens = lexer.tokenize()
        errors = lexer.errors
    except LexerError as e:
        errors.append(str(e))

    error_output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    return tokens, errors, error_output


def run_parser(tokens: list) -> tuple:
    """运行语法分析器，返回 (ast, errors, ast_output, error_output)"""
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    parser = Parser(tokens)
    ast = None
    ast_output = ""

    try:
        ast = parser.parse()
        if not parser.has_error:
            printer = ASTPrinter()
            ast_output = printer.print_ast(ast)
    except Exception as e:
        pass

    error_output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    return ast, parser.errors, ast_output, error_output


def run_semantic(ast) -> tuple:
    """运行语义分析器，返回 (success, errors, error_output)"""
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    analyzer = SemanticAnalyzer()
    success = False

    try:
        success = analyzer.analyze(ast)
    except Exception as e:
        pass

    error_output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    return success, analyzer.errors, error_output


def main():
    st.set_page_config(page_title="SysY 编译器前端演示", page_icon="🔧", layout="wide")

    st.title("🔧 SysY 编译器前端演示")
    st.caption("实时展示词法分析、语法分析、语义分析的完整编译流程")

    # 侧边栏 - 测试用例选择
    st.sidebar.header("📁 测试用例")
    selected_test = st.sidebar.selectbox("选择预设测试用例", ["自定义输入"] + list(TEST_CASES.keys()))

    # 加载代码
    if selected_test == "自定义输入":
        initial_code = DEFAULT_CODE
    else:
        filepath = TEST_CASES[selected_test]
        initial_code = load_test_file(filepath)

    # 侧边栏 - 分析选项
    st.sidebar.header("⚙️ 分析选项")
    show_lexer = st.sidebar.checkbox("显示词法分析", value=True)
    show_parser = st.sidebar.checkbox("显示语法分析", value=True)
    show_semantic = st.sidebar.checkbox("显示语义分析", value=True)

    # 主区域 - 代码编辑器
    st.subheader("📝 源代码编辑器")
    source_code = st.text_area(
        "SysY 源代码", value=initial_code, height=300, help="在此输入或编辑 SysY 代码，修改后自动重新分析"
    )

    if not source_code.strip():
        st.info("请在上方输入 SysY 代码以开始分析")
        return

    # 运行分析
    st.divider()

    # ========== 词法分析 ==========
    if show_lexer:
        st.subheader("🔤 任务 4.2: 词法分析")

        tokens, lex_errors, lex_error_output = run_lexer(source_code)

        if lex_error_output:
            st.error("词法错误 (Error type A)")
            st.code(lex_error_output, language="text")

        if tokens:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Token 列表
                token_lines = []
                for token in tokens:
                    token_lines.append(token.to_string())

                with st.expander(f"Token 列表 ({len(tokens)} 个)", expanded=True):
                    st.code("\n".join(token_lines[:50]), language="text")
                    if len(tokens) > 50:
                        st.caption(f"... 还有 {len(tokens) - 50} 个 token")

            with col2:
                # 统计信息
                st.metric("Token 数量", len(tokens))

                # Token 类型分布
                type_count = {}
                for token in tokens:
                    t = token.type.name
                    type_count[t] = type_count.get(t, 0) + 1

                with st.expander("Token 类型分布"):
                    for t, count in sorted(type_count.items(), key=lambda x: -x[1])[:10]:
                        st.text(f"{t}: {count}")

        if lex_errors:
            st.warning(f"发现 {len(lex_errors)} 个词法错误")
            return

        st.success("✅ 词法分析完成")
    else:
        tokens, _, lex_error_output = run_lexer(source_code)
        if lex_error_output:
            st.error("词法错误，无法继续")
            st.code(lex_error_output, language="text")
            return

    st.divider()

    # ========== 语法分析 ==========
    if show_parser:
        st.subheader("🌳 任务 4.3: 语法分析")

        ast, parse_errors, ast_output, parse_error_output = run_parser(tokens)

        if parse_error_output:
            st.error("语法错误 (Error type B)")
            st.code(parse_error_output, language="text")

        if ast_output:
            with st.expander("抽象语法树 (AST)", expanded=True):
                # 限制显示行数
                lines = ast_output.split("\n")
                if len(lines) > 100:
                    st.code("\n".join(lines[:100]), language="text")
                    st.caption(f"... 还有 {len(lines) - 100} 行")
                else:
                    st.code(ast_output, language="text")

        if parse_errors:
            st.warning(f"发现 {len(parse_errors)} 个语法错误")
            if not show_semantic:
                return
        else:
            st.success("✅ 语法分析完成")
    else:
        ast, parse_errors, ast_output, parse_error_output = run_parser(tokens)
        if parse_error_output:
            st.error("语法错误，无法继续语义分析")
            st.code(parse_error_output, language="text")
            if show_semantic:
                return

    st.divider()

    # ========== 语义分析 ==========
    if show_semantic and ast and not parse_errors:
        st.subheader("🔍 任务 4.4: 语义分析")

        success, semantic_errors, semantic_error_output = run_semantic(ast)

        if semantic_error_output:
            st.error("语义错误")
            st.code(semantic_error_output, language="text")

            # 错误类型说明
            with st.expander("错误类型说明"):
                st.markdown(
                    """
                | 错误类型 | 描述 |
                |---------|------|
                | Error type 1 | 使用未定义的变量 |
                | Error type 2 | 变量/函数重复定义 |
                | Error type 3 | 调用未定义的函数 |
                | Error type 9 | 函数参数数量不匹配 |
                | Error type 10 | return 类型与函数返回类型不匹配 |
                """
                )

        if success:
            st.success("✅ 语义分析完成 - 程序无语义错误!")
            st.balloons()
        else:
            st.warning(f"发现 {len(semantic_errors)} 个语义错误")

    # 底部说明
    st.divider()
    with st.expander("📖 使用说明"):
        st.markdown(
            """
        ### 功能说明
        
        本应用展示 SysY 语言编译器前端的三个核心阶段：
        
        1. **词法分析 (Lexical Analysis)**
           - 将源代码分割成 Token 序列
           - 识别关键字、标识符、常量、运算符
           - 支持八进制(0123)、十六进制(0xFF) 数值转换
           - 检测非法字符、未闭合注释等错误 (Error type A)
        
        2. **语法分析 (Syntax Analysis)**
           - 使用递归下降解析器
           - 根据 SysY 文法构建抽象语法树 (AST)
           - 检测语法结构错误 (Error type B)
        
        3. **语义分析 (Semantic Analysis)**
           - 建立符号表，管理作用域
           - 检测变量未定义、重复定义
           - 检测函数未定义、参数不匹配
           - 检测 return 类型不匹配
        
        ### 测试用例说明
        
        - **test_01 ~ test_08**: 正确的 SysY 程序
        - **test_09**: 包含词法错误 (Error type A)
        - **test_10**: 包含语法错误 (Error type B)
        - **test_11**: 包含语义错误 (Error type 1, 2, 3, 9, 10)
        """
        )


if __name__ == "__main__":
    main()
