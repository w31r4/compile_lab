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
    "⭐ 全部13种语义错误 (test_12)": "test_cases/test_12_all_semantic_errors.sy",
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
    """运行词法分析器，返回 (tokens, has_error, error_output)"""
    # 捕获错误输出
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    lexer = Lexer(source_code)
    tokens = []
    has_error = False

    try:
        tokens = lexer.tokenize()
        has_error = lexer.has_error
    except LexerError as e:
        has_error = True

    error_output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    return tokens, has_error, error_output


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

    # 顶部：支持的错误类型说明
    with st.expander("📋 支持识别的错误类型 (共17种语义错误)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
            **词法错误 (Error type A)**
            - 非法字符
            - 未闭合的注释
            - 非法的十六进制浮点数 (如 0x1.5p10)
            
            **语法错误 (Error type B)**
            - 缺少分号、括号等
            - 语法结构不完整
            - 表达式语法错误
            
            **语义错误 (17种)**
            | 类型 | 描述 |
            |------|------|
            | Error 1 | 变量未声明 |
            | Error 2 | 变量重复声明 |
            | Error 3 | 调用未定义的函数 |
            | Error 4 | 函数重复定义 |
            | Error 5 | 把变量当做函数调用 |
            | Error 6 | 函数名当普通变量引用 |
            | Error 7 | 数组下标不是整型 |
            | Error 8 | 非数组变量使用数组访问 |
            | Error 9 | 函数参数数量或类型不匹配 |
            """
            )
        with col2:
            st.markdown(
                """
            | 类型 | 描述 |
            |------|------|
            | Error 10 | return类型与函数返回类型不匹配 |
            | Error 11 | 操作数类型不匹配 |
            | Error 12 | break语句不在循环体内 |
            | Error 13 | continue语句不在循环体内 |
            | **Error 14** | **数组越界访问** |
            | **Error 15** | **修改常量** |
            | **Error 16** | **void函数返回值被使用** |
            | **Error 17** | **缺少main函数** |
            """
            )

    # 侧边栏 - 测试用例选择
    st.sidebar.header("📁 导入测试用例")
    selected_test = st.sidebar.selectbox(
        "选择预设测试用例", ["自定义输入"] + list(TEST_CASES.keys()), help="选择后代码将导入编辑器，可自由修改"
    )

    # 导入按钮
    if st.sidebar.button("📥 导入到编辑器", use_container_width=True):
        if selected_test != "自定义输入":
            filepath = TEST_CASES[selected_test]
            st.session_state.source_code = load_test_file(filepath)
        else:
            st.session_state.source_code = DEFAULT_CODE

    # 初始化 session state
    if "source_code" not in st.session_state:
        if selected_test == "自定义输入":
            st.session_state.source_code = DEFAULT_CODE
        else:
            filepath = TEST_CASES[selected_test]
            st.session_state.source_code = load_test_file(filepath)

    # 侧边栏 - 分析选项
    st.sidebar.header("⚙️ 分析选项")
    show_lexer = st.sidebar.checkbox("显示词法分析详情", value=False)
    show_parser = st.sidebar.checkbox("显示语法分析详情", value=False)
    show_semantic = st.sidebar.checkbox("显示语义分析详情", value=False)

    # 主区域 - 代码编辑器（更大的输入框）
    st.subheader("📝 源代码编辑器")
    source_code = st.text_area(
        "SysY 源代码",
        value=st.session_state.source_code,
        height=400,  # 更大的高度
        help="在此输入或编辑 SysY 代码，修改后自动重新分析",
    )

    # 同步到 session state
    st.session_state.source_code = source_code

    if not source_code.strip():
        st.info("请在上方输入 SysY 代码以开始分析")
        return

    # ========== 实时错误显示（紧贴代码编辑器下方）==========
    all_errors = []

    # 运行词法分析
    tokens, lex_has_error, lex_error_output = run_lexer(source_code)
    if lex_error_output:
        for line in lex_error_output.strip().split("\n"):
            if line.strip():
                all_errors.append(("A", line.strip()))

    # 运行语法分析
    ast = None
    parse_errors = []
    ast_output = ""
    parse_error_output = ""
    if tokens:
        ast, parse_errors, ast_output, parse_error_output = run_parser(tokens)
        if parse_error_output:
            for line in parse_error_output.strip().split("\n"):
                if line.strip():
                    all_errors.append(("B", line.strip()))

    # 运行语义分析
    semantic_errors = []
    semantic_error_output = ""
    semantic_success = True
    if ast and not parse_errors:
        semantic_success, semantic_errors, semantic_error_output = run_semantic(ast)
        if semantic_error_output:
            for line in semantic_error_output.strip().split("\n"):
                if line.strip():
                    all_errors.append(("语义", line.strip()))

    # 显示错误面板（像IDE一样紧贴编辑器下方）
    if all_errors:
        st.markdown("---")
        st.markdown("### ❌ 问题面板")
        for err_type, err_msg in all_errors:
            if err_type == "A":
                st.error(f"🔤 词法错误: {err_msg}")
            elif err_type == "B":
                st.error(f"🌳 语法错误: {err_msg}")
            else:
                st.warning(f"🔍 语义错误: {err_msg}")
    else:
        st.success("✅ 无错误 - 程序分析通过!")

    # 运行分析详情
    st.divider()

    # ========== 词法分析详情 ==========
    if show_lexer:
        st.subheader("🔤 任务 4.2: 词法分析详情")

        if tokens:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Token 列表
                token_lines = []
                for token in tokens:
                    token_lines.append(token.to_string())

                with st.expander(f"Token 列表 ({len(tokens)} 个)", expanded=False):
                    st.code("\n".join(token_lines), language="text")

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

        st.divider()

    # ========== 语法分析详情 ==========
    if show_parser:
        st.subheader("🌳 任务 4.3: 语法分析详情")

        if ast_output:
            with st.expander("抽象语法树 (AST)", expanded=False):
                st.code(ast_output, language="text")

        st.divider()

    # ========== 语义分析详情 ==========
    if show_semantic:
        st.subheader("🔍 任务 4.4: 语义分析详情")

        if ast and not parse_errors:
            if semantic_success:
                st.info("符号表构建成功，无语义错误")
            else:
                st.info(f"发现 {len(semantic_errors)} 个语义错误（详见上方问题面板）")
        else:
            st.info("语法分析未完成，无法进行语义分析")

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
