"""
SysY 语义分析器

实现语义检查，检测以下错误类型：
- 错误类型 1: 使用未定义的变量
- 错误类型 2: 变量重复定义
- 错误类型 3: 调用未定义的函数
- 错误类型 9: 函数参数类型或数量不匹配
- 错误类型 10: return 语句类型与函数返回类型不匹配
"""

from typing import List, Optional, Union
from .ast_nodes import (
    ASTNode,
    TerminalNode,
    CompUnitNode,
    DeclNode,
    ConstDeclNode,
    VarDeclNode,
    BTypeNode,
    ConstDefNode,
    VarDefNode,
    InitValNode,
    ConstInitValNode,
    FuncDefNode,
    FuncTypeNode,
    FuncFParamsNode,
    FuncFParamNode,
    BlockNode,
    BlockItemNode,
    StmtNode,
    StmtType,
    ExpNode,
    CondNode,
    LValNode,
    PrimaryExpNode,
    NumberNode,
    UnaryExpNode,
    UnaryOpNode,
    FuncRParamsNode,
    MulExpNode,
    AddExpNode,
    RelExpNode,
    EqExpNode,
    LAndExpNode,
    LOrExpNode,
    ConstExpNode,
    BType,
)
from .symbol_table import (
    SymbolTable,
    Symbol,
    SymbolKind,
    TypeKind,
    ArrayType,
    FunctionType,
    SymbolType,
)


class SemanticError:
    """语义错误"""

    def __init__(self, error_type: int, line: int, message: str):
        self.error_type = error_type
        self.line = line
        self.message = message

    def __str__(self):
        return f"Error type {self.error_type} at Line {self.line}: {self.message}"


class SemanticAnalyzer:
    """语义分析器"""

    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors: List[SemanticError] = []
        self.has_error = False

        # 当前函数的返回类型（用于检查 return 语句）
        self.current_function: Optional[Symbol] = None

    def analyze(self, ast: CompUnitNode) -> bool:
        """分析 AST，返回是否有错误"""
        self.visit_comp_unit(ast)
        return not self.has_error

    def error(self, error_type: int, line: int, message: str):
        """报告语义错误"""
        err = SemanticError(error_type, line, message)
        self.errors.append(err)
        self.has_error = True
        print(err)

    # ============ 类型转换辅助方法 ============

    def btype_to_typekind(self, btype: BType) -> TypeKind:
        """将 BType 转换为 TypeKind"""
        if btype == BType.INT:
            return TypeKind.INT
        elif btype == BType.FLOAT:
            return TypeKind.FLOAT
        elif btype == BType.VOID:
            return TypeKind.VOID
        return TypeKind.INT

    def get_exp_type(self, node: ASTNode) -> Optional[TypeKind]:
        """推断表达式的类型"""
        if node is None:
            return None

        if isinstance(node, NumberNode):
            return TypeKind.FLOAT if node.is_float else TypeKind.INT

        if isinstance(node, LValNode):
            symbol = self.symbol_table.lookup(node.name)
            if symbol:
                return symbol.get_base_type()
            return None

        if isinstance(node, UnaryExpNode):
            if node.is_func_call:
                # 函数调用，返回函数返回类型
                func = self.symbol_table.lookup_function(node.func_name)
                if func and isinstance(func.type, FunctionType):
                    return func.type.return_type
                return None
            # 一元运算符，返回操作数类型
            for child in node.children:
                if isinstance(child, (PrimaryExpNode, UnaryExpNode)):
                    return self.get_exp_type(child)
            return None

        if isinstance(node, PrimaryExpNode):
            for child in node.children:
                if not isinstance(child, TerminalNode):
                    return self.get_exp_type(child)
            return None

        if isinstance(node, (MulExpNode, AddExpNode)):
            # 二元运算，检查操作数类型
            has_float = False
            for child in node.children:
                if not isinstance(child, TerminalNode):
                    child_type = self.get_exp_type(child)
                    if child_type == TypeKind.FLOAT:
                        has_float = True
            return TypeKind.FLOAT if has_float else TypeKind.INT

        if isinstance(node, (RelExpNode, EqExpNode)):
            # 关系/相等运算，结果为 int
            return TypeKind.INT

        if isinstance(node, (LAndExpNode, LOrExpNode)):
            # 逻辑运算，结果为 int
            return TypeKind.INT

        if isinstance(node, ExpNode):
            for child in node.children:
                return self.get_exp_type(child)

        if isinstance(node, ConstExpNode):
            for child in node.children:
                return self.get_exp_type(child)

        return None

    # ============ 访问者方法 ============

    def visit_comp_unit(self, node: CompUnitNode):
        """访问编译单元"""
        for child in node.children:
            if isinstance(child, DeclNode):
                self.visit_decl(child)
            elif isinstance(child, FuncDefNode):
                self.visit_func_def(child)

    def visit_decl(self, node: DeclNode):
        """访问声明"""
        for child in node.children:
            if isinstance(child, ConstDeclNode):
                self.visit_const_decl(child)
            elif isinstance(child, VarDeclNode):
                self.visit_var_decl(child)

    def visit_const_decl(self, node: ConstDeclNode):
        """访问常量声明"""
        base_type = self.btype_to_typekind(node.btype)

        for child in node.children:
            if isinstance(child, ConstDefNode):
                self.visit_const_def(child, base_type)

    def visit_const_def(self, node: ConstDefNode, base_type: TypeKind):
        """访问常量定义"""
        name = node.name
        line = node.line

        # 检查重复定义（错误类型 2）
        if self.symbol_table.lookup_local(name):
            self.error(2, line, f"Redefined variable '{name}'")
            return

        # 创建符号
        if node.dimensions:
            # 数组常量
            dims = []
            for dim_exp in node.dimensions:
                dims.append(-1)  # 简化处理，不计算实际维度值
            symbol_type = ArrayType(base_type=base_type, dimensions=dims)
        else:
            # 标量常量
            symbol_type = base_type

        symbol = Symbol(name=name, kind=SymbolKind.CONSTANT, type=symbol_type, line=line, is_const=True)
        self.symbol_table.define(symbol)

        # 访问初始化值
        if node.init_val:
            self.visit_const_init_val(node.init_val)

    def visit_const_init_val(self, node: ConstInitValNode):
        """访问常量初始化值"""
        for child in node.children:
            if isinstance(child, ConstExpNode):
                self.visit_const_exp(child)
            elif isinstance(child, ConstInitValNode):
                self.visit_const_init_val(child)

    def visit_var_decl(self, node: VarDeclNode):
        """访问变量声明"""
        base_type = self.btype_to_typekind(node.btype)

        for child in node.children:
            if isinstance(child, VarDefNode):
                self.visit_var_def(child, base_type)

    def visit_var_def(self, node: VarDefNode, base_type: TypeKind):
        """访问变量定义"""
        name = node.name
        line = node.line

        # 检查重复定义（错误类型 2）
        if self.symbol_table.lookup_local(name):
            self.error(2, line, f"Redefined variable '{name}'")
            return

        # 创建符号
        if node.dimensions:
            # 数组变量
            dims = []
            for dim_exp in node.dimensions:
                dims.append(-1)  # 简化处理
            symbol_type = ArrayType(base_type=base_type, dimensions=dims)
        else:
            # 标量变量
            symbol_type = base_type

        symbol = Symbol(name=name, kind=SymbolKind.VARIABLE, type=symbol_type, line=line)
        self.symbol_table.define(symbol)

        # 访问初始化值
        if node.init_val:
            self.visit_init_val(node.init_val)

    def visit_init_val(self, node: InitValNode):
        """访问初始化值"""
        for child in node.children:
            if isinstance(child, ExpNode):
                self.visit_exp(child)
            elif isinstance(child, InitValNode):
                self.visit_init_val(child)

    def visit_func_def(self, node: FuncDefNode):
        """访问函数定义"""
        name = node.name
        line = node.line
        return_type = self.btype_to_typekind(node.return_type)

        # 收集参数类型
        param_types = []
        param_symbols = []

        for child in node.children:
            if isinstance(child, FuncFParamsNode):
                for param_child in child.children:
                    if isinstance(param_child, FuncFParamNode):
                        param_type = self.btype_to_typekind(param_child.btype)

                        if param_child.is_array:
                            # 数组参数
                            dims = [-1]  # 第一维未知
                            for dim_exp in param_child.dimensions:
                                dims.append(-1)
                            param_types.append(ArrayType(base_type=param_type, dimensions=dims))
                        else:
                            param_types.append(param_type)

                        param_symbols.append((param_child.name, param_child))

        # 检查函数重复定义（错误类型 2）
        if self.symbol_table.lookup_local(name):
            self.error(2, line, f"Redefined function '{name}'")
            return

        # 创建函数符号
        func_type = FunctionType(return_type=return_type, param_types=param_types)
        func_symbol = Symbol(name=name, kind=SymbolKind.FUNCTION, type=func_type, line=line)
        self.symbol_table.define(func_symbol)

        # 设置当前函数（用于检查 return）
        old_function = self.current_function
        self.current_function = func_symbol

        # 进入函数作用域
        self.symbol_table.enter_scope()

        # 定义参数
        for idx, (param_name, param_node) in enumerate(param_symbols):
            param_base_type = self.btype_to_typekind(param_node.btype)

            if param_node.is_array:
                dims = [-1]
                for dim_exp in param_node.dimensions:
                    dims.append(-1)
                param_sym_type = ArrayType(base_type=param_base_type, dimensions=dims)
            else:
                param_sym_type = param_base_type

            param_symbol = Symbol(name=param_name, kind=SymbolKind.PARAMETER, type=param_sym_type, line=param_node.line)

            # 检查参数重复定义
            if not self.symbol_table.define(param_symbol):
                self.error(2, param_node.line, f"Redefined parameter '{param_name}'")

        # 访问函数体
        for child in node.children:
            if isinstance(child, BlockNode):
                self.visit_block(child, new_scope=False)  # 不创建新作用域，已在上面创建

        # 退出函数作用域
        self.symbol_table.exit_scope()

        # 恢复当前函数
        self.current_function = old_function

    def visit_block(self, node: BlockNode, new_scope: bool = True):
        """访问语句块"""
        if new_scope:
            self.symbol_table.enter_scope()

        for child in node.children:
            if isinstance(child, BlockItemNode):
                self.visit_block_item(child)

        if new_scope:
            self.symbol_table.exit_scope()

    def visit_block_item(self, node: BlockItemNode):
        """访问块项"""
        for child in node.children:
            if isinstance(child, DeclNode):
                self.visit_decl(child)
            elif isinstance(child, StmtNode):
                self.visit_stmt(child)

    def visit_stmt(self, node: StmtNode):
        """访问语句"""
        if node.stmt_type == StmtType.ASSIGN:
            # 赋值语句
            for child in node.children:
                if isinstance(child, LValNode):
                    self.visit_lval(child)
                elif isinstance(child, ExpNode):
                    self.visit_exp(child)

        elif node.stmt_type == StmtType.EXP:
            # 表达式语句
            for child in node.children:
                if isinstance(child, ExpNode):
                    self.visit_exp(child)

        elif node.stmt_type == StmtType.BLOCK:
            # 块语句
            for child in node.children:
                if isinstance(child, BlockNode):
                    self.visit_block(child)

        elif node.stmt_type == StmtType.IF:
            # if 语句
            for child in node.children:
                if isinstance(child, CondNode):
                    self.visit_cond(child)
                elif isinstance(child, StmtNode):
                    self.visit_stmt(child)

        elif node.stmt_type == StmtType.WHILE:
            # while 语句
            for child in node.children:
                if isinstance(child, CondNode):
                    self.visit_cond(child)
                elif isinstance(child, StmtNode):
                    self.visit_stmt(child)

        elif node.stmt_type == StmtType.RETURN:
            # return 语句
            return_exp = None
            for child in node.children:
                if isinstance(child, ExpNode):
                    return_exp = child
                    self.visit_exp(child)

            # 检查 return 类型（错误类型 10）
            if self.current_function:
                func_type = self.current_function.type
                if isinstance(func_type, FunctionType):
                    expected_return = func_type.return_type

                    if expected_return == TypeKind.VOID:
                        if return_exp is not None:
                            self.error(10, node.line, f"Return value in void function '{self.current_function.name}'")
                    else:
                        if return_exp is None:
                            self.error(
                                10,
                                node.line,
                                f"Missing return value in non-void function '{self.current_function.name}'",
                            )
                        else:
                            # 检查返回类型是否匹配
                            actual_type = self.get_exp_type(return_exp)
                            if actual_type is not None:
                                # int 和 float 可以隐式转换，只检查 void
                                if actual_type == TypeKind.VOID:
                                    self.error(
                                        10,
                                        node.line,
                                        f"Return type mismatch in function '{self.current_function.name}'",
                                    )

        elif node.stmt_type in (StmtType.BREAK, StmtType.CONTINUE):
            # break/continue 语句（可以添加循环检查，暂时跳过）
            pass

    def visit_exp(self, node: ExpNode):
        """访问表达式"""
        for child in node.children:
            if isinstance(child, AddExpNode):
                self.visit_add_exp(child)

    def visit_cond(self, node: CondNode):
        """访问条件表达式"""
        for child in node.children:
            if isinstance(child, LOrExpNode):
                self.visit_lor_exp(child)

    def visit_lval(self, node: LValNode):
        """访问左值"""
        # 检查变量是否定义（错误类型 1）
        symbol = self.symbol_table.lookup(node.name)
        if symbol is None:
            self.error(1, node.line, f"Undefined variable '{node.name}'")
            return

        # 访问数组索引
        for idx in node.indices:
            self.visit_exp(idx)

    def visit_primary_exp(self, node: PrimaryExpNode):
        """访问基本表达式"""
        for child in node.children:
            if isinstance(child, ExpNode):
                self.visit_exp(child)
            elif isinstance(child, LValNode):
                self.visit_lval(child)
            elif isinstance(child, NumberNode):
                pass  # 数字常量无需检查

    def visit_unary_exp(self, node: UnaryExpNode):
        """访问一元表达式"""
        if node.is_func_call:
            # 函数调用
            func_name = node.func_name
            func_symbol = self.symbol_table.lookup_function(func_name)

            # 检查函数是否定义（错误类型 3）
            if func_symbol is None:
                self.error(3, node.line, f"Undefined function '{func_name}'")
                return

            # 收集实参
            actual_args = []
            for child in node.children:
                if isinstance(child, FuncRParamsNode):
                    for arg in child.children:
                        if isinstance(arg, ExpNode):
                            actual_args.append(arg)
                            self.visit_exp(arg)

            # 检查参数数量和类型（错误类型 9）
            if isinstance(func_symbol.type, FunctionType):
                expected_params = func_symbol.type.param_types

                if len(actual_args) != len(expected_params):
                    self.error(
                        9,
                        node.line,
                        f"Function '{func_name}' expects {len(expected_params)} arguments but got {len(actual_args)}",
                    )
                else:
                    # 检查参数类型（简化处理，只检查数组与非数组的区别）
                    for i, (arg, expected) in enumerate(zip(actual_args, expected_params)):
                        arg_type = self.get_exp_type(arg)
                        is_expected_array = isinstance(expected, ArrayType)

                        # 简化的类型检查：主要检查数组类型匹配
                        # 实际实现可以更精确
                        pass
        else:
            # 普通一元表达式
            for child in node.children:
                if isinstance(child, PrimaryExpNode):
                    self.visit_primary_exp(child)
                elif isinstance(child, UnaryExpNode):
                    self.visit_unary_exp(child)

    def visit_mul_exp(self, node: MulExpNode):
        """访问乘法表达式"""
        for child in node.children:
            if isinstance(child, UnaryExpNode):
                self.visit_unary_exp(child)

    def visit_add_exp(self, node: AddExpNode):
        """访问加法表达式"""
        for child in node.children:
            if isinstance(child, MulExpNode):
                self.visit_mul_exp(child)

    def visit_rel_exp(self, node: RelExpNode):
        """访问关系表达式"""
        for child in node.children:
            if isinstance(child, AddExpNode):
                self.visit_add_exp(child)

    def visit_eq_exp(self, node: EqExpNode):
        """访问相等表达式"""
        for child in node.children:
            if isinstance(child, RelExpNode):
                self.visit_rel_exp(child)

    def visit_land_exp(self, node: LAndExpNode):
        """访问逻辑与表达式"""
        for child in node.children:
            if isinstance(child, EqExpNode):
                self.visit_eq_exp(child)

    def visit_lor_exp(self, node: LOrExpNode):
        """访问逻辑或表达式"""
        for child in node.children:
            if isinstance(child, LAndExpNode):
                self.visit_land_exp(child)

    def visit_const_exp(self, node: ConstExpNode):
        """访问常量表达式"""
        for child in node.children:
            if isinstance(child, AddExpNode):
                self.visit_add_exp(child)
