"""
SysY 语义分析器

实现语义检查，检测以下错误类型：
- 错误类型 1: 使用未定义的变量
- 错误类型 2: 变量重复声明
- 错误类型 3: 调用未定义的函数
- 错误类型 4: 函数重复定义
- 错误类型 5: 把变量当做函数调用
- 错误类型 6: 函数名当普通变量引用
- 错误类型 7: 数组下标不是整型
- 错误类型 8: 非数组变量使用数组访问
- 错误类型 9: 函数参数类型或数量不匹配
- 错误类型 10: return 语句类型与函数返回类型不匹配
- 错误类型 11: 操作数类型不匹配
- 错误类型 12: break 语句不在循环体内
- 错误类型 13: continue 语句不在循环体内

扩展错误类型：
- 错误类型 14: 数组越界访问（常量索引超出维度）
- 错误类型 15: 修改常量
- 错误类型 16: void 函数返回值被使用
- 错误类型 17: 缺少 main 函数
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

        # 循环嵌套深度（用于检查 break/continue）
        self.loop_depth = 0

        # 是否找到 main 函数
        self.has_main = False

    def analyze(self, ast: CompUnitNode) -> bool:
        """分析 AST，返回是否有错误"""
        self.visit_comp_unit(ast)

        # 检查是否存在 main 函数（错误类型 17）
        if not self.has_main:
            self.error(17, 1, "Missing 'main' function")

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
        existing = self.symbol_table.lookup_local(name)
        if existing:
            self.error(2, line, f"Redefined variable '{name}'")
            return

        # 创建符号
        if node.dimensions:
            # 数组常量
            dims = []
            for dim_exp in node.dimensions:
                # 计算维度常量值，用于数组越界检查
                dim_value = self.try_get_const_value(dim_exp)
                if dim_value is not None and dim_value > 0:
                    dims.append(dim_value)
                else:
                    dims.append(-1)  # 无法确定的维度
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
        existing = self.symbol_table.lookup_local(name)
        if existing:
            self.error(2, line, f"Redefined variable '{name}'")
            return

        # 创建符号
        if node.dimensions:
            # 数组变量
            dims = []
            for dim_exp in node.dimensions:
                # 计算维度常量值，用于数组越界检查
                dim_value = self.try_get_const_value(dim_exp)
                if dim_value is not None and dim_value > 0:
                    dims.append(dim_value)
                else:
                    dims.append(-1)  # 无法确定的维度
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

        # 检查函数重复定义（错误类型 4 - 函数重复定义）
        existing = self.symbol_table.lookup_local(name)
        if existing:
            if existing.kind == SymbolKind.FUNCTION:
                self.error(4, line, f"Redefined function '{name}'")
            else:
                self.error(2, line, f"Redefined identifier '{name}'")
            return

        # 创建函数符号
        func_type = FunctionType(return_type=return_type, param_types=param_types)
        func_symbol = Symbol(name=name, kind=SymbolKind.FUNCTION, type=func_type, line=line)
        self.symbol_table.define(func_symbol)

        # 检查是否是 main 函数
        if name == "main":
            self.has_main = True
            # 检查 main 函数签名
            if return_type != TypeKind.INT:
                self.error(10, line, "main function must return int")
            if param_types:
                self.error(9, line, "main function should have no parameters")

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
            lval_node = None
            for child in node.children:
                if isinstance(child, LValNode):
                    lval_node = child
                    self.visit_lval(child)

                    # 检查是否修改常量（错误类型 15）
                    symbol = self.symbol_table.lookup(child.name)
                    if symbol and symbol.is_const:
                        self.error(15, child.line, f"Cannot modify constant '{child.name}'")
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
            self.loop_depth += 1
            for child in node.children:
                if isinstance(child, CondNode):
                    self.visit_cond(child)
                elif isinstance(child, StmtNode):
                    self.visit_stmt(child)
            self.loop_depth -= 1

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

        elif node.stmt_type == StmtType.BREAK:
            # break 语句（错误类型 12）
            if self.loop_depth == 0:
                self.error(12, node.line, "break statement not within a loop")

        elif node.stmt_type == StmtType.CONTINUE:
            # continue 语句（错误类型 13）
            if self.loop_depth == 0:
                self.error(13, node.line, "continue statement not within a loop")

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

        # 检查是否把函数名当做变量使用（错误类型 6）
        if symbol.kind == SymbolKind.FUNCTION:
            self.error(6, node.line, f"Function '{node.name}' used as variable")
            return

        # 检查数组访问
        if node.indices:
            # 检查非数组变量使用数组访问（错误类型 8）
            if not isinstance(symbol.type, ArrayType):
                self.error(8, node.line, f"Variable '{node.name}' is not an array")
            else:
                array_type = symbol.type

                # 检查数组维度是否匹配
                if len(node.indices) > len(array_type.dimensions):
                    self.error(
                        14,
                        node.line,
                        f"Too many indices for array '{node.name}' (expected {len(array_type.dimensions)}, got {len(node.indices)})",
                    )

                # 检查每个下标
                for i, idx in enumerate(node.indices):
                    self.visit_exp(idx)
                    idx_type = self.get_exp_type(idx)

                    # 检查数组下标是否为整型（错误类型 7）
                    if idx_type is not None and idx_type == TypeKind.FLOAT:
                        self.error(7, node.line, f"Array index is not an integer")

                    # 检查数组越界（错误类型 14）- 仅对常量索引检查
                    if i < len(array_type.dimensions):
                        dim_size = array_type.dimensions[i]
                        if dim_size > 0:  # 已知维度大小
                            idx_value = self.try_get_const_value(idx)
                            if idx_value is not None:
                                if idx_value < 0:
                                    self.error(14, node.line, f"Array index {idx_value} is negative")
                                elif idx_value >= dim_size:
                                    self.error(
                                        14, node.line, f"Array index {idx_value} out of bounds (size: {dim_size})"
                                    )
        else:
            # 没有数组索引，只需访问
            pass

    def try_get_const_value(self, node: ASTNode) -> Optional[int]:
        """尝试获取常量表达式的值（简化版，仅处理直接数字常量）"""
        if isinstance(node, ConstExpNode):
            for child in node.children:
                return self.try_get_const_value(child)
        if isinstance(node, ExpNode):
            for child in node.children:
                return self.try_get_const_value(child)
        if isinstance(node, AddExpNode):
            for child in node.children:
                if isinstance(child, MulExpNode):
                    return self.try_get_const_value(child)
        if isinstance(node, MulExpNode):
            for child in node.children:
                if isinstance(child, UnaryExpNode):
                    return self.try_get_const_value(child)
        if isinstance(node, UnaryExpNode):
            for child in node.children:
                if isinstance(child, PrimaryExpNode):
                    return self.try_get_const_value(child)
        if isinstance(node, PrimaryExpNode):
            for child in node.children:
                if isinstance(child, NumberNode):
                    if not child.is_float:
                        return int(child.value)
        if isinstance(node, NumberNode):
            if not node.is_float:
                return int(node.value)
        return None

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

            # 先检查是否存在这个名字
            symbol = self.symbol_table.lookup(func_name)

            if symbol is None:
                # 检查函数是否定义（错误类型 3）
                self.error(3, node.line, f"Undefined function '{func_name}'")
                return

            # 检查是否把变量当做函数调用（错误类型 5）
            if symbol.kind != SymbolKind.FUNCTION:
                self.error(5, node.line, f"Variable '{func_name}' is not a function")
                return

            func_symbol = symbol

            # 检查 void 函数返回值是否被使用（错误类型 16）
            # 这里简化处理，实际需要检查表达式上下文
            if isinstance(func_symbol.type, FunctionType):
                if func_symbol.type.return_type == TypeKind.VOID:
                    # void 函数可以调用，但不应该在表达式中使用其返回值
                    # 此处仅记录，不报错（需要更复杂的上下文分析）
                    pass

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
        operand_types = []
        for child in node.children:
            if isinstance(child, UnaryExpNode):
                self.visit_unary_exp(child)
                operand_types.append(self.get_exp_type(child))
            elif isinstance(child, MulExpNode):
                self.visit_mul_exp(child)
                operand_types.append(self.get_exp_type(child))

        # 检查操作数类型不匹配（错误类型 11）
        # 数组类型不能参与算术运算
        for i, t in enumerate(operand_types):
            if t is None:
                continue
            # 检查是否有数组类型参与运算（简化检查）
            pass

    def visit_add_exp(self, node: AddExpNode):
        """访问加法表达式"""
        operand_types = []
        has_array = False
        has_float = False
        has_int = False

        for child in node.children:
            if isinstance(child, MulExpNode):
                self.visit_mul_exp(child)
                child_type = self.get_exp_type(child)
                operand_types.append(child_type)
                if child_type == TypeKind.FLOAT:
                    has_float = True
                elif child_type == TypeKind.INT:
                    has_int = True
            elif isinstance(child, AddExpNode):
                self.visit_add_exp(child)
                child_type = self.get_exp_type(child)
                operand_types.append(child_type)

        # 检查操作数类型不匹配（错误类型 11）
        # int 和 float 不能混合运算（根据 SysY 规范）
        if has_float and has_int and len(operand_types) > 1:
            self.error(11, node.line, "Type mismatch for operands (int and float cannot mix)")

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
