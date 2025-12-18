"""
AST 节点定义 - SysY 语法分析器

基于 SysY 2022 文法定义的抽象语法树节点类型。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from enum import Enum, auto


class NodeType(Enum):
    """AST 节点类型枚举"""

    # 顶层结构
    COMP_UNIT = auto()

    # 声明
    DECL = auto()
    CONST_DECL = auto()
    VAR_DECL = auto()
    CONST_DEF = auto()
    VAR_DEF = auto()
    CONST_INIT_VAL = auto()
    INIT_VAL = auto()

    # 类型
    BTYPE = auto()
    FUNC_TYPE = auto()

    # 函数
    FUNC_DEF = auto()
    FUNC_FPARAMS = auto()
    FUNC_FPARAM = auto()
    FUNC_RPARAMS = auto()

    # 语句块
    BLOCK = auto()
    BLOCK_ITEM = auto()

    # 语句
    STMT = auto()

    # 表达式
    EXP = auto()
    COND = auto()
    LVAL = auto()
    PRIMARY_EXP = auto()
    NUMBER = auto()
    UNARY_EXP = auto()
    UNARY_OP = auto()
    MUL_EXP = auto()
    ADD_EXP = auto()
    REL_EXP = auto()
    EQ_EXP = auto()
    LAND_EXP = auto()
    LOR_EXP = auto()
    CONST_EXP = auto()

    # 终结符/词法单元
    TERMINAL = auto()


class BType(Enum):
    """基本类型"""

    INT = "int"
    FLOAT = "float"
    VOID = "void"


@dataclass
class ASTNode:
    """AST 节点基类"""

    node_type: NodeType = NodeType.COMP_UNIT  # 默认值，子类的 __post_init__ 会覆盖
    line: int = 0  # 行号（对应第一个词法单元的行号）
    children: List["ASTNode"] = field(default_factory=list)

    def add_child(self, child: "ASTNode"):
        if child is not None:
            self.children.append(child)

    def add_children(self, children: List["ASTNode"]):
        for child in children:
            if child is not None:
                self.children.append(child)


@dataclass
class TerminalNode(ASTNode):
    """终结符节点 - 代表词法单元"""

    token_type: str = ""  # 词法单元类型名称 (如 "INTTK", "IDENFR")
    value: str = ""  # 词素值

    def __post_init__(self):
        self.node_type = NodeType.TERMINAL


@dataclass
class CompUnitNode(ASTNode):
    """编译单元节点 - CompUnit → [CompUnit] (Decl | FuncDef)"""

    def __post_init__(self):
        self.node_type = NodeType.COMP_UNIT


@dataclass
class DeclNode(ASTNode):
    """声明节点 - Decl → ConstDecl | VarDecl"""

    def __post_init__(self):
        self.node_type = NodeType.DECL


@dataclass
class ConstDeclNode(ASTNode):
    """常量声明节点 - ConstDecl → 'const' BType ConstDef { ',' ConstDef } ';'"""

    btype: BType = BType.INT

    def __post_init__(self):
        self.node_type = NodeType.CONST_DECL


@dataclass
class VarDeclNode(ASTNode):
    """变量声明节点 - VarDecl → BType VarDef { ',' VarDef } ';'"""

    btype: BType = BType.INT

    def __post_init__(self):
        self.node_type = NodeType.VAR_DECL


@dataclass
class BTypeNode(ASTNode):
    """基本类型节点"""

    btype: BType = BType.INT

    def __post_init__(self):
        self.node_type = NodeType.BTYPE


@dataclass
class ConstDefNode(ASTNode):
    """常量定义节点 - ConstDef → Ident { '[' ConstExp ']' } '=' ConstInitVal"""

    name: str = ""
    dimensions: List["ASTNode"] = field(default_factory=list)  # 数组维度
    init_val: Optional["ASTNode"] = None

    def __post_init__(self):
        self.node_type = NodeType.CONST_DEF


@dataclass
class VarDefNode(ASTNode):
    """变量定义节点 - VarDef → Ident { '[' ConstExp ']' } ['=' InitVal]"""

    name: str = ""
    dimensions: List["ASTNode"] = field(default_factory=list)  # 数组维度
    init_val: Optional["ASTNode"] = None

    def __post_init__(self):
        self.node_type = NodeType.VAR_DEF


@dataclass
class InitValNode(ASTNode):
    """初始值节点 - InitVal → Exp | '{' [InitVal {',' InitVal}] '}'"""

    is_array: bool = False  # 是否是数组初始化

    def __post_init__(self):
        self.node_type = NodeType.INIT_VAL


@dataclass
class ConstInitValNode(ASTNode):
    """常量初始值节点"""

    is_array: bool = False

    def __post_init__(self):
        self.node_type = NodeType.CONST_INIT_VAL


@dataclass
class FuncDefNode(ASTNode):
    """函数定义节点 - FuncDef → FuncType Ident '(' [FuncFParams] ')' Block"""

    return_type: BType = BType.INT
    name: str = ""
    params: List["ASTNode"] = field(default_factory=list)
    body: Optional["ASTNode"] = None

    def __post_init__(self):
        self.node_type = NodeType.FUNC_DEF


@dataclass
class FuncTypeNode(ASTNode):
    """函数类型节点"""

    return_type: BType = BType.INT

    def __post_init__(self):
        self.node_type = NodeType.FUNC_TYPE


@dataclass
class FuncFParamsNode(ASTNode):
    """函数形参列表节点"""

    def __post_init__(self):
        self.node_type = NodeType.FUNC_FPARAMS


@dataclass
class FuncFParamNode(ASTNode):
    """函数形参节点 - FuncFParam → BType Ident ['[' ']' { '[' Exp ']' }]"""

    btype: BType = BType.INT
    name: str = ""
    is_array: bool = False
    dimensions: List["ASTNode"] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = NodeType.FUNC_FPARAM


@dataclass
class BlockNode(ASTNode):
    """语句块节点 - Block → '{' { BlockItem } '}'"""

    def __post_init__(self):
        self.node_type = NodeType.BLOCK


@dataclass
class BlockItemNode(ASTNode):
    """语句块项节点 - BlockItem → Decl | Stmt"""

    def __post_init__(self):
        self.node_type = NodeType.BLOCK_ITEM


class StmtType(Enum):
    """语句类型"""

    ASSIGN = auto()  # LVal '=' Exp ';'
    EXP = auto()  # [Exp] ';'
    BLOCK = auto()  # Block
    IF = auto()  # 'if' '(' Cond ')' Stmt ['else' Stmt]
    WHILE = auto()  # 'while' '(' Cond ')' Stmt
    BREAK = auto()  # 'break' ';'
    CONTINUE = auto()  # 'continue' ';'
    RETURN = auto()  # 'return' [Exp] ';'


@dataclass
class StmtNode(ASTNode):
    """语句节点"""

    stmt_type: StmtType = StmtType.EXP

    def __post_init__(self):
        self.node_type = NodeType.STMT


@dataclass
class ExpNode(ASTNode):
    """表达式节点 - Exp → AddExp"""

    def __post_init__(self):
        self.node_type = NodeType.EXP


@dataclass
class CondNode(ASTNode):
    """条件表达式节点 - Cond → LOrExp"""

    def __post_init__(self):
        self.node_type = NodeType.COND


@dataclass
class LValNode(ASTNode):
    """左值表达式节点 - LVal → Ident {'[' Exp ']'}"""

    name: str = ""
    indices: List["ASTNode"] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = NodeType.LVAL


@dataclass
class PrimaryExpNode(ASTNode):
    """基本表达式节点 - PrimaryExp → '(' Exp ')' | LVal | Number"""

    def __post_init__(self):
        self.node_type = NodeType.PRIMARY_EXP


@dataclass
class NumberNode(ASTNode):
    """数值节点 - Number → IntConst | FloatConst"""

    value: Union[int, float] = 0
    is_float: bool = False

    def __post_init__(self):
        self.node_type = NodeType.NUMBER


@dataclass
class UnaryExpNode(ASTNode):
    """一元表达式节点 - UnaryExp → PrimaryExp | Ident '(' [FuncRParams] ')' | UnaryOp UnaryExp"""

    op: Optional[str] = None  # '+', '-', '!'
    is_func_call: bool = False
    func_name: str = ""

    def __post_init__(self):
        self.node_type = NodeType.UNARY_EXP


@dataclass
class UnaryOpNode(ASTNode):
    """单目运算符节点"""

    op: str = ""

    def __post_init__(self):
        self.node_type = NodeType.UNARY_OP


@dataclass
class FuncRParamsNode(ASTNode):
    """函数实参列表节点 - FuncRParams → Exp { ',' Exp }"""

    def __post_init__(self):
        self.node_type = NodeType.FUNC_RPARAMS


@dataclass
class MulExpNode(ASTNode):
    """乘除模表达式节点 - MulExp → UnaryExp | MulExp ('*' | '/' | '%') UnaryExp"""

    op: Optional[str] = None  # '*', '/', '%'

    def __post_init__(self):
        self.node_type = NodeType.MUL_EXP


@dataclass
class AddExpNode(ASTNode):
    """加减表达式节点 - AddExp → MulExp | AddExp ('+' | '−') MulExp"""

    op: Optional[str] = None  # '+', '-'

    def __post_init__(self):
        self.node_type = NodeType.ADD_EXP


@dataclass
class RelExpNode(ASTNode):
    """关系表达式节点 - RelExp → AddExp | RelExp ('<' | '>' | '<=' | '>=') AddExp"""

    op: Optional[str] = None

    def __post_init__(self):
        self.node_type = NodeType.REL_EXP


@dataclass
class EqExpNode(ASTNode):
    """相等性表达式节点 - EqExp → RelExp | EqExp ('==' | '!=') RelExp"""

    op: Optional[str] = None

    def __post_init__(self):
        self.node_type = NodeType.EQ_EXP


@dataclass
class LAndExpNode(ASTNode):
    """逻辑与表达式节点 - LAndExp → EqExp | LAndExp '&&' EqExp"""

    def __post_init__(self):
        self.node_type = NodeType.LAND_EXP


@dataclass
class LOrExpNode(ASTNode):
    """逻辑或表达式节点 - LOrExp → LAndExp | LOrExp '||' LAndExp"""

    def __post_init__(self):
        self.node_type = NodeType.LOR_EXP


@dataclass
class ConstExpNode(ASTNode):
    """常量表达式节点 - ConstExp → AddExp"""

    def __post_init__(self):
        self.node_type = NodeType.CONST_EXP


# ============ AST 打印器 ============


class ASTPrinter:
    """AST 先序遍历打印器"""

    # 节点类型到输出名称的映射
    NODE_NAMES = {
        NodeType.COMP_UNIT: "CompUnit",
        NodeType.DECL: "Decl",
        NodeType.CONST_DECL: "ConstDecl",
        NodeType.VAR_DECL: "VarDecl",
        NodeType.CONST_DEF: "ConstDef",
        NodeType.VAR_DEF: "VarDef",
        NodeType.CONST_INIT_VAL: "ConstInitVal",
        NodeType.INIT_VAL: "InitVal",
        NodeType.BTYPE: "BType",
        NodeType.FUNC_TYPE: "FuncType",
        NodeType.FUNC_DEF: "FuncDef",
        NodeType.FUNC_FPARAMS: "FuncFParams",
        NodeType.FUNC_FPARAM: "FuncFParam",
        NodeType.FUNC_RPARAMS: "FuncRParams",
        NodeType.BLOCK: "Block",
        NodeType.BLOCK_ITEM: "BlockItem",
        NodeType.STMT: "Stmt",
        NodeType.EXP: "Exp",
        NodeType.COND: "Cond",
        NodeType.LVAL: "LVal",
        NodeType.PRIMARY_EXP: "PrimaryExp",
        NodeType.NUMBER: "Number",
        NodeType.UNARY_EXP: "UnaryExp",
        NodeType.UNARY_OP: "UnaryOp",
        NodeType.MUL_EXP: "MulExp",
        NodeType.ADD_EXP: "AddExp",
        NodeType.REL_EXP: "RelExp",
        NodeType.EQ_EXP: "EqExp",
        NodeType.LAND_EXP: "LAndExp",
        NodeType.LOR_EXP: "LOrExp",
        NodeType.CONST_EXP: "ConstExp",
    }

    def __init__(self):
        self.output_lines = []

    def print_ast(self, node: ASTNode, indent: int = 0) -> str:
        """先序遍历打印 AST"""
        self.output_lines = []
        self._visit(node, indent)
        return "\n".join(self.output_lines)

    def _visit(self, node: ASTNode, indent: int):
        """访问节点"""
        if node is None:
            return

        prefix = "  " * indent

        if isinstance(node, TerminalNode):
            # 终结符节点
            self._print_terminal(node, prefix)
        else:
            # 非终结符节点
            self._print_non_terminal(node, prefix)
            # 递归访问子节点
            for child in node.children:
                self._visit(child, indent + 1)

    def _print_terminal(self, node: TerminalNode, prefix: str):
        """打印终结符节点"""
        # 终结符只打印类型名称，不打印行号
        # 特殊处理：ID打印词素，TYPE打印具体类型，常数打印值
        if node.token_type == "IDENFR":
            self.output_lines.append(f"{prefix}Ident: {node.value}")
        elif node.token_type in ("INTTK", "FLOATTK", "VOIDTK"):
            self.output_lines.append(f"{prefix}Type: {node.value}")
        elif node.token_type == "INTCON":
            self.output_lines.append(f"{prefix}INTCON: {node.value}")
        elif node.token_type == "FLOATCON":
            self.output_lines.append(f"{prefix}FLOATCON: {node.value}")
        else:
            self.output_lines.append(f"{prefix}{node.token_type}")

    def _print_non_terminal(self, node: ASTNode, prefix: str):
        """打印非终结符节点"""
        name = self.NODE_NAMES.get(node.node_type, node.node_type.name)

        # 检查是否产生了 ε（空节点没有子节点且不是特殊节点）
        if not node.children and node.node_type not in (NodeType.NUMBER,):
            # 产生 ε，只打印名称不打印行号
            self.output_lines.append(f"{prefix}{name}")
        else:
            # 非 ε，打印名称和行号
            self.output_lines.append(f"{prefix}{name} ({node.line})")
