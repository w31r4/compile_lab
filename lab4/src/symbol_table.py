"""
SysY 符号表 - 用于语义分析

实现符号表和作用域栈，支持变量/函数的声明、查找和作用域管理。
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Union


class SymbolKind(Enum):
    """符号类型"""

    VARIABLE = auto()  # 变量
    CONSTANT = auto()  # 常量
    FUNCTION = auto()  # 函数
    PARAMETER = auto()  # 函数参数


class TypeKind(Enum):
    """数据类型"""

    INT = auto()
    FLOAT = auto()
    VOID = auto()


@dataclass
class ArrayType:
    """数组类型"""

    base_type: TypeKind
    dimensions: List[int]  # 每维的大小，-1 表示未知（如函数参数 int a[]）

    def __str__(self):
        dims = "".join(f"[{d if d >= 0 else ''}]" for d in self.dimensions)
        return f"{self.base_type.name.lower()}{dims}"


@dataclass
class FunctionType:
    """函数类型"""

    return_type: TypeKind
    param_types: List[Union[TypeKind, "ArrayType"]]  # 参数类型列表

    def __str__(self):
        params = ", ".join(str(p) for p in self.param_types)
        return f"{self.return_type.name.lower()}({params})"


# 符号的类型可以是基本类型、数组类型或函数类型
SymbolType = Union[TypeKind, ArrayType, FunctionType]


@dataclass
class Symbol:
    """符号表条目"""

    name: str  # 符号名称
    kind: SymbolKind  # 符号类型（变量/常量/函数/参数）
    type: SymbolType  # 数据类型
    line: int  # 声明行号
    scope_level: int = 0  # 作用域层级
    is_const: bool = False  # 是否是常量
    const_value: Optional[Union[int, float, List]] = None  # 常量值

    def is_array(self) -> bool:
        """判断是否是数组"""
        return isinstance(self.type, ArrayType)

    def is_function(self) -> bool:
        """判断是否是函数"""
        return isinstance(self.type, FunctionType)

    def get_base_type(self) -> TypeKind:
        """获取基础类型"""
        if isinstance(self.type, TypeKind):
            return self.type
        elif isinstance(self.type, ArrayType):
            return self.type.base_type
        elif isinstance(self.type, FunctionType):
            return self.type.return_type
        return TypeKind.INT


@dataclass
class Scope:
    """作用域"""

    symbols: Dict[str, Symbol] = field(default_factory=dict)
    level: int = 0
    parent: Optional["Scope"] = None

    def define(self, symbol: Symbol) -> bool:
        """在当前作用域定义符号，返回是否成功（重复定义返回False）"""
        if symbol.name in self.symbols:
            return False
        symbol.scope_level = self.level
        self.symbols[symbol.name] = symbol
        return True

    def lookup_local(self, name: str) -> Optional[Symbol]:
        """在当前作用域查找符号"""
        return self.symbols.get(name)

    def lookup(self, name: str) -> Optional[Symbol]:
        """在当前及父作用域查找符号"""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None


class SymbolTable:
    """符号表管理器 - 实现作用域栈"""

    def __init__(self):
        # 全局作用域
        self.global_scope = Scope(level=0)
        self.current_scope = self.global_scope
        self.scope_stack: List[Scope] = [self.global_scope]

        # 预定义的库函数
        self._init_builtin_functions()

    def _init_builtin_functions(self):
        """初始化内置库函数"""
        # getint: int getint()
        self.define(
            Symbol(
                name="getint",
                kind=SymbolKind.FUNCTION,
                type=FunctionType(return_type=TypeKind.INT, param_types=[]),
                line=0,
            )
        )

        # getch: int getch()
        self.define(
            Symbol(
                name="getch",
                kind=SymbolKind.FUNCTION,
                type=FunctionType(return_type=TypeKind.INT, param_types=[]),
                line=0,
            )
        )

        # getfloat: float getfloat()
        self.define(
            Symbol(
                name="getfloat",
                kind=SymbolKind.FUNCTION,
                type=FunctionType(return_type=TypeKind.FLOAT, param_types=[]),
                line=0,
            )
        )

        # getarray: int getarray(int a[])
        self.define(
            Symbol(
                name="getarray",
                kind=SymbolKind.FUNCTION,
                type=FunctionType(
                    return_type=TypeKind.INT, param_types=[ArrayType(base_type=TypeKind.INT, dimensions=[-1])]
                ),
                line=0,
            )
        )

        # getfarray: int getfarray(float a[])
        self.define(
            Symbol(
                name="getfarray",
                kind=SymbolKind.FUNCTION,
                type=FunctionType(
                    return_type=TypeKind.INT, param_types=[ArrayType(base_type=TypeKind.FLOAT, dimensions=[-1])]
                ),
                line=0,
            )
        )

        # putint: void putint(int n)
        self.define(
            Symbol(
                name="putint",
                kind=SymbolKind.FUNCTION,
                type=FunctionType(return_type=TypeKind.VOID, param_types=[TypeKind.INT]),
                line=0,
            )
        )

        # putch: void putch(int c)
        self.define(
            Symbol(
                name="putch",
                kind=SymbolKind.FUNCTION,
                type=FunctionType(return_type=TypeKind.VOID, param_types=[TypeKind.INT]),
                line=0,
            )
        )

        # putfloat: void putfloat(float f)
        self.define(
            Symbol(
                name="putfloat",
                kind=SymbolKind.FUNCTION,
                type=FunctionType(return_type=TypeKind.VOID, param_types=[TypeKind.FLOAT]),
                line=0,
            )
        )

        # putarray: void putarray(int n, int a[])
        self.define(
            Symbol(
                name="putarray",
                kind=SymbolKind.FUNCTION,
                type=FunctionType(
                    return_type=TypeKind.VOID,
                    param_types=[TypeKind.INT, ArrayType(base_type=TypeKind.INT, dimensions=[-1])],
                ),
                line=0,
            )
        )

        # putfarray: void putfarray(int n, float a[])
        self.define(
            Symbol(
                name="putfarray",
                kind=SymbolKind.FUNCTION,
                type=FunctionType(
                    return_type=TypeKind.VOID,
                    param_types=[TypeKind.INT, ArrayType(base_type=TypeKind.FLOAT, dimensions=[-1])],
                ),
                line=0,
            )
        )

        # starttime: void starttime()
        self.define(
            Symbol(
                name="starttime",
                kind=SymbolKind.FUNCTION,
                type=FunctionType(return_type=TypeKind.VOID, param_types=[]),
                line=0,
            )
        )

        # stoptime: void stoptime()
        self.define(
            Symbol(
                name="stoptime",
                kind=SymbolKind.FUNCTION,
                type=FunctionType(return_type=TypeKind.VOID, param_types=[]),
                line=0,
            )
        )

    def enter_scope(self):
        """进入新作用域"""
        new_level = len(self.scope_stack)
        new_scope = Scope(level=new_level, parent=self.current_scope)
        self.scope_stack.append(new_scope)
        self.current_scope = new_scope

    def exit_scope(self):
        """退出当前作用域"""
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()
            self.current_scope = self.scope_stack[-1]

    def define(self, symbol: Symbol) -> bool:
        """在当前作用域定义符号"""
        return self.current_scope.define(symbol)

    def lookup(self, name: str) -> Optional[Symbol]:
        """查找符号（从当前作用域向上查找）"""
        return self.current_scope.lookup(name)

    def lookup_local(self, name: str) -> Optional[Symbol]:
        """只在当前作用域查找符号"""
        return self.current_scope.lookup_local(name)

    def lookup_function(self, name: str) -> Optional[Symbol]:
        """查找函数（只在全局作用域）"""
        symbol = self.global_scope.lookup_local(name)
        if symbol and symbol.kind == SymbolKind.FUNCTION:
            return symbol
        return None

    def get_current_level(self) -> int:
        """获取当前作用域层级"""
        return self.current_scope.level
