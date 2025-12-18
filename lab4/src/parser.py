"""
SysY 语法分析器 - 递归下降解析器

基于 SysY 2022 文法实现的递归下降语法分析器。
"""

from typing import List, Optional
from .token_type import Token, TokenType
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
    ASTPrinter,
)


class ParserError(Exception):
    """语法分析错误"""

    def __init__(self, message: str, line: int):
        self.message = message
        self.line = line
        super().__init__(f"Error type B at Line {line}: {message}")


class Parser:
    """SysY 递归下降语法分析器"""

    # 最大错误数量限制，防止无限循环报错
    MAX_ERRORS = 10

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.errors: List[ParserError] = []
        self.has_error = False
        self.error_count = 0

    # ============ 辅助方法 ============

    def current_token(self) -> Optional[Token]:
        """获取当前 token"""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def peek(self, offset: int = 0) -> Optional[Token]:
        """向前看 token"""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def current_line(self) -> int:
        """获取当前行号"""
        token = self.current_token()
        if token:
            return token.line
        if self.tokens:
            return self.tokens[-1].line
        return 1

    def advance(self) -> Optional[Token]:
        """前进到下一个 token"""
        token = self.current_token()
        if token:
            self.pos += 1
        return token

    def match(self, *types: TokenType) -> bool:
        """检查当前 token 是否匹配指定类型"""
        token = self.current_token()
        if token and token.type in types:
            return True
        return False

    def expect(self, token_type: TokenType, error_msg: str = None) -> Token:
        """期望并消费指定类型的 token"""
        if self.match(token_type):
            return self.advance()

        # 语法错误
        if error_msg is None:
            error_msg = f"Expected '{token_type.name}'"
        self.error(error_msg)
        # 尝试同步：跳过当前token继续
        self.advance()
        return None

    def error(self, message: str):
        """报告语法错误"""
        self.error_count += 1
        if self.error_count > self.MAX_ERRORS:
            # 达到错误上限，停止报告
            return

        line = self.current_line()
        error = ParserError(message, line)
        self.errors.append(error)
        self.has_error = True
        print(f"Error type B at Line {line}: {message}")

    def synchronize(self):
        """错误恢复：同步到下一个语句边界"""
        self.advance()

        while self.current_token() is not None:
            # 上一个token是分号，可以开始新语句
            if self.pos > 0 and self.tokens[self.pos - 1].type == TokenType.SEMICOLON:
                return

            # 当前token是语句开始关键字
            token = self.current_token()
            if token.type in (
                TokenType.INT,
                TokenType.FLOAT,
                TokenType.VOID,
                TokenType.CONST,
                TokenType.IF,
                TokenType.WHILE,
                TokenType.BREAK,
                TokenType.CONTINUE,
                TokenType.RETURN,
                TokenType.LBRACE,
                TokenType.RBRACE,
            ):
                return

            self.advance()

    def make_terminal(self, token: Token) -> TerminalNode:
        """创建终结符节点"""
        if token is None:
            return None

        # 获取输出类型名称
        type_map = {
            TokenType.CONST: "CONSTTK",
            TokenType.INT: "INTTK",
            TokenType.FLOAT: "FLOATTK",
            TokenType.VOID: "VOIDTK",
            TokenType.IF: "IFTK",
            TokenType.ELSE: "ELSETK",
            TokenType.WHILE: "WHILETK",
            TokenType.BREAK: "BREAKTK",
            TokenType.CONTINUE: "CONTINUETK",
            TokenType.RETURN: "RETURNTK",
            TokenType.PLUS: "PLUS",
            TokenType.MINUS: "MINU",
            TokenType.MUL: "MULT",
            TokenType.DIV: "DIV",
            TokenType.MOD: "MOD",
            TokenType.ASSIGN: "ASSIGN",
            TokenType.EQ: "EQL",
            TokenType.NEQ: "NEQ",
            TokenType.LT: "LSS",
            TokenType.GT: "GRE",
            TokenType.LE: "LEQ",
            TokenType.GE: "GEQ",
            TokenType.NOT: "NOT",
            TokenType.AND: "AND",
            TokenType.OR: "OR",
            TokenType.LPAREN: "LPARENT",
            TokenType.RPAREN: "RPARENT",
            TokenType.LBRACKET: "LBRACK",
            TokenType.RBRACKET: "RBRACK",
            TokenType.LBRACE: "LBRACE",
            TokenType.RBRACE: "RBRACE",
            TokenType.COMMA: "COMMA",
            TokenType.SEMICOLON: "SEMICN",
            TokenType.ID: "IDENFR",
            TokenType.INT_CONST: "INTCON",
            TokenType.FLOAT_CONST: "FLOATCON",
        }

        type_name = type_map.get(token.type, token.type.name)

        # 对于数值常量，使用转换后的十进制值
        if token.type in (TokenType.INT_CONST, TokenType.FLOAT_CONST):
            value = str(token.numeric_value) if token.numeric_value is not None else token.value
        else:
            value = token.value

        node = TerminalNode(line=token.line, token_type=type_name, value=value)
        return node

    # ============ 语法分析方法 ============

    def parse(self) -> CompUnitNode:
        """解析入口 - CompUnit → [CompUnit] (Decl | FuncDef)"""
        return self.parse_comp_unit()

    def parse_comp_unit(self) -> CompUnitNode:
        """CompUnit → { Decl | FuncDef }"""
        node = CompUnitNode(line=self.current_line())

        while self.current_token() is not None:
            # 判断是 Decl 还是 FuncDef
            # FuncDef: FuncType Ident '(' ...
            # Decl: 'const' BType ... 或 BType Ident '[' 或 BType Ident ',' 或 BType Ident ';' 或 BType Ident '='

            if self.match(TokenType.CONST):
                # 常量声明
                decl = self.parse_decl()
                node.add_child(decl)
            elif self.match(TokenType.VOID):
                # void 只能是函数定义
                func_def = self.parse_func_def()
                node.add_child(func_def)
            elif self.match(TokenType.INT, TokenType.FLOAT):
                # 需要向前看判断
                # BType Ident '(' -> FuncDef
                # BType Ident '[' 或 ',' 或 ';' 或 '=' -> VarDecl
                if self.is_func_def():
                    func_def = self.parse_func_def()
                    node.add_child(func_def)
                else:
                    decl = self.parse_decl()
                    node.add_child(decl)
            else:
                # 未知 token，尝试恢复
                self.error(f"Unexpected token '{self.current_token().value}'")
                self.advance()

        return node

    def is_func_def(self) -> bool:
        """判断是否是函数定义（向前看）"""
        # BType Ident '(' -> FuncDef
        if self.peek(1) and self.peek(1).type == TokenType.ID:
            if self.peek(2) and self.peek(2).type == TokenType.LPAREN:
                return True
        return False

    def parse_decl(self) -> DeclNode:
        """Decl → ConstDecl | VarDecl"""
        node = DeclNode(line=self.current_line())

        if self.match(TokenType.CONST):
            const_decl = self.parse_const_decl()
            node.add_child(const_decl)
        else:
            var_decl = self.parse_var_decl()
            node.add_child(var_decl)

        return node

    def parse_const_decl(self) -> ConstDeclNode:
        """ConstDecl → 'const' BType ConstDef { ',' ConstDef } ';'"""
        line = self.current_line()
        node = ConstDeclNode(line=line)

        # 'const'
        const_token = self.expect(TokenType.CONST, "Missing 'const'")
        node.add_child(self.make_terminal(const_token))

        # BType
        btype_node = self.parse_btype()
        node.add_child(btype_node)
        node.btype = btype_node.btype if btype_node else BType.INT

        # ConstDef
        const_def = self.parse_const_def()
        node.add_child(const_def)

        # { ',' ConstDef }
        while self.match(TokenType.COMMA):
            comma = self.advance()
            node.add_child(self.make_terminal(comma))
            const_def = self.parse_const_def()
            node.add_child(const_def)

        # ';'
        semi = self.expect(TokenType.SEMICOLON, "Missing ';'")
        node.add_child(self.make_terminal(semi))

        return node

    def parse_btype(self) -> BTypeNode:
        """BType → 'int' | 'float'"""
        line = self.current_line()
        node = BTypeNode(line=line)

        if self.match(TokenType.INT):
            token = self.advance()
            node.btype = BType.INT
            node.add_child(self.make_terminal(token))
        elif self.match(TokenType.FLOAT):
            token = self.advance()
            node.btype = BType.FLOAT
            node.add_child(self.make_terminal(token))
        else:
            self.error("Expected type 'int' or 'float'")

        return node

    def parse_const_def(self) -> ConstDefNode:
        """ConstDef → Ident { '[' ConstExp ']' } '=' ConstInitVal"""
        line = self.current_line()
        node = ConstDefNode(line=line)

        # Ident
        ident = self.expect(TokenType.ID, "Expected identifier")
        node.add_child(self.make_terminal(ident))
        node.name = ident.value if ident else ""

        # { '[' ConstExp ']' }
        while self.match(TokenType.LBRACKET):
            lbracket = self.advance()
            node.add_child(self.make_terminal(lbracket))

            const_exp = self.parse_const_exp()
            node.add_child(const_exp)
            node.dimensions.append(const_exp)

            rbracket = self.expect(TokenType.RBRACKET, "Missing ']'")
            node.add_child(self.make_terminal(rbracket))

        # '='
        assign = self.expect(TokenType.ASSIGN, "Missing '=' in const definition")
        node.add_child(self.make_terminal(assign))

        # ConstInitVal
        init_val = self.parse_const_init_val()
        node.add_child(init_val)
        node.init_val = init_val

        return node

    def parse_const_init_val(self) -> ConstInitValNode:
        """ConstInitVal → ConstExp | '{' [ ConstInitVal { ',' ConstInitVal } ] '}'"""
        line = self.current_line()
        node = ConstInitValNode(line=line)

        if self.match(TokenType.LBRACE):
            # 数组初始化
            node.is_array = True
            lbrace = self.advance()
            node.add_child(self.make_terminal(lbrace))

            if not self.match(TokenType.RBRACE):
                init_val = self.parse_const_init_val()
                node.add_child(init_val)

                while self.match(TokenType.COMMA):
                    comma = self.advance()
                    node.add_child(self.make_terminal(comma))
                    init_val = self.parse_const_init_val()
                    node.add_child(init_val)

            rbrace = self.expect(TokenType.RBRACE, "Missing '}'")
            node.add_child(self.make_terminal(rbrace))
        else:
            # 单个表达式
            const_exp = self.parse_const_exp()
            node.add_child(const_exp)

        return node

    def parse_var_decl(self) -> VarDeclNode:
        """VarDecl → BType VarDef { ',' VarDef } ';'"""
        line = self.current_line()
        node = VarDeclNode(line=line)

        # BType
        btype_node = self.parse_btype()
        node.add_child(btype_node)
        node.btype = btype_node.btype if btype_node else BType.INT

        # VarDef
        var_def = self.parse_var_def()
        node.add_child(var_def)

        # { ',' VarDef }
        while self.match(TokenType.COMMA):
            comma = self.advance()
            node.add_child(self.make_terminal(comma))
            var_def = self.parse_var_def()
            node.add_child(var_def)

        # ';'
        semi = self.expect(TokenType.SEMICOLON, "Missing ';'")
        node.add_child(self.make_terminal(semi))

        return node

    def parse_var_def(self) -> VarDefNode:
        """VarDef → Ident { '[' ConstExp ']' } [ '=' InitVal ]"""
        line = self.current_line()
        node = VarDefNode(line=line)

        # Ident
        ident = self.expect(TokenType.ID, "Expected identifier")
        node.add_child(self.make_terminal(ident))
        node.name = ident.value if ident else ""

        # { '[' ConstExp ']' }
        while self.match(TokenType.LBRACKET):
            lbracket = self.advance()
            node.add_child(self.make_terminal(lbracket))

            const_exp = self.parse_const_exp()
            node.add_child(const_exp)
            node.dimensions.append(const_exp)

            rbracket = self.expect(TokenType.RBRACKET, "Missing ']'")
            node.add_child(self.make_terminal(rbracket))

        # [ '=' InitVal ]
        if self.match(TokenType.ASSIGN):
            assign = self.advance()
            node.add_child(self.make_terminal(assign))

            init_val = self.parse_init_val()
            node.add_child(init_val)
            node.init_val = init_val

        return node

    def parse_init_val(self) -> InitValNode:
        """InitVal → Exp | '{' [ InitVal { ',' InitVal } ] '}'"""
        line = self.current_line()
        node = InitValNode(line=line)

        if self.match(TokenType.LBRACE):
            node.is_array = True
            lbrace = self.advance()
            node.add_child(self.make_terminal(lbrace))

            if not self.match(TokenType.RBRACE):
                init_val = self.parse_init_val()
                node.add_child(init_val)

                while self.match(TokenType.COMMA):
                    comma = self.advance()
                    node.add_child(self.make_terminal(comma))
                    init_val = self.parse_init_val()
                    node.add_child(init_val)

            rbrace = self.expect(TokenType.RBRACE, "Missing '}'")
            node.add_child(self.make_terminal(rbrace))
        else:
            exp = self.parse_exp()
            node.add_child(exp)

        return node

    def parse_func_def(self) -> FuncDefNode:
        """FuncDef → FuncType Ident '(' [FuncFParams] ')' Block"""
        line = self.current_line()
        node = FuncDefNode(line=line)

        # FuncType
        func_type = self.parse_func_type()
        node.add_child(func_type)
        node.return_type = func_type.return_type if func_type else BType.INT

        # Ident
        ident = self.expect(TokenType.ID, "Expected function name")
        node.add_child(self.make_terminal(ident))
        node.name = ident.value if ident else ""

        # '('
        lparen = self.expect(TokenType.LPAREN, "Missing '('")
        node.add_child(self.make_terminal(lparen))

        # [FuncFParams]
        if not self.match(TokenType.RPAREN):
            params = self.parse_func_fparams()
            node.add_child(params)
            node.params = params.children if params else []

        # ')'
        rparen = self.expect(TokenType.RPAREN, "Missing ')'")
        node.add_child(self.make_terminal(rparen))

        # Block
        block = self.parse_block()
        node.add_child(block)
        node.body = block

        return node

    def parse_func_type(self) -> FuncTypeNode:
        """FuncType → 'void' | 'int' | 'float'"""
        line = self.current_line()
        node = FuncTypeNode(line=line)

        if self.match(TokenType.VOID):
            token = self.advance()
            node.return_type = BType.VOID
            node.add_child(self.make_terminal(token))
        elif self.match(TokenType.INT):
            token = self.advance()
            node.return_type = BType.INT
            node.add_child(self.make_terminal(token))
        elif self.match(TokenType.FLOAT):
            token = self.advance()
            node.return_type = BType.FLOAT
            node.add_child(self.make_terminal(token))
        else:
            self.error("Expected function return type")

        return node

    def parse_func_fparams(self) -> FuncFParamsNode:
        """FuncFParams → FuncFParam { ',' FuncFParam }"""
        line = self.current_line()
        node = FuncFParamsNode(line=line)

        param = self.parse_func_fparam()
        node.add_child(param)

        while self.match(TokenType.COMMA):
            comma = self.advance()
            node.add_child(self.make_terminal(comma))
            param = self.parse_func_fparam()
            node.add_child(param)

        return node

    def parse_func_fparam(self) -> FuncFParamNode:
        """FuncFParam → BType Ident ['[' ']' { '[' Exp ']' }]"""
        line = self.current_line()
        node = FuncFParamNode(line=line)

        # BType
        btype_node = self.parse_btype()
        node.add_child(btype_node)
        node.btype = btype_node.btype if btype_node else BType.INT

        # Ident
        ident = self.expect(TokenType.ID, "Expected parameter name")
        node.add_child(self.make_terminal(ident))
        node.name = ident.value if ident else ""

        # ['[' ']' { '[' Exp ']' }]
        if self.match(TokenType.LBRACKET):
            node.is_array = True

            # 第一维 '[]'
            lbracket = self.advance()
            node.add_child(self.make_terminal(lbracket))
            rbracket = self.expect(TokenType.RBRACKET, "Missing ']'")
            node.add_child(self.make_terminal(rbracket))

            # { '[' Exp ']' }
            while self.match(TokenType.LBRACKET):
                lbracket = self.advance()
                node.add_child(self.make_terminal(lbracket))

                exp = self.parse_exp()
                node.add_child(exp)
                node.dimensions.append(exp)

                rbracket = self.expect(TokenType.RBRACKET, "Missing ']'")
                node.add_child(self.make_terminal(rbracket))

        return node

    def parse_block(self) -> BlockNode:
        """Block → '{' { BlockItem } '}'"""
        line = self.current_line()
        node = BlockNode(line=line)

        # '{'
        lbrace = self.expect(TokenType.LBRACE, "Missing '{'")
        node.add_child(self.make_terminal(lbrace))

        # { BlockItem }
        while not self.match(TokenType.RBRACE) and self.current_token() is not None:
            block_item = self.parse_block_item()
            node.add_child(block_item)

        # '}'
        rbrace = self.expect(TokenType.RBRACE, "Missing '}'")
        node.add_child(self.make_terminal(rbrace))

        return node

    def parse_block_item(self) -> BlockItemNode:
        """BlockItem → Decl | Stmt"""
        line = self.current_line()
        node = BlockItemNode(line=line)

        if self.match(TokenType.CONST, TokenType.INT, TokenType.FLOAT):
            # Decl
            decl = self.parse_decl()
            node.add_child(decl)
        else:
            # Stmt
            stmt = self.parse_stmt()
            node.add_child(stmt)

        return node

    def parse_stmt(self) -> StmtNode:
        """
        Stmt → LVal '=' Exp ';' | [Exp] ';' | Block
             | 'if' '(' Cond ')' Stmt [ 'else' Stmt ]
             | 'while' '(' Cond ')' Stmt
             | 'break' ';' | 'continue' ';'
             | 'return' [Exp] ';'
        """
        line = self.current_line()
        node = StmtNode(line=line)

        if self.match(TokenType.LBRACE):
            # Block
            node.stmt_type = StmtType.BLOCK
            block = self.parse_block()
            node.add_child(block)

        elif self.match(TokenType.IF):
            # 'if' '(' Cond ')' Stmt [ 'else' Stmt ]
            node.stmt_type = StmtType.IF
            if_token = self.advance()
            node.add_child(self.make_terminal(if_token))

            lparen = self.expect(TokenType.LPAREN, "Missing '(' after 'if'")
            node.add_child(self.make_terminal(lparen))

            cond = self.parse_cond()
            node.add_child(cond)

            rparen = self.expect(TokenType.RPAREN, "Missing ')' after condition")
            node.add_child(self.make_terminal(rparen))

            then_stmt = self.parse_stmt()
            node.add_child(then_stmt)

            if self.match(TokenType.ELSE):
                else_token = self.advance()
                node.add_child(self.make_terminal(else_token))
                else_stmt = self.parse_stmt()
                node.add_child(else_stmt)

        elif self.match(TokenType.WHILE):
            # 'while' '(' Cond ')' Stmt
            node.stmt_type = StmtType.WHILE
            while_token = self.advance()
            node.add_child(self.make_terminal(while_token))

            lparen = self.expect(TokenType.LPAREN, "Missing '(' after 'while'")
            node.add_child(self.make_terminal(lparen))

            cond = self.parse_cond()
            node.add_child(cond)

            rparen = self.expect(TokenType.RPAREN, "Missing ')' after condition")
            node.add_child(self.make_terminal(rparen))

            body_stmt = self.parse_stmt()
            node.add_child(body_stmt)

        elif self.match(TokenType.BREAK):
            # 'break' ';'
            node.stmt_type = StmtType.BREAK
            break_token = self.advance()
            node.add_child(self.make_terminal(break_token))

            semi = self.expect(TokenType.SEMICOLON, "Missing ';' after 'break'")
            node.add_child(self.make_terminal(semi))

        elif self.match(TokenType.CONTINUE):
            # 'continue' ';'
            node.stmt_type = StmtType.CONTINUE
            continue_token = self.advance()
            node.add_child(self.make_terminal(continue_token))

            semi = self.expect(TokenType.SEMICOLON, "Missing ';' after 'continue'")
            node.add_child(self.make_terminal(semi))

        elif self.match(TokenType.RETURN):
            # 'return' [Exp] ';'
            node.stmt_type = StmtType.RETURN
            return_token = self.advance()
            node.add_child(self.make_terminal(return_token))

            if not self.match(TokenType.SEMICOLON):
                exp = self.parse_exp()
                node.add_child(exp)

            semi = self.expect(TokenType.SEMICOLON, "Missing ';' after 'return'")
            node.add_child(self.make_terminal(semi))

        elif self.match(TokenType.SEMICOLON):
            # 空语句 ';'
            node.stmt_type = StmtType.EXP
            semi = self.advance()
            node.add_child(self.make_terminal(semi))

        else:
            # LVal '=' Exp ';' 或 Exp ';'
            # 需要先尝试解析 LVal，如果后面是 '='，则是赋值语句
            # 否则回退，解析为 Exp

            # 保存当前位置用于回溯
            saved_pos = self.pos

            # 尝试解析 LVal
            lval = self.parse_lval()

            if self.match(TokenType.ASSIGN):
                # LVal '=' Exp ';'
                node.stmt_type = StmtType.ASSIGN
                node.add_child(lval)

                assign = self.advance()
                node.add_child(self.make_terminal(assign))

                exp = self.parse_exp()
                node.add_child(exp)

                semi = self.expect(TokenType.SEMICOLON, "Missing ';'")
                node.add_child(self.make_terminal(semi))
            else:
                # 回溯并解析为 Exp
                self.pos = saved_pos
                node.stmt_type = StmtType.EXP

                exp = self.parse_exp()
                node.add_child(exp)

                semi = self.expect(TokenType.SEMICOLON, "Missing ';'")
                node.add_child(self.make_terminal(semi))

        return node

    def parse_exp(self) -> ExpNode:
        """Exp → AddExp"""
        line = self.current_line()
        node = ExpNode(line=line)

        add_exp = self.parse_add_exp()
        node.add_child(add_exp)

        return node

    def parse_cond(self) -> CondNode:
        """Cond → LOrExp"""
        line = self.current_line()
        node = CondNode(line=line)

        lor_exp = self.parse_lor_exp()
        node.add_child(lor_exp)

        return node

    def parse_lval(self) -> LValNode:
        """LVal → Ident { '[' Exp ']' }"""
        line = self.current_line()
        node = LValNode(line=line)

        # Ident
        ident = self.expect(TokenType.ID, "Expected identifier")
        node.add_child(self.make_terminal(ident))
        node.name = ident.value if ident else ""

        # { '[' Exp ']' }
        while self.match(TokenType.LBRACKET):
            lbracket = self.advance()
            node.add_child(self.make_terminal(lbracket))

            exp = self.parse_exp()
            node.add_child(exp)
            node.indices.append(exp)

            rbracket = self.expect(TokenType.RBRACKET, "Missing ']'")
            node.add_child(self.make_terminal(rbracket))

        return node

    def parse_primary_exp(self) -> PrimaryExpNode:
        """PrimaryExp → '(' Exp ')' | LVal | Number"""
        line = self.current_line()
        node = PrimaryExpNode(line=line)

        if self.match(TokenType.LPAREN):
            # '(' Exp ')'
            lparen = self.advance()
            node.add_child(self.make_terminal(lparen))

            exp = self.parse_exp()
            node.add_child(exp)

            rparen = self.expect(TokenType.RPAREN, "Missing ')'")
            node.add_child(self.make_terminal(rparen))

        elif self.match(TokenType.INT_CONST, TokenType.FLOAT_CONST):
            # Number
            number = self.parse_number()
            node.add_child(number)

        elif self.match(TokenType.ID):
            # LVal
            lval = self.parse_lval()
            node.add_child(lval)

        else:
            self.error("Expected expression")

        return node

    def parse_number(self) -> NumberNode:
        """Number → IntConst | FloatConst"""
        line = self.current_line()
        node = NumberNode(line=line)

        if self.match(TokenType.INT_CONST):
            token = self.advance()
            node.value = token.numeric_value if token.numeric_value is not None else int(token.value)
            node.is_float = False
            node.add_child(self.make_terminal(token))
        elif self.match(TokenType.FLOAT_CONST):
            token = self.advance()
            node.value = token.numeric_value if token.numeric_value is not None else float(token.value)
            node.is_float = True
            node.add_child(self.make_terminal(token))
        else:
            self.error("Expected number")

        return node

    def parse_unary_exp(self) -> UnaryExpNode:
        """UnaryExp → PrimaryExp | Ident '(' [FuncRParams] ')' | UnaryOp UnaryExp"""
        line = self.current_line()
        node = UnaryExpNode(line=line)

        if self.match(TokenType.PLUS, TokenType.MINUS, TokenType.NOT):
            # UnaryOp UnaryExp
            op_token = self.advance()
            node.op = op_token.value

            unary_op = UnaryOpNode(line=op_token.line, op=op_token.value)
            unary_op.add_child(self.make_terminal(op_token))
            node.add_child(unary_op)

            unary_exp = self.parse_unary_exp()
            node.add_child(unary_exp)

        elif self.match(TokenType.ID) and self.peek(1) and self.peek(1).type == TokenType.LPAREN:
            # Ident '(' [FuncRParams] ')'
            node.is_func_call = True

            ident = self.advance()
            node.func_name = ident.value
            node.add_child(self.make_terminal(ident))

            lparen = self.advance()
            node.add_child(self.make_terminal(lparen))

            if not self.match(TokenType.RPAREN):
                params = self.parse_func_rparams()
                node.add_child(params)

            rparen = self.expect(TokenType.RPAREN, "Missing ')' in function call")
            node.add_child(self.make_terminal(rparen))

        else:
            # PrimaryExp
            primary_exp = self.parse_primary_exp()
            node.add_child(primary_exp)

        return node

    def parse_func_rparams(self) -> FuncRParamsNode:
        """FuncRParams → Exp { ',' Exp }"""
        line = self.current_line()
        node = FuncRParamsNode(line=line)

        exp = self.parse_exp()
        node.add_child(exp)

        while self.match(TokenType.COMMA):
            comma = self.advance()
            node.add_child(self.make_terminal(comma))
            exp = self.parse_exp()
            node.add_child(exp)

        return node

    def parse_mul_exp(self) -> MulExpNode:
        """MulExp → UnaryExp | MulExp ('*' | '/' | '%') UnaryExp"""
        line = self.current_line()
        node = MulExpNode(line=line)

        # 左结合，先解析第一个 UnaryExp
        unary_exp = self.parse_unary_exp()
        node.add_child(unary_exp)

        while self.match(TokenType.MUL, TokenType.DIV, TokenType.MOD):
            op_token = self.advance()
            node.op = op_token.value
            node.add_child(self.make_terminal(op_token))

            unary_exp = self.parse_unary_exp()
            node.add_child(unary_exp)

        return node

    def parse_add_exp(self) -> AddExpNode:
        """AddExp → MulExp | AddExp ('+' | '−') MulExp"""
        line = self.current_line()
        node = AddExpNode(line=line)

        mul_exp = self.parse_mul_exp()
        node.add_child(mul_exp)

        while self.match(TokenType.PLUS, TokenType.MINUS):
            op_token = self.advance()
            node.op = op_token.value
            node.add_child(self.make_terminal(op_token))

            mul_exp = self.parse_mul_exp()
            node.add_child(mul_exp)

        return node

    def parse_rel_exp(self) -> RelExpNode:
        """RelExp → AddExp | RelExp ('<' | '>' | '<=' | '>=') AddExp"""
        line = self.current_line()
        node = RelExpNode(line=line)

        add_exp = self.parse_add_exp()
        node.add_child(add_exp)

        while self.match(TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE):
            op_token = self.advance()
            node.op = op_token.value
            node.add_child(self.make_terminal(op_token))

            add_exp = self.parse_add_exp()
            node.add_child(add_exp)

        return node

    def parse_eq_exp(self) -> EqExpNode:
        """EqExp → RelExp | EqExp ('==' | '!=') RelExp"""
        line = self.current_line()
        node = EqExpNode(line=line)

        rel_exp = self.parse_rel_exp()
        node.add_child(rel_exp)

        while self.match(TokenType.EQ, TokenType.NEQ):
            op_token = self.advance()
            node.op = op_token.value
            node.add_child(self.make_terminal(op_token))

            rel_exp = self.parse_rel_exp()
            node.add_child(rel_exp)

        return node

    def parse_land_exp(self) -> LAndExpNode:
        """LAndExp → EqExp | LAndExp '&&' EqExp"""
        line = self.current_line()
        node = LAndExpNode(line=line)

        eq_exp = self.parse_eq_exp()
        node.add_child(eq_exp)

        while self.match(TokenType.AND):
            op_token = self.advance()
            node.add_child(self.make_terminal(op_token))

            eq_exp = self.parse_eq_exp()
            node.add_child(eq_exp)

        return node

    def parse_lor_exp(self) -> LOrExpNode:
        """LOrExp → LAndExp | LOrExp '||' LAndExp"""
        line = self.current_line()
        node = LOrExpNode(line=line)

        land_exp = self.parse_land_exp()
        node.add_child(land_exp)

        while self.match(TokenType.OR):
            op_token = self.advance()
            node.add_child(self.make_terminal(op_token))

            land_exp = self.parse_land_exp()
            node.add_child(land_exp)

        return node

    def parse_const_exp(self) -> ConstExpNode:
        """ConstExp → AddExp"""
        line = self.current_line()
        node = ConstExpNode(line=line)

        add_exp = self.parse_add_exp()
        node.add_child(add_exp)

        return node
