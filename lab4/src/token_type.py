from enum import Enum, auto


class TokenType(Enum):
    # Keywords
    CONST = auto()
    INT = auto()
    FLOAT = auto()
    VOID = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    BREAK = auto()
    CONTINUE = auto()
    RETURN = auto()

    # Operators
    PLUS = auto()  # +
    MINUS = auto()  # -
    MUL = auto()  # *
    DIV = auto()  # /
    MOD = auto()  # %
    ASSIGN = auto()  # =
    EQ = auto()  # ==
    NEQ = auto()  # !=
    LT = auto()  # <
    GT = auto()  # >
    LE = auto()  # <=
    GE = auto()  # >=
    NOT = auto()  # !
    AND = auto()  # &&
    OR = auto()  # ||

    # Delimiters
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    COMMA = auto()  # ,
    SEMICOLON = auto()  # ;

    # Literals and Identifiers
    ID = auto()
    INT_CONST = auto()
    FLOAT_CONST = auto()

    # End of File
    EOF = auto()


class Token:
    def __init__(self, type: TokenType, value: str, line: int, column: int):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type.name}, '{self.value}', line={self.line}, col={self.column})"

    def to_string(self):
        # Format output as required by the lab: <Type, Value>
        # Note: The lab spec might require specific type names (e.g., INTTK instead of INT)
        # We will map them here.
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

        type_str = type_map.get(self.type, self.type.name)
        return f"{type_str} {self.value}"
