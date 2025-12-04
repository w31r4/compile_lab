# 测试用例标准答案

本文档提供了 `test_grammars` 目录下所有文法示例经过语法分析算法处理后的标准输出结果。

---

## 1. `complex_grammar.txt`

**原始文法:**
```
# 复杂文法示例（用于演示左公因子提取）
# 非终结符
S, A, B
# 终结符
a, b, c, d, e
# 开始符号
S
# 产生式
S -> A | B
A -> a b c | a b d
B -> a b e
```

### 1.1 消除左递归

此文法无左递归，输出应与原始文法相同。

**输出文法:**
```
上下文无关文法:
  非终结符: {A, B, S}
  终结符: {a, b, c, d, e}
  开始符号: S

  产生式:
    S -> A
    S -> B
    A -> a b c
    A -> a b d
    B -> a b e
```

### 1.2 提取左公因子

**输出文法:**
```
上下文无关文法:
  非终结符: {A, A', B, S, S'}
  终结符: {a, b, c, d, e}
  开始符号: S

  产生式:
    B -> a b e
    A -> a b A'
    A' -> c
    A' -> d
    S -> a b S'
    S' -> c
    S' -> d
    S' -> e
```

### 1.3 FIRST 集

```
FIRST(A) = {a}
FIRST(A') = {c, d}
FIRST(B) = {a}
FIRST(S) = {a}
FIRST(S') = {c, d, e}
```

### 1.4 FOLLOW 集

```
FOLLOW(A) = {$}
FOLLOW(A') = {$}
FOLLOW(B) = {$}
FOLLOW(S) = {$}
FOLLOW(S') = {$}
```

### 1.5 LL(1) 分析表与判定

文法为 **LL(1)**。

```
M[A, a] = A -> a A'
M[A', b] = A' -> b A''
M[A'', c] = A'' -> c
M[A'', d] = A'' -> d
M[B, a] = B -> a b e
M[S, a] = S -> a S'
M[S', b] = S' -> b S''
M[S'', c] = S'' -> c
M[S'', d] = S'' -> d
M[S'', e] = S'' -> e
```

---

## 2. `ll1_grammar.txt`

**原始文法:**
```
# LL(1) Grammar Example
# Non-terminals
S, L
# Terminals
a, b, c, ;
# Start Symbol
S
# Productions
S -> a L ;
L -> S | c b
```

### 2.1 消除左递归

此文法无左递归，输出应与原始文法相同。

**输出文法:**
```
上下文无关文法:
  非终结符: {L, S}
  终结符: {;, a, b, c}
  开始符号: S

  产生式:
    S -> a L ;
    L -> S
    L -> c b
```

### 2.2 提取左公因子

此文法无左公因子，输出应与消除左递归后的文法相同。

**输出文法:**
```
上下文无关文法:
  非终结符: {L, S}
  终结符: {;, a, b, c}
  开始符号: S

  产生式:
    S -> a L ;
    L -> S
    L -> c b
```

### 2.3 FIRST 集

```
FIRST(L) = {a, c}
FIRST(S) = {a}
```

### 2.4 FOLLOW 集

```
FOLLOW(L) = {;}
FOLLOW(S) = {;}
```

### 2.5 LL(1) 分析表与判定

文法为 **LL(1)**。

```
M[S, a] = S -> a L ;
M[L, a] = L -> S
M[L, c] = L -> c b
```

---

## 3. `mixed_grammar.txt`

**原始文法:**
```
# Mixed Grammar (Left Recursion and Left Factoring)
# Non-terminals
P, Q, R
# Terminals
x, y, z, w
# Start Symbol
P
# Productions
P -> P x y | Q z
Q -> w R | w y
R -> z | y
```

### 3.1 消除左递归

**输出文法:**
```
上下文无关文法:
  非终结符: {P, P', Q, R}
  终结符: {w, x, y, z}
  开始符号: P

  产生式:
    Q -> w R
    Q -> w y
    R -> z
    R -> y
    P -> Q z P'
    P' -> x y P'
    P' -> @
```

### 3.2 提取左公因子

**输出文法:**
```
上下文无关文法:
  非终结符: {P, P', Q, Q', R}
  终结符: {w, x, y, z}
  开始符号: P

  产生式:
    R -> z
    R -> y
    P -> Q z P'
    P' -> x y P'
    P' -> @
    Q -> w Q'
    Q' -> R
    Q' -> y
```

### 3.3 FIRST 集

```
FIRST(P) = {w}
FIRST(P') = {@, x}
FIRST(Q) = {w}
FIRST(Q') = {y, z}
FIRST(R) = {y, z}
```

### 3.4 FOLLOW 集

```
FOLLOW(P) = {$}
FOLLOW(P') = {$}
FOLLOW(Q) = {z}
FOLLOW(Q') = {z}
FOLLOW(R) = {z}
```

### 3.5 LL(1) 分析表与判定

文法 **不是 LL(1)**，在 `Q'` 行对 `y/z` 会产生冲突。

当前预测分析表填入的产生式为：
```
M[P, w] = P -> Q z P'
M[P', x] = P' -> x y P'
M[P', $] = P' -> @
M[Q, w] = Q -> w Q'
M[Q', y] = Q' -> R
M[Q', z] = Q' -> R
M[R, y] = R -> y
M[R, z] = R -> z
```

---

## 4. `expr_grammar_raw.txt`

**原始文法:**
```
# 原始表达式文法（带左递归）
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
```

### 4.1 消除左递归

**输出文法:**
```
上下文无关文法:
  非终结符: {E, E', F, T, T'}
  终结符: {(, ), *, +, id}
  开始符号: E

  产生式:
    F -> ( E )
    F -> id
    T -> F T'
    T' -> * F T'
    T' -> @
    E -> T E'
    E' -> + T E'
    E' -> @
```

### 4.2 提取左公因子

此文法在消除左递归后已无左公因子，输出应与上一步相同。

**输出文法:**
```
上下文无关文法:
  非终结符: {E, E', F, T, T'}
  终结符: {(, ), *, +, id}
  开始符号: E

  产生式:
    F -> ( E )
    F -> id
    T -> F T'
    T' -> * F T'
    T' -> @
    E -> T E'
    E' -> + T E'
    E' -> @
```

### 4.3 FIRST 集

```
FIRST(E) = {(, id}
FIRST(E') = {+, @}
FIRST(F) = {(, id}
FIRST(T) = {(, id}
FIRST(T') = {*, @}
```

### 4.4 FOLLOW 集

```
FOLLOW(E) = {$, )}
FOLLOW(E') = {$, )}
FOLLOW(F) = {$, ), *, +}
FOLLOW(T) = {$, ), +}
FOLLOW(T') = {$, ), +}
```

### 4.5 LL(1) 分析表与判定

文法为 **LL(1)**。

```
M[E, (] = E -> T E'
M[E, id] = E -> T E'
M[E', +] = E' -> + T E'
M[E', $] = E' -> @
M[E', )] = E' -> @
M[T, (] = T -> ( E ) T'
M[T, id] = T -> id T'
M[T', *] = T' -> * F T'
M[T', +] = T' -> @
M[T', $] = T' -> @
M[T', )] = T' -> @
M[F, (] = F -> ( E )
M[F, id] = F -> id
```

---

## 5. `expr_grammar.txt`

**原始文法:**
```
# 表达式文法（带左递归）
# 非终结符
E, E', T, T', F
# 终结符
+, *, (, ), id
# 开始符号
E
# 产生式
E -> T E'
E' -> + T E' | @
T -> F T'
T' -> * F T' | @
F -> ( E ) | id
```

### 5.1 消除左递归

此文法无左递归，输出应与原始文法相同。

**输出文法:**
```
上下文无关文法:
  非终结符: {E, E', F, T, T'}
  终结符: {(, ), *, +, id}
  开始符号: E

  产生式:
    E -> T E'
    E' -> + T E'
    E' -> @
    T -> F T'
    T' -> * F T'
    T' -> @
    F -> ( E )
    F -> id
```

### 5.2 提取左公因子

此文法无左公因子，输出与上一步相同。

### 5.3 FIRST 集

```
FIRST(E) = {(, id}
FIRST(E') = {+, @}
FIRST(F) = {(, id}
FIRST(T) = {(, id}
FIRST(T') = {*, @}
```

### 5.4 FOLLOW 集

```
FOLLOW(E) = {$, )}
FOLLOW(E') = {$, )}
FOLLOW(F) = {$, ), *, +}
FOLLOW(T) = {$, ), +}
FOLLOW(T') = {$, ), +}
```

### 5.5 LL(1) 分析表与判定

文法为 **LL(1)**。

```
M[E, (] = E -> T E'
M[E, id] = E -> T E'
M[E', +] = E' -> + T E'
M[E', $] = E' -> @
M[E', )] = E' -> @
M[T, (] = T -> ( E ) T'
M[T, id] = T -> id T'
M[T', *] = T' -> * F T'
M[T', +] = T' -> @
M[T', $] = T' -> @
M[T', )] = T' -> @
M[F, (] = F -> ( E )
M[F, id] = F -> id
```
