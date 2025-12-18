## Lab2-new：正则表达式 -> NFA -> DFA -> 最小 DFA

本目录按 `lab3/`、`lab4/` 的组织方式重写 `lab2/`，并引入 Streamlit 做“前后端分离”：
- 后端：`lab2-new/src/`（算法与导出能力）
- 前端：`lab2-new/app.py`（Streamlit UI）
- CLI：`lab2-new/main.py`（批处理/交互菜单，与 lab2 功能对齐）

### 运行 Streamlit（推荐）
在 `lab2-new/` 目录下：
```bash
pip install streamlit
streamlit run app.py
```

### 运行 CLI（兼容 lab2 的批处理/交互）
```bash
python main.py
```

批处理模式（不进入交互菜单）示例：
```bash
python main.py --regex "(a|b)*abb" --strings abb aabb ab --output-html visualization.html --no-interactive
```
