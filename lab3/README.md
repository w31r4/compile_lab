# 语法分析算法可视化工具

本项目是一个基于 Streamlit 的 Web 应用，用于可视化编译原理中的核心语法分析算法，包括：
- 消除左递归
- 提取左公因子
- 计算 FIRST 集
- 计算 FOLLOW 集

应用界面经过优化，支持实时处理：当您在文本框中输入或修改文法时，所有算法将自动执行并立即展示结果，无需手动点击。

## 技术栈
- Python 3.11+
- Streamlit
- uv (用于项目和依赖管理)

## 环境设置与运行

本项目使用 `uv` 作为包管理器，以实现快速的依赖安装和环境管理。

### 1. 安装 uv
如果您尚未安装 `uv`，请根据您的操作系统执行相应的命令：

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

### 2. 创建虚拟环境
在项目根目录下，使用 `uv` 创建一个名为 `.venv` 的虚拟环境：
```bash
uv venv
```

### 3. 激活虚拟环境
**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\Activate.ps1
```

### 4. 安装依赖
激活环境后，使用 `uv` 安装项目所需的依赖：
```bash
uv pip install -r requirements.txt
```
*注意：如果 `requirements.txt` 不存在，请根据 `pyproject.toml` 手动安装：*
```bash
uv pip install streamlit
```

### 5. 运行应用
一切就绪后，使用 `uvx` 直接运行 Streamlit 应用：
```bash
uvx streamlit run app.py
```
`uvx` 会自动在当前 `uv` 所管理的虚拟环境中执行命令。应用启动后，您可以在浏览器中打开指定的本地地址进行访问和演示。

## 使用方法
1. 启动应用后，您会看到一个文本输入框，其中预填了一个表达式文法。
2. 您可以直接修改文本框中的文法，或者将您自己的文法粘贴进去。
3. 文法格式请遵循四段式：
    - `# 非终结符`
    - `# 终结符`
    - `# 开始符号`
    - `# 产生式`
4. 每当您修改文法内容，应用都会自动重新解析并执行所有算法，实时展示更新后的结果。
5. 项目中还包含两个示例文件，您可以将其内容复制到输入框中进行测试：
    - `test_grammars/expr_grammar.txt` (默认示例)
    - `test_grammars/complex_grammar.txt` (用于演示左公因子提取)

## 运行测试

本项目包含一个自动化测试脚本 `run_tests.py`，用于验证所有算法在不同文法下的正确性。

使用 `uv` 运行测试脚本：
```bash
uv run run_tests.py
```

该脚本会自动读取 `test_grammars/` 目录下的所有测试用例，依次执行解析、消除左递归、提取左公因子、计算 FIRST/FOLLOW 集，并打印详细的执行结果。