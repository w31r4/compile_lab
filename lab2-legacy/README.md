## 词法分析实验辅助工具

- 正则表达式 -> NFA -> DFA -> 最小 DFA 的参考实现
- 交互式 CLI，支持批量测试多个字符串
- 自动导出状态转移图（DOT），若安装 Graphviz 可同时生成 PNG
- 可查看 NFA / DFA / 最小 DFA 的转移表，演示匹配路径

### 快速开始
1. 准备 Python 3 环境（可直接用 `python3` 运行）。
2. 可选：安装 Graphviz 以生成 PNG 图：
   ```bash
   sudo apt-get install graphviz  # 或使用对应系统的包管理器
   ```
3. 运行：
   ```bash
   python3 main.py
   ```
4. 按提示输入正则表达式（默认 `(a|b)*abb`），程序会自动在当前目录导出 DOT（以及在装好 Graphviz 时自动生成 PNG）。

### 功能菜单
- 批量测试字符串：多行输入要测试的字符串，空行结束，显示 DFA 与最小 DFA 的接受情况。
- 查看转移表：打印 NFA、DFA、最小 DFA 的状态转移表。
- 显示匹配路径：逐步展示某个字符串在 DFA 中经过的状态序列。
- 导出图：重新输出 DOT/PNG（可自定义文件名前缀）。
- 切换正则：重新构造并导出对应自动机。

### 一键生成前端可视化
- 非交互模式生成可展示的 HTML（页面内可修改正则、批量/单串检测、渲染+下载图）：
  ```bash
  python3 main.py --regex "(a|b)*abb" --output-html visualization.html --no-interactive
  # 生成 DOT/PNG 以及 visualization.html，直接用浏览器打开即可交互查看
  ```

### HTML 页面使用说明（visualization.html）
- 打开：直接双击或浏览器打开 `visualization.html`，无需后台服务。
- 修改正则：顶部输入新正则并点“应用正则”，页面会重新构造 NFA/DFA/最小 DFA，刷新图和检测逻辑。
- 图形展示：下拉切换 NFA/DFA/最小 DFA；“重新渲染”刷新当前图；“下载当前图为 PNG” 直接导出图片。
- 单个检测：输入一个串点“检测”，显示 DFA/最小 DFA 结果与最小 DFA 路径。
- 多行批量检测：每行一个串点“批量检测”，表格刷新结果与最小 DFA 路径。页面不再有预设结果，全部由你输入决定。

### 生成的文件
- `automaton_<正则清洗>.dot/png`：NFA、DFA、最小 DFA 的状态转移图。
- 若未安装 Graphviz，已生成的 DOT 文件可后续通过 `dot -Tpng xxx.dot -o xxx.png` 转为 PNG。
