# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def generate_frontend_html(
    regex: str,
    test_results: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    meta = meta or {}
    regex_json = json.dumps(regex, ensure_ascii=False)
    results_json = json.dumps(test_results, ensure_ascii=False)
    meta_json = json.dumps(meta, ensure_ascii=False)

    # 说明：该 HTML 为 lab2 的“离线可视化页面”原样移植（浏览器侧自带算法实现）。
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Regex → NFA → DFA → MinDFA 可视化</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #0f1730;
      --text: #e8eefc;
      --muted: #97a3c0;
      --accent: #7c5cff;
      --ok: #2bd576;
      --bad: #ff5c7a;
      --border: rgba(255,255,255,.1);
    }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI"; background: var(--bg); color: var(--text); }}
    header {{ padding: 16px 20px; border-bottom: 1px solid var(--border); background: linear-gradient(180deg, rgba(124,92,255,.25), transparent); }}
    h1 {{ margin: 0; font-size: 18px; }}
    main {{ display: grid; grid-template-columns: 340px 1fr; gap: 16px; padding: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    label {{ display: block; font-size: 13px; color: var(--muted); margin: 10px 0 6px; }}
    input, textarea, select, button {{
      width: 100%; box-sizing: border-box;
      background: rgba(255,255,255,.04); color: var(--text);
      border: 1px solid var(--border); border-radius: 10px;
      padding: 10px 12px; outline: none;
    }}
    textarea {{ min-height: 120px; resize: vertical; }}
    button {{ cursor: pointer; background: rgba(124,92,255,.2); border-color: rgba(124,92,255,.35); }}
    button:hover {{ background: rgba(124,92,255,.28); }}
    .row {{ display: flex; gap: 10px; }}
    .row > * {{ flex: 1; }}
    .pill {{ display:inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; border: 1px solid var(--border); }}
    .ok {{ color: var(--ok); border-color: rgba(43,213,118,.35); background: rgba(43,213,118,.1); }}
    .bad {{ color: var(--bad); border-color: rgba(255,92,122,.35); background: rgba(255,92,122,.1); }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--border); padding: 8px 6px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .graph {{ background: rgba(0,0,0,.25); border: 1px solid var(--border); border-radius: 12px; overflow: auto; min-height: 420px; padding: 10px; }}
    .small {{ font-size: 12px; }}
  </style>
  <script src="https://unpkg.com/viz.js@2.1.2/viz.js"></script>
  <script src="https://unpkg.com/viz.js@2.1.2/full.render.js"></script>
</head>
<body>
  <header>
    <h1>Regex → NFA → DFA → 最小 DFA 可视化（离线 HTML）</h1>
    <div class="muted">本页面在浏览器内实现 Thompson / 子集构造 / DFA 最小化，支持改正则后即时刷新。</div>
  </header>
  <main>
    <section class="card">
      <div class="muted">当前 meta：<span id="meta"></span></div>
      <label>正则表达式</label>
      <input id="regex" />
      <div class="row" style="margin-top:10px;">
        <button id="apply">应用正则</button>
        <button id="rerender">重新渲染</button>
      </div>
      <label style="margin-top:14px;">图形类型</label>
      <select id="graphType">
        <option value="nfa">NFA</option>
        <option value="dfa">DFA</option>
        <option value="mdfa">MinDFA</option>
      </select>
      <label style="margin-top:14px;">单串检测</label>
      <input id="single" placeholder="输入一个字符串，如 abb" />
      <button id="checkSingle" style="margin-top:10px;">检测</button>
      <div id="singleOut" class="small" style="margin-top:10px;"></div>

      <label style="margin-top:14px;">多行批量检测（每行一个）</label>
      <textarea id="multi" placeholder="abb\naabb\nab"></textarea>
      <button id="checkBatch" style="margin-top:10px;">批量检测</button>
      <div style="margin-top:10px;" class="muted small">提示：页面初始不再展示预设结果，全部以你的输入为准。</div>
    </section>

    <section class="card">
      <div class="split">
        <div>
          <div class="muted">渲染图（Viz.js）</div>
          <div id="graph" class="graph"></div>
        </div>
        <div>
          <div class="muted">批量结果</div>
          <div class="graph" style="min-height:420px;">
            <table id="resultTable">
              <thead>
                <tr>
                  <th>输入</th>
                  <th>NFA</th>
                  <th>DFA</th>
                  <th>MinDFA</th>
                  <th>MinDFA 路径</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
      </div>
      <div style="margin-top:12px;" class="muted small">
        初始正则：<code id="initRegex"></code>
      </div>
    </section>
  </main>

  <script>
    const INIT_REGEX = {regex_json};
    const INIT_META = {meta_json};
    const INIT_RESULTS = {results_json};

    const EPS_KEY = "__EPS__";
    const EPS = EPS_KEY;

    const $ = (id) => document.getElementById(id);
    const setPill = (ok) => `<span class="pill ${{ok ? "ok" : "bad"}}">${{ok ? "ACCEPT" : "REJECT"}}</span>`;
    const fmtPath = (path) => Array.isArray(path) ? path.join(" → ") : "";

    // ==== Regex -> postfix -> NFA (Thompson) ====
    function insertConcat(re) {{
      const out = [];
      for (let i = 0; i < re.length; i++) {{
        const c = re[i];
        out.push(c);
        if (c === " ") continue;
        if (c !== "|" && c !== "(") {{
          const d = re[i + 1];
          if (d && d !== " " && d !== "|" && d !== ")" && d !== "*") out.push(".");
        }}
      }}
      return out.join("");
    }}

    function regexToPostfix(re) {{
      const prec = {{ "*": 3, ".": 2, "|": 1 }};
      const output = [];
      const stack = [];
      re = insertConcat(re || "");
      for (const c of re) {{
        if (c === " ") continue;
        if (c === "(") stack.push(c);
        else if (c === ")") {{
          while (stack.length && stack[stack.length - 1] !== "(") output.push(stack.pop());
          if (stack.length && stack[stack.length - 1] === "(") stack.pop();
        }} else if (prec[c]) {{
          while (stack.length && stack[stack.length - 1] !== "(" && (prec[stack[stack.length - 1]] || 0) >= prec[c]) output.push(stack.pop());
          stack.push(c);
        }} else {{
          output.push(c);
        }}
      }}
      while (stack.length) output.push(stack.pop());
      return output.join("");
    }}

    function postfixToNfa(post) {{
      let sid = 0;
      const newState = () => sid++;
      const addEdge = (trans, src, sym, dst) => {{
        trans[src] = trans[src] || {{}};
        trans[src][sym] = trans[src][sym] || new Set();
        trans[src][sym].add(dst);
      }};
      const merge = (a, b) => {{
        const out = {{}};
        for (const [k, v] of Object.entries(a)) {{
          out[k] = {{}};
          for (const [s, dsts] of Object.entries(v)) out[k][s] = new Set(dsts);
        }}
        for (const [k, v] of Object.entries(b)) {{
          out[k] = out[k] || {{}};
          for (const [s, dsts] of Object.entries(v)) {{
            out[k][s] = out[k][s] || new Set();
            for (const d of dsts) out[k][s].add(d);
          }}
        }}
        return out;
      }};

      const stack = [];
      const ops = new Set(["*", ".", "|"]);
      for (const c of post) {{
        if (!ops.has(c)) {{
          const start = newState();
          const accept = newState();
          const trans = {{}};
          addEdge(trans, start, c, accept);
          stack.push({{ start, accept, transitions: trans }});
        }} else if (c === "*") {{
          const nfa1 = stack.pop();
          const start = newState();
          const accept = newState();
          const trans = merge(nfa1.transitions, {{}});
          addEdge(trans, start, EPS, nfa1.start);
          addEdge(trans, start, EPS, accept);
          addEdge(trans, nfa1.accept, EPS, nfa1.start);
          addEdge(trans, nfa1.accept, EPS, accept);
          stack.push({{ start, accept, transitions: trans }});
        }} else if (c === ".") {{
          const nfa2 = stack.pop();
          const nfa1 = stack.pop();
          const trans = merge(nfa1.transitions, nfa2.transitions);
          addEdge(trans, nfa1.accept, EPS, nfa2.start);
          stack.push({{ start: nfa1.start, accept: nfa2.accept, transitions: trans }});
        }} else if (c === "|") {{
          const nfa2 = stack.pop();
          const nfa1 = stack.pop();
          const start = newState();
          const accept = newState();
          const trans = merge(nfa1.transitions, nfa2.transitions);
          addEdge(trans, start, EPS, nfa1.start);
          addEdge(trans, start, EPS, nfa2.start);
          addEdge(trans, nfa1.accept, EPS, accept);
          addEdge(trans, nfa2.accept, EPS, accept);
          stack.push({{ start, accept, transitions: trans }});
        }}
      }}
      return stack.pop();
    }}

    const regexToNfa = (re) => postfixToNfa(regexToPostfix(re));

    // ==== NFA -> DFA ====
    const setKey = (set) => Array.from(set).sort((a, b) => a - b).join(",");

    function epsilonClosure(trans, states) {{
      const stack = [...states];
      const closure = new Set(states);
      while (stack.length) {{
        const s = stack.pop();
        const eps = trans[s]?.[EPS] || new Set();
        for (const t of eps) if (!closure.has(t)) {{ closure.add(t); stack.push(t); }}
      }}
      return closure;
    }}

    function move(trans, states, sym) {{
      const res = new Set();
      for (const s of states) {{
        const dsts = trans[s]?.[sym] || new Set();
        dsts.forEach((d) => res.add(d));
      }}
      return res;
    }}

    function nfaToDfa(nfa) {{
      const alphabet = new Set();
      for (const mapping of Object.values(nfa.transitions)) {{
        for (const sym of Object.keys(mapping)) if (sym !== EPS_KEY) alphabet.add(sym);
      }}
      const startSet = epsilonClosure(nfa.transitions, new Set([nfa.start]));
      const dfaStates = new Map();
      const queue = [];
      const trans = {{}};
      dfaStates.set(setKey(startSet), 0);
      queue.push(startSet);
      const accepts = new Set();
      if (startSet.has(nfa.accept)) accepts.add(0);
      while (queue.length) {{
        const S = queue.shift();
        const sid = dfaStates.get(setKey(S));
        trans[sid] = trans[sid] || {{}};
        for (const sym of alphabet) {{
          const U = epsilonClosure(nfa.transitions, move(nfa.transitions, S, sym));
          if (U.size === 0) continue;
          const key = setKey(U);
          if (!dfaStates.has(key)) {{
            dfaStates.set(key, dfaStates.size);
            queue.push(U);
            if (U.has(nfa.accept)) accepts.add(dfaStates.size - 1);
          }}
          trans[sid][sym] = dfaStates.get(key);
        }}
      }}
      return {{ start: 0, accepts, transitions: trans, alphabet }};
    }}

    // ==== DFA 最小化 ====
    function reachableStates(dfa) {{
      const vis = new Set();
      const stack = [dfa.start];
      while (stack.length) {{
        const s = stack.pop();
        if (vis.has(s)) continue;
        vis.add(s);
        const m = dfa.transitions[s] || {{}};
        for (const nxt of Object.values(m)) stack.push(nxt);
      }}
      return vis;
    }}

    function makeTotalDfa(dfa) {{
      const states = reachableStates(dfa);
      const alpha = Array.from(dfa.alphabet);
      let sink = null;
      for (const s of states) {{
        dfa.transitions[s] = dfa.transitions[s] || {{}};
        for (const a of alpha) {{
          if (dfa.transitions[s][a] === undefined) {{
            if (sink === null) {{
              sink = Math.max(...states) + 1;
              dfa.transitions[sink] = {{}};
              for (const a2 of alpha) dfa.transitions[sink][a2] = sink;
            }}
            dfa.transitions[s][a] = sink;
          }}
        }}
      }}
      if (sink !== null) states.add(sink);
      return new Set(states);
    }}

    function minimizeDfa(dfa) {{
      const states = Array.from(makeTotalDfa(dfa)).sort((a, b) => a - b);
      const idx = new Map(states.map((s, i) => [s, i]));
      const n = states.length;
      const table = Array.from({{ length: n }}, () => Array(n).fill(false));
      for (let i = 0; i < n; i++) {{
        for (let j = 0; j < i; j++) {{
          if (dfa.accepts.has(states[i]) !== dfa.accepts.has(states[j])) table[i][j] = true;
        }}
      }}
      let changed = true;
      const alpha = Array.from(dfa.alphabet);
      while (changed) {{
        changed = false;
        for (let i = 0; i < n; i++) {{
          for (let j = 0; j < i; j++) {{
            if (table[i][j]) continue;
            for (const a of alpha) {{
              const p = dfa.transitions[states[i]][a];
              const q = dfa.transitions[states[j]][a];
              if (p === q) continue;
              const ii = idx.get(Math.max(p, q));
              const jj = idx.get(Math.min(p, q));
              if (table[ii][jj]) {{ table[i][j] = true; changed = true; break; }}
            }}
          }}
        }}
      }}
      const parent = new Map(states.map((s) => [s, s]));
      const find = (x) => {{ while (parent.get(x) !== x) {{ parent.set(x, parent.get(parent.get(x))); x = parent.get(x); }} return x; }};
      const unite = (x, y) => {{ const rx = find(x), ry = find(y); if (rx !== ry) parent.set(ry, rx); }};
      for (let i = 0; i < n; i++) {{
        for (let j = 0; j < i; j++) if (!table[i][j]) unite(states[i], states[j]);
      }}
      const reps = new Map();
      let newId = 0;
      for (const s of states) {{
        const r = find(s);
        if (!reps.has(r)) reps.set(r, newId++);
      }}
      const mapState = (s) => reps.get(find(s));
      const newTrans = {{}};
      for (const s of states) {{
        const ns = mapState(s);
        newTrans[ns] = newTrans[ns] || {{}};
        for (const [a, t] of Object.entries(dfa.transitions[s] || {{}})) newTrans[ns][a] = mapState(t);
      }}
      const newAccepts = new Set(Array.from(dfa.accepts).map(mapState));
      const newStart = mapState(dfa.start);
      return {{ start: newStart, accepts: newAccepts, transitions: newTrans, alphabet: new Set(dfa.alphabet) }};
    }}

    // ==== 识别与 DOT ====
    function collectAlphabet(nfa) {{
      const alpha = new Set();
      for (const mapping of Object.values(nfa.transitions)) {{
        for (const sym of Object.keys(mapping)) if (sym !== EPS_KEY) alpha.add(sym);
      }}
      return alpha;
    }}

    function traceNfa(nfa, str) {{
      const alphabet = collectAlphabet(nfa);
      for (const ch of str) {{
        if (!alphabet.has(ch)) return {{ path: [nfa.start], ok: false, reason: `字符 '${{ch}}' 不在字母表中` }};
      }}
      const key = (s, i) => `${{s}}@${{i}}`;
      const prev = new Map();
      const visited = new Set();
      const queue = [];
      const push = (state, pos, fromKey) => {{
        const k = key(state, pos);
        if (visited.has(k)) return;
        visited.add(k);
        if (fromKey) prev.set(k, fromKey);
        queue.push({{ state, pos }});
      }};
      push(nfa.start, 0, null);
      let lastKey = key(nfa.start, 0);

      const buildPath = (endKey) => {{
        if (!endKey) return [nfa.start];
        const seq = [];
        let cur = endKey;
        while (cur) {{
          const [stateStr] = cur.split("@");
          seq.push(Number(stateStr));
          cur = prev.get(cur) || null;
        }}
        return seq.reverse();
      }};

      while (queue.length) {{
        const {{ state, pos }} = queue.shift();
        const currKey = key(state, pos);
        lastKey = currKey;
        if (state === nfa.accept && pos === str.length) {{
          return {{ ok: true, path: buildPath(currKey), reason: "找到接受路径" }};
        }}
        const eps = nfa.transitions[state]?.[EPS] || new Set();
        for (const nxt of eps) push(nxt, pos, currKey);
        if (pos < str.length) {{
          const ch = str[pos];
          const dsts = nfa.transitions[state]?.[ch] || new Set();
          for (const nxt of dsts) push(nxt, pos + 1, currKey);
        }}
      }}
      return {{ ok: false, path: buildPath(lastKey), reason: "未找到接受路径" }};
    }}

    function dfaMatch(dfa, str) {{
      let state = dfa.start;
      for (const ch of str) {{
        if (!dfa.alphabet.has(ch)) return false;
        state = dfa.transitions[state]?.[ch];
        if (state === undefined) return false;
      }}
      return dfa.accepts.has(state);
    }}

    function traceDfa(dfa, str) {{
      let state = dfa.start;
      const path = [state];
      for (const ch of str) {{
        if (!dfa.alphabet.has(ch)) return {{ path, ok: false, reason: `字符 '${{ch}}' 不在字母表中` }};
        const nxt = dfa.transitions[state]?.[ch];
        if (nxt === undefined) return {{ path, ok: false, reason: `状态 ${{state}} 遇到 '${{ch}}' 无转移` }};
        path.push((state = nxt));
      }}
      return {{ path, ok: dfa.accepts.has(state), reason: "完成匹配" }};
    }}

    function automatonToDot(trans, start, accepts, isNfa, name) {{
      const lines = [`digraph ${{name}} {{`, "  rankdir=LR;", '  node [shape=circle];', '  __start [shape=point];', `  __start -> ${{start}};`];
      accepts.forEach((a) => lines.push(`  ${{a}} [shape=doublecircle];`));
      for (const [src, mapping] of Object.entries(trans)) {{
        for (const [sym, dsts] of Object.entries(mapping)) {{
          const label = sym === EPS_KEY ? "ε" : sym;
          if (isNfa) {{
            (Array.from(dsts)).forEach((d) => lines.push(`  ${{src}} -> ${{d}} [label="${{label}}"];`));
          }} else {{
            lines.push(`  ${{src}} -> ${{dsts}} [label="${{label}}"];`);
          }}
        }}
      }}
      lines.push("}}");
      return lines.join("\\n");
    }}

    function cloneDfa(dfa) {{
      const copy = {{
        start: dfa.start,
        accepts: new Set(dfa.accepts),
        alphabet: new Set(dfa.alphabet),
        transitions: {{}},
      }};
      for (const [s, mapping] of Object.entries(dfa.transitions)) {{
        copy.transitions[s] = {{}};
        for (const [a, t] of Object.entries(mapping)) copy.transitions[s][a] = t;
      }}
      return copy;
    }}

    function normalizeNfa(nfa) {{
      const trans = {{}};
      for (const [s, mapping] of Object.entries(nfa.transitions)) {{
        trans[s] = {{}};
        for (const [a, dsts] of Object.entries(mapping)) {{
          trans[s][a] = Array.from(dsts);
        }}
      }}
      return {{ start: nfa.start, accept: nfa.accept, transitions: trans }};
    }}

    function buildAll(regex) {{
      const nfa = regexToNfa(regex || "");
      const dfa = nfaToDfa(nfa);
      const mdfa = minimizeDfa(cloneDfa(dfa));
      const dots = {{
        nfa: automatonToDot(normalizeNfa(nfa).transitions, nfa.start, new Set([nfa.accept]), true, "NFA"),
        dfa: automatonToDot(dfa.transitions, dfa.start, dfa.accepts, false, "DFA"),
        mdfa: automatonToDot(mdfa.transitions, mdfa.start, mdfa.accepts, false, "MinDFA"),
      }};
      return {{ nfa, dfa, mdfa, dots }};
    }}

    const viz = new Viz();
    let CURRENT = null;

    async function renderGraph() {{
      if (!CURRENT) return;
      const type = $("graphType").value;
      const dot = CURRENT.dots[type];
      try {{
        const svg = await viz.renderString(dot);
        $("graph").innerHTML = svg;
      }} catch (e) {{
        $("graph").innerHTML = `<pre>渲染失败：${{String(e)}}</pre><pre>${{dot}}</pre>`;
      }}
    }}

    function renderBatch(strings) {{
      const tbody = $("resultTable").querySelector("tbody");
      tbody.innerHTML = "";
      for (const s of strings) {{
        const nfaRes = traceNfa(CURRENT.nfa, s);
        const dfaOk = dfaMatch(CURRENT.dfa, s);
        const mdfaRes = traceDfa(CURRENT.mdfa, s);
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><code>${{s}}</code></td>
          <td>${{setPill(nfaRes.ok)}}</td>
          <td>${{setPill(dfaOk)}}</td>
          <td>${{setPill(mdfaRes.ok)}}</td>
          <td class="small"><code>${{fmtPath(mdfaRes.path)}}</code></td>
        `;
        tbody.appendChild(tr);
      }}
    }}

    function applyRegex() {{
      const re = $("regex").value || "";
      CURRENT = buildAll(re);
      renderGraph();
      const lines = ($("multi").value || "").split(/\\r?\\n/).map((x) => x.trim()).filter(Boolean);
      if (lines.length) renderBatch(lines);
    }}

    $("apply").addEventListener("click", applyRegex);
    $("rerender").addEventListener("click", renderGraph);
    $("graphType").addEventListener("change", renderGraph);
    $("checkBatch").addEventListener("click", () => {{
      const lines = ($("multi").value || "").split(/\\r?\\n/).map((x) => x.trim()).filter(Boolean);
      renderBatch(lines);
    }});
    $("checkSingle").addEventListener("click", () => {{
      const s = $("single").value || "";
      const nfaRes = traceNfa(CURRENT.nfa, s);
      const dfaOk = dfaMatch(CURRENT.dfa, s);
      const mdfaRes = traceDfa(CURRENT.mdfa, s);
      $("singleOut").innerHTML = `
        <div>NFA：${{setPill(nfaRes.ok)}} <span class="muted small">${{nfaRes.reason}}</span></div>
        <div>DFA：${{setPill(dfaOk)}}</div>
        <div>MinDFA：${{setPill(mdfaRes.ok)}} <span class="muted small">${{mdfaRes.reason}}</span></div>
        <div class="muted small">MinDFA 路径：<code>${{fmtPath(mdfaRes.path)}}</code></div>
      `;
    }});

    // init
    $("regex").value = INIT_REGEX;
    $("initRegex").textContent = INIT_REGEX;
    $("meta").textContent = JSON.stringify(INIT_META);
    applyRegex();
  </script>
</body>
</html>
"""


def write_frontend_html(
    regex: str,
    test_results: List[Dict[str, Any]],
    output_path: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    html = generate_frontend_html(regex=regex, test_results=test_results, meta=meta)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

