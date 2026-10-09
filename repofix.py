"""
repofix.py — RepoFix-Mini benchmark + agent harness for studying how the
*representation of repository structure* affects Gemma 4 coding agents.

Pipeline
  1. build_benchmark(): clone pinned, permissively-licensed Python repos,
     inject single-site AST mutations into library code, keep only mutants
     that break 1..MAX_F2P previously-passing tests (verifiable by pytest).
  2. Representations (the experimental factor), all injected into the first
     user turn under an equal token budget:
        none    : no structural context (agent must explore with tools)
        tree    : file tree of the package (+ line counts)
        repomap : file tree + class/function signatures (test-anchored order)
        graph   : test-anchored static call graph (nodes + edges, depth-2)
  3. Agent loop with a fixed tool set (list_dir, search, view, edit,
     run_tests, finish) and a JSON action protocol, identical across
     conditions. Evaluation = originally failing tests (F2P) pass and the
     other tests in the same files (P2P) still pass.

License: Apache-2.0 (competition winner-license compatible).
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

# --------------------------------------------------------------------------
# 1. Repository specs (pinned tags, OSI licenses that allow commercial use)
# --------------------------------------------------------------------------
REPOS = {
    "toolz": dict(url="https://github.com/pytoolz/toolz.git", tag="1.1.0",
                  license="BSD-3-Clause", pythonpath=".", pkg="toolz",
                  tests=["toolz/tests"], pytest_args=[]),
    "boltons": dict(url="https://github.com/mahmoud/boltons.git", tag="26.2.0",
                    license="BSD-3-Clause", pythonpath=".", pkg="boltons",
                    tests=["tests"], pytest_args=[]),
    "click": dict(url="https://github.com/pallets/click.git", tag="8.5.0",
                  license="BSD-3-Clause", pythonpath="src", pkg="src/click",
                  tests=["tests"], pytest_args=[]),
    "marshmallow": dict(url="https://github.com/marshmallow-code/marshmallow.git",
                        tag="4.3.1", license="MIT", pythonpath="src",
                        pkg="src/marshmallow", tests=["tests"], pytest_args=[]),
    "jinja": dict(url="https://github.com/pallets/jinja.git", tag="3.1.6",
                  license="BSD-3-Clause", pythonpath="src", pkg="src/jinja2",
                  tests=["tests"],
                  pytest_args=["--ignore=tests/test_async.py",
                               "--ignore=tests/test_async_filters.py"]),
}
MAX_F2P = 8          # a mutant must break between 1 and MAX_F2P tests
TEST_TIMEOUT = 120   # seconds per pytest run


def approx_tokens(text: str) -> int:
    """Cheap, tokenizer-free estimate (~4 chars/token) used for budgets."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# 2. Git + pytest utilities
# --------------------------------------------------------------------------
def clone_repo(name: str, root: Path) -> Path:
    spec = REPOS[name]
    dest = Path(root) / name
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "-q", "--depth", "1", "--branch",
                        spec["tag"], spec["url"], str(dest)], check=True)
    else:  # restore pristine state if a previous run was interrupted
        subprocess.run(["git", "checkout", "-q", "--", "."], cwd=dest, check=True)
    return dest


def _pytest_env(repo_dir: Path, name: str) -> dict:
    env = dict(os.environ)
    pp = str((repo_dir / REPOS[name]["pythonpath"]).resolve())
    env["PYTHONPATH"] = pp + os.pathsep + env.get("PYTHONPATH", "")
    env["COLUMNS"] = "300"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


_SUMMARY_RE = re.compile(r"^(PASSED|FAILED|ERROR|XPASS|XFAIL|SKIPPED)\s+(\S.*)$")


def run_pytest(repo_dir: Path, name: str, targets=None, timeout=TEST_TIMEOUT):
    """Run pytest; return dict nodeid -> (status, message) and raw output."""
    spec = REPOS[name]
    targets = targets or spec["tests"]
    cmd = [sys.executable, "-m", "pytest", "-q", "-rA", "--tb=no",
           "-p", "no:cacheprovider", "-p", "no:randomly",
           "--continue-on-collection-errors", *spec["pytest_args"], *targets]
    try:
        p = subprocess.run(cmd, cwd=repo_dir, env=_pytest_env(repo_dir, name),
                           stdin=subprocess.DEVNULL, capture_output=True,
                           text=True, timeout=timeout)
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return {}, "TIMEOUT"
    results = {}
    for line in out.splitlines():
        m = _SUMMARY_RE.match(line.strip())
        if not m:
            continue
        status, rest = m.group(1), m.group(2)
        nodeid, _, msg = rest.partition(" - ")
        results[nodeid.strip()] = (status, msg.strip())
    return results, out


def passing(results: dict) -> set:
    return {k for k, (s, _) in results.items() if s == "PASSED"}


def failing(results: dict) -> set:
    return {k for k, (s, _) in results.items() if s in ("FAILED", "ERROR")}


# --------------------------------------------------------------------------
# 3. Mutation operators (single-line, single-site, semantics-changing)
# --------------------------------------------------------------------------
_CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
             ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn,
             ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
_BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add}


@dataclasses.dataclass
class Mutation:
    file: str            # path relative to repo root
    lineno: int
    col: int
    end_col: int
    original: str
    mutated: str
    operator: str
    function: str        # qualified name of the enclosing function


def _qualnames(tree):
    """Map each FunctionDef node -> qualified name."""
    out = {}

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}.{ch.name}" if prefix else ch.name
                if not isinstance(ch, ast.ClassDef):
                    out[ch] = q
                walk(ch, q)
            else:
                walk(ch, prefix)
    walk(tree, "")
    return out


def _candidates_for_node(node):
    """Yield (operator_name, mutated_node) for one AST node."""
    import copy
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        op = type(node.ops[0])
        if op in _CMP_SWAP:
            m = copy.deepcopy(node)
            m.ops = [_CMP_SWAP[op]()]
            yield "cmp_swap", m
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_SWAP:
        m = copy.deepcopy(node)
        m.op = _BIN_SWAP[type(node.op)]()
        yield "arith_swap", m
    if isinstance(node, ast.BoolOp) and len(node.values) == 2:
        m = copy.deepcopy(node)
        m.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        yield "bool_swap", m
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        yield "drop_not", copy.deepcopy(node.operand)
    if (isinstance(node, ast.Constant) and type(node.value) is int
            and node.value in (0, 1, 2)):
        yield "off_by_one", ast.Constant(node.value + 1)


def enumerate_mutations(repo_dir: Path, name: str):
    spec = REPOS[name]
    pkg = repo_dir / spec["pkg"]
    muts = []
    for f in sorted(pkg.rglob("*.py")):
        rel = f.relative_to(repo_dir).as_posix()
        if "/tests/" in "/" + rel or rel.split("/")[-1].startswith("test_"):
            continue
        src = f.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.splitlines(keepends=True)
        qn = _qualnames(tree)
        for fn, qual in qn.items():
            body = fn.body
            # skip docstring
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], "value", None), ast.Constant):
                body = body[1:]
            for stmt in body:
                for node in ast.walk(stmt):
                    # only nodes directly owned by this function (not nested defs)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                         ast.ClassDef, ast.Lambda)):
                        continue
                    if not hasattr(node, "lineno") or node.lineno != getattr(
                            node, "end_lineno", -1):
                        continue
                    line = lines[node.lineno - 1]
                    seg = line.encode()[node.col_offset:node.end_col_offset].decode(
                        errors="ignore")
                    if not seg or "f'" in seg or 'f"' in seg:
                        continue
                    for op_name, mnode in _candidates_for_node(node):
                        try:
                            new = ast.unparse(mnode)
                        except Exception:
                            continue
                        if isinstance(node, ast.BinOp) or isinstance(node, ast.BoolOp):
                            new = f"({new})" if seg.startswith("(") else new
                        if new == seg:
                            continue
                        muts.append(Mutation(rel, node.lineno, node.col_offset,
                                             node.end_col_offset, seg, new,
                                             op_name, qual))
    return muts


def apply_mutation(repo_dir: Path, m: Mutation, revert=False) -> bool:
    f = repo_dir / m.file
    lines = f.read_text(encoding="utf-8").splitlines(keepends=True)
    line_b = lines[m.lineno - 1].encode()
    old, new = (m.mutated, m.original) if revert else (m.original, m.mutated)
    start = m.col
    end = m.col + len(old.encode()) if revert else m.end_col
    if line_b[start:end].decode(errors="ignore") != old:
        return False
    lines[m.lineno - 1] = (line_b[:start] + new.encode() + line_b[end:]).decode()
    text = "".join(lines)
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    f.write_text(text, encoding="utf-8")
    return True


# --------------------------------------------------------------------------
# 4. Benchmark construction
# --------------------------------------------------------------------------
def build_benchmark(names, root: Path, per_repo=20, max_tries=150, seed=0,
                    log=print, out_path=None):
    """Return list of task dicts. Deterministic given (names, seed).
    If out_path is given, tasks are appended per repo and repos already
    present in the file are skipped (safe to resume after interruption)."""
    tasks, done_repos = [], set()
    if out_path and Path(out_path).exists():
        tasks = load_tasks(out_path)
        done_repos = {t["repo"] for t in tasks}
    for name in names:
        if name in done_repos:
            log(f"[{name}] already built, skipping")
            continue
        new_tasks = []
        repo = clone_repo(name, root)
        t0 = time.time()
        base, _ = run_pytest(repo, name)
        base_pass = passing(base)
        # mutants can create infinite loops: cap each run at ~4x baseline
        mut_timeout = max(15, int(4 * (time.time() - t0)) + 5)
        log(f"[{name}] baseline: {len(base_pass)} passing tests")
        muts = enumerate_mutations(repo, name)
        rng = random.Random(f"{seed}-{name}")
        rng.shuffle(muts)
        # at most one task per function, to spread tasks over the repo
        seen_fn, kept, tries = set(), 0, 0
        for m in muts:
            if kept >= per_repo or tries >= max_tries:
                break
            key = (m.file, m.function)
            if key in seen_fn:
                continue
            if not apply_mutation(repo, m):
                continue
            tries += 1
            try:
                res, out = run_pytest(repo, name, timeout=mut_timeout)
            finally:
                assert apply_mutation(repo, m, revert=True), "revert failed"
            if out == "TIMEOUT":
                continue
            broken = sorted(base_pass & failing(res))
            if not (1 <= len(broken) <= MAX_F2P):
                continue
            test_files = sorted({b.split("::")[0] for b in broken})
            if len(test_files) > 2:
                continue
            seen_fn.add(key)
            p2p = sorted(t for t in base_pass - set(broken)
                         if t.split("::")[0] in test_files)
            msgs = {b: res[b][1] for b in broken}
            tid = f"{name}-{hashlib.sha1(json.dumps(dataclasses.asdict(m)).encode()).hexdigest()[:8]}"
            new_tasks.append(dict(task_id=tid, repo=name, **{"tag": REPOS[name]["tag"]},
                              mutation=dataclasses.asdict(m), f2p=broken,
                              p2p=p2p, test_files=test_files, messages=msgs,
                              issue=make_issue(broken, msgs)))
            kept += 1
            log(f"  + {tid}  {m.operator:11s} {m.file}:{m.lineno} "
                f"({m.function})  breaks {len(broken)}")
        log(f"[{name}] kept {kept} tasks after {tries} mutants")
        tasks += new_tasks
        if out_path:
            with open(out_path, "a", encoding="utf-8") as fh:
                for t in new_tasks:
                    fh.write(json.dumps(t) + "\n")
    return tasks


def make_issue(f2p, messages) -> str:
    lines = ["The test suite reports the following failures after a recent change "
             "to the library source code:", ""]
    for t in f2p:
        msg = messages.get(t, "")
        lines.append(f"FAILED {t}" + (f" - {msg[:200]}" if msg else ""))
    lines += ["", "Find the bug in the library source (not in the tests) and fix it "
              "so that these tests pass without breaking other tests."]
    return "\n".join(lines)


def save_tasks(tasks, path):
    with open(path, "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")


def load_tasks(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


# --------------------------------------------------------------------------
# 5. Static analysis for representations
# --------------------------------------------------------------------------
class RepoIndex:
    """Definitions, signatures and a name-resolved static call graph."""

    def __init__(self, repo_dir: Path, name: str):
        self.repo_dir, self.name = Path(repo_dir), name
        pkg = self.repo_dir / REPOS[name]["pkg"]
        self.files = {}          # rel -> n_lines
        self.defs = {}           # qualname@file -> dict(file,line,sig,kind,short)
        self.by_short = defaultdict(list)
        self.calls = defaultdict(set)   # def key -> set(short names called)
        for f in sorted(pkg.rglob("*.py")):
            rel = f.relative_to(self.repo_dir).as_posix()
            if "/tests/" in "/" + rel:
                continue
            src = f.read_text(encoding="utf-8")
            self.files[rel] = src.count("\n") + 1
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            self._index(tree, rel, "")

    def _index(self, node, rel, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}.{ch.name}" if prefix else ch.name
                key = f"{q}@{rel}"
                if isinstance(ch, ast.ClassDef):
                    bases = ", ".join(ast.unparse(b) for b in ch.bases)
                    sig = f"class {ch.name}({bases})" if bases else f"class {ch.name}"
                    kind = "class"
                else:
                    try:
                        args = ast.unparse(ch.args)
                    except Exception:
                        args = "..."
                    sig = f"def {ch.name}({args})"
                    kind = "def"
                    for sub in ast.walk(ch):
                        if isinstance(sub, ast.Call):
                            fn = sub.func
                            if isinstance(fn, ast.Name):      # plain call
                                self.calls[key].add((fn.id, False))
                            elif isinstance(fn, ast.Attribute):  # obj.method()
                                self.calls[key].add((fn.attr, True))
                self.defs[key] = dict(file=rel, line=ch.lineno, sig=sig,
                                      kind=kind, short=ch.name, qual=q)
                self.by_short[ch.name].append(key)
                self._index(ch, rel, q)

    # -- anchoring: which library symbols does the failing test mention? --
    def seeds_from_tests(self, task) -> list:
        names = set()
        for tf in task["test_files"]:
            p = self.repo_dir / tf
            if not p.exists():
                continue
            src = p.read_text(encoding="utf-8")
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            wanted = {t.split("::")[-1].split("[")[0] for t in task["f2p"]}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and node.name in wanted:
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Name):
                            names.add(sub.id)
                        elif isinstance(sub, ast.Attribute):
                            names.add(sub.attr)
        seeds = []
        for n in sorted(names):
            for key in self.by_short.get(n, [])[:3]:
                seeds.append(key)
        return seeds

    def file_relevance(self, seeds) -> dict:
        score = defaultdict(int)
        for k in seeds:
            score[self.defs[k]["file"]] += 1
        return score


# --------------------------------------------------------------------------
# 6. Representations (experimental conditions)
# --------------------------------------------------------------------------
CONDITIONS = ["none", "tree", "repomap", "graph"]


def _clip(text, budget_tokens):
    if approx_tokens(text) <= budget_tokens:
        return text
    cut = text[: budget_tokens * 4]
    return cut[: cut.rfind("\n")] + "\n... [truncated to budget]"


def build_representation(kind, index: RepoIndex, task, budget=1500) -> str:
    if kind == "none":
        return ""
    seeds = index.seeds_from_tests(task)
    rel = index.file_relevance(seeds)
    files = sorted(index.files, key=lambda f: (-rel.get(f, 0), f))
    if kind == "tree":
        body = "\n".join(f"{f}  ({index.files[f]} lines)" for f in sorted(index.files))
        return _clip("# Repository files\n" + body, budget)
    if kind == "repomap":
        out = ["# Repository map (files ordered by relevance to the failing tests)"]
        for f in files:
            out.append(f"{f}:")
            ds = sorted((d for d in index.defs.values() if d["file"] == f),
                        key=lambda d: d["line"])
            for d in ds:
                indent = "  " * (1 + d["qual"].count("."))
                out.append(f"{indent}L{d['line']}: {d['sig'][:120]}")
        return _clip("\n".join(out), budget)
    if kind == "graph":
        # BFS from test-referenced symbols, depth <= 2, over two edge types:
        #   contains : class -> its methods      calls : function -> callee
        methods = defaultdict(list)
        for k, d in index.defs.items():
            if "." in d["qual"]:
                parent = f"{d['qual'].rsplit('.', 1)[0]}@{d['file']}"
                if parent in index.defs and index.defs[parent]["kind"] == "class":
                    methods[parent].append(k)
        order, depth, edges = [], {}, []
        q = deque()
        for s in seeds:
            if s not in depth:
                depth[s] = 0
                q.append(s)
        while q:
            k = q.popleft()
            order.append(k)
            if depth[k] >= 2:
                continue
            nxt = [(t, "contains") for t in methods.get(k, [])]
            for callee, is_attr in sorted(index.calls.get(k, ())):
                targets = index.by_short.get(callee, [])
                if not is_attr:   # bare names resolve to module-level defs/classes
                    targets = [t for t in targets if "." not in index.defs[t]["qual"]]
                if targets and len(targets) <= 4:      # skip ambiguous names
                    nxt += [(t, "calls") for t in targets]
            for t, kind_e in nxt:
                edges.append((k, t, kind_e))
                if t not in depth:
                    depth[t] = depth[k] + 1
                    q.append(t)
        out = ["# Call graph anchored on the failing tests",
               "## Nodes  [depth]  qualified_name  file:line  signature",
               "## (depth 0 = symbols used directly by the failing tests)"]
        for k in order:
            d = index.defs[k]
            out.append(f"[{depth[k]}] {d['qual']}  {d['file']}:L{d['line']}  "
                       f"{d['sig'][:100]}")
        out.append("## Edges")
        seen = set()
        for a, b, kind_e in edges:
            e = (index.defs[a]["qual"], index.defs[b]["qual"], kind_e)
            if e not in seen:
                seen.add(e)
                arrow = "-calls->" if kind_e == "calls" else "-has->"
                out.append(f"{e[0]} {arrow} {e[1]}")
        return _clip("\n".join(out), budget)
    raise ValueError(kind)


def gold_in_representation(rep: str, task) -> bool:
    """Is the buggy function locatable from the context alone?
    (gold file listed AND its top-level symbol named)."""
    m = task["mutation"]
    top = m["function"].split(".")[0]
    return (m["file"] in rep) and bool(re.search(rf"\b{re.escape(top)}\b", rep))


# --------------------------------------------------------------------------
# 7. Agent environment (tools)
# --------------------------------------------------------------------------
class RepoEnv:
    MAX_OBS = 3000

    def __init__(self, task, clean_root: Path, work_root: Path):
        self.task = task
        self.name = task["repo"]
        self.dir = (Path(work_root) / f"ep_{task['task_id']}_{os.getpid()}").resolve()
        if self.dir.exists():
            shutil.rmtree(self.dir)
        shutil.copytree(Path(clean_root) / self.name, self.dir,
                        ignore=shutil.ignore_patterns(".git"))
        m = Mutation(**task["mutation"])
        assert apply_mutation(self.dir, m), "could not apply mutation"
        self.viewed, self.edited, self.n_test_runs = set(), set(), 0
        self.cur_step, self.gold = 0, task["mutation"]["file"]
        self.first_gold_view = self.first_gold_edit = None
        self.test_dirs = REPOS[self.name]["tests"]

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _safe(self, path):
        p = (self.dir / str(path).lstrip("/")).resolve()
        if not str(p).startswith(str(self.dir.resolve())):
            raise ValueError("path outside repository")
        return p

    def _is_test(self, rel):
        rel = rel.replace("\\", "/")
        return any(rel.startswith(t) for t in self.test_dirs) or \
            rel.split("/")[-1].startswith("test_") or "/tests/" in "/" + rel

    def _cut(self, s):
        return s if len(s) <= self.MAX_OBS else s[: self.MAX_OBS] + "\n...[output truncated]"

    # ---- tools ----
    def list_dir(self, path="."):
        p = self._safe(path)
        if not p.is_dir():
            return f"Error: {path} is not a directory"
        items = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir()
                       if not x.name.startswith(".") and x.name != "__pycache__")
        return self._cut("\n".join(items))

    def search(self, pattern, path="."):
        p = self._safe(path)
        hits = []
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))
        for f in sorted(p.rglob("*.py")) if p.is_dir() else [p]:
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{f.relative_to(self.dir).as_posix()}:{i}: {line.strip()[:160]}")
                        if len(hits) >= 40:
                            break
            except Exception:
                continue
            if len(hits) >= 40:
                break
        return self._cut("\n".join(hits) if hits else "No matches.")

    def view(self, path, start=1, end=None):
        p = self._safe(path)
        if not p.is_file():
            return f"Error: file not found: {path}"
        lines = p.read_text(encoding="utf-8").splitlines()
        start = max(1, int(start or 1))
        end = min(len(lines), int(end) if end else start + 99, start + 149)
        rel = p.relative_to(self.dir).as_posix()
        self.viewed.add(rel)
        if rel == self.gold and self.first_gold_view is None:
            self.first_gold_view = self.cur_step
        body = "\n".join(f"{i:5d} {lines[i-1]}" for i in range(start, end + 1))
        return self._cut(f"{path} (lines {start}-{end} of {len(lines)})\n{body}")

    def edit(self, path, old, new):
        p = self._safe(path)
        rel = p.relative_to(self.dir).as_posix()
        if self._is_test(rel):
            return "Error: editing test files is not allowed."
        if not p.is_file():
            return f"Error: file not found: {path}"
        text = p.read_text(encoding="utf-8")
        new_text, how = _apply_edit(text, old, new)
        if new_text is None:
            return f"Error: {how} Copy 'old' exactly from a view (without line numbers)."
        try:
            ast.parse(new_text)
        except SyntaxError as e:
            return f"Error: edit would produce a SyntaxError: {e}"
        p.write_text(new_text, encoding="utf-8")
        self.edited.add(rel)
        if rel == self.gold and self.first_gold_edit is None:
            self.first_gold_edit = self.cur_step
        return f"Edited {rel} successfully."

    def run_tests(self):
        self.n_test_runs += 1
        res, out = run_pytest(self.dir, self.name, self.task["test_files"])
        if out == "TIMEOUT":
            return "Test run timed out."
        fails = sorted(failing(res))
        npass = len(passing(res))
        if not fails:
            return f"All {npass} tests in {', '.join(self.task['test_files'])} passed."
        lines = [f"{npass} passed, {len(fails)} failed:"]
        lines += [f"FAILED {t} - {res[t][1][:160]}" for t in fails[:15]]
        return self._cut("\n".join(lines))

    # ---- evaluation ----
    def evaluate(self):
        res, out = run_pytest(self.dir, self.name, self.task["test_files"])
        ok_f2p = all(res.get(t, ("MISSING",))[0] == "PASSED" for t in self.task["f2p"])
        # P2P ids absent from this run are ignored: a few parametrized ids embed
        # the current date/time and therefore differ between runs.
        ok_p2p = all(res[t][0] == "PASSED" for t in self.task["p2p"] if t in res)
        return dict(resolved=bool(ok_f2p and ok_p2p), f2p_pass=ok_f2p, p2p_pass=ok_p2p)


_LINENO = re.compile(r"^\s*\d+ ?")


def _strip_linenos(s):
    lines = s.split("\n")
    if lines and all(_LINENO.match(l) for l in lines if l.strip()):
        return "\n".join(_LINENO.sub("", l, count=1) for l in lines)
    return s


def _apply_edit(text, old, new):
    """Exact unique match first; then tolerate pasted line numbers and
    trailing-whitespace differences (line-level match). Returns (text, how)."""
    if not old:
        return None, "'old' is empty."
    n = text.count(old)
    if n == 1:
        return text.replace(old, new, 1), "exact"
    if n > 1:
        return None, f"'old' matches {n} times; include more surrounding lines."
    old2, new2 = _strip_linenos(old), _strip_linenos(new)
    if text.count(old2) == 1:
        return text.replace(old2, new2, 1), "linenos"
    flines = text.split("\n")
    olines = [l.rstrip() for l in old2.strip("\n").split("\n")]
    k = len(olines)
    hits = [i for i in range(len(flines) - k + 1)
            if [l.rstrip() for l in flines[i:i + k]] == olines]
    if len(hits) == 1:
        i = hits[0]
        out = flines[:i] + new2.strip("\n").split("\n") + flines[i + k:]
        return "\n".join(out), "fuzzy"
    # last resort: ignore trailing whitespace everywhere (benign normalisation)
    norm = lambda x: "\n".join(l.rstrip() for l in x.split("\n"))
    tn, on = norm(text), norm(old2)
    if on.strip() and tn.count(on) == 1:
        return tn.replace(on, norm(new2), 1), "rstrip"
    return None, "'old' was not found in the file."


# --------------------------------------------------------------------------
# 8. Agent protocol
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an autonomous software engineering agent working inside a Python repository.
Your goal: fix the bug in the library source code so that the failing tests pass. Do not edit tests.

You act by emitting exactly ONE action per turn as a JSON object inside a ```json code block.
Every action starts with a short "thought" (at most two sentences), then "action" and its fields, e.g.
  {"thought": "The failing test calls merge; inspect it.", "action": "view", "path": "pkg/core.py", "start": 1, "end": 80}
Available actions:
  {"action": "list_dir", "path": "<dir>"}
  {"action": "search", "pattern": "<regex>", "path": "<dir or file, optional>"}
  {"action": "view", "path": "<file>", "start": <line>, "end": <line>}
  {"action": "edit", "path": "<file>", "old": "<exact existing text>", "new": "<replacement text>"}
  {"action": "run_tests"}
  {"action": "finish"}

Rules:
- "old" must match the file exactly once (copy it from a view, without line numbers).
- Keep edits minimal: the bug is a small change somewhere in the library code.
- You MAY view test files to see exactly what the failing tests expect; you may NOT edit them.
- Write nothing outside the JSON block.
- Call run_tests to verify your fix, then finish."""


def build_first_message(task, representation: str) -> str:
    parts = [f"## Issue\n{task['issue']}"]
    if representation:
        parts.append(f"## Repository structure\n{representation}")
    parts.append("Begin. Emit your first action.")
    return "\n\n".join(parts)


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def parse_action(text: str):
    """Lenient parser: fenced JSON first, then the last {...} object."""
    cands = _JSON_BLOCK.findall(text)
    if not cands:
        # fallback: last balanced-looking object
        starts = [m.start() for m in re.finditer(r"\{", text)]
        for s in reversed(starts):
            chunk = text[s:]
            depth = 0
            for i, ch in enumerate(chunk):
                depth += ch == "{"
                depth -= ch == "}"
                if depth == 0:
                    cands.append(chunk[: i + 1])
                    break
            if cands:
                break
    for c in reversed(cands):
        try:
            obj = json.loads(c, strict=False)   # tolerate raw newlines/tabs
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except json.JSONDecodeError:
            try:  # tolerate python-style literals
                obj = ast.literal_eval(c)
                if isinstance(obj, dict) and "action" in obj:
                    return obj
            except Exception:
                continue
    return None


def execute(env: RepoEnv, act: dict) -> str:
    a = act.get("action")
    try:
        if a == "list_dir":
            return env.list_dir(act.get("path", "."))
        if a == "search":
            return env.search(str(act.get("pattern", "")), act.get("path", "."))
        if a == "view":
            return env.view(act["path"], act.get("start", 1), act.get("end"))
        if a == "edit":
            return env.edit(act["path"], act.get("old", ""), act.get("new", ""))
        if a == "run_tests":
            return env.run_tests()
        return f"Error: unknown action '{a}'."
    except KeyError as e:
        return f"Error: missing field {e} for action '{a}'."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _free_cuda():
    try:
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def compact_history(messages, max_tokens):
    """Elide old observations (keep system, first user, last 6 turns)."""
    def total():
        return sum(approx_tokens(m["content"]) for m in messages)
    i = 2
    while total() > max_tokens and i < len(messages) - 6:
        if messages[i]["role"] == "user" and not messages[i]["content"].startswith("[elided"):
            messages[i]["content"] = "[elided old observation]"
        i += 1
    return messages


def run_episode(llm, task, condition, index: RepoIndex, clean_root, work_root,
                max_steps=20, budget=1500, ctx_tokens=6000):
    t0 = time.time()
    rep = build_representation(condition, index, task, budget)
    env = RepoEnv(task, clean_root, work_root)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_first_message(task, rep)}]
    rec = dict(task_id=task["task_id"], repo=task["repo"], condition=condition,
               model=getattr(llm, "name", "llm"), rep_tokens=approx_tokens(rep) if rep else 0,
               gold_in_context=gold_in_representation(rep, task) if rep else False,
               steps=0, format_errors=0, finished=False, prompt_tokens=0,
               gen_tokens=0, error=None, actions=[], oom_retries=0,
               invalid_samples=[])
    try:
        for step in range(max_steps):
            compact_history(messages, ctx_tokens)
            rec["prompt_tokens"] += sum(approx_tokens(m["content"]) for m in messages)
            try:
                reply = llm(messages)
            except Exception as e:                       # CUDA OOM -> retry once
                if "out of memory" not in str(e).lower():
                    raise
                rec["oom_retries"] += 1
                _free_cuda()
                compact_history(messages, ctx_tokens // 2)
                reply = llm(messages)
            rec["gen_tokens"] += approx_tokens(reply)
            rec["steps"] = step + 1
            messages.append({"role": "assistant", "content": reply})
            act = parse_action(reply)
            if act is None:
                rec["format_errors"] += 1
                rec["actions"].append("INVALID")
                if len(rec["invalid_samples"]) < 3:
                    rec["invalid_samples"].append(reply[-400:])
                messages.append({"role": "user", "content":
                                 "Error: no valid JSON action found. Emit exactly one "
                                 "action as a JSON object in a ```json block."})
                continue
            rec["actions"].append(act.get("action"))
            if act.get("action") == "finish":
                rec["finished"] = True
                break
            env.cur_step = step + 1
            obs = execute(env, act)
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})
        rec.update(env.evaluate())
    except Exception as e:
        rec.update(resolved=False, f2p_pass=False, p2p_pass=False,
                   error=f"{type(e).__name__}: {e}")
    finally:
        g = task["mutation"]["file"]
        rec["viewed_gold_file"] = g in env.viewed
        rec["edited_gold_file"] = g in env.edited
        rec["n_edited_files"] = len(env.edited)
        rec["test_runs"] = env.n_test_runs
        rec["first_gold_view"] = env.first_gold_view
        rec["first_gold_edit"] = env.first_gold_edit
        rec["max_steps"] = max_steps
        rec["wall_s"] = round(time.time() - t0, 1)
        env.close()
    return rec


# --------------------------------------------------------------------------
# 9. Reference policies (sanity checks, no LLM needed)
# --------------------------------------------------------------------------
class OraclePolicy:
    """Emits the gold fix. Expected resolve rate: 100%."""
    name = "oracle"

    def __init__(self, task):
        self.task, self.i = task, 0

    def __call__(self, messages):
        m = self.task["mutation"]
        self.i += 1
        if self.i == 1:
            return '```json\n{"action": "view", "path": "%s", "start": %d, "end": %d}\n```' % (
                m["file"], max(1, m["lineno"] - 3), m["lineno"] + 3)
        if self.i == 2:
            # rebuild the 3-line window around the mutated line from the view
            shown = {}
            for ln in messages[-1]["content"].splitlines()[1:]:
                mm = re.match(r"^\s*(\d+) (.*)$", ln)
                if mm:
                    shown[int(mm.group(1))] = mm.group(2)
            L = m["lineno"]
            bad = shown[L].encode()
            fixed = (bad[:m["col"]] + m["original"].encode()
                     + bad[m["col"] + len(m["mutated"].encode()):]).decode()
            win = [shown[i] for i in (L - 1, L, L + 1) if i in shown]
            new = [shown[i] if i != L else fixed for i in (L - 1, L, L + 1) if i in shown]
            return "```json\n" + json.dumps({"action": "edit", "path": m["file"],
                                             "old": "\n".join(win),
                                             "new": "\n".join(new)}) + "\n```"
        if self.i == 3:
            return '```json\n{"action": "run_tests"}\n```'
        return '```json\n{"action": "finish"}\n```'


class NullPolicy:
    """Finishes immediately. Expected resolve rate: 0%."""
    name = "null"

    def __call__(self, messages):
        return '```json\n{"action": "finish"}\n```'


# --------------------------------------------------------------------------
# 10. Gemma 4 wrapper (Hugging Face transformers)
# --------------------------------------------------------------------------
def find_local_model(pattern="gemma-4", roots=("/kaggle/input",)):
    """Locate a model directory mounted via 'Add Input -> Models' on Kaggle."""
    hits = []
    for r in roots:
        for cfg in Path(r).rglob("config.json") if Path(r).exists() else []:
            if pattern.lower() in str(cfg).lower():
                hits.append(cfg.parent)
    return sorted(hits)


class GemmaLLM:
    PREFILL = '```json\n{"thought": "'

    def __init__(self, model_path, name=None, max_new_tokens=512, quant4=False,
                 enable_thinking=False, temperature=0.0, prefill=True):
        import torch
        from transformers import AutoTokenizer
        self.name = name or Path(str(model_path)).name
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.enable_thinking = enable_thinking
        self.prefill = self.PREFILL if prefill else ""
        try:
            from transformers import AutoProcessor
            self.proc = AutoProcessor.from_pretrained(model_path)
            self.tok = getattr(self.proc, "tokenizer", self.proc)
        except Exception:
            self.proc = self.tok = AutoTokenizer.from_pretrained(model_path)
        kw = dict(device_map="auto")
        if quant4:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4")
        else:
            kw["dtype"] = (torch.bfloat16 if torch.cuda.is_available()
                                 and torch.cuda.is_bf16_supported() else torch.float16)
        self.model = None
        errs = []
        import transformers
        for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            try:
                cls = getattr(transformers, cls_name)
                try:
                    self.model = cls.from_pretrained(model_path,
                                                     attn_implementation="sdpa", **kw)
                except (ValueError, TypeError):
                    self.model = cls.from_pretrained(model_path, **kw)
                break
            except Exception as e:
                errs.append(f"{cls_name}: {e}")
        if self.model is None:
            raise RuntimeError("Could not load model:\n" + "\n".join(errs))
        self.model.eval()

    def __call__(self, messages):
        import torch
        try:
            prompt = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=self.enable_thinking)
        except TypeError:
            prompt = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
        # Prefill forces the reply to open as a JSON action with a short thought,
        # which removes free-form preambles that small models let run into the
        # token limit (the main cause of malformed actions in the pilots).
        enc = self.tok(prompt + self.prefill, return_tensors="pt",
                       add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        gen = dict(max_new_tokens=self.max_new_tokens,
                   do_sample=self.temperature > 0)
        if self.temperature > 0:
            gen["temperature"] = self.temperature
        with torch.no_grad():
            try:   # stop right after the closing code fence
                out = self.model.generate(**enc, **gen, stop_strings=["```"],
                                          tokenizer=self.tok)
            except (TypeError, ValueError):
                out = self.model.generate(**enc, **gen)
        new = out[0, enc["input_ids"].shape[1]:]
        text = self.prefill + self.tok.decode(new, skip_special_tokens=True)
        del out, enc
        _free_cuda()
        return text


# --------------------------------------------------------------------------
# 11. Experiment runner with resume
# --------------------------------------------------------------------------
def _result_files(results_path):
    """results.jsonl plus any per-GPU shard files next to it."""
    p = Path(results_path)
    return sorted(set(p.parent.glob(p.stem + "*.jsonl")) | ({p} if p.exists() else set()))


def load_results(results_path):
    rows = []
    for f in _result_files(results_path):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                try:
                    rows.append(json.loads(l))
                except json.JSONDecodeError:
                    pass
    seen, out = set(), []
    for r in rows:   # de-duplicate (model, condition, task)
        k = (r["model"], r["condition"], r["task_id"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def done_keys(results_path):
    return {(r["model"], r["condition"], r["task_id"]) for r in load_results(results_path)}


def _unused_done_keys(results_path):
    keys = set()
    if Path(results_path).exists():
        for l in open(results_path, encoding="utf-8"):
            try:
                r = json.loads(l)
                keys.add((r["model"], r["condition"], r["task_id"]))
            except Exception:
                pass
    return keys


def run_experiment(llm, tasks, conditions, clean_root, work_root, results_path,
                   indexes=None, time_budget_s=None, log=print, only_keys=None,
                   write_path=None, **episode_kw):
    indexes = indexes or {}
    done = done_keys(results_path)
    write_path = write_path or results_path
    t_start = time.time()
    n = 0
    # interleave conditions per task so partial runs stay balanced
    for task in tasks:
        if task["repo"] not in indexes:
            indexes[task["repo"]] = RepoIndex(Path(clean_root) / task["repo"], task["repo"])
        for cond in conditions:
            key = (llm.name, cond, task["task_id"])
            if key in done or (only_keys is not None and key not in only_keys):
                continue
            if time_budget_s and time.time() - t_start > time_budget_s:
                log("Time budget reached; stopping (resume later).")
                return
            rec = run_episode(llm, task, cond, indexes[task["repo"]],
                              clean_root, work_root, **episode_kw)
            with open(write_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            n += 1
            log(f"[{n}] {llm.name:14s} {cond:8s} {task['task_id']:22s} "
                f"resolved={rec['resolved']!s:5s} steps={rec['steps']:2d} "
                f"{rec['wall_s']:6.1f}s {rec['error'] or ''}")


# --------------------------------------------------------------------------
# 12. Statistics
# --------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarize(results_path, baseline="none"):
    """Per (model, condition). Primary outcome (pre-registered): step at which
    the agent first views the buggy file, censored at max_steps + 1, compared
    with the baseline by a paired Wilcoxon signed-rank test. Secondary: rates of
    editing the buggy file and of resolving the task (Wilson CI, exact McNemar)."""
    import pandas as pd
    df = pd.DataFrame(load_results(results_path))
    for col, default in (("first_gold_view", None), ("first_gold_edit", None),
                         ("max_steps", 15)):
        if col not in df:
            df[col] = default
    cens = df.max_steps.fillna(15).astype(int) + 1
    df["t_view"] = df.first_gold_view.fillna(cens).astype(float)
    df["t_edit"] = df.first_gold_edit.fillna(cens).astype(float)
    rows = []
    for (model, cond), g in df.groupby(["model", "condition"]):
        k, n = int(g.resolved.sum()), len(g)
        p, lo, hi = wilson(k, n)
        row = dict(model=model, condition=cond, n=n,
                   t_first_view=round(g.t_view.mean(), 2),
                   t_first_edit=round(g.t_edit.mean(), 2),
                   viewed_gold=round(g.viewed_gold_file.mean(), 3),
                   edited_gold=round(g.edited_gold_file.mean(), 3),
                   resolved=k, rate=round(p, 3), ci_low=round(lo, 3), ci_high=round(hi, 3),
                   gold_in_context=round(g.gold_in_context.mean(), 3),
                   steps=round(g.steps.mean(), 2),
                   format_err=round(g.format_errors.mean(), 2),
                   rep_tokens=round(g.rep_tokens.mean()),
                   prompt_tokens=round(g.prompt_tokens.mean()),
                   wall_s=round(g.wall_s.mean(), 1),
                   crashed=round(g.error.notna().mean(), 3))
        if cond != baseline:
            base = df[(df.model == model) & (df.condition == baseline)]
            m = g.merge(base, on="task_id", suffixes=("", "_b"))
            if len(m):
                d = m.t_view - m.t_view_b
                try:
                    from scipy.stats import wilcoxon
                    pw = 1.0 if (d == 0).all() else float(
                        wilcoxon(m.t_view, m.t_view_b, zero_method="zsplit").pvalue)
                except Exception:
                    pw = float("nan")
                b = int((m.resolved & ~m.resolved_b.astype(bool)).sum())
                c = int((~m.resolved & m.resolved_b.astype(bool)).sum())
                row.update(d_first_view=round(d.mean(), 2), p_wilcoxon=round(pw, 4),
                           wins=b, losses=c, p_mcnemar=round(mcnemar_exact(b, c), 4))
        rows.append(row)
    out = pd.DataFrame(rows)
    # Holm correction of the primary test within each model
    if "p_wilcoxon" in out:
        out["p_wilcoxon_holm"] = float("nan")
        for model, g in out[out.p_wilcoxon.notna()].groupby("model"):
            ps = g.p_wilcoxon.sort_values()
            m_ = len(ps)
            adj, run = {}, 0.0
            for i, (idx, pv) in enumerate(ps.items()):
                run = max(run, min(1.0, (m_ - i) * pv))
                adj[idx] = round(run, 4)
            for idx, v in adj.items():
                out.loc[idx, "p_wilcoxon_holm"] = v
    order = {c: i for i, c in enumerate(CONDITIONS)}
    return out.sort_values(["model", "condition"],
                           key=lambda s: s.map(order) if s.name == "condition" else s)


# --------------------------------------------------------------------------
# 13. Parallel execution: one worker process per GPU
# --------------------------------------------------------------------------
class _OracleRouter:
    """Routes each episode to an OraclePolicy (identified by its issue text)."""
    def __init__(self, tasks, name):
        self.by_issue = {t["issue"]: t for t in tasks}
        self.name, self.cur, self.pol = name, None, None

    def __call__(self, messages):
        first = messages[1]["content"]
        if first is not self.cur:
            self.cur = first
            t = next(t for i, t in self.by_issue.items() if i in first)
            self.pol = OraclePolicy(t)
        return self.pol(messages)


def _worker_main(cfg_path):
    """Entry point of a worker process (CUDA_VISIBLE_DEVICES set by parent)."""
    cfg = json.load(open(cfg_path))
    tasks = load_tasks(cfg["tasks_path"])
    if cfg["model_path"] == "__oracle__":          # harness self-test, no GPU
        llm = _OracleRouter(tasks, cfg["name"])
    else:
      llm = GemmaLLM(cfg["model_path"], name=cfg["name"],
                     max_new_tokens=cfg["max_new_tokens"], quant4=cfg["quant4"],
                     enable_thinking=cfg.get("enable_thinking", False))
    keys = {tuple(k) for k in cfg["keys"]}
    run_experiment(llm, tasks, cfg["conditions"], Path(cfg["clean"]),
                   Path(cfg["ep_dir"]), cfg["results"], only_keys=keys,
                   write_path=cfg["write_path"], time_budget_s=cfg["budget_s"],
                   max_steps=cfg["max_steps"], budget=cfg["rep_budget"])


def run_parallel(name, model_path, tasks, conditions, clean_root, work_root,
                 results_path, n_gpus=2, quant4=False, time_budget_s=None,
                 max_new_tokens=512, max_steps=20, rep_budget=1500,
                 poll_s=120, log=print):
    """Split the remaining (task, condition) episodes of one model over GPUs,
    one process per GPU. Returns True if every episode is done afterwards."""
    work_root = Path(work_root)
    done = done_keys(results_path)
    todo = [(name, c, t["task_id"]) for t in tasks for c in conditions
            if (name, c, t["task_id"]) not in done]
    if not todo:
        return True
    tasks_path = work_root / "tasks_selected.jsonl"
    save_tasks(tasks, tasks_path)
    procs = []
    for g in range(n_gpus):
        shard = todo[g::n_gpus]              # interleaved: balanced repos/conditions
        cfg = dict(tasks_path=str(tasks_path), model_path=str(model_path), name=name,
                   max_new_tokens=max_new_tokens, quant4=quant4, keys=shard,
                   conditions=conditions, clean=str(clean_root),
                   ep_dir=str(work_root / f"work_gpu{g}"), results=str(results_path),
                   write_path=str(Path(results_path).with_name(
                       f"{Path(results_path).stem}_gpu{g}.jsonl")),
                   budget_s=time_budget_s, max_steps=max_steps, rep_budget=rep_budget)
        Path(cfg["ep_dir"]).mkdir(parents=True, exist_ok=True)
        cfg_path = work_root / f"worker_gpu{g}.json"
        json.dump(cfg, open(cfg_path, "w"))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                   PYTORCH_ALLOC_CONF="expandable_segments:True")
        logf = open(work_root / f"worker_gpu{g}.log", "w")
        code = ("import sys, repofix; repofix._worker_main(sys.argv[1])")
        procs.append((subprocess.Popen([sys.executable, "-c", code, str(cfg_path)],
                                       cwd=os.getcwd(), env=env, stdout=logf,
                                       stderr=subprocess.STDOUT), logf, g))
        log(f"[{name}] GPU {g}: {len(shard)} episodes")
    t0 = time.time()
    while any(p.poll() is None for p, _, _ in procs):
        time.sleep(poll_s)
        n = len(done_keys(results_path)) - len(done)
        log(f"[{name}] {n}/{len(todo)} episodes done  ({(time.time()-t0)/3600:.1f} h)")
    for p, f, g in procs:
        f.close()
        if p.returncode != 0:
            tail = open(work_root / f"worker_gpu{g}.log").read()[-1500:]
            log(f"[{name}] worker GPU {g} exited with code {p.returncode}:\n{tail}")
    left = [k for k in todo if k not in done_keys(results_path)]
    log(f"[{name}] parallel phase finished; {len(left)} episodes remain")
    return not left
