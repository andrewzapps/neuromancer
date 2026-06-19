#!/usr/bin/env python3

import ast
import json
import os
import re
from pathlib import Path
import fnmatch

from tqdm import tqdm

from ipynb_filter import convert_ipynb

ignore_directory_patterns = {
    "*/assistant",
    "*/build",
    "*/docs",
    "*/figs",
    "*/.git",
    "*/.github",
    "*/.venv",
    "*/.pytest_cache",
    "*/__pycache__",
    "*/scratch",
    "*/data",
    "*/tests",
    "*.egg-info",
}
ignore_file_patterns = {
    "*.pkl",
    "*.pyc",
    "*.jpg",
    "*.png",
    "*.gif",
    "*.yml",
    "*.toml",
    "*.env",
    "*.DS_Store",
    "*.env.leave",
    "*.gitignore",
    "__init__.py",
}

# autodoc detection
AUTODOC_MIN_DIRECTIVES = 1
AUTODOC_MAX_PROSE_LINES = 8

EXAMPLE_SPLIT_LINE_THRESHOLD = 600

AUTODOC_DIRECTIVE_RE = re.compile(
    r"^\s*\.\.\s+auto(function|class|module|method|data|attribute)::", re.I
)
SPHINX_OPTION_RE = re.compile(r"^\s*:\w+")
RST_UNDERLINE_CHARS = set('=-~^"')


def should_skip_directory(d):
    for p in ignore_directory_patterns:
        if fnmatch.fnmatch(d, p):
            return True
    return False


def should_skip_file(fp):
    for p in ignore_file_patterns:
        if fnmatch.fnmatch(fp, p):
            return True
    return False


def walk_directory(path, callback, skip_dirs=None):

    if skip_dirs is None:
        skip_dirs = list()

    # walk directory twice so we can monitor progress
    total_dirs = 0
    for root, dirs, files in tqdm(os.walk(path, topdown=True)):
        if should_skip_directory(root) or (root in skip_dirs):
            dirs.clear()
        total_dirs += 1

    with tqdm(total=total_dirs, desc="Processing directory", unit="directory") as pbar:

        for root, dirs, files in tqdm(os.walk(path, topdown=True)):
            pbar.update(1)

            if should_skip_directory(root) or (root in skip_dirs):
                dirs.clear()

            else:

                for file_name in files:

                    if not should_skip_file(file_name):

                        callback(os.path.join(root, file_name))


def normalize_rel_path(file_path, root_path):
    return Path(file_path).relative_to(root_path).as_posix()


def make_record(rel_path, source_type, symbol_name, start_line, end_line, content):
    return {
        "id": f"{rel_path}:{symbol_name or ''}:{start_line}",
        "source_type": source_type,
        "file_path": rel_path,
        "symbol_name": symbol_name,
        "start_line": start_line,
        "end_line": end_line,
        "content": content,
    }


def read_file_text(file_path):
    if file_path.endswith(".ipynb"):
        return convert_ipynb(file_path)
    with open(file_path) as f:
        try:
            return f.read()
        except Exception as e:
            print(file_path)
            raise e


def write_jsonl(records, outfile):
    with open(outfile, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_chunks(root_path, process_file, skip_dirs=None):
    records = []
    root_path = Path(root_path).resolve()

    def callback(file_path):
        records.extend(process_file(file_path, root_path))

    walk_directory(str(root_path), callback, skip_dirs)
    return records


def is_rst_underline(line):
    s = line.strip()
    return len(s) >= 2 and all(c in RST_UNDERLINE_CHARS for c in s)


def is_autodoc_shell(text):
    """Returns True when a doc file is Sphinx autodoc directives."""
    directive_lines = 0
    prose_lines = 0

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if AUTODOC_DIRECTIVE_RE.match(line):
            directive_lines += 1
            i += 1
            continue

        if SPHINX_OPTION_RE.match(line):
            i += 1
            continue

        if stripped.startswith(".. "):
            i += 1
            continue

        if is_rst_underline(line):
            i += 1
            continue

        if i + 1 < len(lines) and stripped and is_rst_underline(lines[i + 1]):
            i += 2
            continue

        prose_lines += 1
        i += 1

    return (
        directive_lines >= AUTODOC_MIN_DIRECTIVES
        and prose_lines <= AUTODOC_MAX_PROSE_LINES
    )


def find_rst_headers(lines):
    #find header to chunk by sections
    headers = []
    i = 0
    while i < len(lines) - 1:
        title = lines[i].strip()
        if title and is_rst_underline(lines[i + 1]):
            headers.append((i, title))
            i += 2
        else:
            i += 1
    return headers


def find_md_headers(lines):
    headers = []
    for i, line in enumerate(lines):
        match = re.match(r"^#{1,6}\s+(.+)$", line)
        if match:
            headers.append((i, match.group(1).strip()))
    return headers


def split_by_headers(text, file_path, headers):
    """Chunks by headings"""
    lines = text.splitlines()
    if not lines:
        return []

    if not headers:
        return [
            {
                "symbol_name": Path(file_path).stem,
                "start_line": 1,
                "end_line": max(1, len(lines)),
                "content": text,
            }
        ]

    sections = []
    #chunk anything before first heading
    if headers[0][0] > 0:
        sections.append(
            {
                "symbol_name": Path(file_path).stem,
                "start_line": 1,
                "end_line": headers[0][0],
                "content": "\n".join(lines[: headers[0][0]]),
            }
        )

    for idx, (start, title) in enumerate(headers):
        end = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
        sections.append(
            {
                "symbol_name": title,
                "start_line": start + 1,
                "end_line": end,
                "content": "\n".join(lines[start:end]),
            }
        )

    return sections


def split_doc_sections(text, file_path):
    """Split documentation files into section chunks at RST or Markdown headings."""
    lines = text.splitlines()
    suffix = Path(file_path).suffix.lower()

    if suffix == ".rst":
        headers = find_rst_headers(lines)
    elif suffix in (".md", ".txt"):
        headers = find_md_headers(lines)
    else:
        headers = find_md_headers(lines)
        if not headers:
            headers = find_rst_headers(lines)

    return split_by_headers(text, file_path, headers)


def chunk_doc_file(file_path, root_path):
    """Split hand-written documentation into section chunks"""
    suffix = Path(file_path).suffix.lower()
    if suffix not in (".rst", ".md", ".txt"):
        return []

    rel_path = normalize_rel_path(file_path, root_path)
    text = read_file_text(file_path)

    if is_autodoc_shell(text):
        return []

    records = []
    for section in split_doc_sections(text, file_path):
        if not section["content"].strip():
            continue
        records.append(
            make_record(
                rel_path,
                "doc",
                section["symbol_name"],
                section["start_line"],
                section["end_line"],
                section["content"],
            )
        )
    return records


def is_main_guard(node):
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    if not test.comparators:
        return False
    comp = test.comparators[0]
    if isinstance(comp, ast.Constant):
        return comp.value == "__main__"
    return False


def node_end_line(node, source_lines, siblings, idx):
    if getattr(node, "end_lineno", None):
        return node.end_lineno
    if idx + 1 < len(siblings) and hasattr(siblings[idx + 1], "lineno"):
        return siblings[idx + 1].lineno - 1
    return len(source_lines)


def split_large_python_example(source_lines, rel_path):
    """Split large .py examples on top-level functions"""
    source = "\n".join(source_lines)
    tree = ast.parse(source)
    records = []

    for idx, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbol_name = node.name
        elif is_main_guard(node):
            symbol_name = "__main__"
        else:
            continue

        start = node.lineno
        end = node_end_line(node, source_lines, tree.body, idx)
        content = "\n".join(source_lines[start - 1 : end])
        records.append(
            make_record(rel_path, "example", symbol_name, start, end, content)
        )

    return records


def chunk_example_file(file_path, root_path):
    """Chunk per example file and split only when very large"""
    rel_path = normalize_rel_path(file_path, root_path)
    text = read_file_text(file_path)
    source_lines = text.splitlines()
    line_count = len(source_lines) or 1

    if line_count <= EXAMPLE_SPLIT_LINE_THRESHOLD:
        return [
            make_record(rel_path, "example", None, 1, line_count, text)
        ]

    suffix = Path(file_path).suffix.lower()
    if suffix == ".py":
        return split_large_python_example(source_lines, rel_path)

    sections = split_doc_sections(text, file_path)
    records = []
    for section in sections:
        if not section["content"].strip():
            continue
        records.append(
            make_record(
                rel_path,
                "example",
                section["symbol_name"],
                section["start_line"],
                section["end_line"],
                section["content"],
            )
        )
    return records


def should_index_symbol(name, node):
    if name.startswith("_") and name != "__init__":
        return False
    if name == "__init__":
        doc = ast.get_docstring(node)
        has_params = (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and len(node.args.args) > 1
        )
        return bool(doc and doc.strip()) or has_params
    return True


def qualified_name(parent_class, name):
    if parent_class:
        return f"{parent_class}.{name}"
    return name


def extract_signature_and_doc(node, source_lines):
    start = node.lineno
    if node.decorator_list:
        start = node.decorator_list[0].lineno

    end = node.lineno
    while end <= len(source_lines):
        line = source_lines[end - 1]
        if line.rstrip().endswith(":") and not line.strip().startswith("@"):
            break
        end += 1

    parts = list(source_lines[start - 1 : end])
    doc = ast.get_docstring(node)
    if doc:
        parts.append("")
        parts.append(doc)
    return "\n".join(parts)


def iter_src_symbols(tree):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if should_index_symbol(node.name, node):
                yield None, node
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if should_index_symbol(child.name, child):
                            yield node.name, child


def chunk_src_file(file_path, root_path):
    """AST extraction of API signatures/docstrings and code implementations"""
    if not file_path.endswith(".py"):
        return []

    rel_path = normalize_rel_path(file_path, root_path)
    with open(file_path) as f:
        source = f.read()
    source_lines = source.splitlines()

    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        print(f"SyntaxError parsing {rel_path}")
        return []

    records = []
    for parent_class, node in iter_src_symbols(tree):
        symbol_name = qualified_name(parent_class, node.name)
        start = node.lineno
        end = node.end_lineno or start
        impl_content = "\n".join(source_lines[start - 1 : end])
        api_content = extract_signature_and_doc(node, source_lines)

        records.append(
            make_record(rel_path, "api", symbol_name, start, end, api_content)
        )
        records.append(
            make_record(rel_path, "impl", symbol_name, start, end, impl_content)
        )

    return records


def run(root_path: str):

    outdir = Path("knowledge")
    outdir.mkdir(exist_ok=True)

    repo_root = Path(root_path).expanduser().resolve()
    skip_dirs = [str(repo_root / x) for x in ["src", "examples"]]

    print("reading documentation")
    docs = collect_chunks(repo_root, chunk_doc_file, skip_dirs=skip_dirs)
    docs_path = outdir / "docs.jsonl"
    write_jsonl(docs, docs_path)
    print(f"wrote {docs_path} ({len(docs)} chunks)")

    print("reading src files")
    src = collect_chunks(repo_root / "src", chunk_src_file)
    src_path = outdir / "src.jsonl"
    write_jsonl(src, src_path)
    print(f"wrote {src_path} ({len(src)} chunks)")

    print("reading examples and converting .ipynb -> markdown")
    examples = collect_chunks(repo_root / "examples", chunk_example_file)
    examples_path = outdir / "examples.jsonl"
    write_jsonl(examples, examples_path)
    print(f"wrote {examples_path} ({len(examples)} chunks)")


if __name__ == "__main__":

    # Get the parent directory of the current file
    neuromancer_root_directory = Path(__file__).resolve().parent.parent
    run(str(neuromancer_root_directory))
