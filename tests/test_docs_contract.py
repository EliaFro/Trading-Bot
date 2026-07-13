"""Docs stay true: links resolve, .env.example matches what the code reads,
cited make targets exist, README's reproduce commands are real files.

Added in the final QA pass as regression tests for the doc-drift findings
(.env.example listing unread vars / missing read vars; OPERATING.md citing
the retired Docker deployment).
"""

import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Vars that come from the OS/harness, not from .env
_ENV_WHITELIST = {
    'PATH', 'HOME', 'PYTHONPATH', 'TZ', 'USER', 'SHELL', 'LANG',
    'VIRTUAL_ENV', 'PYTEST_CURRENT_TEST',
}


def _md_files():
    return [f for f in glob.glob(os.path.join(ROOT, '**/*.md'), recursive=True)
            if '_quarantine' not in f and '.venv' not in f]


def test_every_relative_markdown_link_resolves():
    bad = []
    for md in _md_files():
        base = os.path.dirname(md)
        text = open(md, encoding='utf-8').read()
        for m in re.finditer(r'\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)', text):
            target = m.group(1)
            if target.startswith(('http://', 'https://', 'mailto:')):
                continue
            path = os.path.normpath(os.path.join(base, target))
            if not os.path.exists(path):
                bad.append(f'{os.path.relpath(md, ROOT)}: {target}')
    assert not bad, 'dangling markdown links:\n' + '\n'.join(bad)


def _code_files():
    files = []
    for sub in ('src', 'scripts'):
        files += glob.glob(os.path.join(ROOT, sub, '**/*.py'), recursive=True)
    return files


def test_env_example_covers_every_var_the_code_reads():
    direct = re.compile(
        r"os\.(?:getenv|environ\.get)\(\s*['\"]([A-Z][A-Z0-9_]+)['\"]"
        r"|os\.environ\[\s*['\"]([A-Z][A-Z0-9_]+)['\"]")
    # config.py folds env via mapping tuples: ('MAX_POSITION_SIZE', 'max_position_size')
    mapped = re.compile(r"\(\s*['\"]([A-Z][A-Z0-9_]+)['\"]\s*,\s*['\"][a-z_]+['\"]\s*\)")
    read_vars = set()
    for py in _code_files():
        text = open(py, encoding='utf-8').read()
        for m in direct.finditer(text):
            read_vars.add(m.group(1) or m.group(2))
        if 'os.getenv(env_var)' in text or 'os.getenv(name)' in text:
            read_vars.update(m.group(1) for m in mapped.finditer(text))
    read_vars -= _ENV_WHITELIST

    example = open(os.path.join(ROOT, '.env.example'), encoding='utf-8').read()
    documented = set(re.findall(r'^([A-Z][A-Z0-9_]+)=', example, re.M))

    missing = sorted(read_vars - documented)
    assert not missing, f'.env.example is missing vars the code reads: {missing}'

    unread = sorted(documented - read_vars)
    assert not unread, f'.env.example documents vars nothing reads: {unread}'


def test_make_targets_cited_in_docs_exist():
    makefile = open(os.path.join(ROOT, 'Makefile'), encoding='utf-8').read()
    targets = set(re.findall(r'^([a-z][a-z0-9-]*):', makefile, re.M))
    bad = []
    for md in _md_files():
        text = open(md, encoding='utf-8').read()
        # only commands, not prose: fenced blocks + inline code spans
        code = '\n'.join(re.findall(r'```(?:bash|sh|makefile)?\n(.*?)```', text, re.S)
                         + re.findall(r'`([^`\n]+)`', text))
        for m in re.finditer(r'(?:^|[;&|(\s])make\s+([a-z][a-z0-9-]*)', code, re.M):
            if m.group(1) not in targets:
                bad.append(f'{os.path.relpath(md, ROOT)}: make {m.group(1)}')
    assert not bad, 'docs cite nonexistent make targets:\n' + '\n'.join(bad)


def test_readme_reproduce_scripts_exist():
    readme = open(os.path.join(ROOT, 'README.md'), encoding='utf-8').read()
    cited = re.findall(r'python (scripts/\w+\.py)', readme)
    assert cited, 'README reproduce section lost its commands'
    missing = [s for s in cited if not os.path.exists(os.path.join(ROOT, s))]
    assert not missing, f'README cites missing scripts: {missing}'


def test_operating_doc_does_not_route_users_through_docker():
    """The supported deployment is launchd; OPERATING.md's runnable command
    blocks must not send an operator through docker (QA finding: stale §4)."""
    text = open(os.path.join(ROOT, 'OPERATING.md'), encoding='utf-8').read()
    for block in re.findall(r'```bash\n(.*?)```', text, re.S):
        assert 'docker' not in block, f'docker command in OPERATING.md: {block!r}'
