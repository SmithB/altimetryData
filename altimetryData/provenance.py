"""
Provenance of the code that writes the database: the git repository, commit, and whether the working tree
matched the commit.

Every data, stats, and index file records code_info() in its metadata; scripts/process_granules.py and
scripts/build_database.py refuse to run from uncommitted or unpushed code (unless --allow_dirty), so the commit
recorded in a file is the code that made it and can be found in the public repository.
"""

import functools
import os
import re
import subprocess

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# paths whose state defines the code (tests and docs do not change the output)
CODE_PATHS = ["altimetryData", "scripts", "pyproject.toml"]


def _git(*args):
    return subprocess.run(["git", "-C", REPO_DIR, *args], capture_output=True, text=True, check=True).stdout.strip()


def _https(url):
    """https form of a git remote url (git@github.com:user/repo.git -> https://github.com/user/repo)"""
    m = re.match(r"git@([^:]+):(.+)", url)
    if m:
        url = f"https://{m.group(1)}/{m.group(2)}"
    return re.sub(r"\.git$", "", url)


@functools.lru_cache(maxsize=1)
def code_info():
    """{"repository", "commit", "commit_url", "branch", "dirty", "pushed"}; values are None outside a git checkout"""
    try:
        commit = _git("rev-parse", "HEAD")
        repository = _https(_git("remote", "get-url", "origin"))
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--untracked-files=normal", "--", *CODE_PATHS))
        pushed = bool(_git("branch", "-r", "--contains", commit))
    except (OSError, subprocess.CalledProcessError):
        return {"repository": None, "commit": None, "commit_url": None, "branch": None, "dirty": None,
                "pushed": None}
    return {"repository": repository, "commit": commit, "commit_url": f"{repository}/tree/{commit}",
            "branch": branch, "dirty": dirty, "pushed": pushed}


def require_committed(allow_dirty=False):
    """code_info(), or exit if the code is uncommitted or the commit is not on the remote"""
    info = code_info()
    problems = []
    if info["commit"] is None:
        problems.append(f"{REPO_DIR} is not a git checkout")
    else:
        if info["dirty"]:
            problems.append(f"uncommitted changes under {', '.join(CODE_PATHS)}")
        if not info["pushed"]:
            problems.append(f"commit {info['commit'][:12]} is not on any remote branch (push it first)")
    if problems and not allow_dirty:
        raise SystemExit("refusing to write database files: " + "; ".join(problems) +
                         ".  Commit and push, or pass --allow_dirty (the files then record dirty: true).")
    return info
