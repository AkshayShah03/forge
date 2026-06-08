"""Unit tests for the incident-investigation tools added to the registry."""
from __future__ import annotations

import os
import tempfile

import pytest

from agent_system.tools.registry import list_log_files, query_git_log, read_log_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def _invoke(tool, **kwargs) -> str:
    """Call a LangChain @tool via ainvoke and return the string result."""
    return await tool.ainvoke(kwargs)


# ---------------------------------------------------------------------------
# read_log_file
# ---------------------------------------------------------------------------

class TestReadLogFile:
    async def test_reads_file_contents(self, tmp_path):
        """Returns all lines from a plain log file."""
        log = tmp_path / "app.log"
        log.write_text("line1\nline2\nline3\n")
        result = await _invoke(read_log_file, path=str(log), tail_lines=500, pattern="")
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    async def test_tail_lines_limits_output(self, tmp_path):
        """tail_lines keeps only the last N lines."""
        log = tmp_path / "big.log"
        log.write_text("\n".join(f"line{i}" for i in range(100)) + "\n")
        result = await _invoke(read_log_file, path=str(log), tail_lines=5, pattern="")
        lines = [l for l in result.splitlines() if l.startswith("line")]
        assert len(lines) == 5
        assert "line99" in result
        assert "line0" not in result

    async def test_pattern_filters_matching_lines(self, tmp_path):
        """Pattern filters lines by regex (case-insensitive)."""
        log = tmp_path / "mixed.log"
        log.write_text("INFO normal request\nERROR checkout failed\nINFO another\n")
        result = await _invoke(read_log_file, path=str(log), tail_lines=500, pattern="error")
        assert "ERROR checkout failed" in result
        assert "INFO normal request" not in result

    async def test_pattern_is_case_insensitive(self, tmp_path):
        """Pattern matching is case-insensitive."""
        log = tmp_path / "case.log"
        log.write_text("Error: something\nerror: other\nINFO: fine\n")
        result = await _invoke(read_log_file, path=str(log), tail_lines=500, pattern="ERROR")
        assert "Error: something" in result
        assert "error: other" in result
        assert "INFO: fine" not in result

    async def test_file_not_found_returns_error_string(self, tmp_path):
        """Non-existent path returns a descriptive error, not an exception."""
        result = await _invoke(read_log_file, path="/nonexistent/path/to/nothing.log", tail_lines=100, pattern="")
        assert "not found" in result.lower() or "error" in result.lower()

    async def test_empty_file(self, tmp_path):
        """Empty file returns gracefully without crashing."""
        log = tmp_path / "empty.log"
        log.write_text("")
        result = await _invoke(read_log_file, path=str(log), tail_lines=100, pattern="")
        assert isinstance(result, str)

    async def test_pattern_no_matches_returns_empty_match(self, tmp_path):
        """Pattern that matches nothing returns a result without the original lines."""
        log = tmp_path / "nomatch.log"
        log.write_text("hello world\nfoo bar\n")
        result = await _invoke(read_log_file, path=str(log), tail_lines=500, pattern="zzznomatch")
        assert "hello world" not in result
        assert "foo bar" not in result


# ---------------------------------------------------------------------------
# list_log_files
# ---------------------------------------------------------------------------

class TestListLogFiles:
    async def test_finds_log_files_recursively(self, tmp_path):
        """Discovers .log files nested in subdirectories."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "app.log").write_text("x")
        (tmp_path / "subdir" / "worker.log").write_text("y")
        result = await _invoke(list_log_files, path=str(tmp_path))
        assert "app.log" in result
        assert "worker.log" in result

    async def test_finds_multiple_extensions(self, tmp_path):
        """Discovers .json and .yaml files alongside .log."""
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "deploy.yaml").write_text("version: 1")
        (tmp_path / "notes.txt").write_text("note")
        result = await _invoke(list_log_files, path=str(tmp_path))
        assert "config.json" in result
        assert "deploy.yaml" in result
        assert "notes.txt" in result

    async def test_skips_hidden_and_cache_dirs(self, tmp_path):
        """Does not descend into .git, __pycache__, node_modules, .venv, venv."""
        for skip_dir in [".git", "__pycache__", "node_modules", ".venv", "venv"]:
            d = tmp_path / skip_dir
            d.mkdir()
            (d / "hidden.log").write_text("should not appear")
        (tmp_path / "real.log").write_text("should appear")
        result = await _invoke(list_log_files, path=str(tmp_path))
        assert "real.log" in result
        assert "hidden.log" not in result

    async def test_no_matching_files_returns_message(self, tmp_path):
        """Empty directory (or no log-type files) returns a descriptive message."""
        (tmp_path / "readme.md").write_text("# readme")
        result = await _invoke(list_log_files, path=str(tmp_path))
        assert "no" in result.lower() or "found 0" in result.lower() or "not found" in result.lower()

    async def test_caps_output_at_100_files(self, tmp_path):
        """Does not return more than 100 file paths."""
        for i in range(120):
            (tmp_path / f"app_{i}.log").write_text("")
        result = await _invoke(list_log_files, path=str(tmp_path))
        lines = [l for l in result.splitlines() if l.strip().endswith(".log")]
        assert len(lines) <= 100

    async def test_nonexistent_path_returns_error(self):
        """Non-existent directory returns an error string."""
        result = await _invoke(list_log_files, path="/nonexistent/xyz/abc")
        assert "error" in result.lower() or "not found" in result.lower()


# ---------------------------------------------------------------------------
# query_git_log
# ---------------------------------------------------------------------------

class TestQueryGitLog:
    async def test_returns_commits_for_valid_repo(self):
        """Returns commit history lines for the project's own repo."""
        result = await _invoke(query_git_log, repo_path=REPO_ROOT, since_hours=8760)
        # The repo has at least the initial commit
        assert "Initial commit" in result or "|" in result

    async def test_commit_format_includes_hash_and_subject(self):
        """Each commit line contains a hash, author, date, and subject separated by '|'."""
        result = await _invoke(query_git_log, repo_path=REPO_ROOT, since_hours=8760)
        # Format: %h|%ae|%ad|%s  — each line has 3 pipe chars
        commit_lines = [l for l in result.splitlines() if "|" in l]
        assert len(commit_lines) >= 1
        for line in commit_lines[:5]:
            parts = line.split("|")
            assert len(parts) >= 4, f"Expected 4 pipe-separated parts, got: {line}"

    async def test_zero_hours_returns_no_commits_message(self):
        """since_hours=0 returns no commits (nothing committed in last 0 hours)."""
        result = await _invoke(query_git_log, repo_path=REPO_ROOT, since_hours=0)
        assert "no commits" in result.lower() or result.strip() == ""

    async def test_bad_repo_path_returns_error_string(self, tmp_path):
        """Non-git directory returns an error string, not an exception."""
        result = await _invoke(query_git_log, repo_path=str(tmp_path), since_hours=24)
        assert isinstance(result, str)
        # Either "no commits" or an error — not a crash
        assert len(result) > 0
