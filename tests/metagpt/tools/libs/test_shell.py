#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest

from metagpt.tools.libs.shell import shell_execute


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ["command", "expect_stdout", "expect_stderr"],
    [
        (["file", f"{__file__}"], "Python script text executable, ASCII text", ""),
        (f"file {__file__}", "Python script text executable, ASCII text", ""),
    ],
)
async def test_shell(command, expect_stdout, expect_stderr):
    stdout, stderr, returncode = await shell_execute(command)
    assert returncode == 0
    assert expect_stdout in stdout
    assert stderr == expect_stderr


@pytest.mark.asyncio
async def test_shell_execute_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        await shell_execute(command="sleep 3", timeout=1)


@pytest.mark.asyncio
async def test_shell_execute_timeout_list_command():
    with pytest.raises(subprocess.TimeoutExpired):
        await shell_execute(command=["sleep", "3"], timeout=1)


@pytest.mark.asyncio
async def test_shell_execute_error_returncode_string_command():
    stdout, stderr, returncode = await shell_execute(command="exit 1")
    assert returncode != 0
    # Depending on the shell, stderr might be empty for "exit 1"
    # If a command that specifically writes to stderr is needed, this can be adjusted
    # For now, just checking the return code is the primary goal.


@pytest.mark.asyncio
async def test_shell_execute_error_returncode_list_command():
    # "false" is a standard utility that does nothing and exits with a non-zero status
    stdout, stderr, returncode = await shell_execute(command=["false"])
    assert returncode != 0
    # stderr for "false" is typically empty.


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
