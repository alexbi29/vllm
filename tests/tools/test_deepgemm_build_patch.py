# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise native patch application in checkout, archive and Docker contexts."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "tools/apply_deepgemm_patch.sh"
PATCH = ROOT / "cmake/patches/deepgemm-sm120-fp8-fp4-m32.patch"


def native_preimages(directory: Path) -> dict[str, bytes]:
    """Minimal preimages of the real patch; no native build/download required."""
    files: dict[str, list[str]] = {}
    path = None
    cursor = 0
    for line in PATCH.read_text().splitlines():
        if line.startswith("diff --git"):
            path = None
        elif line.startswith("+++ b/"):
            path = line[6:]
            files[path] = []
        elif line.startswith("@@"):
            match = re.match(r"@@ -(\d+)", line)
            assert match is not None
            cursor = int(match[1]) - 1
        elif path and line and line[0] in " -":
            content = files[path]
            content.extend(["\n"] * (cursor - len(content)))
            content.append(line[1:] + "\n")
            cursor += 1
    originals = {name: "".join(lines).encode() for name, lines in files.items()}
    for name, data in originals.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return originals


@pytest.mark.parametrize("layout", ["archive", "nested_archive", "checkout"])
def test_patch_changes_all_files_and_refuses_partial_state(tmp_path, layout):
    outer = tmp_path / "outer"
    outer.mkdir()
    source = outer / "native"
    source.mkdir()
    if layout == "nested_archive":
        subprocess.run(["git", "init", "-q", str(outer)], check=True)
    elif layout == "checkout":
        subprocess.run(["git", "init", "-q", str(source)], check=True)
    originals = native_preimages(source)
    command = ["bash", str(HELPER), str(source)]
    # Build hooks can export Git context; it must not redirect application.
    env = {**os.environ, "GIT_WORK_TREE": str(outer), "GIT_PREFIX": "wrong/"}
    if layout == "nested_archive":
        env["GIT_DIR"] = str(outer / ".git")
    subprocess.run(command, check=True, env=env)
    patched = {name: (source / name).read_bytes() for name in originals}
    assert all(patched[name] != original for name, original in originals.items())
    subprocess.run(command, check=True, env=env)
    assert patched == {name: (source / name).read_bytes() for name in originals}

    name = "csrc/apis/config.hpp"
    (source / name).write_bytes(originals[name])
    partial = {name: (source / name).read_bytes() for name in originals}
    result = subprocess.run(command, capture_output=True, env=env)
    assert result.returncode != 0
    assert partial == {name: (source / name).read_bytes() for name in originals}


def test_docker_csrc_context_can_configure_native_patch(tmp_path):
    stage = ""
    for line in (ROOT / "docker/Dockerfile").read_text().splitlines():
        if line.startswith("FROM "):
            stage = line.split()[-1]
        if stage != "csrc-build" or not line.startswith("COPY "):
            continue
        *sources, destination = shlex.split(line)[1:]
        for source in sources:
            if source != "cmake" and not source.startswith("tools/"):
                continue
            target = tmp_path / destination
            if destination.endswith("/") and not (ROOT / source).is_dir():
                target /= Path(source).name
            if (ROOT / source).is_dir():
                shutil.copytree(ROOT / source, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / source, target)
    assert (tmp_path / "tools/apply_deepgemm_patch.sh").is_file()
    assert (tmp_path / PATCH.relative_to(ROOT)).is_file()
    if shutil.which("cmake") is None:
        pytest.skip("Docker input contract checked; configure also requires CMake")
    native_preimages(tmp_path / "native")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        "project(DeepGEMMBuildContext NONE)\n"
        'set(DEEPGEMM_SRC_DIR "${CMAKE_SOURCE_DIR}/native")\n'
        "set(CMAKE_CUDA_COMPILER_VERSION 13.4)\n"
        f'set(Python_EXECUTABLE "{sys.executable}")\n'
        'set(CUDA_ARCHS "12.0f")\n'
        "function(cuda_archs_loose_intersection result)\n"
        '  set(${result} "12.0f" PARENT_SCOPE)\n'
        "endfunction()\n"
        'include("${CMAKE_SOURCE_DIR}/cmake/external_projects/deepgemm.cmake")\n'
    )
    subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "build")], check=True
    )
