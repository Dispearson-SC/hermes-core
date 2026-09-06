"""Skill discovery and progressive disclosure, end to end.

Covers three real lifted modules:

- ``hermes_core.agent.skill_utils`` -- directory scanning, frontmatter parsing,
  platform/description helpers. Works as extracted; no fixes needed.
- ``hermes_core.agent.prompt_builder`` (``build_skills_system_prompt``) -- builds
  the compact ``## Skills`` index that goes into the system prompt. Also works as
  extracted, via ``skills_dir_override`` or via the workspace seam's
  ``get_skills_dir()``.
- ``hermes_core.tools.skills_tool`` (the ``skill_view`` / ``skills_list`` tool
  implementations). This USED to be unimportable --
  ``ModuleNotFoundError: No module named 'hermes_core.tools.skills_tool_setup'`` --
  because ``tools/lift.py``'s ``MANIFEST`` was missing three sibling modules
  (``skills_tool_setup.py``, ``skills_tool_plugin.py``, ``skills_tool_dedup.py``) plus
  ``tools/path_security.py``. That manifest gap has since been closed upstream and the
  module now imports and runs cleanly -- the sections below exercise it directly,
  including through a real agent turn.

Everywhere this file needs "the full body of a named skill" WITHOUT going through the
tool (e.g. to establish the frontmatter/body split works before layering the tool on
top of it), it uses a tiny test-local helper, ``_load_skill_body``, built on the same
primitives (``parse_frontmatter`` + ``iter_skill_index_files``) ``skill_view`` itself
calls. That is not a reimplementation of the tool -- it is proof that the pieces the
tool is built from are themselves sound. The tool's own behavior -- JSON shape, error
handling, path-traversal rejection, dedup -- is covered separately, against the real
``hermes_core.tools.skills_tool`` module.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from hermes_core.agent import prompt_builder as pb
from hermes_core.agent.skill_utils import (
    SKILL_PROMPT_DESC_LIMIT,
    extract_skill_description,
    is_skill_description_truncated_for_prompt,
    iter_skill_index_files,
    parse_frontmatter,
)
from hermes_core.seams.config import DictConfigSource, set_config_source
from hermes_core.seams.credentials import StaticCredentials, set_credential_source
from hermes_core.seams.paths import DirectoryWorkspace, get_skills_dir, set_workspace
from hermes_core.testing import Script, install_fake_client
from hermes_core.tools.registry import registry

FIXTURE_SKILLS = Path(__file__).parent / "fixtures" / "skills"

# Copied verbatim from this machine's real ~/.claude/skills/ -- Claude Code skills,
# not authored for this test. "malformed-skill" is the one skill written for this
# suite; everything else is realistic input.
REAL_SKILL_NAMES = {"branch-pr", "go-testing", "judgment-day"}
ALL_FIXTURE_SKILL_NAMES = REAL_SKILL_NAMES | {"malformed-skill"}


@pytest.fixture(autouse=True)
def isolated_core():
    """A scratch workspace (+ config/credentials, for the end-to-end test) per test.

    Skill discovery must never touch this machine's real ~/.claude or ~/.hermes_core --
    the whole point of the fixtures directory is that it does not.
    """
    set_workspace(DirectoryWorkspace(tempfile.mkdtemp()))
    set_config_source(DictConfigSource({"model": {"default": "fake-model", "provider": "openai"}}))
    set_credential_source(StaticCredentials("sk-test"))
    yield


@pytest.fixture
def tmp_skills_dir(tmp_path):
    """A throwaway copy of the fixture skills, never the checked-in fixtures path.

    build_skills_system_prompt(skills_dir_override=...) temporarily binds HERMES_HOME to
    skills_dir.parent and writes a ``.skills_prompt_snapshot.json`` cache file there on
    first build (see the report: this is a real, if minor, footgun for a host that
    points skills_dir_override at a plain directory rather than a `<home>/skills`
    layout). Copying to tmp_path keeps that side effect out of the repo.
    """
    dest = tmp_path / "skills"
    shutil.copytree(FIXTURE_SKILLS, dest)
    return dest


def _find_skill_md(name: str, skills_dir: Path):
    for skill_md in iter_skill_index_files(skills_dir, "SKILL.md"):
        frontmatter, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        if frontmatter.get("name") == name or skill_md.parent.name == name:
            return skill_md
    return None


def _load_skill_body(name: str, skills_dir: Path) -> str:
    """The full markdown body of a skill by name -- see module docstring."""
    skill_md = _find_skill_md(name, skills_dir)
    if skill_md is None:
        raise LookupError(f"Skill '{name}' not found under {skills_dir}")
    _, body = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
    return body


# -- discovery -------------------------------------------------------------------


def test_discovery_finds_skills_from_a_configured_directory():
    """Directory scanning works against an arbitrary host-chosen directory."""
    found = {p.parent.name for p in iter_skill_index_files(FIXTURE_SKILLS, "SKILL.md")}
    assert found == ALL_FIXTURE_SKILL_NAMES


def test_host_can_point_the_core_at_its_own_skills_directory(tmp_path):
    """set_workspace + get_skills_dir() is the seam a host uses to redirect skills.

    No skills_dir_override needed: build_skills_system_prompt() falls back to
    get_skills_dir(), which follows whatever workspace the host installed.
    """
    workspace_root = tmp_path / "my-app-workspace"
    shutil.copytree(FIXTURE_SKILLS, workspace_root / "skills")
    set_workspace(DirectoryWorkspace(workspace_root))

    assert get_skills_dir() == workspace_root / "skills"

    prompt = pb.build_skills_system_prompt(available_tools={"skill_view", "skills_list"})
    for name in REAL_SKILL_NAMES:
        assert name in prompt


def test_malformed_frontmatter_does_not_crash_discovery(tmp_skills_dir):
    """A broken SKILL.md must not take the whole index down with it."""
    names = {p.parent.name for p in iter_skill_index_files(tmp_skills_dir, "SKILL.md")}
    assert "malformed-skill" in names  # scan survives and still finds the file

    # Building the full index must not raise either, and the skill still gets listed
    # (with whatever fallback description the malformed YAML produced).
    prompt = pb.build_skills_system_prompt(
        available_tools={"skill_view", "skills_list"}, skills_dir_override=tmp_skills_dir
    )
    assert "malformed-skill" in prompt


# -- the index: names + short descriptions, never full bodies --------------------


def test_index_has_names_and_short_descriptions_not_full_bodies(tmp_skills_dir):
    prompt = pb.build_skills_system_prompt(
        available_tools={"skill_view", "skills_list"}, skills_dir_override=tmp_skills_dir
    )
    for name in REAL_SKILL_NAMES:
        assert name in prompt

    # Phrases that exist only in a skill's body, well past its description, must
    # never leak into the compact index -- that is the whole point of progressive
    # disclosure.
    assert "Prefer table-driven tests" not in prompt
    assert "Bubbletea/TUI" not in prompt  # body-only text from go-testing (its description says "Bubbletea teatest", no slash)


def test_skill_view_returns_the_full_body(tmp_skills_dir):
    body = _load_skill_body("go-testing", tmp_skills_dir)
    assert "Prefer table-driven tests" in body
    assert "teatest" in body

    # And that full body is genuinely absent from the compact index built from the
    # same directory -- the two tiers stay separate.
    prompt = pb.build_skills_system_prompt(
        available_tools={"skill_view", "skills_list"}, skills_dir_override=tmp_skills_dir
    )
    assert "Prefer table-driven tests" not in prompt


def test_unknown_skill_name_fails_cleanly(tmp_skills_dir):
    with pytest.raises(LookupError):
        _load_skill_body("no-such-skill", tmp_skills_dir)


# -- Claude Code skill format compatibility --------------------------------------


def test_claude_code_descriptions_commonly_exceed_the_prompt_budget(tmp_skills_dir):
    """Real finding: Claude Code's `description:` convention packs a "Trigger: ..."
    hint into the same field hermes_core caps at 60 chars for the index. All three
    real fixture skills exceed the cap, and the truncation lands mid-word -- for
    branch-pr it removes the trigger hint entirely (only description text remains).
    This is a format mismatch to flag to hosts wanting to reuse Claude Code skills,
    not a hermes_core bug: SKILL_PROMPT_DESC_LIMIT is a deliberate prompt-budget
    cap, and truncation is graceful (no crash, valid index either way).
    """
    for name in REAL_SKILL_NAMES:
        frontmatter, _ = parse_frontmatter((tmp_skills_dir / name / "SKILL.md").read_text(encoding="utf-8"))
        assert is_skill_description_truncated_for_prompt(frontmatter), name
        indexed = extract_skill_description(frontmatter)
        assert len(indexed) <= SKILL_PROMPT_DESC_LIMIT
        assert indexed.endswith("...")

    branch_pr_fm, _ = parse_frontmatter((tmp_skills_dir / "branch-pr" / "SKILL.md").read_text(encoding="utf-8"))
    # The trigger hint that made this skill discoverable upstream is gone entirely.
    assert "Trigger" not in extract_skill_description(branch_pr_fm)


# -- the real skill_view / skills_list tool --------------------------------------


@pytest.fixture
def skills_tool_ready(tmp_skills_dir):
    """Point the workspace the real ``skills_tool`` module reads (``get_hermes_home() /
    "skills"``) at a throwaway copy of the fixtures, and hand back the real module.

    ``skills_tool.py`` has no ``skills_dir_override`` of its own (unlike
    ``build_skills_system_prompt``) -- it always resolves through the workspace seam --
    so redirecting it means pointing the workspace root at ``tmp_skills_dir``'s parent.
    """
    set_workspace(DirectoryWorkspace(tmp_skills_dir.parent))
    from hermes_core.tools import skills_tool

    return skills_tool


def test_skill_view_tool_returns_the_full_body_absent_from_the_index(skills_tool_ready, tmp_skills_dir):
    """The claim this whole file is about, proven against the REAL tool this time.

    skill_view's ``content`` carries the entire SKILL.md (frontmatter and all -- the
    tool does not strip it), including body text the compact index never carries. The
    contrast is the test: the same phrase must be present in one and absent from the
    other, built from the very same fixture directory.
    """
    payload = json.loads(skills_tool_ready.skill_view("go-testing"))
    assert payload["success"] is True
    assert "Prefer table-driven tests" in payload["content"]
    assert "teatest" in payload["content"]

    prompt = pb.build_skills_system_prompt(
        available_tools={"skill_view", "skills_list"}, skills_dir_override=tmp_skills_dir
    )
    assert "go-testing" in prompt
    assert "Prefer table-driven tests" not in prompt


def test_skills_list_tool_returns_names_and_descriptions_only(skills_tool_ready):
    """Tier 1, from the real tool: no ``content`` key, no body text, anywhere."""
    payload = json.loads(skills_tool_ready.skills_list())
    assert payload["success"] is True
    names = {s["name"] for s in payload["skills"]}
    assert names == ALL_FIXTURE_SKILL_NAMES
    for skill in payload["skills"]:
        assert "content" not in skill
    assert "Prefer table-driven tests" not in json.dumps(payload)


def test_skill_view_tool_unknown_skill_name_fails_cleanly(skills_tool_ready):
    payload = json.loads(skills_tool_ready.skill_view("no-such-skill"))
    assert payload["success"] is False
    assert "not found" in payload["error"]
    assert set(payload["available_skills"]) == ALL_FIXTURE_SKILL_NAMES


def test_skill_view_tool_survives_malformed_frontmatter(skills_tool_ready):
    """Adversarial finding: the malformed fixture does not crash skill_view, and its
    full (unparseable-as-YAML) body is still served -- graceful, not a silent drop."""
    payload = json.loads(skills_tool_ready.skill_view("malformed-skill"))
    assert payload["success"] is True
    assert "This body should never reach the index" in payload["content"]


@pytest.mark.parametrize(
    "bad_name",
    [
        "../etc/passwd",
        "go-testing/../../../etc/passwd",
        "/etc/passwd",
        "C:\\Windows\\system32",
    ],
)
def test_skill_view_tool_rejects_traversal_and_absolute_names(skills_tool_ready, bad_name):
    """Adversarial finding (negative -- nothing broken): every escape shape tried --
    a bare '..' component, one buried mid-path, a POSIX absolute path, and a Windows
    drive path -- is rejected before any directory lookup happens."""
    payload = json.loads(skills_tool_ready.skill_view(bad_name))
    assert payload["success"] is False
    assert "available_skills" not in payload  # rejected pre-lookup, not a not-found


def test_skill_view_tool_rejects_traversal_in_file_path(skills_tool_ready):
    """The ``file_path`` parameter (linked-file access) gets the same treatment as
    ``name``, via the shared ``path_security`` helper."""
    payload = json.loads(
        skills_tool_ready.skill_view("go-testing", file_path="../../go-testing/SKILL.md")
    )
    assert payload["success"] is False
    assert "traversal" in payload["error"].lower() or ".." in payload["error"]


def test_skill_view_tool_confines_an_absolute_file_path_even_though_pathlib_join_would_not(
    skills_tool_ready,
):
    """Adversarial finding (negative): ``Path(skill_dir) / "C:\\Windows\\win.ini"``
    structurally discards ``skill_dir`` -- pathlib replaces the left side of ``/``
    when the right side is absolute. ``has_traversal_component`` alone would not
    catch this (no literal '..'). It is still refused: ``validate_within_dir``
    resolves the joined path and rejects anything outside the skill's own directory,
    so the pathlib quirk is not an actual escape."""
    payload = json.loads(
        skills_tool_ready.skill_view("go-testing", file_path="C:\\Windows\\win.ini")
    )
    assert payload["success"] is False
    assert "escapes allowed directory" in payload["error"]


def test_skill_view_tool_returns_a_very_large_body_in_full(skills_tool_ready, tmp_skills_dir):
    """No hidden truncation of skill content (only the compact-index DESCRIPTION is
    capped -- the full body is not)."""
    huge_dir = tmp_skills_dir / "huge-skill"
    huge_dir.mkdir()
    huge_body = "# Huge\n\n" + ("word " * 200_000)  # ~1 MB
    (huge_dir / "SKILL.md").write_text(
        f"---\nname: huge-skill\ndescription: big one\n---\n\n{huge_body}", encoding="utf-8"
    )

    payload = json.loads(skills_tool_ready.skill_view("huge-skill"))
    assert payload["success"] is True
    assert payload["content"].endswith(huge_body[-200:])
    assert len(payload["content"]) > 1_000_000


# -- end to end: the index reaches the system prompt in a real turn --------------


@pytest.fixture
def skills_tools():
    """Register minimal skills_list/skill_view schemas under toolset "skills".

    Deliberately a stub, not hermes_core.tools.skills_tool: only the tool *names*
    matter here -- system_prompt.py's ``_skills_prompt`` gates the whole ``## Skills``
    block on ``skill_view`` / ``skills_list`` / ``skill_manage`` being among the
    agent's valid tool names, and the index text itself comes from the real, working
    build_skills_system_prompt. This proves discovery-to-system-prompt wiring in
    isolation from tool dispatch, which the next test covers with the real tool.
    """
    def handler(args, **_kwargs):
        return "{}"

    for name in ("skills_list", "skill_view"):
        registry.register(
            name=name, toolset="skills",
            schema={"name": name, "description": "test stub", "parameters": {"type": "object", "properties": {}}},
            handler=handler, override=True,
        )
    yield
    for name in ("skills_list", "skill_view"):
        registry.deregister(name)


def test_skills_index_reaches_the_system_prompt_in_a_real_turn(skills_tools):
    from hermes_core.run_agent import AIAgent

    shutil.copytree(FIXTURE_SKILLS, get_skills_dir())

    agent = AIAgent(
        api_key="sk-test", base_url="https://example.invalid/v1", provider="openai",
        model="fake-model", enabled_toolsets=["skills"], quiet_mode=True, max_iterations=5,
    )
    client = install_fake_client(agent, Script().text("ok"))

    agent.run_conversation("hola")

    system = client.last_request.system_prompt
    assert "## Skills" in system
    for name in REAL_SKILL_NAMES:
        assert name in system
    # Progressive disclosure, proven at the point that matters: the model's actual
    # system prompt carries the index, not any skill's full body.
    assert "Prefer table-driven tests" not in system


@pytest.fixture
def real_skills_tools():
    """Register skill_view/skills_list with their ACTUAL handlers from the real,
    now-importable ``hermes_core.tools.skills_tool`` -- unlike ``skills_tools`` above,
    ``skill_view`` here really reads a SKILL.md and returns its full content."""
    from hermes_core.tools import skills_tool as st

    registered = {
        "skills_list": (
            st.SKILLS_LIST_SCHEMA,
            lambda args, **kw: st.skills_list(category=args.get("category"), task_id=kw.get("task_id")),
        ),
        "skill_view": (st.SKILL_VIEW_SCHEMA, st._skill_view_with_bump),
    }
    for name, (schema, handler) in registered.items():
        registry.register(name=name, toolset="skills", schema=schema, handler=handler, override=True)
    yield
    for name in registered:
        registry.deregister(name)


def test_skill_view_reaches_the_model_as_a_full_body_tool_result_in_a_real_turn(real_skills_tools):
    """The feature working end to end, not just its parts: the model calls
    skill_view, the REAL tool runs, and the full body -- absent from the same
    request's system prompt -- comes back as the tool-role message content.
    """
    from hermes_core.run_agent import AIAgent

    shutil.copytree(FIXTURE_SKILLS, get_skills_dir())

    agent = AIAgent(
        api_key="sk-test", base_url="https://example.invalid/v1", provider="openai",
        model="fake-model", enabled_toolsets=["skills"], quiet_mode=True, max_iterations=5,
    )
    client = install_fake_client(
        agent,
        Script().calls(("skill_view", {"name": "go-testing"})).text("Listo."),
    )

    result = agent.run_conversation("mostrame el detalle del skill go-testing")

    assert result["completed"] is True

    tool_msgs = [m for m in client.last_request.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["name"] == "skill_view"
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["success"] is True
    assert "Prefer table-driven tests" in payload["content"]

    # Same request, both tiers present, never mixed: the compact index in the
    # system prompt, the full body only in the tool result the model asked for.
    system = client.last_request.system_prompt
    assert "## Skills" in system
    assert "Prefer table-driven tests" not in system
