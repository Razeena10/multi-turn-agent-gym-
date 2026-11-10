"""Wave 7: Tests for demo, report generation, and README.

Verifies:
    - generate_reports.py: build_report(), write_json_report(),
      validate_json_report(), write_markdown_report()
    - JSON structure, serialisability, and no hidden state
    - Markdown structure and actual-data values
    - Report values match evaluator output
    - demo/run_demo.py imports and non-interactive entry point
    - README.md structure and required content
"""

from __future__ import annotations

import json
import os

import pytest

from src.config import DEFAULT_SEED, DEFAULT_NUM_EPISODES
from src.support_scenarios import NUM_SCENARIOS, SCENARIOS


# ---------------------------------------------------------------------------
# Helpers — locate the reports directory
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(__file__))
_REPORTS_DIR = os.path.join(_ROOT, "reports")


def _reports_path(filename: str) -> str:
    return os.path.join(_REPORTS_DIR, filename)


# ---------------------------------------------------------------------------
# 1. build_report() — structure and correctness
# ---------------------------------------------------------------------------


class TestBuildReport:
    """Verify build_report() produces a correct, complete dict."""

    @pytest.fixture(scope="class")
    @classmethod
    def report(cls):
        from demo.generate_reports import build_report
        return build_report(num_episodes=5, seed=42)

    def test_returns_dict(self, report):
        assert isinstance(report, dict)

    def test_has_all_top_level_keys(self, report):
        required = {
            "generated_at", "environment", "agent",
            "config", "metrics", "scenario_breakdown", "episodes",
        }
        assert required <= set(report.keys())

    def test_environment_name(self, report):
        assert report["environment"] == "CustomerSupportEnv"

    def test_agent_name(self, report):
        assert report["agent"] == "BaselineSupportAgent"

    def test_config_keys(self, report):
        cfg = report["config"]
        assert cfg["num_episodes"] == 5
        assert cfg["seed"] == 42
        assert cfg["max_turns"] > 0
        assert cfg["num_scenarios"] == NUM_SCENARIOS

    def test_episodes_count_matches_config(self, report):
        assert len(report["episodes"]) == report["config"]["num_episodes"]

    def test_metrics_has_required_keys(self, report):
        required = {
            "total_episodes", "successful_resolutions", "failed_resolutions",
            "success_rate", "failure_rate", "mean_reward", "min_reward",
            "max_reward", "std_reward", "mean_turns",
            "terminated_episodes", "truncated_episodes",
        }
        assert required <= set(report["metrics"].keys())

    def test_metrics_total_matches_episodes(self, report):
        assert report["metrics"]["total_episodes"] == 5

    def test_scenario_breakdown_has_all_scenarios(self, report):
        keys = set(report["scenario_breakdown"].keys())
        expected = {str(i) for i in range(NUM_SCENARIOS)}
        assert keys == expected

    def test_scenario_breakdown_values_have_required_fields(self, report):
        required = {"name", "count", "successes", "failures",
                    "success_rate", "mean_reward", "mean_turns"}
        for sb in report["scenario_breakdown"].values():
            assert required <= set(sb.keys())

    def test_no_hidden_state_in_episodes(self, report):
        hidden = {
            "hidden_issue", "diagnosed", "solution_proposed",
            "solution_rejected", "last_action", "conversation_state",
        }
        for ep in report["episodes"]:
            for key in hidden:
                assert key not in ep, f"Hidden key '{key}' found in episode record"

    def test_generated_at_is_iso_string(self, report):
        ts = report["generated_at"]
        assert isinstance(ts, str)
        assert "T" in ts  # ISO 8601 contains 'T' separator

    def test_metrics_success_rate_in_range(self, report):
        m = report["metrics"]
        assert 0.0 <= m["success_rate"] <= 1.0

    def test_metrics_success_plus_failure_equals_total(self, report):
        m = report["metrics"]
        assert m["successful_resolutions"] + m["failed_resolutions"] == m["total_episodes"]

    def test_episode_records_have_required_keys(self, report):
        required = {
            "episode_id", "scenario_id", "total_reward",
            "number_of_turns", "success", "terminated",
            "truncated", "termination_reason",
        }
        for ep in report["episodes"]:
            assert required <= set(ep.keys())


# ---------------------------------------------------------------------------
# 2. write_json_report() and validate_json_report()
# ---------------------------------------------------------------------------


class TestJsonReport:
    """Verify JSON writing, parsing, and validation."""

    def test_write_and_read_roundtrip(self, tmp_path):
        from demo.generate_reports import build_report, write_json_report

        path = str(tmp_path / "test_report.json")
        report = build_report(num_episodes=3, seed=42)
        write_json_report(report, path)

        assert os.path.exists(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["config"]["num_episodes"] == 3

    def test_json_is_valid(self, tmp_path):
        from demo.generate_reports import build_report, write_json_report

        path = str(tmp_path / "valid.json")
        report = build_report(num_episodes=3, seed=42)
        write_json_report(report, path)

        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_validate_passes_on_correct_report(self, tmp_path):
        from demo.generate_reports import (
            build_report, write_json_report, validate_json_report
        )
        path = str(tmp_path / "to_validate.json")
        report = build_report(num_episodes=3, seed=42)
        write_json_report(report, path)
        result = validate_json_report(path)
        assert result["config"]["num_episodes"] == 3

    def test_validate_fails_on_missing_key(self, tmp_path):
        from demo.generate_reports import validate_json_report

        path = str(tmp_path / "bad.json")
        with open(path, "w") as fh:
            json.dump({"generated_at": "now", "environment": "Env"}, fh)

        with pytest.raises(AssertionError, match="missing keys"):
            validate_json_report(path)

    def test_creates_directory_if_absent(self, tmp_path):
        from demo.generate_reports import build_report, write_json_report

        path = str(tmp_path / "subdir" / "report.json")
        report = build_report(num_episodes=2, seed=42)
        write_json_report(report, path)
        assert os.path.exists(path)

    def test_no_numpy_types_in_json(self, tmp_path):
        """All values must be JSON-native (no numpy scalars)."""
        from demo.generate_reports import build_report, write_json_report

        path = str(tmp_path / "types.json")
        report = build_report(num_episodes=3, seed=42)
        # write_json_report must not raise TypeError for numpy types
        write_json_report(report, path)
        with open(path) as fh:
            data = json.load(fh)
        # mean_reward should be a plain float
        assert isinstance(data["metrics"]["mean_reward"], float)
        assert isinstance(data["metrics"]["total_episodes"], int)


# ---------------------------------------------------------------------------
# 3. write_markdown_report()
# ---------------------------------------------------------------------------


class TestMarkdownReport:
    """Verify the Markdown report structure and value accuracy."""

    @pytest.fixture(scope="class")
    @classmethod
    def md_content(cls, tmp_path_factory):
        from demo.generate_reports import build_report, write_markdown_report

        tmp = tmp_path_factory.mktemp("md")
        path = str(tmp / "summary.md")
        report = build_report(num_episodes=10, seed=42)
        write_markdown_report(report, path)
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def test_has_title(self, md_content):
        assert "# Evaluation Summary" in md_content

    def test_has_configuration_section(self, md_content):
        assert "## Configuration" in md_content

    def test_has_aggregate_metrics_section(self, md_content):
        assert "## Aggregate Metrics" in md_content

    def test_has_per_scenario_section(self, md_content):
        assert "## Per-Scenario Results" in md_content

    def test_contains_success_rate(self, md_content):
        # Success rate should appear in the document
        assert "success_rate" in md_content.lower() or "Success rate" in md_content

    def test_contains_all_scenario_names(self, md_content):
        for s in SCENARIOS:
            assert s.name in md_content, f"Scenario '{s.name}' missing from Markdown"

    def test_generated_at_present(self, md_content):
        assert "Generated:" in md_content

    def test_values_from_real_data(self, tmp_path):
        """Values in Markdown match values from build_report() for same seed."""
        from demo.generate_reports import build_report, write_markdown_report

        path = str(tmp_path / "check.md")
        report = build_report(num_episodes=5, seed=42)
        write_markdown_report(report, path)
        with open(path) as fh:
            content = fh.read()

        # The total episodes count must appear in the document
        assert str(report["metrics"]["total_episodes"]) in content

    def test_creates_directory_if_absent(self, tmp_path):
        from demo.generate_reports import build_report, write_markdown_report

        path = str(tmp_path / "newdir" / "report.md")
        report = build_report(num_episodes=2, seed=42)
        write_markdown_report(report, path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# 4. Report values match evaluator output
# ---------------------------------------------------------------------------


class TestReportValuesMatchEvaluator:
    """Cross-verify report values against direct evaluator output."""

    def test_mean_reward_matches_evaluator(self, tmp_path):
        from demo.generate_reports import build_report
        from src.evaluator import evaluate_agent
        from src.metrics import compute_metrics
        from src.support_env import CustomerSupportEnv
        from src.baseline_agent import BaselineSupportAgent

        # Generate report
        report = build_report(num_episodes=10, seed=42)

        # Run evaluator independently
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        records = evaluate_agent(env, agent, num_episodes=10, seed=42)
        metrics = compute_metrics(records)

        assert report["metrics"]["mean_reward"] == pytest.approx(
            metrics["mean_reward"], abs=1e-6
        )

    def test_total_episodes_correct(self):
        from demo.generate_reports import build_report

        report = build_report(num_episodes=7, seed=42)
        assert report["metrics"]["total_episodes"] == 7
        assert len(report["episodes"]) == 7

    def test_scenario_breakdown_counts_sum_to_total(self):
        from demo.generate_reports import build_report

        report = build_report(num_episodes=20, seed=42)
        total = sum(v["count"] for v in report["scenario_breakdown"].values())
        assert total == 20


# ---------------------------------------------------------------------------
# 5. demo/run_demo.py — import and non-interactive checks
# ---------------------------------------------------------------------------


class TestDemoModule:
    """Verify demo/run_demo.py is importable and non-interactive paths work."""

    def test_demo_importable(self):
        """demo.run_demo imports without error."""
        import demo.run_demo  # noqa: F401

    def test_demo_has_main(self):
        """demo.run_demo exposes a main() callable."""
        from demo import run_demo
        assert callable(run_demo.main)

    def test_run_episode_runs_silently(self):
        """_run_episode() with verbose=False returns a valid record."""
        from demo.run_demo import _run_episode
        from src.support_env import CustomerSupportEnv
        from src.baseline_agent import BaselineSupportAgent

        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        record = _run_episode(env, agent, seed=42, verbose=False)

        assert isinstance(record["total_reward"], float)
        assert isinstance(record["turns"], int)
        assert record["turns"] >= 1
        assert isinstance(record["success"], bool)
        assert isinstance(record["termination_reason"], str)

    def test_run_episode_no_hidden_state_in_output(self):
        """_run_episode() does not expose hidden state in its return value."""
        from demo.run_demo import _run_episode
        from src.support_env import CustomerSupportEnv
        from src.baseline_agent import BaselineSupportAgent

        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        record = _run_episode(env, agent, seed=42, verbose=False)

        hidden = {"hidden_issue", "diagnosed", "solution_proposed",
                  "solution_rejected", "last_action"}
        for key in hidden:
            assert key not in record

    def test_run_episode_verbose_output(self, capsys):
        """_run_episode() with verbose=True prints output."""
        from demo.run_demo import _run_episode
        from src.support_env import CustomerSupportEnv
        from src.baseline_agent import BaselineSupportAgent

        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        _run_episode(env, agent, seed=42, verbose=True)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_run_episode_verbose_no_hidden_words(self, capsys):
        """Verbose output never contains hidden state field names."""
        from demo.run_demo import _run_episode
        from src.support_env import CustomerSupportEnv
        from src.baseline_agent import BaselineSupportAgent

        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        _run_episode(env, agent, seed=42, verbose=True)
        output = capsys.readouterr().out
        assert "hidden_issue" not in output
        assert "expected_resolution" not in output
        assert "solution_proposed" not in output
        assert "diagnosed" not in output

    def test_demo_episode_terminates(self):
        """An episode run via the demo terminates or truncates."""
        from demo.run_demo import _run_episode
        from src.support_env import CustomerSupportEnv
        from src.baseline_agent import BaselineSupportAgent

        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        record = _run_episode(env, agent, seed=42, verbose=False)
        assert record["termination_reason"] in {"resolved", "escalated", "truncated"}


# ---------------------------------------------------------------------------
# 6. README.md — structure and content
# ---------------------------------------------------------------------------


class TestReadme:
    """Verify README.md contains all required sections and accurate content."""

    @pytest.fixture(scope="class")
    @classmethod
    def readme(cls):
        readme_path = os.path.join(_ROOT, "README.md")
        with open(readme_path, encoding="utf-8") as fh:
            return fh.read()

    def test_readme_exists(self):
        assert os.path.exists(os.path.join(_ROOT, "README.md"))

    def test_has_project_title(self, readme):
        assert "multi-turn-agent-gym" in readme

    def test_has_technology_stack(self, readme):
        assert "Gymnasium" in readme
        assert "NumPy" in readme or "numpy" in readme.lower()
        assert "pytest" in readme.lower()

    def test_has_observation_space_section(self, readme):
        assert "Observation" in readme

    def test_has_action_space_section(self, readme):
        assert "Action" in readme

    def test_has_reward_structure(self, readme):
        assert "Reward" in readme

    def test_has_termination_section(self, readme):
        assert "termination" in readme.lower() or "Termination" in readme

    def test_has_scenario_descriptions(self, readme):
        for s in SCENARIOS:
            assert s.name in readme, f"Scenario '{s.name}' missing from README"

    def test_has_installation_instructions(self, readme):
        assert "pip install" in readme

    def test_has_run_commands(self, readme):
        assert "pytest" in readme
        assert "run_demo" in readme

    def test_no_fake_performance_claims(self, readme):
        """README must not claim SageMaker was tested."""
        # The README mentions SageMaker as a future improvement, not a completed test
        if "SageMaker" in readme:
            assert "tested" not in readme.split("SageMaker")[0][-50:], (
                "README should not claim SageMaker was tested"
            )

    def test_scope_note_present(self, readme):
        """README must clarify this is not an LLM trained with RL."""
        lower = readme.lower()
        assert "rule-based" in lower or "rule based" in lower

    def test_has_limitations_section(self, readme):
        assert "Limitations" in readme or "limitations" in readme.lower()

    def test_has_evaluation_results(self, readme):
        """README must include actual 100-episode evaluation results."""
        assert "100" in readme
        assert "10.5" in readme or "+10.5" in readme


# ---------------------------------------------------------------------------
# 7. Integration — generate reports command produces valid files
# ---------------------------------------------------------------------------


class TestReportGenerationIntegration:
    """Run the full report generation pipeline and verify outputs."""

    def test_full_pipeline_creates_both_files(self, tmp_path):
        """build_report → write_json → validate + write_md all succeed."""
        from demo.generate_reports import (
            build_report, write_json_report,
            validate_json_report, write_markdown_report,
        )

        json_path = str(tmp_path / "eval.json")
        md_path = str(tmp_path / "summary.md")

        report = build_report(num_episodes=5, seed=42)
        write_json_report(report, json_path)
        validate_json_report(json_path)
        write_markdown_report(report, md_path)

        assert os.path.exists(json_path)
        assert os.path.exists(md_path)
        assert os.path.getsize(json_path) > 100
        assert os.path.getsize(md_path) > 100

    def test_episode_count_configurable(self, tmp_path):
        """build_report respects custom num_episodes."""
        from demo.generate_reports import build_report

        r3 = build_report(num_episodes=3, seed=42)
        r8 = build_report(num_episodes=8, seed=42)
        assert len(r3["episodes"]) == 3
        assert len(r8["episodes"]) == 8

    def test_seed_configurable(self, tmp_path):
        """Different seeds produce different scenario sequences."""
        from demo.generate_reports import build_report

        r1 = build_report(num_episodes=20, seed=42)
        r2 = build_report(num_episodes=20, seed=99)
        s1 = [ep["scenario_id"] for ep in r1["episodes"]]
        s2 = [ep["scenario_id"] for ep in r2["episodes"]]
        assert s1 != s2

    def test_existing_reports_are_valid(self):
        """If reports already exist, they pass JSON validation."""
        from demo.generate_reports import validate_json_report

        json_path = _reports_path("evaluation_report.json")
        if os.path.exists(json_path):
            result = validate_json_report(json_path)
            assert "metrics" in result
            assert result["metrics"]["total_episodes"] > 0
