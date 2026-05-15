import unittest

from agent_core import AgentLoop, Decision, GuardResult, Harness, Tool, ToolResult, ValidationResult


class ScriptedModel:
    def __init__(self, *decisions):
        self.decisions = list(decisions)

    def decide(self, context, state):
        if not self.decisions:
            raise AssertionError("model was called more times than expected")
        return self.decisions.pop(0)


class RepeatingModel:
    def __init__(self, decision):
        self.decision = decision

    def decide(self, context, state):
        return self.decision


class AgentCoreLoopTests(unittest.TestCase):
    def test_completes_after_tool_result_satisfies_validator(self):
        calls = []

        def echo(args, state):
            calls.append(args)
            return ToolResult(tool_name="echo", ok=True, output={"seen": args["text"]})

        harness = Harness(
            tools=[Tool(name="echo", description="Echo input", handler=echo)],
            validate_progress=lambda state: ValidationResult(done=True, summary="done"),
        )
        model = ScriptedModel(Decision.call_tool("echo", {"text": "hello"}))

        result = AgentLoop(model=model, harness=harness).run("echo hello")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.summary, "done")
        self.assertEqual(calls, [{"text": "hello"}])
        self.assertEqual(result.state.tool_history[0].output, {"seen": "hello"})

    def test_final_answer_stops_loop_without_tool_work(self):
        model = ScriptedModel(Decision.final_answer("ready"))

        result = AgentLoop(model=model, harness=Harness()).run("answer directly")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.summary, "ready")
        self.assertEqual(result.state.tool_history, [])

    def test_approval_pause_can_resume_approved_tool_call(self):
        calls = []

        def risky(args, state):
            calls.append(args)
            return ToolResult(tool_name="risky", ok=True, output="applied")

        harness = Harness(
            tools=[Tool(name="risky", description="Risky action", handler=risky)],
            guard_action=lambda decision, state: GuardResult(requires_approval=True, reason="review required"),
            validate_progress=lambda state: ValidationResult(
                done=bool(state.tool_history),
                summary="approved action completed",
            ),
        )
        loop = AgentLoop(
            model=ScriptedModel(Decision.call_tool("risky", {"id": 1})),
            harness=harness,
        )

        paused = loop.run("do risky thing")
        resumed = loop.resume(paused.state, approved=True)

        self.assertEqual(paused.status, "approval_required")
        self.assertEqual(paused.reason, "review required")
        self.assertEqual(resumed.status, "completed")
        self.assertEqual(resumed.summary, "approved action completed")
        self.assertEqual(calls, [{"id": 1}])

    def test_tool_error_becomes_feedback_for_next_iteration(self):
        attempts = []

        def unstable(args, state):
            attempts.append(args)
            if len(attempts) == 1:
                raise RuntimeError("temporary failure")
            return ToolResult(tool_name="unstable", ok=True, output="recovered")

        def validate(state):
            if state.tool_history and state.tool_history[-1].ok:
                return ValidationResult(done=True, summary="recovered")
            return ValidationResult(feedback="retry after tool failure")

        harness = Harness(
            tools=[Tool(name="unstable", description="Fails once", handler=unstable)],
            validate_progress=validate,
        )
        model = ScriptedModel(
            Decision.call_tool("unstable", {"attempt": 1}),
            Decision.call_tool("unstable", {"attempt": 2}),
        )

        result = AgentLoop(model=model, harness=harness).run("recover from tool failure")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.summary, "recovered")
        self.assertEqual(len(result.state.tool_history), 2)
        self.assertFalse(result.state.tool_history[0].ok)
        self.assertIn("temporary failure", result.state.tool_history[0].error)
        self.assertEqual(result.state.feedback, ["retry after tool failure"])

    def test_stops_at_max_iterations_when_not_done(self):
        def noop(args, state):
            return ToolResult(tool_name="noop", ok=True, output="still running")

        harness = Harness(
            tools=[Tool(name="noop", description="No-op", handler=noop)],
            validate_progress=lambda state: ValidationResult(feedback="not done yet"),
            max_iterations=2,
        )

        result = AgentLoop(
            model=RepeatingModel(Decision.call_tool("noop")),
            harness=harness,
        ).run("never finishes")

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.reason, "max_iterations reached")
        self.assertEqual(result.state.iteration, 2)
        self.assertEqual(len(result.state.tool_history), 2)


if __name__ == "__main__":
    unittest.main()
