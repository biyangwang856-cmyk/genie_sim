"""Recovery gates for measured state and real command evidence."""
import os
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from skill_recovery import FailureKind, PhysicalState, RecoveryCandidate, RecoveryFailure, run_recoverable_skill


def pose_at(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


class PhysicalFixture:
    def __init__(self):
        self.payload_x = .1
        self.hand_x = .1
        self.offset = 0
        self.sim_time = 0.
        self.holding = {"operator": "sample"}

    def observe(self):
        return PhysicalState(
            {"sample": pose_at(self.payload_x), "hand": pose_at(self.hand_x)},
            {"arm": self.hand_x}, self.offset, self.holding, self.sim_time,
        )

    def move(self, hand_x, payload_x):
        self.hand_x, self.payload_x = hand_x, payload_x
        self.offset += 123
        self.sim_time += .03

    @staticmethod
    def verify(before, after, _result):
        return bool(np.isclose(after.poses["sample"][0, 3], .3))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = PhysicalFixture()
        self.events = []

    def event(self, name, **data):
        self.events.append({"event": name, **data})

    def run_candidates(self, *candidates, **options):
        defaults = {
            "observe": self.fixture.observe,
            "verify": self.fixture.verify,
            "event": self.event,
            "deadline": time.monotonic() + 5.,
        }
        return run_recoverable_skill("place-sample", candidates, **{**defaults, **options})

    def test_confirmed_refusal_uses_fresh_native_candidate(self):
        def rejected(_before):
            exc = RuntimeError("The native follower found no path.")
            exc.kind = "local_waypoint_plan_or_tracking_failed"
            exc.details = {"planner_accepted": False}
            raise exc

        def alternate(before):
            self.assertAlmostEqual(before.poses["sample"][0, 3], .1)
            self.fixture.move(.3, .3)
            return "placed"

        result = self.run_candidates(RecoveryCandidate("direct", rejected), RecoveryCandidate("over", alternate))
        self.assertEqual(result, "placed")
        self.assertEqual([row["event"] for row in self.events], [
            "skill_attempt_started", "skill_attempt_failed", "skill_attempt_started", "skill_attempt_recovered",
        ])
        self.assertEqual(self.events[1]["failure_kind"], "planning_refused")
        self.assertEqual(self.events[-1]["change"]["native_command_bytes"], 123)

    def test_false_with_native_commands_cannot_dispatch_planning_retry(self):
        def rejected_after_command(_before):
            # A hand can be blocked although the underlying command was already submitted.
            self.fixture.move(.1, .1)
            return False

        def blind_retry(_before):
            self.fail("An ambiguous false result with native commands must not trigger a no-motion retry.")

        with self.assertRaises(RecoveryFailure) as caught:
            self.run_candidates(
                RecoveryCandidate("direct", rejected_after_command), RecoveryCandidate("over", blind_retry),
            )
        self.assertEqual(caught.exception.kind, FailureKind.PHYSICAL_FAILURE)
        self.assertEqual(self.events[-1]["reason"], "failure_not_applicable")

    def test_tracking_remedy_replans_from_measured_held_pose(self):
        def short_move(_before):
            self.fixture.move(.2, .2)
            exc = RuntimeError("Planner accepted, but feedback was outside tolerance.")
            exc.kind, exc.details = "local_waypoint_plan_or_tracking_failed", {"planner_accepted": True}
            raise exc

        def finish(before):
            self.assertAlmostEqual(before.poses["sample"][0, 3], .2)
            self.fixture.move(.3, .3)

        self.run_candidates(
            RecoveryCandidate("direct", short_move),
            RecoveryCandidate("measured-correction", finish, after=frozenset({FailureKind.TRACKING_ERROR}),
                              applicable=lambda state: state.holding.get("operator") == "sample"),
        )
        self.assertTrue(any(row.get("failure_kind") == "tracking_error" for row in self.events))
        self.assertEqual(self.events[-1]["event"], "skill_attempt_recovered")

    def test_material_failure_requires_fresh_precondition_callback(self):
        def slipped(_before):
            self.fixture.move(.2, .1)
            self.fixture.holding.clear()
            raise RecoveryFailure(FailureKind.GRASP_LOST)

        def retry(_before):
            self.fail("A held-object correction requires a fresh precondition.")

        with self.assertRaises(RecoveryFailure) as caught:
            self.run_candidates(
                RecoveryCandidate("direct", slipped),
                RecoveryCandidate("correction", retry, after=frozenset({FailureKind.GRASP_LOST})),
            )
        self.assertEqual(caught.exception.kind, FailureKind.GRASP_LOST)
        self.assertEqual(self.events[-1]["reason"], "missing_recovery_precondition")

    def test_same_candidate_is_not_dispatched_again_while_world_is_unchanged(self):
        calls = []
        history = {}

        def reject(before):
            calls.append(before)
            return False

        candidate = RecoveryCandidate("no-path", reject)
        for _ in range(2):
            with self.assertRaises(RecoveryFailure):
                self.run_candidates(candidate, history=history, retry_key="move:sample:destination")
            self.fixture.sim_time += .1
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.events[-1]["reason"], "same_candidate_without_physical_progress")
        self.fixture.payload_x += .02
        with self.assertRaises(RecoveryFailure):
            self.run_candidates(candidate, history=history, retry_key="move:sample:destination")
        self.assertEqual(len(calls), 2)

    def test_reported_success_does_not_override_observed_effect(self):
        def missed_goal(_before):
            self.fixture.move(.3, .1)
            return True

        with self.assertRaises(RecoveryFailure) as caught:
            self.run_candidates(RecoveryCandidate("miss", missed_goal))
        self.assertEqual(caught.exception.details["reason"], "measured_effect_not_verified")

    def test_missing_evidence_and_reversed_journal_stop_recovery(self):
        for reason in ["missing_native_command_log", "native_command_log_reversed"]:
            with self.subTest(reason=reason):
                states = iter([PhysicalState({"sample": pose_at(.1)}, command_offset=7),
                               PhysicalState({"sample": pose_at(.1)},
                                             command_offset=None if reason.startswith("missing") else 6)])
                with self.assertRaises(RecoveryFailure) as caught:
                    self.run_candidates(RecoveryCandidate("direct", lambda _: True), observe=lambda: next(states))
                self.assertEqual(caught.exception.kind, FailureKind.OBSERVATION_MISSING)
                self.assertEqual(caught.exception.details["reason"], reason)

    def test_budget_blocks_a_candidate_before_dispatch(self):
        def unexpected(_before):
            self.fail("Expired budget cannot issue a native command.")

        with self.assertRaises(RecoveryFailure) as caught:
            self.run_candidates(RecoveryCandidate("direct", unexpected), deadline=time.monotonic() - 1)
        self.assertEqual(caught.exception.kind, FailureKind.BUDGET_EXHAUSTED)


class CapRefusalTests(unittest.TestCase):
    def test_axis_alternative_requires_confirmed_no_command_native_refusal(self):
        from skill_executor import SkillExecutor, SkillFailure

        for accepted, bytes_written, expected in ((False, 0, "planning_refused"),
                                                   (False, 123, "physical_failure"),
                                                   (True, 123, "tracking_error")):
            with self.subTest(accepted=accepted, bytes_written=bytes_written):
                executor = SkillExecutor.__new__(SkillExecutor)
                executor.spec = {"support": {}}
                executor.robot = Mock()
                executor.ops = Mock()
                executor.ops.pose.return_value = np.eye(4)
                executor.robot.get_ee_pose.return_value = np.eye(4)
                executor.arm = lambda _role: "left"
                executor.event = Mock()
                executor.cap_axis_only_candidate = Mock(return_value=(np.eye(4), {}))
                executor.follow_local_pose = Mock(side_effect=[
                    SkillFailure("local_waypoint_plan_or_tracking_failed", {"planner_accepted": accepted}), None,
                ])

                def offset(_path):
                    return 123 + (bytes_written if executor.follow_local_pose.called else 0)

                params = {"object": "cap", "reference": "pen", "role": "assistant"}
                with (
                    patch.dict(os.environ, {"GENIE_EPISODE_EVIDENCE": "test-evidence"}),
                    patch("skill_recovery.native_command_offset", side_effect=offset),
                ):
                    if expected == "planning_refused":
                        executor.follow_cap_axial_step(params, np.eye(4), -.01)
                        executor.cap_axis_only_candidate.assert_called_once()
                    else:
                        with self.assertRaises(SkillFailure):
                            executor.follow_cap_axial_step(params, np.eye(4), -.01)
                        executor.cap_axis_only_candidate.assert_not_called()
                classification = next(call for call in executor.event.call_args_list
                                      if call.args == ("cap_axial_motion_classified",))
                self.assertEqual(classification.kwargs["failure_kind"], expected)


if __name__ == "__main__":
    unittest.main()
