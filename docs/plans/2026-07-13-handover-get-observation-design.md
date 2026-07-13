# Handover `get_observation` Design

## Goal

Expose the standard non-privileged Handover environment observation through
`FrankaHandoverApi`, matching the contract used by Cube Lift and Reduced Handover.

## Design

Add a public `get_observation()` method that returns `self._env.get_observation()` and include it in
`functions()`. Do not add privileged object state, grasp-pose functions, reward configuration, or
rollback behavior. The existing `ApiBase.recovery_observation_functions()` default will then declare
`get_observation` automatically, so Capsule prompt generation, validation, and guards require the
same fresh-state call as other standard APIs.

## Verification

Add a regression test using an uninitialized API instance with a minimal fake environment. Verify
that the public function returns the exact underlying observation and that the inferred Capsule
recovery capability is exactly `{"get_observation"}`. Run the Handover API test and the full
runtime-control regression set in WSL.
