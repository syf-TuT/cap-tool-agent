# LIBERO Text Controller with Molmo Perception Design

## Goal

Run the LIBERO-Object Capsule `llm_step` experiment with a text-only decision
model while retaining Molmo as an internal perception dependency of
`FrankaLiberoApi`.

## Configuration

The experiment preset disables both direct visual-input paths:

- `use_visual_feedback: false` prevents images from being attached to the
  initial program-generation prompt.
- `capsule_action_visual_feedback: false` prevents images from being attached
  to Capsule action prompts.
- `use_wrist_camera: false` disables wrist-image capture and wrist video.

`record_video: true` remains enabled for the main camera. The Molmo endpoint and
model settings remain unchanged, so `FrankaLiberoApi` can continue to call Molmo
for internal object perception without exposing images to the decision model.

## Scope

This is a configuration and documentation change. The existing runtime already
keeps direct prompt imagery independent from API-internal perception, so no
production Python behavior needs to change.

## Verification

The configuration test must parse the YAML and assert that all three direct
visual-input/wrist switches are false while the Molmo URL and model remain
configured. Focused WSL tests must cover the LIBERO preset and Molmo config.
