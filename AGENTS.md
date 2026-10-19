# Nolan development

Use Chinese for user-facing copy. Keep provider/model choices explicit.

- Never auto-approve a person, scene, voice, plan, video segment or episode.
- GPT video reviews are advisory only. Every new video version waits for the
  user's decision. Only explicit human rework requests authorize another video
  submission; resuming a task or a failed report never does.
- Each next episode is blocked until the preceding final render is approved.
- Reserve cost before a paid POST. Unknown submission outcomes must not be
  retried. Persist the provider ID before polling or downloading.
- Do not claim sampled frames establish audio, lip-sync or full-motion quality.
- Never replace actual references with prose while claiming references were sent.
- GPT selects concise system broadcasts and their timing; retain full rules in
  the on-screen system UI and preserve character dialogue. Keep a single human
  approval for the complete final prompt; do not add a seconds-selection gate.
- Keep keys out of project state, logs, tests and version control.
- Use argv for subprocesses, never shell commands from GPT output.
- Keep original media immutable and decisions bound to version and file hash.
- Route segment feedback through the same revision planner for ordinary edits,
  local storyboarding and asset additions. A new storyboard may change only the
  target segment's internal shots and duration within provider limits. GPT
  chooses timing from the user's requested content; update the episode total
  accordingly rather than silently retiming other segments.
  Finish any asset approval first, then require the complete final prompt's
  existing human approval before submitting the new video version.
- Synthetic tests are visibly marked as such. Do not contact paid providers in CI.

Run `python3 -m unittest discover -s tests -v` (FFmpeg integration tests require
ffmpeg and ffprobe). Preserve package data for the optional local dashboard.
