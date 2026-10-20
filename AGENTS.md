# Nolan development

Use Chinese for user-facing copy. Keep provider/model choices explicit.

- Never auto-approve a person, scene, voice, plan, video segment or episode.
- GPT video reviews are advisory only. Every new video version waits for the
  user's decision. Only explicit human rework requests authorize another video
  submission; resuming a task or a failed report never does.
- Each next episode is blocked until the preceding final render is approved.
- Importing a season preserves existing episodes, assets, budget, approvals and
  source/novel.txt. Store immutable per-episode source hashes and retain original
  episode numbers/durations when explicitly chosen. Plan only the next eligible
  episode from its source, a small asset catalog and the previous accepted ending.
- A new episode storyboard is a version/hash-bound human candidate, not permission
  to generate media. Existing asset-direction, asset and video-prompt approvals
  still apply. Reuse completed planning caches after interruption; unknown calls
  never authorize a repeat. Episode navigation must not approve or regenerate.
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
- Show the current final export directly in the episode postproduction panel,
  with its human approval and subtitle/music feedback together. Never substitute
  a dry cut or retired export for the current final preview.
- Before final episode approval, independently check subtitle timing against the
  actual audio. Repair locally at most twice and recheck; a spelling pass, the
  original VAD boundaries, or missing/contradictory evidence cannot pass this gate.
  Bind the report to current audio, captions, source and rendered manifest.
  Continuing or restarting must not reset exhausted automatic repair attempts.
  Subtitle checks never regenerate video, voices or music, or approve an episode.
- Manual postproduction edits retain the user's subtitle times and approved
  track identities. Render a new local preview; do not send them back to GPT or
  silently replace them with automatic timing. Explicit user edits bypass new
  ASR/sync checks: bind the manual adoption to audio, captions, source and export,
  label it as manual rather than machine-passed, then await final human approval.
  Keep prior exports and provenance.
- Prefer nearby scripted dialogue for caption text, then verify against actual
  audio. Do not add unspoken lines or overwrite differing numbers from a script.
- Automatic episode music uses source-hash-bound local acoustic analysis for
  excerpt choices, a shared track loudness reference, and bounded ducking only
  in verified speech windows. Check the rendered mix numerically and allow at
  most two local corrections; never retry video/music generation or treat
  acoustic metrics as human listening approval. Persist exhausted checks and
  bind them to the final export, music, captions and source. Explicit manual
  postproduction edits bypass fresh checks and adaptive changes.
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
