# Nolan development

Use Chinese for user-facing copy. Keep provider/model choices explicit.

- Never auto-approve a person, scene, voice, plan or episode.
- Each next episode is blocked until the preceding final render is approved.
- Reserve cost before a paid POST. Unknown submission outcomes must not be
  retried. Persist the provider ID before polling or downloading.
- Do not claim sampled frames establish audio, lip-sync or full-motion quality.
- Never replace actual references with prose while claiming references were sent.
- Keep keys out of project state, logs, tests and version control.
- Use argv for subprocesses, never shell commands from GPT output.
- Keep original media immutable and decisions bound to version and file hash.
- Synthetic tests are visibly marked as such. Do not contact paid providers in CI.

Run `python3 -m unittest discover -s tests -v` (FFmpeg integration tests require
ffmpeg and ffprobe). Preserve package data for the optional local dashboard.
