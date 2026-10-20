# Nolan development

Use Chinese for user-facing copy. Keep provider/model choices explicit.

- Interactive Nolan uses native Codex file/terminal/image tools with
  workspace-write and on-request approvals. A user-selected working directory
  may contain source files; create production projects in independent empty
  child folders and retain existing projects/media. Keep subscription/API homes
  separate, share login only through the existing auth link, and never inherit
  provider credentials into shell tools or import global hooks/MCP config.
  Ordinary files can be edited with native tools; production approvals, budget,
  provider submissions and ledger changes use Nolan's MCP flow. Never claim the
  desktop app's browser/plugins are automatically available in the CLI.
- Ark credentials resolve from the explicitly selected environment variable,
  then the fixed macOS Keychain item for default ARK_API_KEY. Saving/rotating
  requires the user's local ark-login command and hidden input. Never put keys
  into command arguments, shell tools, project JSON, logs or plaintext config.
  Missing, locked and access-denied are distinct diagnostics. Native shell
  isolation must not lead to a false report that the MCP provider lacks a key;
  get_credentials_status/doctor use the same resolver. Credential checks do not
  generate media, approve anything, clear failures or retry existing jobs.

- Never auto-approve a person, scene, voice, plan, video segment or episode.
- GPT video reviews are advisory only. Every new video version waits for the
  user's decision. Only explicit human rework requests authorize another video
  submission; resuming a task or a failed report never does.
- GPT video reviews default off, including projects without the setting. The
  local UI switch affects future segment and rough-cut advisory submissions,
  never human gates, local evidence/stitching or subtitle checks. Keep existing
  reports; enabling does not backfill skipped versions. Check the setting again
  at reservation, while allowing an already submitted review to finish.
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
- The prompt editor at the pre-generation confirmation step starts with the entire prepared
  prompt. Its explicit Save and Generate action adopts the user's exact text
  and authorizes one video, with current duration/references and any explicitly
  selected video model. Never run
  GPT planning/refinement, append old prompt rules or require another prompt
  approval on this route. Archive original/corrected text with exact hashes;
  stale inputs and unknown outcomes block without discarding the human text or
  silently reverting to GPT. Ordinary video rework retains asset/storyboard gates.
- Per-segment video model selection is bound atomically to the prompt hash,
  human consent, provider request and price snapshot. Do not change project
  defaults, existing media or in-flight jobs. Resume by the saved task ID and
  original request settings; never silently switch models, remove references
  or change duration to make a model compatible. Record effective wire defaults.
- Keep keys out of project state, logs, tests and version control.
- Use argv for subprocesses, never shell commands from GPT output.
- Keep original media immutable and decisions bound to version and file hash.
- Prepare source-hash-bound transport copies without removing/reordering any
  references. Preserve source dimensions where possible, image alpha and audio
  duration/channels; never alter approved originals or official tail frames.
  Validate exact encoded request size before reserving and verify the same
  payload digest immediately before POST. Keep only counts/digests/provenance,
  never base64 bodies or signed video URLs, in the ledger/UI.
  Proven pre-network local refusals release only their reservation. Legacy
  refunds require the exact known local failure and absence of all remote,
  usage/bill/output evidence; preserve the failed record and restore a fresh
  human prompt gate. Never loosen generic spent-accounting or retry protections.
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
  If automatic spelling changes fail acoustic support, a bounded original-ASR
  text recovery must require unique same-clock mapping, full independent
  support and no new problems elsewhere. Preserve failed reports and used
  timing-repair attempts; never apply this to explicit manual edits. A burned
  diagnostic preview may show unresolved captions, but cannot open final approval
  or repeat rendering/arrangement on ordinary Continue. Check subtitle filter
  and CJK font capability before normalizing clips for a requested hard burn.
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
- When a new ending conflicts with the immediate next unstarted plan, let GPT
  draft the minimum continuity edit instead of repeatedly requesting the same
  human feedback. Preserve the next segment's duration, cuts, dialogue, scene
  and assets. Never modify another started segment, prompt candidate or human
  instruction. Bind both drafts atomically; rejecting the source prompt rolls
  back only untouched neighbor drafts. Each segment keeps its own prompt gate.
  Migrate old planning blockers once per policy version and cache completed
  revision results; unknown outcomes never authorize resubmission.
- Synthetic tests are visibly marked as such. Do not contact paid providers in CI.

Run `python3 -m unittest discover -s tests -v` (FFmpeg integration tests require
ffmpeg and ffprobe). Preserve package data for the optional local dashboard.
