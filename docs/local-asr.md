# 本地字幕识别

Nolan 使用 whisper.cpp 在本机识别实际音轨，取得文字和时间候选。有邻近、相似的本集剧本台词时优先作为文字候选，随后由现有 GPT 连接校对，再经独立声学模型核对实际声音。识别过程无需 OpenAI API Key，也不会在失败时切换到收费转录。GPT 校对仍消耗当前连接对应的订阅额度或 API 用量；使用订阅连接时无需另外购买转录 API。

macOS 首次安装：

```bash
bash scripts/install-local-asr.sh
```

脚本通过 Homebrew 安装或升级 `whisper-cpp`，需要同时提供 `whisper-cli` 和 `whisper-vad-speech-segments`。从 whisper.cpp 官方指定的公开仓库下载两个模型：约 466 MiB 的 `ggml-small.bin` 多语言模型，以及约 865 KiB 的 `ggml-silero-v6.2.0.bin` 人声检测模型。两者都固定版本、大小与 SHA256，已校验文件复用、不覆盖已有异样文件。默认目录为 `~/.local/share/norlan/models/`；可用 `NOLAN_ASR_MODEL_DIR` 指定其他目录。普通识别不联网。脚本要求已安装 FFmpeg。

项目配置：

```json
{
  "subtitles": {
    "mode": "local",
    "backend": "whisper_cpp",
    "language": "zh",
    "timeout": 600,
    "threads": 4
  }
}
```

可用 `executable`、`model_path` 指定本地 `whisper-cli` 及多语言 ggml 模型，用 `vad_executable`、`vad_model_path` 指定独立人声检测程序及 Silero 模型。缺少人声检测依赖时会提示安装，不会静默退回旧的粗略时间轴。模型及识别程序不依赖 Nolan 的 Python 虚拟环境，因此升级 Nolan 不会重新安装 Python 深度学习依赖。

`LocalTranscriber(config, cache_dir=path).transcribe(audio_path)` 返回相对输入音轨的秒级 `start`、`end`、`text`。内部先用 FFmpeg 转成 16 kHz 单声道 PCM WAV，再用独立 Silero VAD 检测实际人声区间。检测每 20 秒重新开始一次，相邻窗口重叠 10 秒，避免长段环境声使检测器漏掉后来的轻声对白。重复或重叠的人声区间合并，真实静默间隙保留。人声窗保留原音轨的时钟，每窗按样本裁音、独立识别。识别额外保留前后最多 250 毫秒上下文，且不越过相邻人声窗之间的中点，避免弱声起始字被截掉或串入下一句；字幕显示仍受原始人声窗限制，再将局部时间映回整集时钟。不会将整集静默压缩后推算字幕，也不会给所有字幕统一加一个固定秒数。检测器输出为百分之一秒，Whisper JSON offsets 为毫秒；适配器分别转换。短于两秒的裁音若遇到解码器最短两秒结束标记，以真实人声窗为硬边界。

剧本文字候选不是声音证据；必须经实际声音独立核验。GPT 文字步骤保留这次识别的时间候选，实际对时由本地声学检查器完成，不能复用旧字幕时间或凭剧本补出未说台词。数字或实际改说与剧本不符时，不用剧本强改。可选 `cancel_check` 在每个人声窗之间检查暂停；VAD 与逐窗识别共享一个总超时。

识别缓存绑定实际音轨 SHA256、Whisper 和 Silero 两个模型 SHA256、两个程序 SHA256、语言、线程数、人声检测参数以及对时策略版本，并在同一内容锁内去重。缓存包含实际人声窗，先完整识别再原子发布，源音轨始终不改写。旧版整集粗略时间戳缓存与新策略不相容，升级后自动失效。音轨改变后重新识别；仅改字幕文字或重新混配音乐不应触发视频重生成。缓存仅说明使用了相同输入和识别配置，不代表识别准确。

噪声、重叠对白和生成音轨异常仍可能使识别漏字或错字；本地 ASR 也可能在静音或音乐中产生错误文字。最终仍由用户观看带字幕配乐的整集审批，抽帧审查不能证明字幕与实际语音一致。

## 字幕生成后的独立同步检查

字幕生成、GPT 文字校对完成以后，Nolan 默认用独立的 SenseVoice int8 中文 CTC 模型读取实际音轨，核对每条字幕的开始、结束和人声覆盖；检查器不使用上一轮 Silero 的边界来证明自己的正确性。完整音轨按固定 20 秒窗口、15 秒步长扫描，重叠区各取中间归属，保留原片时钟；避免把长音轨一次送入离线识别器而漏掉早期对白。再以不同上下文检查局部字音时间锚。每条最多四个局部上下文，至少两个不同上下文支持首末边界和字音顺序，才认为证据一致。全轨粗扫某字有误时，也可以由两个实际局部上下文的完整声学证据支持时间，不能用低覆盖字幕的暂定范围遮住漏字幕的人声。短于四个字的字幕要求完整匹配；更长字幕至少匹配三个字音、首末字有支持且覆盖不少于 80%。拼音比较允许同音字，阿拉伯数字转换为明确的中文读法后比较，显示文字不会因此自动被替换。CTC 给出的是声学时间锚，最后一个字的锚点会加受限的结束余量，不能将它冒称为精确的发音终点。

首次安装独立检查器：

```bash
bash scripts/install-subtitle-alignment.sh
```

需要 Python 3.10–3.13（macOS 默认寻找 `python3.13`，可设置 `NOLAN_ALIGN_PYTHON`）。脚本创建独立虚拟环境，仅安装固定版本的 `sherpa-onnx`、NumPy 和拼音比较依赖，不安装 PyTorch。默认下载官方 SenseVoice 约 163 MB 压缩包，校验后保留约 239 MB 模型和词表；同时安装约 26 MB 的 Zipformer 小模型，用于复核覆盖扫描的可疑误识。文件均绑定固定大小及 SHA256，已有正确文件复用。SenseVoice 导出来自官方 Sherpa 项目和其发布者固定版本仓库，原模型许可证来源保留在 `MODEL_SOURCE.txt` / `LICENSE`。

默认主模型目录为 `~/.local/share/norlan/models/sensevoice-zh-int8-2024-07-17/`，辅助模型为 `~/.local/share/norlan/models/zipformer-ctc-zh-small-int8/`，独立 Python 入口为 `~/.local/share/norlan/alignment-runtime/bin/python`。升级时创建新环境并切换入口，现有进程继续使用原环境。可设置 `NOLAN_ALIGN_BASE` 和 `NOLAN_ALIGN_MODEL_DIR` 改变安装位置，再通过字幕配置的 `alignment_python`、`alignment_model_path`、`alignment_coverage_model_path` 对应指定；`alignment_timeout` 默认 300 秒。不需要新的密钥或收费接口。

旧小模型仍可明确选择：安装时设置 `NOLAN_ALIGN_BACKEND=zipformer_ctc`，项目字幕设置 `alignment_backend=zipformer_ctc`。默认值为 `sense_voice`，不会在模型缺失时暗中切回小模型。

同步不合格时，Nolan 最多自动重对齐两次，并重新检查实际音轨；只有机检通过才进入正常整集最终审批。文字校对不能替代同步修复。全音轨检查发现漏字幕、对齐证据矛盾或本地依赖失效时，不会假报通过，也不会无限重跑、重新生成视频或音乐。不支持的疑似噪声字幕只有在完整 CTC、局部 CTC 和额外两种上下文的原 Whisper 识别都不支持时才可排除。覆盖扫描发现的可疑额外对白，也不能凭 SenseVoice 单模型强迫添字：只有另一小 CTC 的全轨和两个局部上下文都零支持、且两个宽上下文 Whisper 也不支持，才允许排除该误报并保留证据。外语不受中文小模型未识别结果否决；任何有支持的可疑真声仍会显示需要关注的区间和原因。排除、时间变更及证据会记录，供最后审片查看。

检查缓存绑定实际音轨、当前字幕文字和时间、模型及词表、检查脚本、运行库版本和策略；时间或文字改变后重新核验。本检查器主要覆盖普通话，不应将它用于证明英文、方言、重叠对白或噪声一定没有说话；无法确认的内容会停止并说明原因。机检通过表示通过公开的机器容差规则，仍不等于人工逐字听验或所有字幕绝对准确。

发现实际声音中多说或漏记的内容时，只允许依据 SenseVoice 与小 CTC 在同一原时钟支持的共同文字提出补条；两个模型还须各有两种不同上下文，完整支持该条字音、首末边界及一致的时间范围。修复仅将实际文字插入已重对齐字幕的非重叠空隙，记录 `added_from_audio` 及全部证据，然后重新检查全片。只在字面上出现相同词、时标不一致、单个不定时长的字、与已有字幕重叠或只有一个模型支持时，均不能自动增添。剧本期望本身不能授权补出实际未说内容。

独立检查官方资料：[SenseVoice API 与中文逐字时间输出](https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html)、[所用固定版本模型](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tree/2365baeacb507f821a0c8120fcee3d484dba7a07)、[辅助 Zipformer 中文 CTC](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-ctc/zipformer-ctc-models.html)、[SenseVoice 时间单位实现](https://github.com/k2-fsa/sherpa-onnx/blob/v1.12.40/sherpa-onnx/csrc/offline-recognizer-sense-voice-impl.h)。

官方依据：[whisper.cpp 项目](https://github.com/ggml-org/whisper.cpp)、[CLI 与 JSON 输出实现](https://github.com/ggml-org/whisper.cpp/tree/master/examples/cli)、[官方模型下载脚本](https://github.com/ggml-org/whisper.cpp/blob/master/models/download-ggml-model.sh)、[独立 VAD 实现与时间单位](https://github.com/ggml-org/whisper.cpp/tree/master/examples/vad-speech-segments)、[官方 Silero 模型仓库](https://huggingface.co/ggml-org/whisper-vad)。
