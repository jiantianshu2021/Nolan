# 本地字幕识别

Nolan 使用 whisper.cpp 在本机识别实际音轨，随后将识别文字、原始时间戳和必要的人名/台词参考交给现有 GPT 连接校对。识别过程无需 OpenAI API Key，也不会在失败时切换到收费转录。GPT 校对仍消耗当前连接对应的订阅额度或 API 用量；使用订阅连接时无需另外购买转录 API。

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

剧本只供人名和同音字校对，不是声音证据；GPT 必须保留这次实际识别的时间，不能复用旧字幕时间或凭剧本补出未说台词。可选 `cancel_check` 在每个人声窗之间检查暂停；VAD 与逐窗识别共享一个总超时。

识别缓存绑定实际音轨 SHA256、Whisper 和 Silero 两个模型 SHA256、两个程序 SHA256、语言、线程数、人声检测参数以及对时策略版本，并在同一内容锁内去重。缓存包含实际人声窗，先完整识别再原子发布，源音轨始终不改写。旧版整集粗略时间戳缓存与新策略不相容，升级后自动失效。音轨改变后重新识别；仅改字幕文字或重新混配音乐不应触发视频重生成。缓存仅说明使用了相同输入和识别配置，不代表识别准确。

噪声、重叠对白和生成音轨异常仍可能使识别漏字或错字；本地 ASR 也可能在静音或音乐中产生错误文字。最终仍由用户观看带字幕配乐的整集审批，抽帧审查不能证明字幕与实际语音一致。

官方依据：[whisper.cpp 项目](https://github.com/ggml-org/whisper.cpp)、[CLI 与 JSON 输出实现](https://github.com/ggml-org/whisper.cpp/tree/master/examples/cli)、[官方模型下载脚本](https://github.com/ggml-org/whisper.cpp/blob/master/models/download-ggml-model.sh)、[独立 VAD 实现与时间单位](https://github.com/ggml-org/whisper.cpp/tree/master/examples/vad-speech-segments)、[官方 Silero 模型仓库](https://huggingface.co/ggml-org/whisper-vad)。
