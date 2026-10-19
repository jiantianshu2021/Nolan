# 本地字幕识别

Nolan 使用 whisper.cpp 在本机识别实际音轨，随后将识别文字、原始时间戳和必要的人名/台词参考交给现有 GPT 连接校对。识别过程无需 OpenAI API Key，也不会在失败时切换到收费转录。GPT 校对仍消耗当前连接对应的订阅额度或 API 用量；使用订阅连接时无需另外购买转录 API。

macOS 首次安装：

```bash
bash scripts/install-local-asr.sh
```

脚本通过 Homebrew 安装 `whisper-cpp`，从 whisper.cpp 官方指定的公开模型仓库下载约 466 MiB 的 `ggml-small.bin` 多语言模型，并核对固定版本、大小与 SHA256。已校验模型复用，不重复下载。默认路径为 `~/.local/share/norlan/models/ggml-small.bin`；可用 `NOLAN_ASR_MODEL_DIR` 指定其他目录。普通识别不联网。脚本要求已安装 FFmpeg。

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

可用 `executable` 指定本地 `whisper-cli` 路径，用 `model_path` 指定已有多语言 ggml 模型。模型及识别程序不依赖 Nolan 的 Python 虚拟环境，因此升级 Nolan 不会重新安装 Python 深度学习依赖。

`LocalTranscriber(config, cache_dir=path).transcribe(audio_path)` 返回相对输入音轨的秒级 `start`、`end`、`text`。内部先用 FFmpeg 转成 16 kHz 单声道 PCM WAV，读取 whisper.cpp JSON 中以毫秒表示的 offsets，不把剧本台词当作已听到的声音。字幕校对必须保留识别时间，不能凭剧本补出漏说台词。

识别缓存绑定实际音轨 SHA256、多语言模型 SHA256、程序 SHA256 和语言设置，并在同一内容锁内去重。音轨改变后重新识别；仅改字幕文字或重新混配音乐不应触发视频重生成。缓存仅说明使用了相同输入和识别配置，不代表识别准确。

噪声、重叠对白和生成音轨异常仍可能使识别漏字或错字；本地 ASR 也可能在静音或音乐中产生错误文字。最终仍由用户观看带字幕配乐的整集审批，抽帧审查不能证明字幕与实际语音一致。

官方依据：[whisper.cpp 项目](https://github.com/ggml-org/whisper.cpp)、[CLI 与 JSON 输出实现](https://github.com/ggml-org/whisper.cpp/tree/master/examples/cli)、[官方模型下载脚本](https://github.com/ggml-org/whisper.cpp/blob/master/models/download-ggml-model.sh)。
