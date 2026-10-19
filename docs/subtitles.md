# 根据实际对白制作字幕

Nolan 的自动字幕从已经生成的视频音轨识别文字和时间，不把剧本台词直接当成实际对白。分镜阶段仍然规划台词；当前主流程先将本集镜头合成，再识别整集实际音轨，时间戳直接对应本集时间轴。镜头重生成或时长变化后，应重新识别合成音轨并重建字幕。

当前适配器使用 OpenAI `POST /v1/audio/transcriptions`，选择 `whisper-1`、`verbose_json` 和分段时间戳。该组合已经依据官方文档核实；当前代码不接受直接替换为没有相同时间戳返回约定的模型。[OpenAI 转录接口](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)、[文件转录指南](https://developers.openai.com/api/docs/guides/speech-to-text)

配置放在项目的 `settings.subtitles` 中：

```json
{
  "mode": "openai",
  "model": "whisper-1",
  "api_key_env": "OPENAI_API_KEY",
  "call_cap": 0,
  "timeout": 120,
  "language": "zh"
}
```

`language` 可以省略以使用自动识别。`call_cap` 是每集转录一次的保守费用上限，使用项目账本币种；示例的 0 必须按实际服务价格和本集长度改成足够的正数，再批准总预算。API 密钥通过环境变量提供，不写入项目配置。请求固定发送到官方 OpenAI 地址，不跟随重定向；ChatGPT/Codex 登录本身不为这个 HTTP 接口提供 API 密钥。

底层接口为 `Transcriber(config).transcribe(audio_path)`，返回 `[{"start": 0.2, "end": 2.1, "text": "实际识别文字"}]`，时间单位为秒，相对于输入音轨起点。接口可以接受单镜或整集音轨，当前主流程每集调用一次。单次文件上限为 25 MB，超过上限会停止，不会默默截断音频；需要先压缩或另行实现分块及时间偏移。无须为此购买另外的配音服务。[文件大小与时间戳说明](https://developers.openai.com/api/docs/guides/speech-to-text)

转录前检查真实音轨与时长，转录后拒绝倒序、明显重叠、负时间、超出音轨和没有时间戳的非空文本。最多 50 毫秒的边界误差按实际长度收齐。未识别出对白时返回空字幕；不会用预先写好的对白补足。超时、断网或无效返回可能发生在计费之后，因此抛出 `ProviderError(uncertain=True)`，交由账本保留待核实费用，绝不自动重发付费 POST。

这些检查只说明时间结构合理，不能证明识别准确。背景音乐、低音量、多人重叠、专有名字和生成语音异常仍会导致漏字、错字或幻觉。ASR 结果是待核对字幕；最终每集审批要同时检查字幕、实际声音及口型。抽帧视觉检查也不能代替试听。

已有字幕的成片可单独校正文字。例如在终端对话中输入 `E001 字幕 把“你好”改成“您好”`，或者运行 `nolan captions 项目目录 E001 --version 1 --correction '把你好改成您好'`。校正针对所见的具体版本，保留视频镜头和已有时间戳，再重新合成供人工审片；它不会为了字幕错字重新付费生成全部视频。网页审片室也有独立的“修改字幕”按钮。对白本身错误、漏说或口型错误应通过镜头返工处理。

没有配置转录时可以选择主流程提供的无字幕或人工对齐方式，但不能将计划台词标成“已与实际声音校准”。当前适配器没有安装或调用本地 Whisper，也没有进行任何付费测试。
