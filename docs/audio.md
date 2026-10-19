# 声音、配乐与字幕

Norlan 在分镜策划时记录对白、环境声、音乐情绪和转场意图，在合成阶段按实际镜头时长统一剪辑。FFmpeg 执行剪辑与混音；它不会作曲，也不会判断剧情。

当前本地媒体模块可以保留镜头原声、给静音镜头补齐静音轨，统一为 48 kHz 双声道并按镜头顺序拼接。画面按首镜头比例归一化，不同比例素材加黑边。视频统一为 30 fps，单镜头时长最多有半帧的舍入，合成清单记录原始时长和实际时间线。不会用计划时长覆盖实际测得的时长。

## 配乐

可以导入一个本地音频文件作为全片配乐。它在连续时间线上播放，开头和结尾淡入淡出，遇到原声音量上升时自动压低；配乐短于画面时尾部补静音，不擅自循环旋律。混音进行一次响度归一化，目标为 −16 LUFS、−1.5 dBTP；这不等于已经通过最终听审。

未导入配乐时会明确记录“无配乐”，不把测试音、原生混音或模型提示词标记为已生成的独立音乐。

独立的 `MusicProvider` 适配器已经实现火山引擎纯音乐 v5.0 的提交、查询和下载，使用 `GenBGMForTime`（后付费）或 `GenBGM`（预付费）与 `QuerySong`。它属于单独开通的火山音乐服务，不能复用即梦网页登录或 Ark API Key。需要在终端环境变量中配置 `VOLC_ACCESS_KEY_ID`、`VOLC_SECRET_ACCESS_KEY`；密钥不会写入项目。这里完成的是按官方协议编写与模拟响应测试，尚未使用真实账户付费验证。

该接口每次生成 30–120 秒的中文描述纯音乐。Norlan 显式提交分段时长之和，避免提示词中另一种时长覆盖预算参数；返回结果仍须核对实际时长与账单。超过 120 秒的整集可先选择本地已备配乐，不能假设一次调用就会生成无限长音乐。重复使用同一音乐素材可以统一主题，但不能保证模型多次生成旋律完全相同。

提交超时或丢失任务 ID 时不自动重发，以防重复扣费；已完成但下载失败时仅重试原任务下载。成功查询返回的官方音频 CDN 地址才会被下载，下载请求不携带 API 凭据。

原生视频若已经把台词、环境声、音乐混为一条音轨，本模块无法无损拆分。压低配乐依据的是整个原声，无法保证只根据对白触发。需要更精细控制时应提供独立声音素材，或要求生成镜头不带背景音乐。声线连续性、口型和自然衔接仍须听审。

## 字幕

`compose` 接收全片绝对秒数的 `start`、`end` 和 `text`。这些时间戳应来自对最终对白的转写和校准。本模块不会识别语音，不会自动把原剧本当作真实台词，也不会声称字幕已经与嘴型对齐。每次改变镜头顺序或时长，都需要重新提供相应时间戳。

字幕总会导出为 UTF-8 SRT；本机同时具备 FFmpeg 的 `subtitles` 滤镜和已知字体时，默认烧录到画面，也可以设置 `burn_subtitles=False` 仅输出 SRT。检测到字体不代表每个特殊字符都有字形，交付前需要检查实际画面。没有时间戳就不生成虚构字幕。

## 检查证据

`extract_evidence` 保存首、中、片尾附近三张画面、原视频哈希、采样时刻和独立 WAV 音轨。这些文件帮助下一镜头参考和审片定位，**三张截图无法证明整段动作、口型或声音连续性**。片尾附近截图不是保证解码出的最后一帧。自动检查应明确覆盖范围，不确定时交给人工观看完整视频。

演示模式的图片和视频带有 `NORLAN DEMO / SYNTHETIC PLACEHOLDER` 字样。演示音轨只是测试音，不包含合成对白、角色声线或生成配乐。

实现依据：[FFmpeg 官方滤镜文档](https://ffmpeg.org/ffmpeg-filters.html) 中的 `concat`、`loudnorm`、`sidechaincompress`、`afade` 与 `subtitles`。具体滤镜和字体由安装的 FFmpeg 构建决定。

音乐接口依据：[生成纯音乐](https://docs.volcengine.com/docs/Audiovideounderstandingandprocessing/Generatepuremusic?lang=zh)、[查询任务](https://docs.volcengine.com/docs/Audiovideounderstandingandprocessing/Querytask-1?lang=zh)、[公共字段](https://docs.volcengine.com/docs/Audiovideounderstandingandprocessing/Publicfields?lang=zh) 与[签名方法](https://docs.volcengine.com/docs/APISignatureCallGuide/Signaturemethod-9?lang=zh)。接口与权限以开通账户时的官方文档为准。
