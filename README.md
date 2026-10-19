# Norlan

在 Mac mini 终端运行的 GPT 对话与 AI 漫剧制作助手。启动就能自由聊天、讨论创意、写故事和修改方案，不需要先提供小说或创建制作项目。你决定制作时，GPT 再调用 Norlan 的工具规划分集、资产和分镜。**每集都交给你完整观看；你批准当前版本后，才开始下一集。**

默认入口采用原生 Codex 对话界面，支持流式回答、多轮上下文和会话恢复，Norlan 制作能力通过专用 MCP 工具提供。本地审片页面用于观看素材和成片、批准版本、提交修改意见。视频生成使用火山引擎官方 Ark API；GPT 可选择 ChatGPT 订阅或 OpenAI API。即梦网页会员、现有即梦 CLI 登录与火山 API 权限互不等同。

本项目已包含真实 API 适配器和本地模拟测试。本机 ChatGPT 登录下的真实 GPT 规划调用已经验证；视频、图片、音乐和转写服务尚未进行付费实测。演示素材带有 `NORLAN DEMO / SYNTHETIC PLACEHOLDER` 标记，只用于检验流程；实际媒体权限、费用和生成质量需要首次小额实测。

## 安装

需要 macOS、Python 3.11 或更新版本、Git、FFmpeg（包含 ffprobe）。默认 GPT 方式还需要已安装并登录的 Codex CLI。

```bash
git clone https://github.com/jiantianshu2021/Norlan.git
cd Norlan
bash scripts/install.sh
norlan doctor
norlan
```

仓库是私有的，克隆需要具有访问权限的 GitHub 登录。安装脚本将入口放在 `~/.local/bin/norlan`；若终端找不到命令，可先使用该完整路径，或把 `~/.local/bin` 加入 `PATH`。程序需要在运行制作任务时保持启动，关闭终端后可以从项目目录继续。

直接输入 `norlan` 打开新的 GPT 对话，不会自动进入旧演示项目。你可以说“先聊聊我的故事，不要开始制作”，也可以贴入故事，或明确提供 TXT/MD 小说路径让它读取。对话可以持续修改创意；决定制作时，直接说“把我们刚才确定的故事创建成漫剧项目”。

```bash
norlan                    # 新的自由对话
norlan --resume           # 恢复最近的 Norlan 对话
norlan chat ./my-drama    # 带着指定制作项目开始对话
norlan setup              # 选择或切换 GPT 连接方式
norlan connection         # 查看当前连接选择
```

原生对话里的 `/new` 是新对话，不是旧控制台的导入小说命令。导入小说直接使用自然语言即可。退出后制作文件与历史仍保留；重新打开可让 Norlan 查看项目并继续。

### 选择 GPT 连接

运行 `norlan setup` 后选择：

- **ChatGPT 订阅**：使用现有 ChatGPT 登录与账户额度，默认保留当前的订阅方式和模型选择。
- **OpenAI API**：填写自己账户可用的模型 ID，在本机终端隐藏输入 API Key；也可从终端的 `OPENAI_API_KEY` 环境变量读取。认证由官方 Codex CLI 保存到 Norlan 的独立 API 登录目录，密钥不会写入项目或仓库。

连接选择会保存，订阅和 API 的登录目录、会话各自独立；切换不会复制两边的登录凭据或静默改用另一个服务。自由对话的 API 费用由 OpenAI 账户单独计费，**不受漫剧项目预算控制**。制作流程内部的 GPT API 调用必须设置项目预算和 `caps.llm`，与媒体调用一起记账。账户级消费限制应在 API 平台设置。

Norlan 的模型只可使用专用制作工具，不能替你批准资产/成片、提高预算或直接改写项目状态。需要审批时，它打开本机审片页供你操作。预算配置仍由你在终端执行 `norlan configure`。

Norlan 会比较 `PATH` 中的 Codex 与 ChatGPT/Codex 桌面应用内置 CLI，自动使用其中较新的已安装版本，不替换系统安装，也不自动更换你的模型。需要固定某个 CLI 时，可设置 `NORLAN_CODEX_PATH` 或 `brain.codex_command`。

## 先走一遍无付费演示

```bash
norlan demo ./demo-project
norlan console ./demo-project
```

演示创建两集占位分镜，保留规划、资产和逐集审批。`console` 是显式保留的旧流程控制台，演示时不调用 GPT 或付费媒体服务；`norlan` / `norlan chat` 始终是实际 GPT 对话入口。可在另一个终端打开审片页面：

```bash
norlan studio ./demo-project
```

根据终端显示的项目状态和版本操作。批准并不会跳过其余审批；继续制作会推进到下一个需要你决定的位置。

## 用小说开始制作

```bash
norlan new ./my-drama --novel ./novel.txt --mode live
norlan chat ./my-drama
```

可在创建时加 `--brief "竖屏国风漫画，每集约一分钟"` 表达创作要求；分集方案仍需要你批准。

小说使用 UTF-8 文本。GPT 规划包括分集、每镜时长、对白、角色和场景资产、声音参考与配乐描述。超出一次输入范围时程序会停止并提示拆分，不会静默截掉小说后半段。

在 GPT 对话中可以查看当前状态、继续制作、提出返工意见、暂停或恢复。人工审批通过审片页或另一个终端进行，例如批准界面上显示的第 1 版规划：

```bash
norlan approve ./my-drama plan plan --version 1
norlan run ./my-drama
```

审批对象可以是 `plan`、`asset`、`episode` 或需要人工复核的 `shot`。ID 和版本必须来自当前项目；旧页面上的批准不会覆盖新版本。审片建议指出镜头 ID、时间位置与具体问题，例如“E001-S002 的人物转身后服装变了，请保留其他镜头”。

```bash
norlan status ./my-drama
norlan preview ./my-drama episode E001
norlan reject ./my-drama episode E001 --version 1 --shots E001-S002 --feedback "转身后服装变化，请修复"
norlan run ./my-drama
```

## 配置账户与预算

复制 [配置示例](examples/settings.example.json)，填写已开通的图片、视频模型 ID 和你核实的保守调用成本上限。示例中的空模型与零成本刻意使付费任务无法启动。

```bash
cp examples/settings.example.json ./settings.json
# 编辑 settings.json 后，在当前终端配置 ARK_API_KEY。
norlan configure ./my-drama --file ./settings.json --budget 30
```

上面 `30` 只是你自行批准的项目预算示例，默认单位为人民币，**不是平台价格报价**。不要原样接受它；按实际模型、分辨率、时长、参考素材与重试需求填写。

| 配置 | 用途 |
| --- | --- |
| `brain.provider: "codex"` | 跟随 `norlan setup` 选定的订阅/API 登录。订阅记录调用次数；API 还须设置 `caps.llm` 和项目预算。 |
| `brain.provider: "openai"` | 可选的直接 Responses API 适配器；需 `OPENAI_API_KEY` 和明确的 `brain.model`。若当前选择订阅，程序会提示连接冲突并停止，不会静默付费。通常保留 `codex` 并用 `norlan setup` 切换即可。 |
| `ark.image_model` / `ark.video_model` | 填写你在火山 Ark 实际开通的模型或接入点 ID，不由程序猜测。 |
| `caps.image` | 每次图片生成的保守费用上限。 |
| `caps.video_second` | 每秒视频生成的保守费用上限；也用于声音试片。 |
| `caps.llm` | 每次 GPT API 调用的保守费用上限；Codex 订阅方式可设为 0。 |
| `max_attempts` / `max_llm_calls` | 每项资产或镜头的累计生成尝试上限（包括人工返工），以及项目 GPT 调用次数上限。 |
| `creative_brief` | 额外创作要求，用于下一次规划；不会自行改写已批准方案。 |

密钥只通过终端环境变量读取，不要放入小说、提示词、配置文件或 Git 仓库。ChatGPT 订阅不包含 OpenAI API 和火山媒体服务费用；即梦网页登录也不能代替 `ARK_API_KEY`。

每次付费提交前，程序先写入预算预留。超时或进程中断导致结果不明时保留额度并停止新提交，不把“没有收到结果”当成“没有扣费”。预算是基于你填写单价的本地控制，不能代替供应商的实时账单和账户消费限额；已经提交的远程任务也不会因本地暂停而自动取消。

## 配乐与字幕

配乐在策划阶段考虑，在整集时间线上统一混音。默认 `music.mode` 为 `none`；设为 `file` 并提供本地 `music.file` 可导入已有配乐。设为 `volc` 可接入火山纯音乐服务，需要单独开通权限、设置 `VOLC_ACCESS_KEY_ID` / `VOLC_SECRET_ACCESS_KEY`，并填写 `music.call_cap`。

火山音乐适配器每次生成 30–120 秒纯音乐。音乐连续播放、淡入淡出，并按原声音量自动压低；不会把短音乐随意循环。台词与 BGM 已经混为一条音轨时，FFmpeg 无法无损分离。详见 [声音与配乐说明](docs/audio.md)。

字幕采用最终音频的真实时间戳，不能把分镜中的计划对白直接当作已核实字幕。默认关闭自动转写；将 `subtitles.mode` 设为 `openai` 并填写 `subtitles.call_cap`，可使用 OpenAI 对实际成片音轨转写。它需要单独的 `OPENAI_API_KEY`，不能使用 ChatGPT 订阅代付；`call_cap` 为每集一次转写的保守费用上限。识别错误仍需在每集人工审片时指出。具体配置与支持范围见 [字幕说明](docs/subtitles.md)。

如果只需改字幕错别字，可在对话中说“将第一集字幕的张三改成张山，只修改字幕”。程序保留原镜头、音轨和字幕时间，只修改文字后重新合成，并要求审批新版本；修改真实对白或镜头后则重新建立字幕时间线。

## 当前检查能做什么

- 已批准的资产文件和声音参考会实际传给视频接口；同场景续镜使用前镜实际片尾附近画面，以及审查记录中的末态。
- 每镜生成后检查媒体参数、抽取首中尾画面，交给 GPT 做画面审查；每隔若干镜头生成粗剪并再次抽样检查。
- 明确失败的单镜可在预算与尝试上限内自动改写并重做。不确定、无法定位或达到上限时停止，交给你决定。
- 修改资产或镜头会让受影响的续镜及审批失效；旧文件与历史记录仍保留。

**抽帧审查不能证明整段动作、口型、音色或配乐衔接已经正确。** 当前 GPT 检查不会声称看完视频或听完音轨；每集人工完整观看与听审是正式交付环节。角色与声音参考增加生成条件，无法保证模型完全一致。

| 能力 | 当前验证范围 |
| --- | --- |
| 原生终端 GPT 自由对话、前文记忆、退出恢复及 MCP 工具调用 | 已使用本机 ChatGPT 订阅完成真实多轮测试，成功查询现有项目。 |
| 订阅/API 选择、独立认证目录及计费隔离 | 已通过离线测试与本机配置验证；用户 API Key 尚未配置，API 实际推理调用待验证。 |
| GPT 小说规划与结构化返回 | 已通过本机现有 ChatGPT 登录真实调用验证。 |
| 审批、预算、恢复、字幕缓存与局部返工 | 使用离线响应和真实状态转换测试。 |
| 抽帧、拼接、原声保留、字幕和混音 | 使用本机 FFmpeg 与合成素材进行实际测试。 |
| Ark 生成、火山配乐、OpenAI 转写 | 已实现官方接口适配和离线测试；真实账户付费链路待小额验收。 |
| 完整视频动作、音色、口型、音乐和字幕内容验收 | 由你逐集观看、听审并批准，未实现完全自动替代。 |

项目状态保存在制作目录的 `project.json`。`source/` 保存输入，`plans/` 保存规划，`media/` 保存素材，`evidence/` 保存检查证据，`exports/` 保存粗剪与各版成片。恢复制作时使用同一个目录，不要重新创建来绕过未知账单或审批。

保存规划失败时，Norlan 会向 GPT 返回出错的字段位置与修复建议，供其修改候选草案。声音资产必须绑定角色，引用声音的镜头也须引用对应角色；画外声源应在提示词中明确不出镜。关联校验不能保证生成模型不会画出画外人物，仍需审片确认。修正保存失败的规划不会自动批准或触发付费生成。升级后退出并运行 `norlan --resume`，即可在原会话中加载更新后的工具。

开发结构和状态约束见 [架构说明](docs/architecture.md)。运行测试：

```bash
python3 -m unittest discover -s tests -v
```

## 思路来源

制作流程参考了“从零开始做 AI 短剧”系列的 [第 1 篇](https://naiyous.com/12276.html)、[第 2 篇](https://naiyous.com/12298.html)、[第 3 篇](https://naiyous.com/12329.html)、[第 4 篇](https://naiyous.com/12363.html)、[第 5 篇](https://naiyous.com/12388.html)、[第 6 篇](https://naiyous.com/12450.html)。它们提供前期规划、资产、异步生成与拼接方面的思路；前期文本审稿不等于已生成视频的完整审片。本仓库代码独立编写，逐集人工审批、版本追踪和结果不明时的预算保留是 Norlan 的具体实现。
