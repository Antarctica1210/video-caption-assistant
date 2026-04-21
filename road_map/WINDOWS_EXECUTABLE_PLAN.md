# Windows 可执行版本升级建议

## 目标定义

当前项目已经具备可用的字幕流水线，但离 Windows 可执行产品还有几步关键升级：

1. 输入输出从 `MinIO bucket` 改为本地文件选择与本地导出
2. 翻译从 `LM Studio API` 改为真正的本地离线模型调用
3. 保留 `transcribe` 与 `translate` 全离线能力
4. 最终交付为 Windows 用户可双击运行的 `.exe`

你的现有仓库已经有这些基础：

- `Python + Typer` CLI 入口
- `LangGraph` 编排转写与翻译流程
- `faster-whisper` 本地转写
- SRT / ASS 导出能力

所以最现实的方向不是推倒重写，而是把当前 Python 核心流程产品化。

---

## 推荐原则

这个项目的升级建议优先遵守以下原则：

- `transcribe` 必须使用离线 ASR 模型
- `translate` 必须使用离线翻译模型
- Windows 端尽量不要求用户额外安装 Python
- 尽量减少运行时依赖外部服务
- 第一版优先“稳定能用”，而不是一开始追求最复杂的界面

---

## 可选技术栈

## 方案 A：`Python + PySide6 + PyInstaller`

### 组成

- UI: `PySide6`
- 核心流程: 继续沿用现有 `Python` 代码
- 打包: `PyInstaller`
- 转写: `faster-whisper` / `ctranslate2`
- 翻译: `CTranslate2` 版 `NLLB` 或 `M2M100`

### 优点

- 和现有仓库最兼容，迁移成本最低
- 你现有的转写、字幕导出、流程编排代码可大量复用
- Python 生态里接 `faster-whisper`、`sentencepiece`、`ctranslate2` 都比较直接
- 最快能做出第一版 Windows 桌面产品

### 缺点

- 打包后体积通常较大
- Windows 依赖管理要认真处理，尤其是 `ffmpeg`、模型文件、CUDA/CPU 兼容
- UI 质感通常不如 Web 技术栈灵活

### 适用判断

如果你的目标是“尽快把当前项目做成可交付的 Windows 桌面工具”，这是最推荐的主路线。

---

## 方案 B：`Python Backend + Tauri Desktop`

### 组成

- UI: `Tauri + HTML/CSS/TypeScript`
- 核心流程: Python 子进程或本地服务
- 打包: `Tauri`
- 转写: `faster-whisper`
- 翻译: `NLLB/M2M100` 离线模型

### 优点

- UI 更容易做得现代
- 桌面壳体更轻
- 后续如果要加任务列表、设置页、进度可视化，会比纯 Qt 灵活

### 缺点

- 架构复杂度更高，需要维护前后端通信
- 打包链路比纯 Python 更复杂
- 对当前仓库的直接复用度低于方案 A

### 适用判断

如果你未来想把它做成一个体验更像成熟商业软件的桌面工具，方案 B 值得考虑；但不建议作为第一阶段落地方案。

---

## 方案 C：`Rust / C++ 重写核心 + 原生桌面应用`

### 组成

- UI: `Tauri`、`Qt` 或原生框架
- 核心推理与流程: `Rust` 或 `C++`
- 模型推理: ONNX Runtime / CTranslate2 / whisper.cpp

### 优点

- 性能和部署可控性最好
- 最适合长期产品化
- 更容易做成真正稳定的单机软件

### 缺点

- 重写成本最高
- 研发周期明显更长
- 你当前 Python 代码复用价值会下降

### 适用判断

这更像第二代或第三代架构，不适合作为你现在的主升级路线。

---

## 离线模型选型建议

## 1. Transcribe 离线方案

### 推荐

- 首选：`faster-whisper`
- 备选：`whisper.cpp`

### 原因

- 你当前已经在用 `faster-whisper`
- 迁移到 Windows 成本最低
- CPU/GPU 双路线都相对成熟
- 输出时间轴和分段能力适合字幕生产

### 建议模型分层

- 高质量：`large-v3`
- 均衡速度：`large-v3-turbo`
- 低配置兼容：`medium` 或 `small`

### 实施建议

- 第一版保留你现有的 `faster-whisper`
- 在设置页提供 `模型大小`、`CPU/GPU`、`快速模式` 选项
- 模型缓存目录改为应用本地目录，例如 `models/whisper/`

---

## 2. Translate 离线方案

当前仓库的翻译层通过 `LM Studio` 走本地 OpenAI 兼容接口，这严格说仍然是“本地服务依赖”，不适合打包成即开即用的桌面软件。

### 推荐优先级

1. 首选：`NLLB-200 distilled 600M` + `CTranslate2`
2. 次选：`M2M100 418M`
3. 不建议第一版使用本地通用大语言模型做翻译主链路

### 为什么推荐 NLLB / M2M100

- 这是更纯粹的离线机器翻译路线
- 输出结构更稳定，适合批量字幕翻译
- 不需要像 LLM 一样频繁处理 JSON 偏航、漏行、幻觉
- 更容易打包成“确定性工具”

### 为什么不建议第一版继续走本地 LLM

- 需要额外模型服务管理
- JSON 输出稳定性仍需要大量兜底
- 资源占用更高
- 对普通 Windows 用户来说部署复杂

### 语言方向建议

如果你的主要目标是：

- `日语/英语 -> 中文`
- `英语 -> 多语`

那么 `NLLB-200 distilled 600M` 是较好的第一版平衡点。

### 翻译层落地建议

- 把当前 `LMStudioClient` 抽象成通用接口，例如 `TranslationBackend`
- 提供两个实现：
  - `OfflineNLLBTranslator`
  - `LMStudioTranslator`（仅开发期保留，可后续移除）
- 在图流程中把翻译节点改成直接调用本地翻译后端，而不是 HTTP API

---

## 最推荐的总体技术路线

## 第一推荐

`Python + PySide6 + PyInstaller + faster-whisper + NLLB(CTranslate2)`

这是最适合你当前仓库的方案，原因是：

- 现有 Python 代码能最大程度保留
- `transcribe` 和 `translate` 都能做到真正离线
- 可以先做可用版本，再逐步优化 UI 和性能
- 技术风险比 Tauri 或重写方案低很多

---

## 建议的产品架构

建议把项目从“脚本流水线”重构成以下分层：

### 1. Core 核心层

负责纯业务能力：

- 视频读取
- 音频抽取
- 分段转写
- 文本规范化
- 离线翻译
- 字幕导出

这一层尽量不依赖 UI。

### 2. Service 服务层

负责统一调度：

- 任务创建
- 进度状态
- 错误恢复
- 模型加载
- 临时文件管理

### 3. Desktop UI 层

负责交互：

- 选择视频文件
- 选择输出目录
- 选择源语言 / 目标语言
- 选择转写模型与翻译模型
- 显示进度条和日志
- 打开输出文件夹

---

## 需要替换或重构的现有部分

## 1. 去掉 MinIO 依赖

当前产品逻辑里 `MinIO` 更像开发环境中的对象存储中转站，但桌面软件不需要它。

建议改为：

- 输入：用户选择本地视频文件
- 输出：直接保存到用户选定目录
- 缓存：本地 `temp` 目录

### 替换方向

- 删除 `input_bucket` / `output_bucket` 思维
- 用 `Path` 和本地文件系统代替上传下载节点
- 把 `uploader` 改成 `local_exporter`

---

## 2. 去掉 LM Studio 依赖

建议把：

- `clients/lm_studio.py`
- `translate_title`
- `translate_segments`

改造成面向本地模型推理的结构。

### 重构方向

- 新建 `clients/translators/` 目录
- 定义统一接口，例如：
  - `translate_text(text, source_lang, target_lang)`
  - `translate_batch(items, source_lang, target_lang)`
- 用离线模型实现主翻译后端

---

## 3. CLI 保留，但不再是唯一入口

当前 `Typer` CLI 可以继续保留，原因是：

- 便于调试
- 便于批处理
- 便于自动化测试

但面向 Windows 用户时，主要入口应改为桌面 UI。

---

## 可行开发路线

## 阶段 1：把现有流程改造成纯本地桌面友好架构

### 目标

先不做 UI，先把“服务依赖版流水线”改成“本地文件版流水线”。

### 任务

1. 去掉 `MinIO` 输入输出，改为本地文件路径
2. 抽象翻译接口，解除 `LM Studio` 绑定
3. 接入 `NLLB` 或 `M2M100` 的本地离线翻译实现
4. 统一模型目录、缓存目录、输出目录结构
5. 保留 CLI 作为调试入口

### 阶段产物

- 能通过命令行处理本地视频文件
- 不依赖 MinIO
- 不依赖 LM Studio
- `transcribe` / `translate` 都完全离线

---

## 阶段 2：补齐 Windows 打包能力

### 目标

把命令行版本做成能在 Windows 上直接运行的应用。

### 任务

1. 引入 `PyInstaller`
2. 处理 `ffmpeg`、模型目录、动态库打包
3. 设计首次启动的模型检查与下载提示
4. 区分 CPU 版与 GPU 版打包策略
5. 输出可分发目录或安装包

### 阶段产物

- 可执行文件 `exe`
- 首次运行可初始化模型目录
- 能在无 Python 环境的 Windows 机器启动

### 现实建议

第一版建议优先做：

- `CPU-only` 稳定版

之后再做：

- `NVIDIA GPU` 加速版

因为 GPU 版会引入更多 CUDA 兼容问题。

---

## 阶段 3：加入桌面界面

### 目标

让非技术用户可直接使用。

### 建议界面功能

- 文件选择
- 输出目录选择
- 源语言选择
- 目标语言选择
- 转写模型选择
- 翻译模型选择
- 字幕格式选择
- 进度显示
- 日志面板
- 打开导出目录

### UI 技术建议

- 第一版使用 `PySide6`
- 不建议一开始就为了 UI 美观切换到更复杂栈

---

## 阶段 4：增强产品化能力

### 可以继续补的能力

- 批量处理多个视频
- 字幕预览与人工微调
- 术语表 / 专有名词替换
- 失败任务断点恢复
- 模型管理器
- 自动检测硬件并推荐模型

---

## 模型与部署策略建议

## 1. 模型不建议直接塞进主程序包

原因：

- 安装包会非常大
- 后续更新困难
- CPU/GPU 版本差异大

### 更合理的做法

- 主程序单独打包
- 模型首次启动下载或单独提供模型包
- 模型目录放到：
  - 应用安装目录下的 `models/`
  - 或用户数据目录下的 `models/`

---

## 2. 建议准备两种发行版本

### 版本 A：轻量安装包

- 不内置大模型
- 首次启动检查并引导下载模型

### 版本 B：离线完整包

- 适合没有网络环境的机器
- 主程序 + 模型一起分发

---

## 风险点

## 1. 翻译模型质量与速度平衡

离线翻译模型通常比云端大模型更稳定，但在自然度上可能略弱，需要你接受“工程稳定性优先”。

## 2. Windows 打包体积会很大

尤其是同时包含：

- Python runtime
- ffmpeg
- whisper 模型
- 翻译模型

最终体积可能达到数 GB。

## 3. GPU 兼容复杂度高

如果用户机器环境不统一，GPU 版问题会比 CPU 版多很多，所以建议 CPU 版先行。

---

## 最终建议结论

如果目标是“尽快把现有项目升级为可交付的 Windows 可执行离线字幕工具”，建议采用：

`Python + PySide6 + PyInstaller + faster-whisper + NLLB(CTranslate2)`

并按下面的顺序推进：

1. 先移除 `MinIO`
2. 再移除 `LM Studio`
3. 完成纯本地 CLI 版本
4. 再做 Windows 打包
5. 最后补桌面 UI

这个顺序最稳，因为它优先解决“架构依赖是否正确”，再解决“产品形态是否美观”。

---

## 对当前仓库的具体落地方向

结合你现在的代码结构，建议后续优先改这些模块：

- `main.py`
  - 从“扫描 MinIO bucket”改为“接收本地文件路径”
- `src/video_caption/graph.py`
  - 去掉与上传下载绑定的节点
- `src/video_caption/clients/lm_studio.py`
  - 替换为离线翻译后端
- `src/video_caption/nodes/translator.py`
  - 改为直接调用本地翻译模型
- `src/video_caption/nodes/uploader.py`
  - 替换为本地导出逻辑
- `src/video_caption/config.py`
  - 增加桌面应用所需的模型路径、缓存路径、输出路径配置

---

## 下一步建议

如果继续往下做，最值得优先推进的不是 UI，而是下面这个里程碑：

### 里程碑

把当前项目改造成：

`输入本地视频 -> 本地转写 -> 本地翻译 -> 本地导出字幕`

只要这个里程碑完成，后面的 `.exe` 打包和桌面 UI 都会顺很多。
