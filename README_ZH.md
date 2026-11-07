# ComfyUI-SeeThrough-TileFlow

[English](README.md) · **中文**

> 分块驱动的 SeeThrough 角色拆层流水线：把高分辨率原图切成带重叠的块，逐块跑
> LayerDiff + Depth + 后处理，再把每块的分层结果拼回一张完整的分层 PSD。

界面与面板默认中文；英文界面由 `locales/en` 与 `web/stlu_i18n.js` 提供。

---

## 特性

- **只跑勾选的块**：未勾选的格子不会进入 GPU，已生成过的格子直接复用。
- **顺序引导回填**：每块生成前，把已经完成的相邻块在重叠区的像素填回去再送进
  模型（`STLU_GuidedLayers` / `STLU_SeqTileLoop`），减轻拼缝。
- **固定输出目录 + 参数指纹**：参数变了才清空重跑；只改合并选项时只重新合并。
- **分层 PSD + 逐块清单**：每块存独立分层 PSD，全部完成后写 `manifest.json` /
  `images.json` / `generation.json`，并合并出整图分层 PSD。
- **两套工作流**：一套四步流水线（节点最少），一套「透明版」把每个阶段摊成子图，
  内部每一步都可见，两套共用同一套底层实现。
- **中英双语面板**：界面默认中文，`Comfy.Locale` 为英文时由 `locales/en` 与
  `web/stlu_i18n.js` 提供英文显示（不改写任何存档值）。

## 环境要求

- ComfyUI（Python ≥ 3.10）。
- ComfyUI-SeeThrough 自定义节点及其 LayerDiff / Depth 模型——工作流里的
  `SeeThrough_LoadLayerDiffModel`、`SeeThrough_GenerateLayers`、
  `SeeThrough_LoadDepthModel`、`SeeThrough_GenerateDepth`、`SeeThrough_PostProcess`
  都来自它。
- ComfyUI 自带的 `torch` / `numpy` / `Pillow` / `OpenCV` / `SciPy`。
- 可选：`opencv-contrib-python`。4K 掩膜节点会优先用 `cv2.ximgproc.guidedFilter`，
  没装时自动退化为双线性插值；边缘吸附需要 `scipy`。
- 本包没有额外的 pip 依赖。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Sirokus/ComfyUI-SeeThrough-TileFlow.git
```

重启 ComfyUI。新节点出现在 **SeeThrough/LayerUpscale** 分类下，旧版节点在
**SeeThrough/LayerUpscale/Legacy**。

## 工作流

| 文件 | 说明 |
|---|---|
| `workflow/seethrough-tile-flow-simple.json` | 四步流水线：`STLU_SimpleInput` → SeeThrough 自动拆层 → `STLU_TilePSD` / `STLU_WaveIndex` → `STLU_TileMerge` → `STLU_PSDPreview`。节点最少，适合直接用。 |
| `workflow/seethrough-tile-flow.json` | 透明版：① 计划与切块、② 顺序拆层、③ 存盘、④ 合并各是一个子图，⑤ 预览单独显示；每个内部步骤都是可见节点。 |

两套工作流读同一份 `settings` 参数、写同一种输出目录，结果可以互相接手。

## 快速开始

1. 打开上面的工作流，在 `LoadImage` 里选原图。
2. 在 **① 用户输入** 面板里：
   - 设置块宽 / 块高 / 重叠（默认 1024 / 1024 / 128）、拆层分辨率与步数（默认 512 / 30）；
   - 在格子面板里勾选要跑的块（不勾不跑）；
   - 选输出根目录；每按一次 Queue 是否新建波次文件夹由设置决定；
   - 选择合并选项（合并名、同名图层是否单独成组、是否导出 depth、是否放入原图）。
3. 按 **Queue**，② 拆层 → ③ 存 PSD / 写清单 → ④ 合并自动接力。
4. 结果在 `output/<输出目录>/`；⑤ 预览面板给出全身总览，以及按块 / 按部位的分组。

## 流水线

| 阶段 | 做什么 | 简化版节点 | 透明版节点 |
|---|---|---|---|
| ① 计划与切块 | 规划网格、选块、上下文外扩、写入参数指纹、裁剪 | `STLU_SimpleInput` / `STLU_PipelineInput` | `STLU_PipelineSettings` → `STLU_PlanGrid` → `STLU_PickTiles` → `STLU_PlanContext` → `STLU_PrepareRun` → `STLU_CropTiles` |
| ② 顺序拆层 | 逐块运行 LayerDiff → Depth → PostProcess，重叠区回填引导 | `STLU_GuidedLayers` | `STLU_SeqTileLoop`（唯一必须留在 Python 里的循环），`STLU_GuideTile` / `STLU_CropBack` 为调试用单块版本 |
| ③ 存盘 | 恢复锁定部件、逐块写分层 PSD 与分层 PNG、存原图、汇总清单 | `STLU_TilePSD` + `STLU_WaveIndex` | `STLU_RestoreLocks` → `STLU_WriteTilePSD` / `STLU_WriteTileImages` → `STLU_SaveOriginal` → `STLU_WriteManifest` |
| ④ 合并 | 接缝归属、缩放、整部件例外、分层组装 | `STLU_TileMerge` | `STLU_MergeTiles`（整块算法，保留为单节点） |
| ⑤ 预览 | 合并 PSD 总览、按块 / 按部位分组 | `STLU_PSDPreview` | 同左，额外接收 ② 的 `guided_inputs` 与 `indices`，显示每块实际喂给模型的输入 |

只在 **①** 上操作；②③④ 的参数 widget 会被锁定，仅显示进度和结果。

## 常用参数

| 参数 | 默认 | 作用 |
|---|---|---|
| 块宽 / 块高 / 重叠 | 1024 / 1024 / 128 | 切格大小与重叠像素；重叠供引导回填使用 |
| 分辨率 / 步数 | 512 / 30 | 单块拆层的分辨率与推理步数 |
| 深度分辨率 | -1 | `-1` 表示跟随拆层分辨率 |
| 种子 | 0 | 点「随机」或手填 |
| 后处理 | — | 左右拆成两半 / LaMa 去背景 |
| 块外上下文 | 0 | 每块额外向外多裁的像素，0 关闭；用于让块内看到更多身体 |
| 透明阈值 | 8 | 高级项，判定透明像素的 alpha 门限 |
| 合并名 / 同名图层单独成组 / depth / 原图 | — | 只影响合并，不触发重新生成 |

## 输出结构

```text
<output>/<输出目录>/[<分类>/]
├─ generation.json         本次运行的参数与指纹
├─ manifest.json           逐块清单（新生成 / 复用 / 恢复的块）
├─ images/
│  ├─ images.json          分层 PNG 清单
│  └─ ...
├─ original.png            原图（如选择保留）
├─ tile_00_r0_c0.psd       每块的分层 PSD
├─ tile_00_r0_c0_preview.png
├─ <合并名>.psd            合并后的整图分层 PSD
└─ <合并名>_depth.psd      可选的独立 depth PSD（永远不混进彩色 PSD）
```

## 缓存、复用与重跑

- 会改变像素的参数（原图、块尺寸 / 重叠、分辨率、步数、深度分辨率、种子、左右
  拆分、LaMa、透明阈值、拆层 / 深度模型）一旦变化，Queue 时会自动清空该输出目录
  并重跑全部块，`generation.json` 记录原因。
- 只改合并选项（同名图层单独成组、是否导出 depth、是否放原图、接缝羽化）不会重跑
  块，只重新合并。
- 同一目录里已经生成的块会自动复用：只勾选要重跑的块即可，未勾选的块仍参与合并。
- 换了原图（即使尺寸相同）不会误用旧块——清单里带源图指纹。

## 节点总览

主流水线之外，包内还提供其他工作流用的节点（分类 **SeeThrough/LayerUpscale**）：

- `STLU_Plan` / `STLU_UpscaleBatch`：按计划批量放大 SeeThrough 图层并组装 PSD。
- `STLU_Mask4K` / `STLU_Mask4KBatch`：纯 CPU 的 4K 部件遮罩（引导滤波 + 边缘吸附）。
- `STLU_EditPartsBatch`：图编辑模型批量抠取 Live2D 部件。
- `STLU_RebuildPSD`：从已导出的分层图片重建合并 PSD（CPU）。
- Legacy 分类下的旧节点（`STLU_LoadLayer`、`STLU_ContextBackdrop`、
  `STLU_FocusCrop`、`STLU_PasteBack`、`STLU_DetailBatch`、`STLU_AssemblePSD`、
  `STLU_ChromaToAlpha`、`STLU_PartsPSD`、`STLU_TileGrid` 等）保持可用，
  但不再出现在新流水线里。

## 工具

```bash
# 构建 / 修复透明版工作流（保留用户调过的 widget 值，可重复执行）
python tools/build_transparent_workflow.py <workflow_dir>
python tools/build_transparent_workflow.py <workflow_dir> --force
python tools/build_transparent_workflow.py <workflow_dir> --validate

# 其他工作流生成器
python tools/build_batch_workflow.py
python tools/build_detail_workflow.py
python tools/build_edit_workflow.py
python tools/build_mask4k_workflow.py
python tools/validate_batch_workflow.py <workflow.json>
```

## 测试

```bash
python tests/run_all.py                 # 全部离线测试
python tests/run_all.py tile mask4k     # 只跑文件名匹配的测试
```

全部测试只跑 CPU，写入临时目录，不需要启动 ComfyUI，也不会加载模型。

## 前端与 HTTP 接口

- `web/stlu.js`：① 输入面板、格子勾选、③ 进度、⑤ PSD 预览、锁定部件面板等。
- `web/stlu_i18n.js` + `locales/en/`：英文界面。
- `routes.py` 向 ComfyUI 的 prompt server 注册 `/stlu/*` 接口：`ls`、`layers`、
  `thumb`、`preview`、`reveal`、`open`、`wf`、`imgsize`、`psd_preview`、`locks`、
  `tiles/*`（grids、grid、images、from_images、dirs、manifest、clear、merge、
  merge-mixed、strips）、`edit_parts`。

## 许可

Apache License 2.0，见 [LICENSE](LICENSE)。

拆层与深度生成基于 ComfyUI-SeeThrough；本仓库只负责分块调度、分层 PSD 的读写与合并。
