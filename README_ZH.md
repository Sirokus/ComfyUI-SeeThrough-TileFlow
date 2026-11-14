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
- **固定输出目录 + 参数指纹**：参数变了只提示、不自动删结果；清空永远是手动操作。
- **分层 PSD + 逐块清单**：每块存独立分层 PSD，全部完成后写 `manifest.json` /
  `images.json` / `generation.json`，并合并出整图分层 PSD。
- **一套分块工作流**：规划切块、顺序拆层、存盘、合并、预览各是一个可见节点，
  每个节点职责单一。
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
| `workflow/seethrough-tile-flow.json` | 分块流水线：① `STLU_PipelineInput` 规划、切块、算指纹 → ② `STLU_GuidedLayers` 逐块带跨块引导拆层 → ③ 每块存 PSD 与清单 → ④ `STLU_TileMerge` 合并 → ⑤ `STLU_PSDPreview` 预览。只跑勾选的块，已生成的块自动复用。 |
| `workflow/seethrough-psd-edit.json` | PSD 图层编辑：① `STLU_EditPrepare` 选成品 PSD、选要重画的图层并给每层写编辑指令（可附参考图）→ ② `STLU_LayerEdit` 按任务克隆 Qwen 编辑图逐层重画 → ③ `STLU_EditAssemble` 按原图层顺序拼回分层 PSD。 |
| `workflow/seethrough-psd-upscale.json` | PSD 逐层放大（精简版）：① `STLU_QwenEditLoader` 一个节点加载 UNET / CLIP / VAE → ② `STLU_PsdUpscalePrepare` 选 PSD、圈定图层、设置放大倍数与可选的高分辨率参考图 → ③ `STLU_PsdUpscaleMin` 逐层放大并重建 PSD。 |

分块工作流依赖本包 + ComfyUI-SeeThrough 节点及其 LayerDiff / Depth 模型；两个
PSD 工作流依赖示例里模型节点加载的 Qwen-Image-2.1，以及作为编辑子图的
`templates/edit/qwen_image_edit_2.1.json`。

## 快速开始

1. 打开上面的工作流，在 `LoadImage` 里选原图。
2. 在 **① 用户输入** 面板里：
   - 设置块宽 / 块高 / 重叠（默认 1024 / 1024 / 128）、拆层分辨率与步数（默认 512 / 30）；
   - 在格子面板里勾选要跑的块（不勾不跑）；
   - 选输出根目录；每按一次 Queue 是否新建波次文件夹由设置决定；
   - 选择合并选项（合并名、同名图层是否单独成组、是否导出 depth、是否放入原图）。
3. 按 **Queue**，② 拆层 → ③ 存 PSD / 写清单 → ④ 合并自动接力。
4. 结果在 `output/<输出目录>/`；⑤ 预览面板给出全身总览，以及按块 / 按部位的分组。

以上步骤是分块工作流；两个 PSD 工作流的用法见各自画布上的说明节点。

## 流水线

| 阶段 | 做什么 | 流水线节点 | 可选的拆分单步节点 |
|---|---|---|---|
| ① 计划与切块 | 规划网格、选块、上下文外扩、写入参数指纹、裁剪 | `STLU_PipelineInput` | `STLU_PipelineSettings` → `STLU_PlanGrid` → `STLU_PickTiles` → `STLU_PlanContext` → `STLU_PrepareRun` → `STLU_CropTiles` |
| ② 顺序拆层 | 逐块运行 LayerDiff → Depth → PostProcess，重叠区回填引导 | `STLU_GuidedLayers` | `STLU_SeqTileLoop`（唯一必须留在 Python 里的循环），`STLU_GuideTile` / `STLU_CropBack` 为调试用单块版本 |
| ③ 存盘 | 逐块写分层 PSD 与分层 PNG、存原图、汇总清单 | `STLU_TilePSD` + `STLU_WaveIndex` | `STLU_WriteTilePSD` / `STLU_WriteTileImages` → `STLU_SaveOriginal` → `STLU_WriteManifest` |
| ④ 合并 | 接缝归属、缩放、整部件例外、分层组装 | `STLU_TileMerge` | `STLU_MergeTiles`（整块算法，保留为单节点） |
| ⑤ 预览 | 合并 PSD 总览、按块 / 按部位分组 | `STLU_PSDPreview` | 同左，直接读取 ④ 的 `psd_path` |

只在 **①** 上操作；②③④ 的参数 widget 会被锁定，仅显示进度和结果。

随附的工作流用的是「流水线节点」一列。第三列是同一套实现拆成的单职责节点，
方便调试或自己搭图，但不再随附示例工作流。

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
  拆分、LaMa、透明阈值、拆层 / 深度模型）发生变化时，`generation.json` 会记录并在
  界面提示，但**不会删除任何结果**：Queue 只重跑你勾选的块并覆盖它们，其余块和
  合并图都保留。想整体重来请手动点「清空此目录」。
- 只改合并选项（同名图层单独成组、是否导出 depth、是否放原图、接缝羽化）不会重跑
  块，只重新合并。
- 同一目录里已经生成的块会自动复用：只勾选要重跑的块即可，未勾选的块仍参与合并。
- 换了原图（即使尺寸相同）不会误用旧块——清单里带源图指纹；旧块会留在目录里，
  需要时手动清空。

## 节点总览

主流水线之外，包内还提供其他工作流用的节点（分类 **SeeThrough/LayerUpscale**）：

- `STLU_Mask4K` / `STLU_Mask4KBatch`：纯 CPU 的 4K 部件遮罩（引导滤波 + 边缘吸附）。
- `STLU_EditPartsBatch`：图编辑模型批量抠取 Live2D 部件。
- `STLU_RebuildPSD`：从已导出的分层图片重建合并 PSD（CPU）。
- Legacy 分类下的旧节点（`STLU_LoadLayer`、`STLU_ContextBackdrop`、
  `STLU_FocusCrop`、`STLU_PasteBack`、`STLU_DetailBatch`、
  `STLU_PartsPSD`、`STLU_TileGrid` 等）保持可用，
  但不再出现在新流水线里。

## 工具

```bash
# 构建 / 修复「逐步骤拆分版」工作流（每个内部步骤一个可见节点；不是默认示例）
# 保留用户调过的 widget 值，可重复执行
python tools/build_transparent_workflow.py <workflow_dir>
python tools/build_transparent_workflow.py <workflow_dir> --force
python tools/build_transparent_workflow.py <workflow_dir> --validate

# 其他工作流生成器
python tools/build_edit_workflow.py
```

## 测试

```bash
python tests/run_all.py                 # 全部离线测试
python tests/run_all.py tile mask4k     # 只跑文件名匹配的测试
```

全部测试只跑 CPU，写入临时目录，不需要启动 ComfyUI，也不会加载模型。

## 前端与 HTTP 接口

- `web/stlu.js`：① 输入面板、格子勾选、③ 进度、⑤ PSD 预览等。
- `web/stlu_i18n.js` + `locales/en/`：英文界面。
- `routes.py` 向 ComfyUI 的 prompt server 注册 `/stlu/*` 接口：`ls`、`layers`、
  `thumb`、`preview`、`reveal`、`open`、`wf`、`imgsize`、`psd_preview`、
  `tiles/*`（grids、grid、images、from_images、dirs、manifest、clear、merge、
  merge-mixed、strips）、`edit_parts`。

## 许可

Apache License 2.0，见 [LICENSE](LICENSE)。

拆层与深度生成基于 ComfyUI-SeeThrough；本仓库只负责分块调度、分层 PSD 的读写与合并。
