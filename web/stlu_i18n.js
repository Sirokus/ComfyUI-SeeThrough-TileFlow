/**
 * English localization for the SeeThrough layer-upscale panels.
 *
 * The node backend and every panel speak Chinese; the stored values (enum
 * strings, widget text, paths) must keep their Chinese spelling because they
 * are serialized into workflows and matched by Python.  So nothing here
 * rewrites a value -- it only rewrites what the user *sees*: DOM text nodes,
 * title/placeholder/aria-label attributes and combo dropdown labels.
 *
 * Chinese locales are left completely untouched.
 *
 * Loaded as a plain side-effect module; exposes ``window.STLU_I18N``.
 */
(function () {
  "use strict";

  function detectLocale() {
    try {
      const app = window.comfyAPI && window.comfyAPI.app && window.comfyAPI.app.app;
      const settings = app && app.ui && app.ui.settings;
      const value = settings && settings.getSettingValue
        ? settings.getSettingValue("Comfy.Locale")
        : null;
      if (value) return String(value);
    } catch (e) { /* app not ready */ }
    try {
      const stored = window.localStorage.getItem("Comfy.Locale");
      if (stored) return stored;
    } catch (e) { /* storage blocked */ }
    const lang = (document.documentElement.getAttribute("lang") || "").trim();
    if (lang) return lang;
    return navigator.language || "";
  }

  function isChinese(locale) {
    return /^zh/i.test(String(locale || "").trim());
  }

  // Resolved once: a locale switch reloads the page (node defs are fetched at
  // startup), so there is no stale state to worry about.
  const ENABLED = !isChinese(detectLocale());
  const CJK = /[　-〿㐀-鿿豈-﫿＀-￯]/;

  // ---------------------------------------------------------------------------
  // structural rules: strings the panels build from numbers, paths and names.
  // Anchored so they only fire on a whole text node / tooltip.
  // ---------------------------------------------------------------------------
  const RULES = [
    // Python-side messages are written "中文 / English" so Chinese locales keep
    // the original; here only the English half is shown, and only when that
    // half is actually English (no CJK left).
    [/^[\s\S]*[一-鿿][\s\S]*$/, (m) => {
      // A Chinese half may itself contain " / " (e.g. "Photoshop / 资源管理器"),
      // so try each separator and take the first joint whose right side is
      // CJK-free; that is the English half.
      let at = m.indexOf(" / ");
      while (at !== -1) {
        const tail = m.slice(at + 3);
        if (!CJK.test(tail)) return tail;
        at = m.indexOf(" / ", at + 3);
      }
      return m;
    }],
    [/^波(\d+)$/, "Wave $1"],
    [/^块 (\d+)$/, "Tile $1"],
    [/^块 (\d+) · 行(\d+) 列(\d+)$/, "Tile $1 · row $2, col $3"],
    [/^合并失败：([\s\S]*)$/, "Merge failed: $1"],
    [/^无法读取：([\s\S]*)$/, "Cannot read: $1"],
    [/^参考图 (\d+)×(\d+) · 原生分辨率裁取 ×([\d.]+)$/,
      "Reference $1×$2 · cropped at native resolution ×$3"],
    [/^参考图 (\d+)×(\d+)（与画布同尺寸）$/, "Reference $1×$2 (same size as the canvas)"],
    [/^输出文件夹：([\s\S]*?)\n([\s\S]*)$/, "Output folder: $1\n$2"],
    [/^点击更改；当前为自动 = 拆层文件夹下的 UpScale。$/, "Click to change; currently auto = UpScale inside the layer-split folder."],
    [/^读取失败：([\s\S]*)$/, "Read failed: $1"],
    [/^提交失败：([\s\S]*)$/, "Submit failed: $1"],
    [/^合成失败：([\s\S]*)$/, "Composite failed: $1"],
    [/^(\d+) 个图层$/, "$1 layers"],
    [/^已选重绘 (\d+) 个$/, "$1 selected for redraw"],
    [/^原图 · (\d+)×(\d+)$/, "Original · $1×$2"],
    [/^无法检查（([\s\S]*)）$/, "Cannot check ($1)"],
    [/^跟随 source 输入（([\s\S]*)）$/, "Follow the source input ($1)"],
    [/^块：只做 (\d+) \/ (\d+) 块（点“选历史结果…”可改）$/,
      "Tiles: only $1 / $2 (click “Choose history…” to change)"],
    [/^✓ 完成：(\d+) 块 → ([\s\S]*)$/, "✓ Done: $1 tiles → $2"],
    [/^失败：([\s\S]*)$/, "Failed: $1"],
    [/^文件夹 ([\s\S]+)$/, "Folder $1"],
    [/^(\d+) 块$/, "$1 tiles"],
    [/^(\d+) 层\/块$/, "$1 layers/tile"],
    [/^合并 (\d+) 层$/, "Merge $1 layers"],
    [/^ · 图片在 ([\s\S]+)$/, " · images in $1"],
    [/^图片目录：([\s\S]*)$/, "Image folder: $1"],
    [/^(\d+) \/ (\d+) 块$/, "$1 / $2 tiles"],
    [/^第 (\d+) 块 · x(\d+) y(\d+) · 点击切换$/, "Tile $1 · x$2 y$3 · click to toggle"],
    [/^读不到原图尺寸（([\s\S]*)），检查 image 输入是否接好$/,
      "Cannot read the original size ($1); check that the image input is connected"],
    [/^原图 (\d+)×(\d+)（后端读不到，用浏览器尺寸）$/,
      "Original $1×$2 (not available from the backend; using the browser size)"],
    [/^原图 (\d+)×(\d+) · 共 (\d+) 块$/, "Original $1×$2 · $3 tiles in total"],
    [/^主体 (\d+)×(\d+)([\s\S]*)$/, "Subject $1×$2$3"],
    [/^([\s\S]+) · (\d+) 条 · 边长 (\d+) · 模型 (\d+)（([\d.]+) 源px\/遮罩px）$/,
      "$1 · $2 strips · side $3 · model $4 ($5 src px / mask px)"],
    [/^读取分区计划失败：([\s\S]*)$/, "Failed to read the partition plan: $1"],
    [/^方形全宽分区：自动计划 (\d+) 条。([\s\S]*)$/, "Square full-width: planner produced $1 strips. $2"],
    [/^自定义分区：(\d+) 个窗口，顺序 = 生成顺序。([\s\S]*)$/,
      "Custom regions: $1 windows; order = generation order. $2"],
    [/^该目录是旧版结果（没有参数指纹），但不会自动清空：Queue 只覆盖选中的 (\d+) 块。$/,
      "This folder holds legacy results (no parameter fingerprint), but nothing is cleared automatically: Queue only overwrites the $1 selected tiles."],
    [/^该目录是旧版结果（没有参数指纹），但不会自动清空：Queue 会重跑全部块并覆盖。$/,
      "This folder holds legacy results (no parameter fingerprint), but nothing is cleared automatically: Queue reruns every tile and overwrites them."],
    [/^检测到生成参数已变：([\s\S]*)。不会自动清空目录：Queue 只重跑选中的 (\d+) 块，其余保留旧结果；要整体重来请点「清空此目录」。$/,
      "Generation settings changed: $1. Nothing is cleared automatically: Queue reruns only the $2 selected tiles and keeps the rest; use “Clear this folder” to start over."],
    [/^检测到生成参数已变：([\s\S]*)。不会自动清空目录：Queue 会重跑全部块并覆盖；要丢弃旧结果请点「清空此目录」。$/,
      "Generation settings changed: $1. Nothing is cleared automatically: Queue reruns every tile and overwrites them; use “Clear this folder” to discard the old results."],
    [/^([\s\S]*)已变：Queue 不清空目录，只重跑选中的 (\d+) 块。([\s\S]*)$/,
      "$1 changed: Queue keeps the folder and reruns only the $2 selected tiles. $3"],
    [/^([\s\S]*)已变：Queue 不清空目录，重跑全部块并覆盖。$/,
      "$1 changed: Queue keeps the folder, reruns every tile and overwrites them."],
    [/^参数变化（([\s\S]*)）：不会清空 ([\s\S]+)，只重跑选中的 (\d+) 块$/,
      "Settings changed ($1): keeping $2, rerunning only the $3 selected tiles"],
    [/^参数变化：不会清空 ([\s\S]+)，只重跑选中的 (\d+) 块$/,
      "Settings changed: keeping $1, rerunning only the $2 selected tiles"],
    [/^参数变化（([\s\S]*)）：不会清空 ([\s\S]+)，正在重跑全部 (\d+) 块并覆盖$/,
      "Settings changed ($1): keeping $2, rerunning all $3 tiles and overwriting them"],
    [/^参数变化：不会清空 ([\s\S]+)，正在重跑全部 (\d+) 块并覆盖$/,
      "Settings changed: keeping $1, rerunning all $2 tiles and overwriting them"],
    [/^第 (\d+) ([\s\S]+) · x(\d+) y(\d+) · 模型 ([\s\S]+) · 点击切换$/,
      "Tile $1 $2 · x$3 y$4 · model $5 · click to toggle"],
    [/^第 (\d+) ([\s\S]+) · x(\d+) y(\d+) · 点击切换$/,
      "Tile $1 $2 · x$3 y$4 · click to toggle"],
    [/^原图 (\d+)×(\d+) · (\d+) 条（边长 (\d+) · 模型 (\d+)）$/,
      "Original $1×$2 · $3 strips (side $4 · model $5)"],
    [/^原图 (\d+)×(\d+) · (\d+) 块（(\d+)×(\d+)）$/,
      "Original $1×$2 · $3 tiles ($4×$5)"],
    [/^原图 (\d+)×(\d+) · (\d+) 块$/, "Original $1×$2 · $3 tiles"],
    [/^正在读取 ([\s\S]+)…$/, "Reading $1…"],
    [/^没有可导入的分区（([\s\S]*)）。$/, "Nothing to import from $1."],
    [/^已从([\s\S]*)导入 (\d+) 个正方形窗口(；(\d+) 个非正方形\/过小的块已取内接正方形或跳过)?。$/,
      "Imported $2 square windows from $1$3."],
    [/^(\d+) 个非正方形\/过小的块已取内接正方形或跳过$/, " ($1 non-square or too-small tiles inset or skipped)"],
    [/^清空 ([\s\S]+) 里已生成的切块结果？此操作不可撤销。$/,
      "Clear the generated tile results in $1? This cannot be undone."],
    [/^已清空 ([\s\S]+)（(\d+) 个文件）。$/,
      "Cleared $1 ($2 files)."],
    [/^来源：([\s\S]*)$/, "Source: $1"],
    [/^本次会写到 → ([\s\S]*)$/, "This run writes to → $1"],
    [/^已存 (\d+) 块$/, "$1 tiles saved"],
    [/^(\d+) 块 · ([\s\S]*)$/, "$1 tiles · $2"],
    [/^（其中 (\d+) 块复用自之前的运行）$/, " ($1 tiles reused from an earlier run)"],
    [/^上次合并：([\s\S]*)$/, "Last merge: $1"],
    [/^选择要拼进成品的块([\s\S]*)$/, "Choose the tiles to merge into the result$1"],
    [/^列目录失败：([\s\S]*)$/, "Cannot list the folder: $1"],
    [/^完成：([\s\S]+) · (\d+) 层([\s\S]*)$/, "Done: $1 · $2 layers$3"],
    [/^✗ PSD 写入失败：([\s\S]*)$/, "✗ PSD write failed: $1"],
    [/^✗ 运行失败：([\s\S]*)$/, "✗ Run failed: $1"],
    [/^开始抠 (\d+) 个部件…$/, "Cutting out $1 parts…"],
    [/^正在抠 (\d+)\/(\d+) · ([\s\S]+)$/, "Cutting out $1/$2 · $3"],
    [/^（(\d+) 层）$/, " ($1 layers)"],
    [/^来自块 (\d+)$/, "From tile $1"],
    [/^(\d+) 部位$/, "$1 parts"],
    [/^(\d+) 份( · ([\s\S]+))?$/, "$1 copies$2"],
    [/^([\s\S]+) · 合并$/, "$1 · merge"],
    [/^([\s\S]+) · 实际输入$/, "$1 · actual input"],
    [/^([\s\S]+) · 引导回填后送进模型的输入 · 点击放大$/,
      "$1 · input sent to the model after guide fill-in · click to enlarge"],
    [/^([\s\S]+) · 实际输入（引导回填后）$/, "$1 · actual input (after guide fill-in)"],
    [/^(\d+) 个([\s\S]*)$/, "$1 $2"],
    [/^0 个([\s\S]*)$/, "0 $1"],
    [/^合并 PSD：([\s\S]*)$/, "Merged PSD: $1"],
    [/^([\s\S]+) 里还没有合并 PSD$/, "No merged PSD in $1 yet"],
    [/^读取(分块|部位)…$/, "Reading $1…"],
  ];

  // ---------------------------------------------------------------------------
  // phrase table: every Chinese literal the panels can show, longest first so
  // that a full sentence always wins over a fragment inside it.
  // ---------------------------------------------------------------------------
  const PHRASES = [
    ["块", "tile"],
    ["输出目录", "Output folder"],
    ["均匀网格", "Uniform grid"],
    ["全选", "Select all"],
    ["自动", "Auto"],
    ["清空", "Clear"],
    ["层", "layer"],
    ["原图 ", "Original "],
    ["刷新", "Refresh"],
    ["打开文件夹", "Open folder"],
    ["块 ", "Tile "],
    ["浏览…", "Browse…"],
    ["等待运行", "Waiting to run"],
    ["反选", "Invert"],
    ["合并", "Merge"],
    ["用户输入", "User input"],
    ["合并失败：", "Merge failed: "],
    ["读取失败：", "Read failed: "],
    ["打开 PSD", "Open PSD"],
    ["选择输出根目录", "Choose the output root"],
    ["没有可选的块", "No tiles to choose"],
    [" · 点击切换", " · click to toggle"],
    [" · 模型 ", " · model "],
    ["种子", "Seed"],
    [" / 深度 ", " / depth "],
    ["点击复制这一块的两个种子", "Click to copy this tile's two seeds"],
    ["透明阈值同时决定每块 PSD 和合并时的透明判定。",
      "The transparency threshold decides alpha for each tile PSD and when merging."],
    ["填到分块 PSD 节点", "Fill into the tile-PSD node"],
    ["当前输出目录为空", "The output folder is empty"],
    ["部位", "part"],
    ["参考图", "Reference image"],
    ["未选择", "Not selected"],
    ["选择…", "Choose…"],
    ["选择全身参考图", "Choose a full-body reference"],
    ["输入目录", "Input folder"],
    ["没有找到 LoadImage 节点", "No LoadImage node found"],
    ["未找到 LoadImage 节点", "No LoadImage node found"],
    ["搜索部件…", "Search parts…"],
    [" 层）", " layers)"],
    ["空层", "Empty layer"],
    ["固定", "Fixed"],
    ["参考图 ", "Reference "],
    ["正在读取部件…", "Reading parts…"],
    ["完成", "Done"],
    ["未选", "Not selected"],
    ["来源", "Source"],
    ["本次结果（实时）", "This run (live)"],
    ["块：全部", "Tiles: all"],
    ["输出根目录", "Output root"],
    ["打开合并 PSD", "Open merged PSD"],
    ["正在读取原图…", "Reading the original image…"],
    [" 块 · x", " tiles · x"],
    ["读不到原图尺寸（", "Cannot read the original size ("],
    ["），检查 image 输入是否接好", "); check that the image input is connected"],
    ["接上 LoadImage 就会显示切格", "Connect a LoadImage node to show the tile grid"],
    ["点图对齐主体", "Align grid to subject"],
    ["个窗口", "windows"],
    ["正在计算方形分区…", "Computing square partitions…"],
    ["重叠", "Overlap"],
    ["分辨率", "Resolution"],
    ["步数", "Steps"],
    ["透明阈值", "Transparency threshold"],
    ["块外上下文", "Context outside the tile"],
    ["分区方式", "Partition mode"],
    ["改位置/边长/分辨率/顺序不会清空该目录，只重跑勾选的窗口。",
      "Changing position, side, resolution or order keeps this folder and reruns only the selected windows."],
    [" 块。", " tiles."],
    ["检测到生成参数已变：", "Generation settings changed: "],
    ["不会清空 ", "keeping "],
    ["跟随全局", "Follow global"],
    ["让某一块完整包住人物后，合并时这个部位会整块采用那一块，不再被接缝切成两半。",
      "When one tile fully contains the subject, merging takes that part wholesale from it instead of cutting it at the seam."],
    ["也可以把重叠调到块宽的一半（块数会增加），让每块看到更多邻居。",
      "You can also raise the overlap to half a tile (more tiles), so each tile sees more neighbours."],
    ["0 个", "0 "],
    ["当前分区", "Current partition"],
    ["上次执行的分区", "Partition from the last run"],
    ["名称", "Name"],
    ["换种子不会清空目录：只重跑勾选的块，未勾选的块保留旧种子结果。",
      "Changing the seed does not clear the folder: only selected tiles rerun, the rest keep their old result."],
    ["合并文件名", "Merge filename"],
    ["同名图层单独成组（不裁剪）", "Group same-named layers (no cropping)"],
    ["depth 另存 PSD", "Export depth as a separate PSD"],
    ["原图作底图", "Use the original as the base layer"],
    ["不裁剪模式不使用接缝过渡", "No-crop mode skips seam blending"],
    ["直出合并时落败块在接缝外侧的淡出宽度（像素）；0 = 全部硬切",
      "Fade width, in pixels, for the losing tile outside the seam in direct mode; 0 = hard cut everywhere"],
    ["修完在 PS 里隐藏/删除不要的副本即可。",
      "Just hide or delete the extra copies in PS when you are done."],
    ["选一个波次文件夹或 *_layers.json；不跑 GPU。",
      "Pick a wave folder or *_layers.json; no GPU involved."],
    [" 个部件…", " parts…"],
    [" · 合并", " · merge"],
    ["方形全宽分区", "Square full-width"],
    ["自定义分区", "Custom regions"],
    ["头发", "Hair"],
    ["面部", "Face"],
    ["服装", "Clothing"],
    ["身体 / 其它", "Body / other"],
    ["自动 · ", "Auto · "],
    ["先在 ③ 里选一份历史结果", "Pick a history result in ③ first"],
    ["还没有可分格合并的波次：先跑一次 ③。", "No waves to merge yet: run ③ first."],
    ["这些波次都没有可读的分块 PSD。", "None of these waves has a readable tile PSD."],
    ["逐块选来源（混合合并）", "Choose the source per tile (mixed merge)"],
    ["每个格子可以来自不同的波次。同一个部件如果在多个波次里都出现，只取你选的那个波次——",
      "Each tile can come from a different wave. If a part appears in several waves, only the wave you pick is used —"],
    ["不会叠出两张脸。", "you never get two faces stacked."],
    ["勾选“同名图层单独成组”时输出不裁剪版：同名部件各自保留完整原件并收进同名文件夹；",
      "With “Group same-named layers” checked you get the no-crop version: same-named parts keep their full originals in a same-named folder;"],
    ["不勾时按接缝归属裁切成直出合并版。",
      "unchecked, they are cut by seam ownership into the direct merged version."],
    ["勾选“导出 depth PSD”时另存为 <名字>_depth.psd，绝不会出现在彩色 PSD 里。",
      "With “Export depth PSD” checked it is saved as <name>_depth.psd and never appears in the colour PSD."],
    ["主波次", "Main wave"],
    ["合并选中的分块", "Merge the selected tiles"],
    ["波", "Wave"],
    ["这个波次没有可合并的块。", "This wave has no tiles to merge."],
    ["正在合并…", "Merging…"],
    ["关闭", "Close"],
    ["预览", "Preview"],
    ["选择", "Select"],
    ["使用此文件夹", "Use this folder"],
    ["LoadImage 只读取 input 根目录；子文件夹里的图请先上传或移动到根目录。",
      "LoadImage only reads the input root; upload or move images from subfolders into the root first."],
    ["加载中…", "Loading…"],
    ["无法读取：", "Cannot read: "],
    ["上一级", "Up"],
    ["工作流", "Workflow"],
    ["图片", "Images"],
    ["空文件夹", "Empty folder"],
    ["拆层文件夹", "Layer-split folder"],
    ["output 下的文件夹，例如 SeeThrough", "A folder under output, e.g. SeeThrough"],
    ["选择拆层文件夹", "Choose the layer-split folder"],
    ["局部裁切重绘", "Crop-edit redraw"],
    ["只在部件周围一小块真实上下文里生成，再按原坐标贴回并恢复原 alpha。",
      "Generate inside a small patch of real context around the part, then paste it back at the original coordinates and restore the original alpha."],
    ["开：小部件不会被补全成整个人；关：整帧编辑（旧行为）。",
      "On: small parts are not completed into a whole person; off: full-frame edit (old behaviour)."],
    ["上下文：合成源层", "Context: composited source layers"],
    ["上下文：参考图", "Context: reference image"],
    ["合成源层：把全部源层拼成一张图再裁，坐标天然对齐（推荐）。",
      "Composited source: stitch every source layer into one image and crop from it — coordinates line up by construction (recommended)."],
    ["参考图：直接用 LoadImage 那张图裁，要求它与拆层画面严格对齐。",
      "Reference image: crop straight from the LoadImage picture; it must line up exactly with the layer-split frame."],
    ["边距", "Margin"],
    ["裁片在部件四周多留多少像素（源图尺度）。",
      "How many extra pixels to keep around the part, in source-image pixels."],
    ["瞳孔这类小部件用 16–64：留得越少，模型越没有空间「联想」出更大的结构。",
      "Use 16–64 for parts like pupils: the less margin, the less room the model has to “imagine” a larger structure."],
    ["px（源图尺度）", "px (source scale)"],
    ["裁片提示词（点开编辑）", "Patch prompt (click to edit)"],
    ["预览合成", "Preview composite"],
    ["用当前勾选和分辨率离线合成一张预览（不跑 GPU）",
      "Composite a preview offline from the current selection and resolution (no GPU)"],
    ["源文件夹", "Source folder"],
    ["在文件管理器中打开拆层文件夹", "Open the layer-split folder in the file manager"],
    ["在文件管理器中打开输出文件夹", "Open the output folder in the file manager"],
    ["从 PNG 合成（无 layers.json）", "Compose from PNGs (no layers.json)"],
    ["全部", "All"],
    ["局部裁切 ", "Crop-edit "],
    ["整帧 ", "Full-frame "],
    ["整帧编辑（关）", "Full-frame edit (off)"],
    [" 层超过源尺寸 8×", " layers exceed 8× the source size"],
    ["⚠ 参考图 ", "⚠ Reference "],
    [" 比例不符，会缩放回画布", " has a different aspect ratio and will be scaled to the canvas"],
    [" · 原生分辨率裁取 ×", " · cropped at native resolution ×"],
    ["PSD 写入失败", "PSD write failed"],
    ["③ 图片 → 分格 PSD", "③ Images → tile PSDs"],
    ["图片来源二选一：① 本次结果（刚跑完 SeeThrough 的输出，跑 GPU）；② 历史结果（点“选历史结果…”挑一个以前的文件夹，不跑 GPU）。",
      "Pick one image source: ① this run (fresh SeeThrough output, uses the GPU); ② history (click “Choose history…” to pick an earlier folder, no GPU)."],
    ["选历史结果…", "Choose history…"],
    ["用本次结果", "Use this run"],
    ["每块都保存 depth 数据和左右/前后子层；是否导出 depth 由 ④ 决定。",
      "Every tile saves depth data and left/right (front/back) sub-layers; ④ decides whether depth is exported."],
    ["分类前缀", "Category prefix"],
    ["可留空 → SeeThrough/<本次 id>/", "Optional → SeeThrough/<this run's id>/"],
    ["生成分格 PSD", "Generate tile PSDs"],
    ["跟随 source 输入（", "Follow the source input ("],
    ["本次结果（实时，来自 SeeThrough）", "This run (live, from SeeThrough)"],
    ["历史", "History"],
    ["历史*", "History*"],
    ["本次", "This run"],
    ["块：只做 ", "Tiles: only "],
    [" 块（点“选历史结果…”可改）", " tiles (click “Choose history…” to change)"],
    ["块：跟随 ① 的选格", "Tiles: follow ①'s selection"],
    ["没有接 parts：请先“选历史结果…”挑一份图片",
      "No parts connected: click “Choose history…” to pick an image set first"],
    ["生成中…", "Generating…"],
    ["✓ 完成：", "✓ Done: "],
    [" 块 → ", " tiles → "],
    ["失败：", "Failed: "],
    ["选择图片波次文件夹（③ 的图片来源）", "Choose an image wave folder (the ③ image source)"],
    ["错误", "Error"],
    ["文件夹 ", "Folder "],
    [" 层/块", " layers/tile"],
    ["合并 ", "Merge "],
    [" · 图片在 ", " · images in "],
    ["图片目录：", "Image folder: "],
    ["打开第一个 PSD", "Open the first PSD"],
    ["选择要生成的块", "Choose the tiles to generate"],
    ["点格子切换；未勾选的块不会跑扩散。", "Click a cell to toggle; unchecked tiles never run diffusion."],
    ["接上原图（image 输入）才能算出分块网格",
      "Connect the original image (image input) to compute the tile grid"],
    ["（后端读不到，用浏览器尺寸）", " (not available from the backend; using the browser size)"],
    [" · 共 ", " · "],
    ["① 用户输入（只改这里）", "① User input (edit here only)"],
    ["① 切格 + 选块", "① Tile grid + tile selection"],
    ["拖框移动；拖四角缩放；双击空白新建；选中后按 Delete 删除",
      "Drag to move; drag a corner to resize; double-click empty space to add; press Delete to remove the selected one"],
    ["点一下画面…", "Click on the canvas…"],
    ["在画面上点人物（例如脸的中心），网格会平移到这个点正好落在一块的中心",
      "Click the subject on the canvas (e.g. the centre of the face); the grid shifts so that point sits at the centre of a tile"],
    ["点一下这个按钮，再点画面上的脸/头部中心：自动平移内部切分线，让某一块完整包住人物",
      "Click this button, then click the face/head centre on the canvas: the internal cut lines shift so one tile fully contains the subject"],
    ["未勾选任何", "Nothing selected of the "],
    ["：Queue 会在这里停下。点「全选」或点格子/画面勾选。",
      ": Queue stops here. Click “Select all” or tick cells / the canvas."],
    ["分区未生效：Queue 会按均匀网格运行", "Partition not active: Queue falls back to the uniform grid"],
    ["无法生成方形分区", "Cannot build square partitions"],
    ["主体 ", "Subject "],
    ["主体已检测", "Subject detected"],
    [" 条 · 边长 ", " strips · side "],
    [" 源px/遮罩px）", " src px / mask px)"],
    ["接上 LoadImage 后才能自动分区。", "Connect LoadImage to enable automatic partitioning."],
    ["分区接口没有返回结果。", "The partition endpoint returned nothing."],
    ["读取分区计划失败：", "Failed to read the partition plan: "],
    ["原图尺寸", "Original size"],
    ["原图", "Original"],
    ["块尺寸", "Tile size"],
    ["网格偏移", "Grid offset"],
    ["深度分辨率", "Depth resolution"],
    ["左右拆分", "Split left/right"],
    ["引导参数", "Guidance settings"],
    ["拆层模型", "Layer-split model"],
    ["深度模型", "Depth model"],
    ["分区位置", "Partition position"],
    ["方形全宽分区：正在计算分区计划…", "Square full-width: computing the partition plan…"],
    ["方形全宽分区未生效：Queue 会按均匀网格运行；",
      "Square full-width did not take effect: Queue runs the uniform grid instead;"],
    ["修改分区参数不会清空该目录。", "Changing partition settings keeps this folder."],
    ["方形全宽分区：自动计划 ", "Square full-width: planner produced "],
    [" 条。", " strips."],
    ["切换分区方式或修改分区参数不会清空该目录，只重跑勾选的条。",
      "Switching the mode or changing partition settings keeps this folder and reruns only the selected strips."],
    ["自定义分区：还没有窗口。双击画面或点「＋ 新窗口」创建后再 Queue。",
      "Custom regions: no windows yet. Double-click the canvas or click “+ New window”, then Queue."],
    ["自定义分区：", "Custom regions: "],
    [" 个窗口，顺序 = 生成顺序。", " windows; order = generation order."],
    ["该目录是旧版结果（没有参数指纹），但不会自动清空：Queue 只覆盖选中的 ",
      "This folder holds legacy results (no parameter fingerprint), but nothing is cleared automatically: Queue overwrites only the selected "],
    ["该目录是旧版结果（没有参数指纹），但不会自动清空：Queue 会重跑全部块并覆盖。",
      "This folder holds legacy results (no parameter fingerprint), but nothing is cleared automatically: Queue reruns every tile and overwrites them."],
    ["。不会自动清空目录：Queue 只重跑选中的 ", ". Nothing is cleared automatically: Queue reruns only the selected "],
    [" 块，其余保留旧结果；要整体重来请点「清空此目录」。",
      " tiles and keeps the rest; use “Clear this folder” to start over."],
    ["。不会自动清空目录：Queue 会重跑全部块并覆盖；要丢弃旧结果请点「清空此目录」。",
      ". Nothing is cleared automatically: Queue reruns every tile and overwrites them; use “Clear this folder” to discard the old results."],
    ["已变：Queue 不清空目录，只重跑选中的 ", " changed: Queue keeps the folder and reruns only the selected "],
    [" 块，未勾选的块保留旧结果。", " tiles; the rest keep their old results."],
    ["已变：Queue 不清空目录，重跑全部块并覆盖。",
      " changed: Queue keeps the folder and reruns every tile, overwriting them."],
    ["参数变化", "Settings changed"],
    ["：不会清空 ", ": keeping "],
    ["，只重跑选中的 ", ", rerunning only the selected "],
    ["，正在重跑全部 ", ", rerunning all "],
    [" 块并覆盖", " tiles and overwriting them"],
    ["（参数变化：", " (settings changed: "],
    ["还没有窗口：双击画面或点「＋ 新窗口」", "No windows yet: double-click the canvas or click “+ New window”"],
    [" 个自定义窗口（拖框移动 · 拖角缩放 · 双击空白新建 · Delete 删除）",
      " custom windows (drag to move · drag a corner to resize · double-click empty space to add · Delete to remove)"],
    [" 条（边长 ", " strips (side "],
    ["方形分区未生效，当前按均匀网格显示（Queue 会回退均匀网格）",
      "Square partitioning is inactive; showing the uniform grid (Queue falls back to it)"],
    [" 块（", " tiles ("],
    ["自定义分区：还没有窗口。双击画面或点「＋ 新窗口」创建。",
      "Custom regions: no windows yet. Double-click the canvas or click “+ New window”."],
    ["接上 LoadImage 后才能摆放窗口。", "Connect LoadImage to place windows."],
    ["正在读取 ", "Reading "],
    ["细节说明（点开）", "Details (click to expand)"],
    ["方式", "Mode"],
    ["方形全宽分区：自动检测主体，按主体宽度做方形条；每条一次推理看到完整左右，",
      "Square full-width: detects the subject and makes square strips at its width; each strip is one inference that sees the full width,"],
    ["不再有竖直缝穿过人物。自定义分区：自己摆放正方形窗口，每个窗口完整包住一个部位。",
      "so no vertical seam cuts through the character. Custom regions: you place square windows yourself, each fully containing one part."],
    ["「方形全宽分区」会忽略块宽/块高/偏移和点图对齐；区域最大边是目标值，",
      "“Square full-width” ignores tile width/height, offset and grid alignment; the region's long side is the target,"],
    ["主体更宽时会自动放大到覆盖全宽，不会竖直切开。",
      "and it grows to cover the full width when the subject is wider, never slicing it vertically."],
    ["「自定义分区」由你在画面上摆放正方形窗口。切换模式不会清空输出目录。",
      "“Custom regions” means you place the square windows on the canvas yourself. Switching mode keeps the output folder."],
    ["切格", "Tile grid"],
    ["块宽", "Tile width"],
    ["块高", "Tile height"],
    ["偏移X", "Offset X"],
    ["偏移Y", "Offset Y"],
    ["接缝切到脸时：点按钮，再点画面上的脸/头部中心。",
      "When a seam cuts the face: click the button, then click the face/head centre on the canvas."],
    ["点图对齐主体：接缝切到脸时，先点「点图对齐主体」，再点画面上的脸/头部中心。",
      "Align grid to subject: when a seam cuts the face, click “Align grid to subject”, then click the face/head centre on the canvas."],
    ["区域最大边", "Max region side"],
    ["主体边距", "Subject margin"],
    ["块外上下文 0 = 关闭；大于 0 时每块向四周多裁这么多像素送进模型，生成后裁回原条带。",
      "Context outside the tile: 0 = off; above 0, each tile is cropped that many pixels larger on every side for the model and cropped back to the strip afterwards."],
    ["让只有腿脚的块也能看到上半身，减少按错误部位补全；不改变分块位置。",
      "Lets a legs-only tile see the upper body too, reducing completion by the wrong body part; the partition stays the same."],
    ["改它会重新生成相关块。", "Changing it regenerates the affected tiles."],
    ["方形分区每条都是方形窗口，自动以主体宽度/位置为中心；头部在第一条内完整。",
      "Every square strip is a square window centred on the subject's width and position; the head is whole in the first strip."],
    ["勾选按原图覆盖区域从旧网格迁移，不会自动扩大；如果迁移后为空，",
      "Ticking migrates from the old grid by original coverage and never widens it; if that leaves it empty,"],
    ["面板会标红提示，点「全选」再 Queue。", "the panel turns red — click “Select all”, then Queue."],
    ["自定义正方形窗口（每个包住一个部位）", "Custom square windows (each contains one part)"],
    ["＋ 新窗口", "+ New window"],
    ["在画面中心新建一个正方形窗口；也可以直接在画面上双击空白处",
      "Add a square window at the canvas centre; you can also double-click empty space on the canvas"],
    ["导入现有分区", "Import an existing partition"],
    ["导入", "Import"],
    ["取消", "Cancel"],
    ["自定义窗口顺序 = 生成顺序；后面的窗口会看到前面窗口在重叠区的结果。",
      "Custom window order = generation order; later windows see the earlier results in overlapping areas."],
    ["分辨率「跟随全局」使用「拆层质量」里的「分辨率」，每个窗口可单独覆盖。",
      "A window set to “Follow global” uses the Resolution under Layer-split quality; each window can override it."],
    [" 个窗口", " windows"],
    ["还没有窗口。点「＋ 新窗口」或在画面上双击空白处创建。",
      "No windows yet. Click “+ New window” or double-click empty space on the canvas."],
    ["勾选：本次生成这个窗口", "Checked: generate this window this run"],
    ["只用于识别，不参与缓存指纹", "Used for identification only; not part of the cache fingerprint"],
    ["边", "Side"],
    ["正方形边长（源图像素，64 的倍数）", "Square side in source pixels (a multiple of 64)"],
    ["左上角坐标（源图像素）", "Top-left corner in source pixels"],
    ["模型", "Model"],
    ["这个窗口的模型分辨率；跟随全局 = 上面的「分辨率」",
      "This window's model resolution; Follow global = the Resolution above"],
    ["上移：提前生成（引导顺序随之改变）", "Move up: generate earlier (the guidance order changes too)"],
    ["下移：延后生成", "Move down: generate later"],
    ["删除这个窗口", "Delete this window"],
    ["当前自定义窗口", "Current custom window"],
    ["当前方形分区", "Current square partition"],
    ["接上 LoadImage 后才能导入。", "Connect LoadImage to import."],
    ["当前均匀网格", "Current uniform grid"],
    ["没有可导入的分区（", "Nothing to import from "],
    ["已从", "Imported from "],
    ["导入 ", " imported "],
    [" 个正方形窗口", " square windows"],
    [" 个非正方形/过小的块已取内接正方形或跳过", " non-square or too-small tiles inset or skipped"],
    ["拆层质量", "Layer-split quality"],
    ["深度分辨率（-1=跟随）", "Depth resolution (-1 = follow)"],
    ["随机", "Random"],
    ["「分辨率」是模型推理分辨率，三种分区都用它：均匀网格直接按块裁到该分辨率，",
      "“Resolution” is the model's inference resolution and all three partition modes use it: the uniform grid crops each tile straight to it,"],
    ["方形分区按它推理（分区摘要里显示实际值），自定义窗口「跟随全局」也用它。",
      "square partitions infer at it (the actual value shows in the summary), and custom windows on “Follow global” use it too."],
    ["后处理", "Post-processing"],
    ["左右拆成两半", "Split into left/right halves"],
    ["用 LaMa 去背景", "Remove the background with LaMa"],
    ["分块引导（只影响生成）", "Tile guidance (generation only)"],
    ["用已生成相邻块的重叠区做引导", "Guide with the overlap of already-generated neighbouring tiles"],
    ["顺序逐块生成时，把先完成的相邻块在重叠区的像素回填到当前块再拆层；",
      "In sequential generation, pixels from finished neighbours in the overlap are filled into the current tile before layer-splitting;"],
    ["只有「引导版」工作流会读取，普通四步流水线忽略。",
      "only the “guided” workflow reads this; the plain four-step pipeline ignores it."],
    ["羽化", "Feather"],
    ["强度%", "Strength %"],
    ["与原图容差", "Tolerance vs. original"],
    ["分块引导：只回填生成结果与原图接近的像素；容差越小越保守，0 = 不过滤（会把补全内容也带过去）。",
      "Tile guidance: only fill back pixels close to the original; a smaller tolerance is more conservative, 0 = no filtering (completions get carried over too)."],
    ["强度 100 会用上一块的补全结果覆盖原图，容易带出不明物体/丢眼睛，建议 30-50。",
      "Strength 100 overwrites the original with the previous tile's completion, which tends to drag in stray objects or lose eyes; 30–50 is recommended."],
    ["输出位置", "Output location"],
    ["根目录", "Root folder"],
    ["SeeThrough/我的自设/", "SeeThrough/my-character/"],
    ["分类（可空）", "Category (optional)"],
    ["分块前缀", "Tile prefix"],
    ["清空此目录", "Clear this folder"],
    ["删除该目录里本流水线生成的切块 PSD、预览、depth、manifest 和 images/；",
      "Deletes the tile PSDs, previews, depth, manifest and images/ this pipeline wrote into the folder;"],
    ["你自己放进去的其他文件会保留。", "any other files you put there are kept."],
    ["清空 ", "Clear "],
    [" 里已生成的切块结果？此操作不可撤销。", " — clear its generated tile results? This cannot be undone."],
    [" 个文件）。", " files)."],
    ["Queue 会重新生成勾选的块。", "Queue regenerates the selected tiles."],
    ["清空失败：", "Clear failed: "],
    ["（直接写入所选文件夹；已有的块会复用，只重跑勾选的块）",
      " (writes straight into the chosen folder; existing tiles are reused and only selected ones rerun)"],
    ["勾选：不裁剪版——各块同名部件保留完整原件，收进同名文件夹，",
      "Checked: the no-crop version — same-named parts keep their full originals in a same-named folder,"],
    ["接缝裁切、幻影清理、完整部件淘汰都不做，便于在 PS 里修；",
      "with no seam cutting, ghost cleanup or whole-part elimination, so they are easy to fix in PS;"],
    ["不勾：直出版——按接缝归属裁切后合成一层。只影响合并，不会重跑生成。",
      "Unchecked: the direct version — cut by seam ownership and merged into one layer. Merging only; nothing is regenerated."],
    ["接缝过渡（仅直出）", "Seam blend (direct only)"],
    ["不裁剪版：同名副本收进同名组，组内是各块自己的完整副本（层名带块号），",
      "No-crop: same-named copies go into a same-named group, each the tile's own full copy (layer name carries the tile number),"],
    ["只有一个副本的部位保持普通图层；重叠处会重复显示——",
      "parts with a single copy stay plain layers; overlaps show twice —"],
    ["直出版：每个像素只归离自己块边更远的一块，另一块在接缝处被裁掉；",
      "Direct: every pixel belongs to the tile whose edge is farther away, and the other is cut at the seam;"],
    ["落败块在接缝外侧按「接缝过渡」淡出。某一块完整看到某部位时整块采用它。",
      "the losing tile fades out beyond the seam by “Seam blend”. When one tile sees a whole part, it is taken wholesale from that tile."],
    ["自定义窗口的「跟随全局」分辨率使用「拆层质量」里的「分辨率」；每个窗口可在列表里单独覆盖。",
      "A custom window's “Follow global” resolution uses the Resolution under Layer-split quality; each window can override it in the list."],
    ["换种子不会清空目录：只重跑勾选的窗口，未勾选的保留旧种子结果。",
      "Changing the seed does not clear the folder: only selected windows rerun, the rest keep their old-seed results."],
    ["「分辨率」是模型推理分辨率，三种分区都用它；方形分区的实际值见分区摘要。",
      "“Resolution” is the model's inference resolution and all three modes use it; the square partition's actual value is in the summary."],
    ["方形全宽分区：每条一次推理看到完整左右，水平缝落在条之间；",
      "Square full-width: one inference per strip sees the full width, and horizontal seams fall between strips;"],
    ["合并仍按完整部件规则优先采用整条看到该部位的副本。",
      "merging still prefers, by the whole-part rule, the copy whose strip fully saw that part."],
    ["自定义分区：每个正方形窗口一次推理，只看到框内完整部位；",
      "Custom regions: one inference per square window, seeing only the parts fully inside it;"],
    ["重叠区按运行顺序把前面窗口的结果回填给后面的窗口。",
      "overlaps are filled back to later windows from earlier ones in run order."],
    ["方形分区不可用，已回退均匀网格。", "Square partitioning unavailable; fell back to the uniform grid."],
    ["① 选块（点格子）", "① Select tiles (click cells)"],
    ["点格子切换；也可以直接改节点上的 select 文字（all / 0-2,5）。",
      "Click a cell to toggle; you can also edit the select text on the node (all / 0-2,5)."],
    ["③ 从图片重建（CPU）", "③ Rebuild from images (CPU)"],
    ["选择图片波次文件夹", "Choose an image wave folder"],
    ["来源：", "Source: "],
    ["③ 输出文件夹（每次运行新建）", "③ Output folder (new on every run)"],
    ["所有结果都存在这里。每按一次 Queue 都新建一个波次文件夹，上一次的结果不会被覆盖。",
      "Everything lands here. Each Queue press creates a new wave folder, so the previous results are never overwritten."],
    ["本次会写到 → ", "This run writes to → "],
    ["③ 每块存一份 PSD（自动逐块）", "③ One PSD per tile (automatic, tile by tile)"],
    ["② 送来几块，本节点就自动执行几次：每块写一份 PSD、预览、depth 和 images/ 里的图层图片。不用手点。",
      "② sends N tiles and this node runs N times: each tile gets a PSD, preview, depth and layer images under images/. Nothing to click."],
    ["运行完成后这里会显示输出文件夹。", "The output folder shows here once the run finishes."],
    ["打开波次文件夹", "Open the wave folder"],
    ["正在逐块写入…", "Writing tile by tile…"],
    ["已存 ", "Saved "],
    ["③ 写清单（全部块完成后）", "③ Write the manifest (after every tile)"],
    ["全部块写完后，这里统一写 manifest.json / images.json / original.png，",
      "Once all tiles are written, this writes manifest.json / images.json / original.png in one go,"],
    ["并把波次文件夹交给 ④ 合并。", "and hands the wave folder to ④ for merging."],
    ["运行完成后这里会显示结果文件夹。", "The result folder shows here once the run finishes."],
    ["打开结果文件夹", "Open the result folder"],
    ["已写入", "Written"],
    ["② 保存分块图片", "② Save tile images"],
    ["把每块的部件图片按块存成文件夹，第三步可以直接读取。",
      "Saves each tile's part images into its own folder, which step three reads directly."],
    ["目录", "Folder"],
    [" 块 · ", " tiles · "],
    ["图里没有分块 PSD 节点", "The workflow has no tile-PSD node"],
    ["已填入 ✓", "Filled in ✓"],
    ["分块 PSD 合并", "Tile PSD merge"],
    ["合并方式", "Merge mode"],
    ["合并所选", "Merge selected"],
    ["选择目录后勾选要拼的块", "Pick a folder, then tick the tiles to merge"],
    ["合并预览", "Merge preview"],
    ["depth 图（单独合并）", "Depth map (merged separately)"],
    ["depth 图", "Depth map"],
    ["打开 depth PSD", "Open the depth PSD"],
    ["这个目录里没有分块", "This folder has no tiles"],
    [" · 单击勾选，双击放大", " · click to toggle, double-click to enlarge"],
    [" 预览", " preview"],
    ["读取分块…", "Reading tiles…"],
    ["（其中 ", " ("],
    [" 块复用自之前的运行）", " tiles reused from an earlier run)"],
    ["这些分块已经把同部件合成一层了；要按部件分文件夹，请在 ③ 里保留子层再生成",
      "These tiles already merged same-named parts into one layer; to get one folder per part, keep sub-layers in ③ and regenerate"],
    ["上次合并：", "Last merge: "],
    ["选择要拼进成品的块", "Choose the tiles to merge into the result"],
    ["还没有分块结果", "No tile results yet"],
    ["先跑一次分块工作流", "Run the tiling workflow once first"],
    ["列目录失败：", "Cannot list the folder: "],
    ["完成：", "Done: "],
    [" 个文件夹", " folders"],
    [" · depth 另存 ", " · depth saved separately as "],
    ["先勾选至少一块", "Select at least one tile first"],
    ["合并中…", "Merging…"],
    ["自动合并：目录和选项来自 ① 用户输入", "Automatic merge: folder and options come from ① User input"],
    ["部件选择", "Part selection"],
    ["正在读取部件表…", "Reading the part list…"],
    ["勾选这次要抠的部件；勾得越多跑得越久。",
      "Tick the parts to cut out this run; the more you tick, the longer it takes."],
    ["读取部件表失败：请重启 ComfyUI 后刷新（仍可手写 select）。",
      "Cannot read the part list: restart ComfyUI and refresh (you can still type select by hand)."],
    ["正在运行…", "Running…"],
    ["已中断", "Interrupted"],
    ["✓ 运行完成", "✓ Run complete"],
    ["未知错误", "Unknown error"],
    ["开始抠 ", "Cutting out "],
    ["正在抠 ", "Cutting out "],
    ["结果预览", "Result preview"],
    ["按块", "By tile"],
    ["按部位", "By part"],
    ["每块一张卡：合并图 + 这块的各个部位",
      "One card per tile: the merged image plus this tile's parts"],
    ["每个部位一张卡：合并图 + 组成它的各块副本",
      "One card per part: the merged image plus the copies it is made of"],
    ["缩小合并图 / 拆分图（下限 40%）", "Shrink the merged / split images (min 40%)"],
    ["放大合并图 / 拆分图（上限 300%）", "Enlarge the merged / split images (max 300%)"],
    [" 张，", ", "],
    [" 张失败：", " failed: "],
    ["（块 ", " (tile "],
    ["跨块合并成的一层", "A layer merged across tiles"],
    ["来自块 ", "From tile "],
    [" 部位", " parts"],
    [" 份", " copies"],
    [" · 实际输入", " · actual input"],
    [" · 引导回填后送进模型的输入 · 点击放大",
      " · input sent to the model after guide fill-in · click to enlarge"],
    [" · 实际输入（引导回填后）", " · actual input (after guide fill-in)"],
    ["这次没有生成这块（未勾选，或用了上次的结果）",
      "This tile was not generated this run (unchecked, or the previous result was used)"],
    ["无可拆分图层", "No layers to split"],
    ["总览", "Overview"],
    ["合并 PSD · 点击放大", "Merged PSD · click to enlarge"],
    ["合并 PSD · 总览", "Merged PSD · overview"],
    ["合并 PSD：", "Merged PSD: "],
    ["没有分块（这不是分块运行文件夹）", "No tiles (this is not a tile-run folder)"],
    ["没有可预览的部位", "No parts to preview"],
    ["输入", "Input"],
    ["引导回填后真正送进模型的那张图", "The image actually sent to the model after guide fill-in"],
    ["拆分图层", "Split layers"],
    ["等待 ④ 的结果（连上并运行一次即可）", "Waiting for ④'s results (connect and run once)"],
    ["读取", "Read"],
    ["分块", "Tiles"],
    ["失败", "Failed"],
    ["接缝过渡", "Seam blend"],
    ["附加输出", "Extra output"],
    ["勾选：不裁剪版——两格里同名部件各自保留完整原件并收进同名文件夹，接缝裁切/幻影清理/完整部件淘汰都不做；不勾：直出版——按接缝归属裁切后合成一层",
      "Checked: no-crop — same-named parts from both tiles keep their full originals in a same-named folder, with no seam cutting, ghost cleanup or whole-part elimination. Unchecked: direct — cut by seam ownership and merged into one layer."],
    ["不裁剪版：同名副本收进同名组，组内是各块自己的完整副本；只有一个副本的部位保持普通图层。重叠处会重复显示，修完在 PS 里隐藏/删除不要的副本即可。",
      "No-crop: same-named copies go into a same-named group, each the tile's own full copy; parts with a single copy stay plain layers. Overlaps show twice — hide or delete the copies you don't want in PS once you are done editing."],
    ["直出版：每个像素只归离自己块边更远的一块，落败块在接缝外侧按该宽度淡出；depth 永远单独一个 PSD，和彩色 PSD 互不相干。",
      "Direct: every pixel belongs to the tile whose edge is farther away, and the losing tile fades out beyond the seam by that width; depth always gets its own PSD, entirely separate from the colour PSD."],
    ["同一文件夹里已有的块会自动参与合并——只重跑一块也能出完整成品。",
      "Tiles already in the same folder join the merge automatically — rerunning a single tile still gives a complete result."],
    ["导出独立的 <名字>_depth.psd；不勾则不导出。depth 永远不会进入彩色 PSD",
      "Exports a separate <name>_depth.psd; uncheck to skip it. depth never enters the colour PSD."],
    ["合并 PSD 底部放入原图图层和 original.png",
      "Puts the original image layer and original.png at the bottom of the merged PSD"],
  ];

  const UNITS = [
    [/(\d[\d.,]*)\s*块/g, "$1 tiles"],
    [/(\d[\d.,]*)\s*层/g, "$1 layers"],
    [/(\d[\d.,]*)\s*个窗口/g, "$1 windows"],
    [/(\d[\d.,]*)\s*张/g, "$1"],
    [/(\d[\d.,]*)\s*条/g, "$1"],
    [/(\d[\d.,]*)\s*份/g, "$1"],
    [/(\d[\d.,]*)\s*个/g, "$1"],
  ];

  PHRASES.sort((a, b) => b[0].length - a[0].length);

  function translate(text) {
    if (!ENABLED || !text || !CJK.test(text)) return text;
    let out = String(text);
    for (let i = 0; i < RULES.length; i++) {
      const rule = RULES[i];
      if (rule[0].test(out)) out = out.replace(rule[0], rule[1]);
    }
    if (!CJK.test(out)) return out;
    for (let i = 0; i < PHRASES.length; i++) {
      const item = PHRASES[i];
      if (out.indexOf(item[0]) !== -1) out = out.split(item[0]).join(item[1]);
    }
    if (!CJK.test(out)) return out;
    for (let i = 0; i < UNITS.length; i++) {
      out = out.replace(UNITS[i][0], UNITS[i][1]);
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // DOM watcher: translate in place, remember the Chinese text so panels that
  // pattern-match their own labels can still read it.
  // ---------------------------------------------------------------------------
  const ATTRS = ["title", "placeholder", "aria-label", "alt"];
  const RAW = new WeakMap();

  function translateNode(node) {
    if (!node) return;
    if (node.nodeType === 3) {
      const text = node.nodeValue;
      if (!text || !CJK.test(text)) return;
      if (!RAW.has(node)) RAW.set(node, text);
      const out = translate(text);
      if (out !== text) node.nodeValue = out;
      return;
    }
    if (node.nodeType !== 1 || !node.getAttribute) return;
    for (let i = 0; i < ATTRS.length; i++) {
      const name = ATTRS[i];
      const value = node.getAttribute(name);
      if (!value || !CJK.test(value)) continue;
      const out = translate(value);
      if (out !== value) RAW.set(node, value);
      if (out !== value) node.setAttribute(name, out);
    }
  }

  function walk(root) {
    translateNode(root);
    if (!root || root.nodeType !== 1 || !root.querySelectorAll) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) translateNode(node);
    const all = root.querySelectorAll("*");
    for (let i = 0; i < all.length; i++) translateNode(all[i]);
  }

  function watch(root) {
    if (!ENABLED || !root || root.__stluI18nWatched) return;
    root.__stluI18nWatched = true;
    walk(root);
    const observer = new MutationObserver((records) => {
      for (let i = 0; i < records.length; i++) {
        const record = records[i];
        if (record.type === "childList") {
          for (let j = 0; j < record.addedNodes.length; j++) walk(record.addedNodes[j]);
        } else {
          translateNode(record.target);
        }
      }
    });
    observer.observe(root, {
      childList: true,
      subtree: true,
      characterData: true,
      attributes: true,
      attributeFilter: ATTRS,
    });
  }

  // The Chinese text a label had before it was translated.
  function rawText(node) {
    if (!node) return "";
    if (node.nodeType === 3) return RAW.has(node) ? RAW.get(node) : (node.nodeValue || "");
    if (node.childNodes && node.childNodes.length === 1
        && node.firstChild.nodeType === 3) return rawText(node.firstChild);
    return node.textContent || "";
  }

  // Record the source text for code that writes a translated string itself,
  // so raw() still answers with the original Chinese.
  function remember(node, text) {
    if (!node || text == null) return;
    const target = node.nodeType === 3 ? node
      : (node.firstChild && node.firstChild.nodeType === 3 ? node.firstChild : null);
    if (target) RAW.set(target, String(text));
  }

  // ---------------------------------------------------------------------------
  // combo labels: the stored value stays Chinese, only the drawn label and the
  // dropdown entries are translated.
  // ---------------------------------------------------------------------------
  const COMBO_LABELS = {
    "均匀网格": "Uniform grid",
    "方形全宽分区": "Square full-width",
    "自定义分区": "Custom regions",
  };

  function labelCombos(node) {
    if (!ENABLED || !node || !node.widgets) return;
    node.widgets.forEach((widget) => {
      const options = widget && widget.options;
      const values = options && options.values;
      if (!Array.isArray(values)) return;
      if (!values.some((v) => typeof v === "string" && COMBO_LABELS[v])) return;
      if (widget.__stluComboLabel) return;
      widget.__stluComboLabel = true;
      const inner = options.getOptionLabel;
      options.getOptionLabel = function (value) {
        if (value != null && COMBO_LABELS[value]) return COMBO_LABELS[value];
        return inner ? inner.call(this, value) : String(value);
      };
      widget.callback = widget.callback; // keep the reference stable
    });
  }

  window.STLU_I18N = {
    enabled: ENABLED,
    t: translate,
    watch: watch,
    raw: rawText,
    remember: remember,
    labelCombos: labelCombos,
    comboLabels: COMBO_LABELS,
  };
})();
