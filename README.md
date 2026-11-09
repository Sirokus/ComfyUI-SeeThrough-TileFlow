# ComfyUI-SeeThrough-TileFlow

**English** · [中文](README_ZH.md)

> Tile-driven SeeThrough character layer/depth generation: slice a high-resolution
> reference into overlapping tiles, run LayerDiff + Depth + post-processing per
> tile, then merge the per-tile layers back into one layered PSD.

Nodes and panels speak Chinese by default; an English localization file ships in
`locales/en`.

---

## Features

- **Only ticked tiles run**: unchecked cells never reach the GPU, already generated
  cells are reused.
- **Sequential guided backfill**: before each tile is generated, pixels from already
  finished neighbouring tiles are filled into the overlap and fed to the model
  (`STLU_GuidedLayers` / `STLU_SeqTileLoop`), softening seams.
- **Fixed output directory + parameter fingerprint**: changed parameters are only
  reported, results are never deleted automatically; clearing is always manual.
- **Layered PSD + per-tile manifests**: each tile is saved as its own layered PSD;
  when all tiles finish, `manifest.json` / `images.json` / `generation.json` are
  written and the whole image is merged into one layered PSD.
- **One tile workflow**: a single six-node pipeline where planning, sequential
  layer generation, saving, merging and preview are each a separate visible node.
  Every node keeps a single responsibility.
- **Bilingual panels**: Chinese by default; when `Comfy.Locale` is English,
  `locales/en` and `web/stlu_i18n.js` provide the English UI (saved values are
  never rewritten).

## Requirements

- ComfyUI (Python ≥ 3.10).
- The ComfyUI-SeeThrough custom nodes and their LayerDiff / Depth models — the
  workflow's `SeeThrough_LoadLayerDiffModel`, `SeeThrough_GenerateLayers`,
  `SeeThrough_LoadDepthModel`, `SeeThrough_GenerateDepth` and
  `SeeThrough_PostProcess` all come from it.
- The `torch` / `numpy` / `Pillow` / `OpenCV` / `SciPy` that ship with ComfyUI.
- Optional: `opencv-contrib-python`. The 4K mask nodes prefer
  `cv2.ximgproc.guidedFilter` and fall back to bilinear interpolation when it is
  missing; edge snapping needs `scipy`.
- This package has no extra pip dependencies.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Sirokus/ComfyUI-SeeThrough-TileFlow.git
```

Restart ComfyUI. New nodes appear under **SeeThrough/LayerUpscale**, legacy
nodes under **SeeThrough/LayerUpscale/Legacy**.

## Workflows

| File | Description |
|---|---|
| `workflow/seethrough-tile-flow.json` | The tile pipeline: ① `STLU_PipelineInput` plans, crops and fingerprints the tiles → ② `STLU_GuidedLayers` generates layers block by block with cross-tile guidance → ③ per-tile PSD + manifest → ④ `STLU_TileMerge` merges → ⑤ `STLU_PSDPreview`. Only the checked tiles are (re)run; existing tiles are reused. |

## Quick start

1. Open one of the workflows above and pick the source image in `LoadImage`.
2. In the **① user input** panel:
   - set tile width / height / overlap (default 1024 / 1024 / 128) and layer
     resolution / steps (default 512 / 30);
   - tick the tiles to run in the grid panel (unticked tiles do not run);
   - choose the output root; whether every Queue creates a new wave folder is a
     setting;
   - choose merge options (merge name, group same-named layers, export depth,
     include the original).
3. Press **Queue**; ② layer generation → ③ PSD / manifest writing → ④ merge
   continue automatically.
4. Results land in `output/<output dir>/`; the ⑤ preview panel shows the full-body
   overview plus per-tile / per-part groupings.

## Pipeline

| Stage | What it does | Pipeline nodes | Optional split-step nodes |
|---|---|---|---|
| ① plan & crop | Plan the grid, pick tiles, expand context, write the parameter fingerprint, crop | `STLU_PipelineInput` | `STLU_PipelineSettings` → `STLU_PlanGrid` → `STLU_PickTiles` → `STLU_PlanContext` → `STLU_PrepareRun` → `STLU_CropTiles` |
| ② sequential layer generation | Run LayerDiff → Depth → PostProcess per tile, guided backfill in the overlap | `STLU_GuidedLayers` | `STLU_SeqTileLoop` (the only loop that must stay in Python); `STLU_GuideTile` / `STLU_CropBack` are single-tile versions for debugging |
| ③ save | Write per-tile layered PSD and layered PNG, save the source, summarize manifests | `STLU_TilePSD` + `STLU_WaveIndex` | `STLU_WriteTilePSD` / `STLU_WriteTileImages` → `STLU_SaveOriginal` → `STLU_WriteManifest` |
| ④ merge | Seam ownership, scaling, whole-part exceptions, layered assembly | `STLU_TileMerge` | `STLU_MergeTiles` (whole-tile algorithm, kept as one node) |
| ⑤ preview | Merged PSD overview, per-tile / per-part grouping | `STLU_PSDPreview` | Same, driven by ④'s `psd_path` |

Only edit **①**; the parameter widgets of ②③④ are locked and only show progress
and results.

The shipped workflow uses the *Pipeline nodes* column. The split-step column is
the same implementation exposed as single-purpose nodes — handy for debugging or
building a custom graph, but no example workflow ships for it.

## Common parameters

| Parameter | Default | Meaning |
|---|---|---|
| Tile width / height / overlap | 1024 / 1024 / 128 | Tile size and overlap pixels; the overlap feeds guided backfill |
| Resolution / steps | 512 / 30 | Per-tile layer resolution and inference steps |
| Depth resolution | -1 | `-1` follows the layer resolution |
| Seed | 0 | Click "random" or type a value |
| Post-process | — | Split left/right into halves / LaMa background removal |
| Context padding | 0 | Extra pixels cropped around each tile, 0 disables; lets a tile see more of the body |
| Alpha threshold | 8 | Advanced; alpha cutoff for transparent pixels |
| Merge name / group same-named layers / depth / original | — | Merge-only, never triggers regeneration |

## Output layout

```text
<output>/<output dir>/[<category>/]
├─ generation.json         parameters and fingerprint of this run
├─ manifest.json           per-tile manifest (new / reused / restored tiles)
├─ images/
│  ├─ images.json          layered PNG manifest
│  └─ ...
├─ original.png            source image (if kept)
├─ tile_00_r0_c0.psd       per-tile layered PSD
├─ tile_00_r0_c0_preview.png
├─ <merge name>.psd        merged full-image layered PSD
└─ <merge name>_depth.psd  optional separate depth PSD (never mixed into the colour PSD)
```

## Cache, reuse and re-runs

- If a parameter that changes pixels (source image, tile size / overlap,
  resolution, steps, depth resolution, seed, left/right split, LaMa, alpha
  threshold, layer / depth model) changes, `generation.json` records it and the
  UI reports it, but **nothing is ever deleted**: Queue re-runs only the ticked
  tiles and overwrites them, keeping every other tile and the merged image. Click
  "Clear this folder" to start over.
- Changing only merge options (group same-named layers, export depth, include the
  original, seam feathering) never re-runs tiles, only re-merges.
- Tiles already generated in the same directory are reused automatically: tick
  only the tiles you want re-run; unticked tiles still take part in the merge.
- Switching the source image (even at the same size) never reuses stale tiles —
  the manifest carries a source fingerprint; the old tiles stay on disk until you
  clear the folder manually.

## Other nodes

Outside the main pipeline, the package also provides nodes for other workflows
(category **SeeThrough/LayerUpscale**):

- `STLU_Mask4K` / `STLU_Mask4KBatch`: pure-CPU 4K part masks (guided filter + edge
  snapping).
- `STLU_EditPartsBatch`: batch-extract Live2D parts with an image-editing model.
- `STLU_RebuildPSD`: rebuild a merged PSD from exported layer images (CPU).
- Legacy nodes under the Legacy category (`STLU_LoadLayer`,
  `STLU_ContextBackdrop`, `STLU_FocusCrop`, `STLU_PasteBack`,
  `STLU_DetailBatch`, `STLU_ChromaToAlpha`, `STLU_PartsPSD`, `STLU_TileGrid`, …)
  remain usable but no longer appear in the new pipeline.

## Tools

```bash
# Build / repair an optional split-step variant (every inner step a visible
# node; not shipped as a default example). Keeps widget values you changed.
python tools/build_transparent_workflow.py <workflow_dir>
python tools/build_transparent_workflow.py <workflow_dir> --force
python tools/build_transparent_workflow.py <workflow_dir> --validate

# Other workflow builders
python tools/build_edit_workflow.py
```

## Tests

```bash
python tests/run_all.py                 # all offline tests
python tests/run_all.py tile mask4k     # only tests whose file name matches
```

All tests run on CPU, write to temporary directories, need no running ComfyUI and
load no models.

## Frontend and HTTP endpoints

- `web/stlu.js`: the ① input panel, tile picking, ③ progress, ⑤ PSD preview,
  etc.
- `web/stlu_i18n.js` + `locales/en/`: English UI.
- `routes.py` registers `/stlu/*` endpoints on ComfyUI's prompt server: `ls`,
  `layers`, `thumb`, `preview`, `reveal`, `open`, `wf`, `imgsize`, `psd_preview`,
  `tiles/*` (grids, grid, images, from_images, dirs, manifest, clear,
  merge, merge-mixed, strips), `edit_parts`.

## License

Apache License 2.0, see [LICENSE](LICENSE).

Layer and depth generation is built on ComfyUI-SeeThrough; this repository only
handles tile scheduling and layered PSD reading, writing and merging.
