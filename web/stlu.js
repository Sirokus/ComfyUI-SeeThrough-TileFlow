/**
 * Front-end for the SeeThrough layer-upscale nodes.
 *
 * STLU_Plan       big panel: folder, reference image, per-part selection,
 *                 resolution, composite preview, run button, live progress.
 * STLU_UpscaleBatch  card showing which subgraph workflow is used.
 * STLU_AssemblePSD   result card with the finished PSD + preview.
 *
 * The raw text widgets stay the source of truth for prompt serialization;
 * this file only writes them through setWidget().
 */

(function () {
  "use strict";

  const { app } = window.comfyAPI.app;
  const { api } = window.comfyAPI.api;

  const PLAN = "STLU_Plan";
  const BATCH = "STLU_UpscaleBatch";
  const ASSEMBLE = "STLU_AssemblePSD";
  const TILES = "STLU_TilesPSD";
  const TILE_SELECT = "STLU_TileSelect";
  const TILE_MERGE = "STLU_TileMerge";
  const TILE_GRID = "STLU_TileGrid";
  const PIPELINE = "STLU_PipelineInput";
  // The transparent variant splits stage ① into visible nodes; the same grid
  // panel lives on its settings node so the user still has one place to edit.
  const PIPELINE_SETTINGS = "STLU_PipelineSettings";
  // The lightweight ① node: same planner, fewer knobs.  Optional controls are
  // hidden by widget presence, so this only needs to be recognized as a
  // pipeline node.
  const SIMPLE = "STLU_SimpleInput";
  // Subject strips: one inference sees the whole subject width.  Kept as a
  // literal so the panel, the route and the Python widget use one spelling.
  const STRIPS_MODE = "方形全宽分区";
  // User-placed square windows.  Mirrors region_plan.CUSTOM_LABEL.
  const CUSTOM_MODE = "自定义分区";
  // Mirrors region_plan's numeric contract: 64 px side grid, 512-2048 model.
  const CUSTOM_STEP = 64;
  const CUSTOM_MIN = 64;
  const CUSTOM_RES_MIN = 512;
  const CUSTOM_RES_MAX = 2048;
  const SELECT_TILES = "STLU_SelectTiles";
  const REBUILD_PSD = "STLU_RebuildPSD";
  const SAVE_TILES = "STLU_SaveTiles";
  const EDIT_BATCH = "STLU_EditPartsBatch";
  const WAVE = "STLU_Wave";
  const TILE_PSD = "STLU_TilePSD";
  const WAVE_INDEX = "STLU_WaveIndex";
  const PSD_PREVIEW = "STLU_PSDPreview";
  const editStates = new Set();
  const mergeStates = new Set();
  const tilesStates = new Set();
  const stage3States = new Set();

  // Last split produced in this session.  Stage one no longer writes a file:
  // ``lastGridDoc`` is the inline document, ``lastGridPath`` stays for nodes
  // saved by older versions that still point at a grids/*.json.
  let lastGridDoc = "";
  let lastGridPath = "";
  let lastImagesDir = "";
  let lastTilesInfo = null;
  let lastMergeInfo = null;
  // Newest wave this session produced, so the merge editor can default to it.
  let lastRunDir = "";

  // Tile grids can be shown twice (the ① preview and the legacy select node);
  // the selection itself lives in one visible widget and is mirrored through
  // this list.
  const gridPanels = [];
  const gridStates = new Set();
  const panelRegistry = new Map();
  let gridBroadcast = false;
  const SESSION_V = Date.now();

  const HIDDEN_PLAN = [
    "layers_json", "select", "exclude", "size_rules", "out_dir",
    "canvas_size", "skip_empty", "empty_coverage",
    "focus_crop", "focus_margin", "focus_context", "focus_prompt",
  ];

  // Mirrors core_plan.DEFAULT_FOCUS_PROMPT.
  const DEFAULT_FOCUS_PROMPT =
    "Upscale <image1>, a cropped region of a character illustration. Keep the exact " +
    "composition, framing, colors and shapes. Do NOT add, remove or extend anything. " +
    "Enhance only sharpness and fine detail.";
  const HIDDEN_BATCH = ["plan", "plan_path", "seed", "workflow_json"];
  const HIDDEN_ASSEMBLE = [
    "plan", "out_dir", "psd_filename", "fallback", "write_layers_json", "alpha_threshold",
  ];

  // Part categories, first match wins. "body" is the catch-all.
  const CATEGORIES = [
    { id: "hair", label: "头发", test: (n) => /hair/i.test(n) },
    { id: "face", label: "面部", test: (n) => /face|eyebrow|eyelash|eyewhite|irides|ears|nose|mouth|eyewear|earwear/i.test(n) },
    { id: "wear", label: "服装", test: (n) => /wear|cloth|skirt|shoe|dress|pants/i.test(n) },
    { id: "body", label: "身体 / 其它", test: () => true },
  ];

  const SIZE_PRESETS = [512, 1024, 1536, 2048, 3072, 4096];

  const CSS = `
  .stlu-panel{display:flex;flex-direction:column;gap:7px;height:100%;padding:9px;box-sizing:border-box;
    background:var(--comfy-menu-bg,#1b1b1c);color:var(--input-text,#ddd);border:1px solid var(--border-color,#333);
    border-radius:8px;font:12px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden;}
  .stlu-card{background:var(--comfy-input-bg,#141415);border:1px solid var(--border-color,#2f2f31);border-radius:6px;
    padding:7px 8px;display:flex;flex-direction:column;gap:6px;min-width:0;}
  .stlu-line{display:flex;gap:6px;align-items:center;min-width:0;}
  .stlu-line .grow{flex:1;}
  .stlu-label{color:#9a9a9a;white-space:nowrap;font-size:11px;}
  .stlu-input{flex:1;min-width:0;background:#0e0e0f;color:#eee;border:1px solid #3a3a3a;border-radius:4px;
    padding:4px 7px;font-size:12px;outline:none;}
  .stlu-input:focus{border-color:#4a90e2;}
  .stlu-btn{background:#2c2c2e;color:#ddd;border:1px solid #444;border-radius:4px;padding:4px 9px;cursor:pointer;
    font-size:12px;white-space:nowrap;}
  .stlu-btn:hover{background:#38383b;}
  .stlu-btn:active{background:#444;}
  .stlu-btn:disabled{opacity:.45;cursor:default;}
  .stlu-btn.primary{background:#2d6cdf;border-color:#2d6cdf;color:#fff;font-weight:600;padding:6px 14px;}
  .stlu-outdir{max-width:260px;overflow:hidden;text-overflow:ellipsis;text-align:left;}
  .stlu-btn.primary:hover{background:#3c7bec;}
  .stlu-mini{background:none;border:none;color:#6cb6ff;cursor:pointer;font-size:11px;padding:0 2px;}
  .stlu-mini:hover{text-decoration:underline;}
  .stlu-hint{color:#8d8d8d;font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .stlu-hint.warn{color:#e0a63c;}
  .stlu-hint.error{color:#e06c6c;}
  .stlu-hint.ok{color:#7fd18f;}
  .stlu-check{display:flex;align-items:center;gap:5px;cursor:pointer;white-space:nowrap;}
  .stlu-check input{margin:0;cursor:pointer;}
  .stlu-focusname{font-weight:600;color:#eaeaea;}
  .stlu-focusctx{flex:0 1 150px;font-size:11px;}
  .stlu-focusprompt{margin-top:2px;}
  .stlu-focusprompt summary{cursor:pointer;}
  .stlu-hints summary{cursor:pointer;color:#9a9a9a;}
  .stlu-hints .stlu-hint{white-space:normal;line-height:1.5;margin:2px 0;}
  .stlu-promptarea{width:100%;box-sizing:border-box;resize:vertical;margin-top:3px;
    font:11px/1.45 ui-monospace,Consolas,monospace;
    background:#0e0e0f;color:#ddd;border:1px solid #3a3a3c;border-radius:4px;padding:4px 6px;}
  .stlu-chips{display:flex;gap:4px;flex-wrap:wrap;}
  .stlu-chip{background:#252527;border:1px solid #3a3a3c;border-radius:999px;color:#bbb;padding:1px 9px;
    font-size:11px;cursor:pointer;white-space:nowrap;}
  .stlu-chip:hover{background:#303033;}
  .stlu-chip.on{background:rgba(45,108,223,.18);border-color:#2d6cdf;color:#9cc4ff;}
  .stlu-chip .n{opacity:.6;margin-left:3px;}
  .stlu-search{flex:1 1 160px;}
  .stlu-bulk{flex:0 1 190px;color:#bbb;}
  .stlu-list{flex:1 1 auto;min-height:110px;overflow-y:auto;border:1px solid #2d2d2f;border-radius:6px;background:#111112;}
  .stlu-sec{display:flex;align-items:center;gap:8px;padding:3px 8px;background:#1a1a1c;color:#9a9a9a;
    font-size:11px;position:sticky;top:0;z-index:1;border-bottom:1px solid #262628;}
  .stlu-row{display:grid;grid-template-columns:18px 42px minmax(120px,1fr) 168px;gap:8px;align-items:center;
    padding:4px 8px;border-bottom:1px solid #1f1f21;}
  .stlu-row:hover{background:#19191b;}
  .stlu-row.off .stlu-thumb{opacity:.45;}
  .stlu-row.off .stlu-name{color:#a9a9a9;font-weight:500;}
  .stlu-row input[type=checkbox]{margin:0;cursor:pointer;}
  .stlu-who{min-width:0;}
  .stlu-name{font-weight:600;color:#eaeaea;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
    display:flex;gap:5px;align-items:center;}
  .stlu-name .t{font-weight:400;}
  .stlu-sub{color:#8d8d8d;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .stlu-thumb{width:38px;height:38px;object-fit:contain;border-radius:3px;border:1px solid #2a2a2c;cursor:zoom-in;
    background:repeating-conic-gradient(#242426 0% 25%,#1a1a1c 0% 50%) 50%/10px 10px;}
  .stlu-tag{font-size:10px;padding:0 4px;border-radius:3px;background:#3a2f12;color:#e6b450;white-space:nowrap;}
  .stlu-tag.ok{background:#16351f;color:#7fd18f;}
  .stlu-tag.error{background:#3a1717;color:#f0a0a0;}
  .stlu-btn.attention{border-color:#e06c6c;color:#ffc0c0;}
  .stlu-sizebox{display:flex;align-items:center;gap:4px;justify-content:flex-end;}
  .stlu-size{width:74px;background:#0e0e0f;color:#eee;border:1px solid #3a3a3a;border-radius:4px;
    padding:3px 5px;font-size:12px;text-align:right;}
  .stlu-auto{border:1px solid #3a3a3c;background:#252527;color:#8d8d8d;border-radius:4px;font-size:10px;
    padding:2px 5px;cursor:pointer;}
  .stlu-auto.on{background:#1d3a26;color:#8fd6a2;border-color:#2f6b42;}
  .stlu-foot{display:flex;align-items:center;gap:6px;flex-wrap:wrap;}
  .stlu-status{flex:1 1 130px;color:#9a9a9a;font-size:11px;min-width:110px;}
  .stlu-status.error{color:#e06c6c;}
  .stlu-status.ok{color:#7fd18f;}
  .stlu-refthumb{width:34px;height:34px;object-fit:contain;border-radius:3px;border:1px solid #2a2a2c;
    background:repeating-conic-gradient(#242426 0% 25%,#1a1a1c 0% 50%) 50%/10px 10px;}
  .stlu-cardtitle{font-weight:600;color:#eaeaea;}
  .stlu-dot{width:8px;height:8px;border-radius:50%;background:#777;display:inline-block;flex:none;}
  .stlu-dot.ok{background:#4cc06a;}
  .stlu-dot.bad{background:#d05a5a;}
  .stlu-psd-prev{max-width:100%;max-height:150px;object-fit:contain;border-radius:4px;align-self:center;
    background:repeating-conic-gradient(#242426 0% 25%,#1a1a1c 0% 50%) 50%/14px 14px;}
  .stlu-overlay{position:fixed;inset:0;background:rgba(0,0,0,.62);z-index:10000;display:flex;
    align-items:center;justify-content:center;}
  .stlu-dialog{width:min(780px,92vw);height:min(620px,88vh);background:#1d1d1f;border:1px solid #444;
    border-radius:8px;display:flex;flex-direction:column;color:#ddd;font-size:13px;
    box-shadow:0 12px 40px rgba(0,0,0,.6);overflow:hidden;}
  .stlu-dialog.wide{width:min(1040px,94vw);}
  .stlu-dialog-head{display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid #333;}
  .stlu-dialog-head .grow{flex:1;}
  .stlu-close{background:none;border:none;color:#aaa;font-size:20px;cursor:pointer;line-height:1;}
  .stlu-crumbs{padding:6px 12px;border-bottom:1px solid #2a2a2c;display:flex;flex-wrap:wrap;gap:2px;
    font-size:12px;color:#9a9a9a;align-items:center;}
  .stlu-crumb{background:none;border:none;color:#6cb6ff;cursor:pointer;padding:1px 2px;font-size:12px;}
  .stlu-tabs{display:flex;gap:6px;padding:6px 12px;border-bottom:1px solid #2a2a2c;}
  .stlu-tab{background:#252527;border:1px solid #3a3a3c;border-radius:4px;color:#bbb;padding:3px 10px;
    font-size:12px;cursor:pointer;}
  .stlu-tab.on{background:rgba(45,108,223,.2);border-color:#2d6cdf;color:#9cc4ff;}
  .stlu-filelist{flex:1;overflow-y:auto;padding:4px 0;}
  .stlu-item{display:flex;align-items:center;gap:9px;padding:5px 14px;cursor:pointer;user-select:none;}
  .stlu-item:hover{background:#2a2a2d;}
  .stlu-item .icon{width:18px;text-align:center;flex:none;}
  .stlu-item .name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .stlu-item .tag{font-size:11px;color:#7fb36a;}
  .stlu-item img{width:34px;height:34px;object-fit:contain;border-radius:3px;
    background:repeating-conic-gradient(#242426 0% 25%,#1a1a1c 0% 50%) 50%/10px 10px;}
  .stlu-dialog-foot{display:flex;align-items:center;gap:10px;padding:10px 12px;border-top:1px solid #333;}
  .stlu-dialog-foot .path{flex:1;color:#9a9a9a;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .stlu-primary{background:#2d6cdf;color:#fff;border:none;border-radius:4px;padding:6px 14px;cursor:pointer;font-size:13px;}
  .stlu-primary:hover{background:#3c7bec;}
  .stlu-prevbox{flex:1;display:flex;align-items:center;justify-content:center;min-height:0;padding:12px;
    background:repeating-conic-gradient(#232325 0% 25%,#19191b 0% 50%) 50%/18px 18px;}
  .stlu-prevbox img{max-width:100%;max-height:100%;object-fit:contain;box-shadow:0 0 0 1px #333;}
  .stlu-prevmeta{padding:8px 14px;border-top:1px solid #333;color:#9a9a9a;font-size:12px;display:flex;gap:14px;
    flex-wrap:wrap;align-items:center;}
  .stlu-notes{margin:6px 0 0;padding:0 0 0 16px;color:#b08b4f;font-size:11px;max-height:64px;overflow:auto;}
  .stlu-layerbox{display:flex;gap:14px;padding:14px;flex:1;min-height:0;align-items:stretch;justify-content:center;
    background:repeating-conic-gradient(#232325 0% 25%,#19191b 0% 50%) 50%/18px 18px;}
  .stlu-layerbox figure{margin:0;flex:1;display:flex;flex-direction:column;align-items:center;min-width:0;gap:6px;}
  .stlu-layerbox img{max-width:100%;max-height:100%;min-height:0;object-fit:contain;}
  .stlu-layerbox figcaption{color:#9a9a9a;font-size:11px;}
  .stlu-prow{display:flex;gap:8px;align-items:flex-start;padding:4px 8px;border-bottom:1px solid #1f1f21;cursor:pointer;}
  .stlu-prow:hover{background:#19191b;}
  .stlu-prow input[type=checkbox]{margin:2px 0 0;cursor:pointer;flex:none;}
  .stlu-prow .who{min-width:0;display:flex;flex-direction:column;gap:1px;}
  .stlu-prow .who .n{font-weight:600;color:#eaeaea;}
  .stlu-prow .who .d{color:#8d8d8d;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .stlu-prow.off .who .n{color:#a9a9a9;font-weight:500;}
  .stlu-prow.hidden{display:none;}
  .stlu-tilegrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(52px,1fr));gap:4px;
    max-height:250px;overflow-y:auto;padding:2px;border:1px solid #2d2d2f;border-radius:6px;
    background:#111112;touch-action:manipulation;}
  .stlu-tcell{position:relative;aspect-ratio:1/1;border:1px solid #3a3a3c;border-radius:4px;
    background-color:#141415;background-repeat:no-repeat;cursor:pointer;user-select:none;
    display:flex;align-items:flex-end;justify-content:center;overflow:hidden;font-size:10px;}
  .stlu-tcell:hover{border-color:#6aa2ea;}
  .stlu-tcell.on{border-color:#5b93f5;box-shadow:inset 0 0 0 2px rgba(91,147,245,.6);}
  .stlu-tcell.on:after{content:"";position:absolute;inset:0;background:rgba(45,108,223,.22);}
  .stlu-tcell span{position:relative;z-index:1;background:rgba(0,0,0,.65);color:#ddd;
    padding:0 3px;border-radius:3px;margin-bottom:2px;}
  .stlu-gridwrap{display:flex;justify-content:center;align-items:center;background:#111112;
    border:1px solid #2d2d2f;border-radius:6px;padding:4px;min-height:60px;overflow:hidden;}
  .stlu-gridwrap canvas{max-width:100%;max-height:300px;display:block;cursor:pointer;border-radius:3px;}
  .stlu-gridwrap .stlu-stack{position:relative;display:inline-block;line-height:0;max-width:100%;}
  .stlu-gridwrap.custom .stlu-stack canvas{max-height:460px;}
  .stlu-gridwrap .stlu-vec{position:absolute;left:0;top:0;width:100%;height:100%;
    max-width:none;max-height:none;cursor:crosshair;outline:none;}
  .stlu-customlist{display:flex;flex-direction:column;gap:3px;max-height:220px;overflow-y:auto;
    padding:2px;border:1px solid #2d2d2f;border-radius:6px;background:#111112;}
  .stlu-crow{display:flex;align-items:center;gap:4px;padding:2px 3px;border-radius:4px;
    border:1px solid transparent;font-size:11px;min-width:0;}
  .stlu-crow:hover{background:#19191b;}
  .stlu-crow.active{border-color:#b8963a;background:rgba(184,150,58,.10);}
  .stlu-crow .idx{flex:none;min-width:18px;text-align:center;font-weight:600;color:#cfcfcf;}
  .stlu-crow input[type=checkbox]{margin:0;flex:none;cursor:pointer;}
  .stlu-crow input[type=number],.stlu-crow input[type=text],.stlu-crow select{
    background:#0e0e0f;color:#eee;border:1px solid #3a3a3a;border-radius:3px;padding:1px 3px;
    font-size:11px;min-width:0;}
  .stlu-crow input[type=number]{width:58px;}
  .stlu-crow input.cname{width:54px;}
  .stlu-crow .move{background:none;border:none;color:#8fb9f0;cursor:pointer;padding:0 2px;font-size:12px;}
  .stlu-crow .move:hover{color:#cfe2ff;}
  .stlu-crow .del{background:none;border:none;color:#d08a8a;cursor:pointer;padding:0 3px;font-size:12px;}
  .stlu-crow .del:hover{color:#f0a0a0;}
  .stlu-tilewall{display:grid;grid-template-columns:repeat(auto-fill,minmax(92px,1fr));gap:6px;
    max-height:330px;overflow-y:auto;padding:4px;border:1px solid #2d2d2f;border-radius:6px;background:#111112;}
  .stlu-tile{position:relative;border:2px solid transparent;border-radius:5px;background:#141415;
    cursor:pointer;padding:2px;user-select:none;}
  .stlu-tile:hover{border-color:#3a3a3c;}
  .stlu-tile.on{border-color:#2d6cdf;background:rgba(45,108,223,.12);}
  .stlu-tile img{width:100%;aspect-ratio:1/1;object-fit:contain;display:block;border-radius:3px;
    background:repeating-conic-gradient(#242426 0% 25%,#1a1a1c 0% 50%) 50%/12px 12px;}
  .stlu-tile .badge{position:absolute;left:3px;top:3px;font-size:10px;line-height:14px;
    background:rgba(0,0,0,.68);color:#ddd;border-radius:3px;padding:0 4px;pointer-events:none;}
  .stlu-tile .chk{position:absolute;right:4px;top:4px;width:12px;height:12px;border-radius:3px;
    border:1px solid #666;background:rgba(0,0,0,.55);pointer-events:none;}
  .stlu-tile.on .chk{background:#2d6cdf;border-color:#5b93f5;}
  .stlu-tile .empty{width:100%;aspect-ratio:1/1;display:flex;align-items:center;justify-content:center;
    color:#666;font-size:10px;}
  .stlu-merge-prev{max-width:100%;max-height:220px;object-fit:contain;align-self:center;border-radius:4px;
    background:repeating-conic-gradient(#242426 0% 25%,#1a1a1c 0% 50%) 50%/14px 14px;}
  .stlu-wavegrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;}
  .stlu-wavegrid-wrap{max-height:280px;overflow-y:auto;padding:2px;}
  .stlu-wavecard{display:flex;flex-direction:column;gap:4px;background:#141415;border:2px solid #2f2f31;
    border-radius:6px;padding:5px;cursor:pointer;user-select:none;min-width:0;}
  .stlu-wavecard:hover{border-color:#3f3f43;}
  .stlu-wavecard.on{border-color:#2d6cdf;background:rgba(45,108,223,.12);}
  .stlu-wavecard img{width:100%;height:96px;object-fit:contain;border-radius:4px;
    background:repeating-conic-gradient(#242426 0% 25%,#1a1a1c 0% 50%) 50%/12px 12px;}
  .stlu-wavecard .n{font-weight:600;color:#eaeaea;font-size:12px;white-space:nowrap;overflow:hidden;
    text-overflow:ellipsis;}
  .stlu-wavecard .m{color:#8d8d8d;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .stlu-wavecard.live{justify-content:center;align-items:center;min-height:128px;text-align:center;}
  .stlu-wavecard .liveicon{font-size:26px;}
  .stlu-tilepick{display:grid;grid-template-columns:repeat(auto-fill,minmax(78px,1fr));gap:5px;
    max-height:230px;overflow-y:auto;padding:3px;border:1px solid #2d2d2f;border-radius:6px;background:#111112;}
  .stlu-tpick{position:relative;border:2px solid transparent;border-radius:5px;background:#141415;
    cursor:pointer;padding:2px;user-select:none;}
  .stlu-tpick:hover{border-color:#3a3a3c;}
  .stlu-tpick.on{border-color:#2d6cdf;background:rgba(45,108,223,.12);}
  .stlu-tpick img{width:100%;aspect-ratio:1/1;object-fit:contain;display:block;border-radius:3px;
    background:repeating-conic-gradient(#242426 0% 25%,#1a1a1c 0% 50%) 50%/10px 10px;}
  .stlu-tpick .b{position:absolute;left:3px;top:3px;font-size:10px;line-height:14px;border-radius:3px;
    padding:0 4px;background:rgba(0,0,0,.68);color:#ddd;pointer-events:none;}
  .stlu-tpick .c{position:absolute;right:4px;top:4px;width:12px;height:12px;border-radius:3px;
    border:1px solid #666;background:rgba(0,0,0,.55);pointer-events:none;}
  .stlu-tpick.on .c{background:#2d6cdf;border-color:#5b93f5;}
  .stlu-sec2{font-weight:600;color:#cfcfcf;font-size:12px;margin-top:2px;}
  .stlu-runrow{display:flex;gap:6px;align-items:center;flex-wrap:wrap;}
  .stlu-runrow .stlu-btn.primary{padding:7px 18px;font-size:13px;}
  `;

  // -------------------------------------------------------------------------
  // small helpers
  // -------------------------------------------------------------------------

  const apiUrl = (p) => (api && api.apiURL ? api.apiURL(p) : p);

  function injectCss() {
    if (document.getElementById("stlu-style")) return;
    const style = document.createElement("style");
    style.id = "stlu-style";
    style.textContent = CSS;
    document.head.appendChild(style);
  }

  async function getJson(path) {
    const res = await fetch(apiUrl(path), { cache: "no-store" });
    if (!res.ok) {
      let msg = res.status + " " + res.statusText;
      try {
        const j = await res.json();
        if (j && j.error) msg = j.error;
      } catch (e) { /* not json */ }
      throw new Error(msg);
    }
    return res.json();
  }

  async function postJson(path, body) {
    const res = await fetch(apiUrl(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    let data = null;
    try { data = await res.json(); } catch (e) { /* not json */ }
    if (!res.ok || (data && data.error)) {
      throw new Error((data && data.error) || (res.status + " " + res.statusText));
    }
    return data || {};
  }

  // stlu_i18n.js loads after this file, so never cache the module at top level.
  function i18n() {
    return window.STLU_I18N || null;
  }

  function tr(text) {
    const m = i18n();
    return m && m.enabled ? m.t(text) : text;
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = tr(text);
    return node;
  }

  function fmtBytes(n) {
    if (!n) return "";
    if (n >= 1024 * 1024) return (n / 1048576).toFixed(1) + " MB";
    return Math.max(1, Math.round(n / 1024)) + " KB";
  }

  function round32(v) {
    return Math.max(32, Math.round(v / 32) * 32);
  }

  function widgetOf(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
  }

  // Values may live in a core Primitive* node parked in the "用户输入" group
  // and arrive through a widget socket.  Read and write through that link so
  // the panels always show the number the queued run will actually use.
  const PRIMITIVE_TYPES = new Set([
    "PrimitiveInt", "PrimitiveFloat", "PrimitiveString",
    "PrimitiveStringMultiline", "PrimitiveBoolean",
  ]);

  function linkedOrigin(node, name) {
    const input = ((node && node.inputs) || []).find((i) => i.name === name);
    if (!input || input.link == null) return null;
    const link = linkById(input.link);
    if (!link) return null;
    const originId = link.origin_id != null ? link.origin_id : link[1];
    if (originId == null) return null;
    return allNodes().find((n) => n.id === originId) || null;
  }

  function primitiveWidget(origin) {
    if (!origin || !PRIMITIVE_TYPES.has(origin.type)) return null;
    return widgetOf(origin, "value") || (origin.widgets || [])[0] || null;
  }

  function widgetValue(node, name, fallback) {
    const pw = primitiveWidget(linkedOrigin(node, name));
    if (pw) return pw.value;
    const w = widgetOf(node, name);
    return w ? w.value : fallback;
  }

  // The widget that actually drives this input: the upstream primitive's
  // "value" when linked, otherwise the node's own widget.
  function valueWidget(node, name) {
    return primitiveWidget(linkedOrigin(node, name)) || widgetOf(node, name);
  }

  // 旧工作流把默认输出目录写死成 SeeThrough/UpScale。现在空值 = 自动，
  // 即「当前拆层文件夹/UpScale」；读到旧默认值就迁移成自动，免得换了个角色
  // 仍把结果混进同一个目录。后端 core_plan.LEGACY_OUT_DIR 有同样的兜底。
  const LEGACY_OUT_DIR = "SeeThrough/UpScale";

  function readOutDir(node) {
    const value = String(widgetValue(node, "out_dir", "") || "").trim().replace(/\\/g, "/");
    if (!value || value === LEGACY_OUT_DIR) return "";
    return value;
  }

  function outDirLabel(ctx) {
    if (ctx.outDir) return ctx.outDir;
    return ctx.autoOutDir ? "自动 · " + ctx.autoOutDir : "自动";
  }

  function outDirPath(ctx) {
    return ctx.outDir || ctx.autoOutDir || "";
  }

  // Stage ③ writes straight into this folder: no per-run subfolder.  Tiles
  // already there are reused by the next partial run.
  function wavePreview(node) {
    const outDir = String(widgetValue(node, "out_dir", "SeeThrough") || "SeeThrough")
      .replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
    const group = String(widgetValue(node, "group", "") || "")
      .replace(/\\/g, "/").replace(/^\/+|\/+$/g, "").trim();
    return "output/" + outDir + (group ? "/" + group : "") + "/";
  }

  function previewSrc(preview) {
    if (!preview || !preview.filename) return "";
    return apiUrl(
      "/view?filename=" + encodeURIComponent(preview.filename) +
      "&subfolder=" + encodeURIComponent(preview.subfolder || "") +
      "&type=" + encodeURIComponent(preview.type || "output") +
      "&v=" + encodeURIComponent(preview.v || SESSION_V)
    );
  }

  function hideWidget(w) {
    if (!w) return;
    // Hide visually only: the value must keep serializing into widgets_values.
    w.hidden = true;
    w.computeSize = () => [0, -4];
    if (w.options) w.options.hidden = true;
    w.draw = () => {};
    for (const linked of w.linkedWidgets || []) hideWidget(linked);
  }

  // Locked nodes are the fixed pipeline: parameters come from the input node,
  // so their own widgets only get in the way.  The workflow marks them with
  // ``properties.stlu_locked``; the values still serialize and execute.
  function applyLockedWidgets(node) {
    if (!node || !node.properties || !node.properties.stlu_locked) return;
    const keep = Array.isArray(node.properties.stlu_show_widgets)
      ? new Set(node.properties.stlu_show_widgets) : null;
    for (const w of node.widgets || []) {
      if (keep && keep.has(w.name)) continue;
      hideWidget(w);
    }
  }

  // Workflows can hide widgets that do nothing on their path (the plain
  // pipeline does not use the guide_* parameters) without removing them:
  // values still serialize, so switching back to the guided workflow keeps
  // its settings.  No property = nothing is hidden.
  function applyHiddenWidgets(node) {
    if (!node || !node.properties) return;
    const hidden = node.properties.stlu_hidden_widgets;
    if (!Array.isArray(hidden) || !hidden.length) return;
    const names = new Set(hidden);
    for (const w of node.widgets || []) {
      if (names.has(w.name)) hideWidget(w);
    }
  }

  function setWidget(node, name, value, depth) {
    const origin = linkedOrigin(node, name);
    const pw = primitiveWidget(origin);
    if (pw && (depth || 0) < 8) {
      // The input is fed by a primitive: update that node instead, otherwise
      // the write would land on a hidden widget the run never reads.
      return setWidget(origin, pw.name, value, (depth || 0) + 1);
    }
    const w = widgetOf(node, name);
    if (!w) return false;
    if (w.value === value) return true;
    w.value = value;
    try {
      if (typeof w.callback === "function") w.callback(value, app.canvas, node, null, null);
    } catch (e) {
      console.warn("[STLU] widget callback failed", name, e);
    }
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    if (app.graph && app.graph.setDirtyCanvas) app.graph.setDirtyCanvas(true, false);
    return true;
  }

  // Run ``cb`` whenever the widget behind an input changes, whether it is
  // the node's own widget or the primitive it is linked to.
  const watchedWidgets = new WeakMap();
  function watchValue(node, name, cb) {
    const w = valueWidget(node, name);
    if (!w) return false;
    let listeners = watchedWidgets.get(w);
    if (!listeners) {
      listeners = new Set();
      watchedWidgets.set(w, listeners);
      const old = w.callback;
      w.callback = function () {
        const result = old ? old.apply(this, arguments) : undefined;
        for (const fn of Array.from(listeners)) {
          try { fn(); } catch (e) { console.warn("[STLU] watchValue", e); }
        }
        return result;
      };
    }
    listeners.add(cb);
    return true;
  }

  function allNodes() {
    const g = app.graph;
    if (!g) return [];
    return g.nodes || g._nodes || [];
  }

  function linkById(id) {
    const g = app.graph;
    if (!g || !g.links || id == null) return null;
    const links = g.links;
    if (typeof links.get === "function") return links.get(id) || null;
    return links[id] || null;
  }

  function findBatchNode() {
    return allNodes().find((n) => n.type === BATCH) || null;
  }

  function findReferenceNode() {
    const batch = findBatchNode();
    if (batch) {
      const input = (batch.inputs || []).find((i) => i.name === "reference");
      const link = input && input.link != null ? linkById(input.link) : null;
      if (link) {
        const originId = link.origin_id != null ? link.origin_id : link[1];
        const origin = allNodes().find((n) => n.id === originId);
        if (origin) return origin;
      }
    }
    return allNodes().find((n) => n.type === "LoadImage") || null;
  }

  async function openPath(path, file) {
    try {
      await getJson("/stlu/" + (file ? "open" : "reveal") + "?path=" + encodeURIComponent(path || ""));
    } catch (e) {
      console.warn("[STLU] open failed", e);
    }
  }

  function addDOM(node, name, root, minHeight, maxHeight) {
    if (typeof node.addDOMWidget !== "function") return null;
    const dom = node.addDOMWidget(name, name, root, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: typeof minHeight === "function" ? minHeight : () => minHeight,
      getMaxHeight: typeof maxHeight === "number" ? () => maxHeight : undefined,
    });
    if (dom) {
      dom.serialize = false;
      dom.options = dom.options || {};
      dom.options.serialize = false;
    }
    const m = i18n();
    if (m && m.enabled && root) m.watch(root);
    return dom;
  }

  // -------------------------------------------------------------------------
  // shared UI plumbing: thumbs, panels, user mode, tile-run actions
  // -------------------------------------------------------------------------

  function thumbUrl(rel, size) {
    if (!rel) return "";
    return apiUrl("/stlu/thumb?scope=output&path=" + encodeURIComponent(rel) +
      "&size=" + (size || 128) + "&v=" + SESSION_V);
  }

  function fmtWhen(ms) {
    if (!ms) return "";
    const date = new Date(Number(ms) * 1000);
    const pad = (n) => String(n).padStart(2, "0");
    return date.getFullYear() + "-" + pad(date.getMonth() + 1) + "-" + pad(date.getDate()) +
      " " + pad(date.getHours()) + ":" + pad(date.getMinutes());
  }

  function parseIndexList(text) {
    const raw = String(text == null ? "" : text).trim().toLowerCase();
    if (!raw || raw === "all") return null;
    if (raw === "none") return [];
    const out = new Set();
    for (const chunk of raw.replace(/\s+/g, "").split(",")) {
      if (!chunk) continue;
      if (chunk.includes("-")) {
        const [a, b] = chunk.split("-").map((v) => parseInt(v, 10));
        if (Number.isFinite(a) && Number.isFinite(b)) {
          for (let i = Math.min(a, b); i <= Math.max(a, b); i++) out.add(i);
        }
      } else {
        const v = parseInt(chunk, 10);
        if (Number.isFinite(v)) out.add(v);
      }
    }
    return Array.from(out).sort((a, b) => a - b);
  }

  function findNodeOfType(type) {
    return allNodes().find((n) => n.type === type) || null;
  }

  function registerPanel(node, kind, dom, root) {
    const element = (dom && (dom.element || dom.el)) || root || null;
    const entry = { node, kind, element, dom };
    panelRegistry.set(node.id, entry);
    return entry;
  }

  function notifyGrid(node, source) {
    if (gridBroadcast) return;
    gridBroadcast = true;
    try {
      for (const panel of Array.from(gridPanels)) {
        if (panel.node !== node || panel === source) continue;
        try { panel.onExternal(); } catch (e) { console.warn("[STLU] grid sync", e); }
      }
    } finally {
      gridBroadcast = false;
    }
  }

  function autoFillWidget(type, widgetName, value) {
    if (!value) return;
    for (const other of allNodes()) {
      if (other.type !== type) continue;
      const widget = widgetOf(other, widgetName);
      if (!widget) continue;
      const input = (other.inputs || []).find((i) => i.name === widgetName);
      if (input && input.link != null) continue;
      if (String(widget.value || "").trim()) continue;
      setWidget(other, widgetName, String(value));
    }
  }

  function refreshTilesPanels() {
    for (const fn of Array.from(tilesStates)) {
      try { fn(); } catch (e) { console.warn("[STLU] tiles panel", e); }
    }
  }

  function linkedImagesDir(node) {
    return upstreamNode(node, "source") ? lastImagesDir : "";
  }

  function tilesNodeSettings(node) {
    const source = String(widgetValue(node, "source", "") || "").trim() || linkedImagesDir(node);
    let gridRef = String(widgetValue(node, "grid", "") || "").trim();
    if (!gridRef && upstreamNode(node, "grid")) gridRef = lastGridDoc || lastGridPath;
    return {
      source,
      grid: gridRef,
      out_dir: String(widgetValue(node, "out_dir", "SeeThrough") || "SeeThrough"),
      group: String(widgetValue(node, "group", "") || "").trim(),
      tiles: parseIndexList(widgetValue(node, "tiles", "")),
      name: String(widgetValue(node, "filename_prefix", "tile") || "tile"),
      merge_parts: !!widgetValue(node, "merge_parts", true),
      merged_psd: !!widgetValue(node, "merged_psd", true),
      write_depth: !!widgetValue(node, "write_depth", true),
      group_layers: !!widgetValue(node, "group_layers", true),
      include_original: !!widgetValue(node, "include_original", true),
      alpha_threshold: Number(widgetValue(node, "alpha_threshold", 8)) || 8,
    };
  }

  async function runTilesFromImages(node) {
    const body = tilesNodeSettings(node);
    if (!body.source) throw new Error("先在 ③ 里选一份历史结果");
    const data = await postJson("/stlu/tiles/from_images", body);
    handleTilesInfo(data);
    return data;
  }

  function handleTilesInfo(data) {
    if (!data) return;
    lastTilesInfo = data;
    if (data.dir) lastRunDir = data.dir;
    autoFillWidget(TILE_MERGE, "run_dir", data.dir || data.subfolder);
    for (const fn of Array.from(tilesStates)) {
      try { fn(); } catch (e) { console.warn("[STLU] tiles panel", e); }
    }
    for (const fn of Array.from(stage3States)) {
      try { fn(data); } catch (e) { console.warn("[STLU] stage 3 panel", e); }
    }
  }

  function handleMergeInfo(data) {
    if (!data) return;
    lastMergeInfo = data;
    if (data.dir) lastRunDir = data.dir;
    autoFillWidget(TILE_MERGE, "run_dir", data.dir);
    for (const fn of Array.from(mergeStates)) {
      try { fn(data); } catch (e) { console.warn("[STLU] merge panel", e); }
    }
  }

  // -------------------------------------------------------------------------
  // mixed merge: every tile may come from a different run
  // -------------------------------------------------------------------------

  async function openMergeEditor(mergeNode) {
    let dirs = [];
    try {
      const data = await getJson("/stlu/tiles/dirs");
      dirs = (data.dirs || []).filter((entry) => entry.dir && entry.count);
    } catch (e) { dirs = []; }
    if (!dirs.length) {
      window.alert(tr("还没有可分格合并的波次：先跑一次 ③。"));
      return;
    }
    const manifests = new Map();
    await Promise.all(dirs.slice(0, 40).map(async (entry) => {
      try {
        const doc = await getJson("/stlu/tiles/manifest?dir=" + encodeURIComponent(entry.dir));
        if (doc && Array.isArray(doc.tiles) && doc.tiles.length) manifests.set(entry.dir, doc);
      } catch (e) { /* skip an unreadable run */ }
    }));
    const usable = dirs.filter((entry) => manifests.has(entry.dir));
    if (!usable.length) {
      window.alert(tr("这些波次都没有可读的分块 PSD。"));
      return;
    }

    const { dialog, close } = openOverlay("wide");
    const head = el("div", "stlu-dialog-head");
    head.appendChild(el("strong", "grow", "逐块选来源（混合合并）"));
    head.appendChild(closeButton(close));
    dialog.appendChild(head);

    const tip = el("div", "stlu-hint",
      "每个格子可以来自不同的波次。同一个部件如果在多个波次里都出现，只取你选的那个波次——" +
      "不会叠出两张脸。" +
      "勾选“同名图层单独成组”时输出不裁剪版：同名部件各自保留完整原件并收进同名文件夹；" +
      "不勾时按接缝归属裁切成直出合并版。" +
      "勾选“导出 depth PSD”时另存为 <名字>_depth.psd，绝不会出现在彩色 PSD 里。");
    tip.style.padding = "6px 14px 0";
    dialog.appendChild(tip);

    const baseLine = el("div", "stlu-line");
    baseLine.style.padding = "6px 14px 0";
    baseLine.appendChild(el("span", "stlu-label", "主波次"));
    const baseChips = el("div", "stlu-chips grow");
    baseLine.appendChild(baseChips);
    dialog.appendChild(baseLine);

    const body = el("div", "stlu-filelist");
    body.style.padding = "8px 14px";
    dialog.appendChild(body);

    const foot = el("div", "stlu-line");
    foot.style.padding = "8px 14px 12px";
    const mergeBtn = el("button", "stlu-btn primary", "合并选中的分块");
    foot.appendChild(mergeBtn);
    dialog.appendChild(foot);

    const member = new Map();
    usable.forEach((entry, i) => member.set(entry.dir, "波" + (i + 1)));

    function shortName(dir) {
      const parts = String(dir).split("/");
      const name = parts[parts.length - 1] || String(dir);
      const m = name.match(/(\d{8}_\d{6})/);
      return m ? m[1] : name.slice(-12);
    }

    const wanted = String(widgetValue(mergeNode, "run_dir", "") || "");
    let base = usable.find((entry) => entry.dir === wanted || entry.dir === lastRunDir) || usable[0];
    const choice = new Map();
    let prioritySeq = 0;

    function resetChoice() {
      choice.clear();
      prioritySeq = 0;
      const doc = manifests.get(base.dir);
      for (const tile of doc.tiles || []) {
        choice.set(Number(tile.index), { dir: base.dir, priority: 0 });
      }
    }

    function tileOptions(index) {
      return usable.filter((entry) => (manifests.get(entry.dir).tiles || [])
        .some((t) => Number(t.index) === Number(index)));
    }

    function thumbOf(dir, index) {
      const doc = manifests.get(dir);
      const tile = (doc.tiles || []).find((t) => Number(t.index) === Number(index));
      if (!tile) return "";
      const file = tile.preview || tile.depth || tile.file || "";
      return file ? thumbUrl(dir + "/" + file, 120) : "";
    }

    function renderBase() {
      baseChips.innerHTML = "";
      for (const entry of usable) {
        const chip = el("button", "stlu-chip" + (entry.dir === base.dir ? " on" : ""),
          member.get(entry.dir) + " · " + shortName(entry.dir));
        chip.title = entry.dir;
        chip.addEventListener("click", () => {
          base = entry;
          resetChoice();
          renderBase();
          renderTiles();
        });
        baseChips.appendChild(chip);
      }
    }

    function renderTiles() {
      body.innerHTML = "";
      const doc = manifests.get(base.dir);
      const tiles = (doc.tiles || []).slice().sort((a, b) => Number(a.index) - Number(b.index));
      for (const tile of tiles) {
        const index = Number(tile.index);
        const row = el("div", "stlu-line");
        row.style.gap = "8px";
        const img = document.createElement("img");
        img.style.cssText = "width:56px;height:56px;object-fit:cover;border-radius:4px;background:#111;";
        const thumb = thumbOf(base.dir, index);
        if (thumb) img.src = thumb;
        row.appendChild(img);
        row.appendChild(el("span", "stlu-label", "块 " + index));
        const box = tile.tile || [0, 0, 0, 0];
        row.appendChild(el("span", "stlu-hint", "x" + box[0] + " y" + box[1]));
        row.appendChild(el("span", "grow"));
        const pick = document.createElement("select");
        pick.className = "stlu-input";
        pick.style.width = "200px";
        for (const entry of tileOptions(index)) {
          const opt = document.createElement("option");
          opt.value = entry.dir;
          opt.textContent = member.get(entry.dir) + " · " + shortName(entry.dir);
          const cur = choice.get(index);
          if (cur && cur.dir === entry.dir) opt.selected = true;
          pick.appendChild(opt);
        }
        pick.addEventListener("change", () => {
          const dir = pick.value;
          choice.set(index, { dir, priority: dir === base.dir ? 0 : ++prioritySeq });
        });
        row.appendChild(pick);
        body.appendChild(row);
      }
      if (!tiles.length) body.appendChild(el("div", "stlu-hint", "这个波次没有可合并的块。"));
    }

    mergeBtn.addEventListener("click", async () => {
      mergeBtn.disabled = true;
      try {
        const entries = [];
        for (const [index, pick] of choice) {
          entries.push({ dir: pick.dir, index, priority: pick.priority });
        }
        mergeBtn.textContent = "正在合并…";
        const data = await postJson("/stlu/tiles/merge-mixed", {
          entries,
          name: "merged_mixed",
          group_layers: !!widgetValue(mergeNode, "group_layers", false),
          include_depth: !!widgetValue(mergeNode, "include_depth", false),
          include_original: !!widgetValue(mergeNode, "include_original", false),
        });
        handleMergeInfo(data);
        refreshTilesPanels();
        close();
      } catch (e) {
        mergeBtn.textContent = "合并失败：" + e.message;
      } finally {
        mergeBtn.disabled = false;
      }
    });

    resetChoice();
    renderBase();
    renderTiles();
  }

  // -------------------------------------------------------------------------
  // picker dialog (folders / images / workflows)
  // -------------------------------------------------------------------------

  function openOverlay(className) {
    injectCss();
    const overlay = el("div", "stlu-overlay");
    const dialog = el("div", "stlu-dialog" + (className ? " " + className : ""));
    overlay.appendChild(dialog);
    const close = () => {
      document.removeEventListener("keydown", onKey);
      overlay.remove();
    };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.addEventListener("keydown", onKey);
    document.body.appendChild(overlay);
    return { overlay, dialog, close };
  }

  function closeButton(close) {
    const btn = el("button", "stlu-close", "×");
    btn.title = "关闭";
    btn.addEventListener("click", close);
    return btn;
  }

  function showBigImage(rel, title) {
    const { dialog, close } = openOverlay();
    const head = el("div", "stlu-dialog-head");
    head.appendChild(el("strong", "grow", title || "预览"));
    head.appendChild(closeButton(close));
    dialog.appendChild(head);
    const box = el("div", "stlu-prevbox");
    const img = document.createElement("img");
    img.src = apiUrl(
      "/stlu/thumb?scope=output&path=" + encodeURIComponent(rel) + "&size=1400&v=" + Date.now()
    );
    box.appendChild(img);
    dialog.appendChild(box);
  }

  /**
   * opts = {
   *   title, mode: "folder"|"image"|"workflow",
   *   scopes: [{scope, label}], start,
   *   onPick(rel, abs, file)
   * }
   */
  function openPicker(opts) {
    const { dialog, close } = openOverlay();
    const head = el("div", "stlu-dialog-head");
    head.appendChild(el("strong", "grow", opts.title || "选择"));
    head.appendChild(closeButton(close));
    dialog.appendChild(head);

    const scopes = opts.scopes && opts.scopes.length ? opts.scopes : [{ scope: "output", label: "输出目录" }];
    let scope = scopes[0].scope;
    let current = cleanRel(opts.start);
    let tabsEl = null;
    if (scopes.length > 1) {
      tabsEl = el("div", "stlu-tabs");
      for (const s of scopes) {
        const b = el("button", "stlu-tab", s.label);
        b.dataset.scope = s.scope;
        b.addEventListener("click", () => {
          scope = s.scope;
          current = "";
          for (const child of tabsEl.children) child.classList.toggle("on", child.dataset.scope === scope);
          nav("");
        });
        tabsEl.appendChild(b);
      }
      tabsEl.firstChild.classList.add("on");
      dialog.appendChild(tabsEl);
    }

    const isImageMode = opts.mode === "image";
    const crumbsEl = el("div", "stlu-crumbs");
    const listEl = el("div", "stlu-filelist");
    const foot = el("div", "stlu-dialog-foot");
    const pathEl = el("span", "path");
    foot.appendChild(pathEl);
    if (opts.mode === "folder") {
      const use = el("button", "stlu-primary", "使用此文件夹");
      use.addEventListener("click", () => {
        const rel = current;
        close();
        opts.onPick(rel, null, null);
      });
      foot.appendChild(use);
    }
    dialog.appendChild(crumbsEl);
    if (isImageMode) {
      const note = el("div", "stlu-hint");
      note.style.padding = "0 14px 4px";
      note.textContent = "LoadImage 只读取 input 根目录；子文件夹里的图请先上传或移动到根目录。";
      dialog.appendChild(note);
    }
    dialog.appendChild(listEl);
    dialog.appendChild(foot);

    function cleanRel(value) {
      let rel = String(value || "").replace(/\\/g, "/").trim();
      if (/^[a-zA-Z]:/.test(rel) || rel.startsWith("/")) rel = "";
      if (/\.(json|png|jpe?g|webp|psd)$/i.test(rel)) {
        rel = rel.includes("/") ? rel.slice(0, rel.lastIndexOf("/")) : "";
      }
      return rel.replace(/^\.?\/+/, "").replace(/\/+$/, "");
    }

    function pickFile(file) {
      close();
      opts.onPick(file.rel, file.abs || null, file);
    }

    async function nav(rel) {
      listEl.innerHTML = "";
      listEl.appendChild(el("div", "stlu-item", "加载中…"));
      let data;
      try {
        data = await getJson(
          "/stlu/ls?scope=" + encodeURIComponent(scope) + "&path=" + encodeURIComponent(rel || "")
        );
      } catch (e) {
        listEl.innerHTML = "";
        const err = el("div", "stlu-item");
        err.appendChild(el("span", "icon", "⚠"));
        err.appendChild(el("span", "name", "无法读取：" + e.message));
        listEl.appendChild(err);
        return;
      }
      current = data.path || "";
      pathEl.textContent = (scope === "input" ? "input/" : scope === "workflows" ? "workflows/" : "output/") +
        (current ? current + "/" : "");

      crumbsEl.innerHTML = "";
      const crumb = (label, target) => {
        const b = el("button", "stlu-crumb", label);
        b.addEventListener("click", () => nav(target));
        crumbsEl.appendChild(b);
      };
      crumb(scope === "input" ? "input" : scope === "workflows" ? "workflows" : "output", "");
      let acc = "";
      for (const part of current ? current.split("/") : []) {
        acc = acc ? acc + "/" + part : part;
        crumbsEl.appendChild(el("span", null, "/"));
        crumb(part, acc);
      }

      listEl.innerHTML = "";
      if (!isImageMode && !data.at_root) {
        const up = el("div", "stlu-item");
        up.appendChild(el("span", "icon", "⬆"));
        up.appendChild(el("span", "name", "上一级"));
        up.addEventListener("click", () => nav(data.parent || ""));
        listEl.appendChild(up);
      }
      if (!isImageMode) {
        for (const dir of data.dirs || []) {
          const item = el("div", "stlu-item");
          item.appendChild(el("span", "icon", "📁"));
          item.appendChild(el("span", "name", dir.name));
          item.addEventListener("click", () => nav(dir.rel));
          listEl.appendChild(item);
        }
      }
      for (const file of data.files || []) {
        const item = el("div", "stlu-item");
        if (file.thumb) {
          const img = document.createElement("img");
          img.loading = "lazy";
          img.src = apiUrl(file.thumb);
          img.addEventListener("error", () => { img.style.visibility = "hidden"; });
          item.appendChild(img);
        } else {
          item.appendChild(el("span", "icon", file.is_layers ? "🧩" : "📄"));
        }
        item.appendChild(el("span", "name", file.name));
        const tag = el("span", "tag",
          file.is_layers ? "layers.json" : file.is_workflow ? "工作流" : file.is_image ? "图片" : fmtBytes(file.size));
        item.appendChild(tag);
        if (opts.mode === "folder") {
          if (file.is_layers) {
            item.addEventListener("click", () => {
              const parent = file.rel.includes("/") ? file.rel.slice(0, file.rel.lastIndexOf("/")) : "";
              close();
              opts.onPick(parent, null, file);
            });
          }
        } else if (opts.mode === "image" && file.is_image) {
          item.addEventListener("click", () => pickFile(file));
        } else if (opts.mode === "workflow" && file.is_workflow) {
          item.addEventListener("click", () => pickFile(file));
        }
        listEl.appendChild(item);
      }
      if (!(data.dirs || []).length && !(data.files || []).length) {
        const empty = el("div", "stlu-item");
        empty.appendChild(el("span", "icon", "·"));
        empty.appendChild(el("span", "name", "空文件夹"));
        listEl.appendChild(empty);
      }
    }

    nav(current);
  }

  // -------------------------------------------------------------------------
  // Plan panel
  // -------------------------------------------------------------------------

  const planStates = new Set();

  function parseSelectNames(text) {
    const t = (text || "").trim();
    if (!t || t.toLowerCase() === "all" || t === "*") return null;
    const names = new Set();
    for (const item of t.split(",")) {
      const name = item.trim();
      if (!name || name.startsWith("!") || name.startsWith("-")) continue;
      if (/[*?[]/.test(name)) continue;
      names.add(name);
    }
    return names;
  }

  function parseSizes(text) {
    try {
      const obj = JSON.parse(text || "{}");
      const out = {};
      for (const [key, value] of Object.entries(obj)) {
        if (key === "*" || !value) continue;
        if (typeof value === "number") out[key] = value;
        else if (typeof value === "object" && value.long_side != null) out[key] = Number(value.long_side);
      }
      return out;
    } catch (e) {
      return {};
    }
  }

  // Rules that are globs stay "template defaults"; literal layer names are
  // turned into pinned rows by parseSizes so "自动" can really reset them.
  function templateRules(base) {
    const out = {};
    try {
      for (const [key, value] of Object.entries(base || {})) {
        if (key === "*" || /[*?[]/.test(key)) out[key] = value;
      }
    } catch (e) { /* ignore */ }
    if (!out["*"]) out["*"] = {};
    return out;
  }

  function categoryOf(name) {
    for (const cat of CATEGORIES) if (cat.test(name)) return cat;
    return CATEGORIES[CATEGORIES.length - 1];
  }

  function entryFor(ctx, layer) {
    let entry = ctx.entries.get(layer.name);
    if (!entry) {
      let checked = !layer.empty;
      let longSide = layer.suggested.long_side;
      let auto = true;
      // Restore the selection / pinned sizes saved in the workflow the first
      // time the layer list is loaded after opening or reloading the graph.
      if (ctx.restore) {
        if (ctx.savedSelectNames) checked = ctx.savedSelectNames.has(layer.name);
        if (ctx.savedSizes && ctx.savedSizes[layer.name] != null) {
          longSide = Number(ctx.savedSizes[layer.name]);
          auto = false;
        }
      }
      entry = { checked, longSide, auto };
      ctx.entries.set(layer.name, entry);
    } else if (entry.auto) {
      entry.longSide = layer.suggested.long_side;
    }
    return entry;
  }

  function buildPlanPanel(node, ctx) {
    const root = el("div", "stlu-panel");

    const sourceCard = el("div", "stlu-card");
    const srcLine = el("div", "stlu-line");
    srcLine.appendChild(el("span", "stlu-label", "拆层文件夹"));
    ctx.pathInput = el("input", "stlu-input stlu-path");
    ctx.pathInput.type = "text";
    ctx.pathInput.spellcheck = false;
    ctx.pathInput.placeholder = "output 下的文件夹，例如 SeeThrough";
    srcLine.appendChild(ctx.pathInput);
    const browseBtn = el("button", "stlu-btn", "浏览…");
    browseBtn.addEventListener("click", () => {
      openPicker({
        title: "选择拆层文件夹",
        mode: "folder",
        scopes: [{ scope: "output", label: "输出目录" }],
        start: ctx.pathInput.value.trim(),
        onPick: (rel) => {
          ctx.pathInput.value = rel;
          setWidget(ctx.node, "layers_json", rel);
          refresh(ctx);
        },
      });
    });
    srcLine.appendChild(browseBtn);
    const refreshBtn = el("button", "stlu-btn", "刷新");
    refreshBtn.addEventListener("click", () => refresh(ctx));
    srcLine.appendChild(refreshBtn);
    sourceCard.appendChild(srcLine);

    const configLine = el("div", "stlu-line");
    configLine.appendChild(el("span", "stlu-label", "PSD 画布"));
    ctx.canvasInput = el("input", "stlu-input stlu-canvas");
    ctx.canvasInput.type = "number";
    ctx.canvasInput.min = "256";
    ctx.canvasInput.max = "8192";
    ctx.canvasInput.step = "256";
    ctx.canvasInput.style.flex = "0 0 92px";
    ctx.canvasInput.title = "最终 PSD 的边长（像素）。部件默认按 画布/源图 的倍率放大。";
    configLine.appendChild(ctx.canvasInput);
    ctx.scaleEl = el("span", "stlu-hint stlu-scale");
    configLine.appendChild(ctx.scaleEl);
    configLine.appendChild(el("span", "grow"));
    configLine.appendChild(el("span", "stlu-label", "输出"));
    ctx.outBtn = el("button", "stlu-btn stlu-outdir");
    ctx.outBtn.title = "输出文件夹（相对 output）。留空 = 当前拆层文件夹下的 UpScale，"
      + "不存在会自动新建；点这里可以改成别的目录。";
    ctx.outBtn.addEventListener("click", () => {
      openPicker({
        title: "选择输出文件夹",
        mode: "folder",
        scopes: [{ scope: "output", label: "输出目录" }],
        start: outDirPath(ctx),
        onPick: (rel) => {
          // 选中的就是自动目录时保持「自动」，拆层文件夹改名后仍能跟随。
          ctx.outDir = !rel || rel === ctx.autoOutDir ? "" : rel;
          setWidget(ctx.node, "out_dir", ctx.outDir);
          refresh(ctx);
        },
      });
    });
    configLine.appendChild(ctx.outBtn);
    sourceCard.appendChild(configLine);

    ctx.metaEl = el("div", "stlu-hint stlu-meta");
    sourceCard.appendChild(ctx.metaEl);
    root.appendChild(sourceCard);

    // reference image -------------------------------------------------------
    const refCard = el("div", "stlu-card stlu-refcard");
    const refLine = el("div", "stlu-line");
    ctx.refThumb = el("img", "stlu-refthumb");
    ctx.refThumb.alt = "";
    ctx.refThumb.title = "当前全身参考图（点击在 LoadImage 中更换）";
    refLine.appendChild(ctx.refThumb);
    const refWho = el("div", "stlu-who");
    refWho.style.flex = "1";
    refWho.appendChild(el("div", "stlu-name", "参考图"));
    ctx.refName = el("div", "stlu-sub", "未选择");
    refWho.appendChild(ctx.refName);
    refLine.appendChild(refWho);
    const pickRef = el("button", "stlu-btn", "选择…");
    pickRef.title = "从 input 目录中选择参考图";
    pickRef.addEventListener("click", () => {
      openPicker({
        title: "选择全身参考图",
        mode: "image",
        scopes: [{ scope: "input", label: "输入目录" }],
        start: "",
        onPick: (rel) => {
          const node = findReferenceNode();
          if (!node) { setStatus(ctx, "没有找到 LoadImage 节点", "error"); return; }
          setWidget(node, "image", rel);
          refreshRef(ctx);
          setStatus(ctx, "参考图已设为 " + rel);
        },
      });
    });
    refLine.appendChild(pickRef);
    const uploadRef = el("button", "stlu-btn", "上传…");
    uploadRef.title = "从电脑上传一张新参考图到 input 目录";
    ctx.refFile = el("input");
    ctx.refFile.type = "file";
    ctx.refFile.accept = "image/*";
    ctx.refFile.style.display = "none";
    uploadRef.addEventListener("click", () => ctx.refFile.click());
    ctx.refFile.addEventListener("change", async () => {
      const file = ctx.refFile.files && ctx.refFile.files[0];
      ctx.refFile.value = "";
      if (!file) return;
      setStatus(ctx, "正在上传 " + file.name + " …");
      try {
        const form = new FormData();
        form.append("image", file, file.name);
        form.append("type", "input");
        form.append("overwrite", "true");
        const res = await fetch(apiUrl("/upload/image"), { method: "POST", body: form });
        if (!res.ok) throw new Error(res.status + " " + res.statusText);
        const data = await res.json();
        const name = data.name || (data.subfolder ? data.subfolder + "/" + data.name : "");
        const node = findReferenceNode();
        if (!node) throw new Error("没有找到 LoadImage 节点");
        setWidget(node, "image", name);
        refreshRef(ctx);
        setStatus(ctx, "参考图已上传");
      } catch (e) {
        setStatus(ctx, "上传失败：" + e.message, "error");
      }
    });
    refLine.appendChild(uploadRef);
    refLine.appendChild(ctx.refFile);
    refCard.appendChild(refLine);
    root.appendChild(refCard);

    // toolbar ---------------------------------------------------------------
    const toolbar = el("div", "stlu-toolbar");
    toolbar.style.display = "flex";
    toolbar.style.flexDirection = "column";
    toolbar.style.gap = "6px";
    const searchLine = el("div", "stlu-line");
    ctx.search = el("input", "stlu-input stlu-search");
    ctx.search.type = "text";
    ctx.search.placeholder = "搜索部件…";
    let searchTimer = null;
    ctx.search.addEventListener("input", () => {
      if (searchTimer) clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        ctx.query = ctx.search.value || "";
        renderList(ctx);
      }, 120);
    });
    searchLine.appendChild(ctx.search);
    ctx.bulk = el("select", "stlu-input stlu-bulk");
    const bulkPlaceholder = el("option", null, "已勾选：批量长边…");
    bulkPlaceholder.value = "";
    ctx.bulk.appendChild(bulkPlaceholder);
    const bulkAuto = el("option", null, "自动（跟随画布）");
    bulkAuto.value = "auto";
    ctx.bulk.appendChild(bulkAuto);
    for (const size of SIZE_PRESETS) {
      const opt = el("option", null, size + " px");
      opt.value = String(size);
      ctx.bulk.appendChild(opt);
    }
    ctx.bulk.title = "把当前勾选的部件统一设为某个长边（“自动”等于跟随 PSD 画布）";
    ctx.bulk.addEventListener("change", () => bulkApply(ctx, ctx.bulk.value));
    searchLine.appendChild(ctx.bulk);
    toolbar.appendChild(searchLine);

    const chipLine = el("div", "stlu-line");
    ctx.chipsEl = el("div", "stlu-chips");
    chipLine.appendChild(ctx.chipsEl);
    chipLine.appendChild(el("span", "grow"));
    const allBtn = el("button", "stlu-btn", "全选");
    allBtn.title = "勾选所有部件";
    allBtn.addEventListener("click", () => setAll(ctx, () => true));
    const nonEmptyBtn = el("button", "stlu-btn", "仅非空");
    nonEmptyBtn.title = "只勾选有实际像素的部件（跳过翅膀/尾巴等空层）";
    nonEmptyBtn.addEventListener("click", () => setAll(ctx, (layer) => !layer.empty));
    const noneBtn = el("button", "stlu-btn", "清空");
    noneBtn.title = "全部不勾选（PSD 仍会用原图合成）";
    noneBtn.addEventListener("click", () => setAll(ctx, () => false));
    chipLine.appendChild(allBtn);
    chipLine.appendChild(nonEmptyBtn);
    chipLine.appendChild(noneBtn);
    toolbar.appendChild(chipLine);

    const legend = el("div", "stlu-hint",
      "勾选 = 本次用 AI 重绘；未勾选的部件仍会以原图放进 PSD（不耗时）。");
    toolbar.appendChild(legend);
    root.appendChild(toolbar);

    // local crop refinement -------------------------------------------------
    const focusCard = el("div", "stlu-card stlu-focus");
    const fLine1 = el("div", "stlu-line");
    const fLabel = el("label", "stlu-check");
    ctx.focusCb = el("input");
    ctx.focusCb.type = "checkbox";
    fLabel.appendChild(ctx.focusCb);
    fLabel.appendChild(el("span", "stlu-focusname", "局部裁切重绘"));
    fLabel.title = "只在部件周围一小块真实上下文里生成，再按原坐标贴回并恢复原 alpha。" +
      "开：小部件不会被补全成整个人；关：整帧编辑（旧行为）。";
    fLine1.appendChild(fLabel);
    fLine1.appendChild(el("span", "grow"));
    ctx.focusCtxSel = el("select", "stlu-input stlu-focusctx");
    const fOptBackdrop = el("option", null, "上下文：合成源层");
    fOptBackdrop.value = "backdrop";
    ctx.focusCtxSel.appendChild(fOptBackdrop);
    const fOptRef = el("option", null, "上下文：参考图");
    fOptRef.value = "reference";
    ctx.focusCtxSel.appendChild(fOptRef);
    ctx.focusCtxSel.title = "合成源层：把全部源层拼成一张图再裁，坐标天然对齐（推荐）。" +
      "参考图：直接用 LoadImage 那张图裁，要求它与拆层画面严格对齐。";
    fLine1.appendChild(ctx.focusCtxSel);
    focusCard.appendChild(fLine1);

    const fLine2 = el("div", "stlu-line");
    fLine2.appendChild(el("span", "stlu-hint", "边距"));
    ctx.focusMargin = el("input", "stlu-size");
    ctx.focusMargin.type = "number";
    ctx.focusMargin.min = "0";
    ctx.focusMargin.max = "256";
    ctx.focusMargin.step = "8";
    ctx.focusMargin.title = "裁片在部件四周多留多少像素（源图尺度）。" +
      "瞳孔这类小部件用 16–64：留得越少，模型越没有空间「联想」出更大的结构。";
    fLine2.appendChild(ctx.focusMargin);
    fLine2.appendChild(el("span", "stlu-hint", "px（源图尺度）"));
    focusCard.appendChild(fLine2);

    ctx.focusHint = el("div", "stlu-hint");
    focusCard.appendChild(ctx.focusHint);

    const focusDetails = el("details", "stlu-focusprompt");
    focusDetails.appendChild(el("summary", "stlu-hint", "裁片提示词（点开编辑）"));
    ctx.focusPrompt = el("textarea", "stlu-input stlu-promptarea");
    ctx.focusPrompt.rows = 4;
    focusDetails.appendChild(ctx.focusPrompt);
    focusCard.appendChild(focusDetails);
    root.appendChild(focusCard);

    ctx.listEl = el("div", "stlu-list");
    root.appendChild(ctx.listEl);

    // footer ----------------------------------------------------------------
    const foot = el("div", "stlu-foot");
    const previewBtn = el("button", "stlu-btn", "预览合成");
    previewBtn.title = "用当前勾选和分辨率离线合成一张预览（不跑 GPU）";
    previewBtn.addEventListener("click", () => openCompositePreview(ctx));
    foot.appendChild(previewBtn);
    const openSrc = el("button", "stlu-btn", "源文件夹");
    openSrc.title = "在文件管理器中打开拆层文件夹";
    openSrc.addEventListener("click", () => {
      const dir = (ctx.source && (ctx.source.source_dir || ctx.source.layers_json)) || ctx.pathInput.value;
      openPath(dir, false);
    });
    foot.appendChild(openSrc);
    const openOut = el("button", "stlu-btn", "输出目录");
    openOut.title = "在文件管理器中打开输出文件夹";
    openOut.addEventListener("click", () => openPath(outDirPath(ctx), false));
    foot.appendChild(openOut);
    ctx.statusEl = el("span", "stlu-status");
    foot.appendChild(ctx.statusEl);
    ctx.runBtn = el("button", "stlu-btn primary", "▶ 开始放大");
    ctx.runBtn.title = "运行整个工作流（模型加载一次，逐层放大）";
    ctx.runBtn.addEventListener("click", () => runWorkflow(ctx));
    foot.appendChild(ctx.runBtn);
    root.appendChild(foot);

    ctx.rows = new Map();
    ctx.entries = new Map();
    ctx.query = "";
    ctx.chip = "all";
    ctx.running = false;
    ctx.done = new Set();
    ctx.total = 0;
    ctx.rowEls = new Map();
    ctx.restore = true;
    ctx.outDir = readOutDir(node);
    ctx.autoOutDir = "";
    ctx.canvas = Number(widgetValue(node, "canvas_size", 2048)) || 2048;

    ctx.pathInput.value = String(widgetValue(node, "layers_json", "") || "");
    ctx.canvasInput.value = String(ctx.canvas);
    ctx.outBtn.textContent = outDirLabel(ctx);
    readFocusWidgets(ctx);

    ctx.focusCb.addEventListener("change", () => writeFocusWidgets(ctx));
    ctx.focusCtxSel.addEventListener("change", () => writeFocusWidgets(ctx));
    ctx.focusMargin.addEventListener("change", () => writeFocusWidgets(ctx));
    ctx.focusPrompt.addEventListener("change", () => writeFocusWidgets(ctx));

    ctx.pathInput.addEventListener("change", () => {
      setWidget(node, "layers_json", ctx.pathInput.value.trim());
      refresh(ctx);
    });
    ctx.canvasInput.addEventListener("change", () => {
      let value = Number(ctx.canvasInput.value) || 2048;
      value = Math.max(256, Math.min(8192, Math.round(value / 8) * 8));
      ctx.canvas = value;
      ctx.canvasInput.value = String(value);
      setWidget(node, "canvas_size", value);
      refresh(ctx);
    });
    ctx.canvasInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") ctx.canvasInput.blur();
    });

    addDOM(node, "stlu_panel", root, 470, undefined);
  }

  function setStatus(ctx, text, cls) {
    if (!ctx.statusEl) return;
    ctx.statusEl.textContent = text || "";
    ctx.statusEl.className = "stlu-status" + (cls ? " " + cls : "");
  }

  function selectedCount(ctx) {
    let n = 0;
    for (const layer of ctx.layers) {
      const entry = ctx.entries.get(layer.name);
      if (entry && entry.checked) n++;
    }
    return n;
  }

  function updateRunState(ctx) {
    if (!ctx.runBtn) return;
    const n = selectedCount(ctx);
    if (ctx.running) {
      ctx.runBtn.disabled = true;
      ctx.runBtn.textContent = "运行中…";
      return;
    }
    ctx.runBtn.disabled = false;
    ctx.runBtn.textContent = n ? "▶ 开始放大（" + n + " 层）" : "▶ 只合成 PSD（不放大）";
    ctx.runBtn.title = n
      ? "运行整个工作流（模型只加载一次，逐层放大）"
      : "没有勾选部件：不调用模型，直接用原图合成 PSD";
  }

  function updateScale(ctx) {
    if (!ctx.scaleEl) return;
    const src = ctx.meta && ctx.meta.source ? Number(ctx.meta.source.canvas_w) || 1024 : 1024;
    const scale = ctx.canvas / src;
    ctx.scaleEl.textContent = "源 " + src + " · " + scale.toFixed(2) + "×";
  }

  function updateSummary(ctx) {
    if (!ctx.metaEl) return;
    const total = ctx.layers.length;
    let checked = 0;
    let inPsd = 0;
    let upscaled = 0;
    for (const layer of ctx.layers) {
      const entry = ctx.entries.get(layer.name);
      if (entry && entry.checked) checked++;
      if (!(layer.empty && !(entry && entry.checked))) inPsd++;
      if (layer.upscaled) upscaled++;
    }
    const bits = [total + " 个部件", "PSD " + inPsd + " 层", "已选 " + checked + " 个重绘"];
    if (upscaled) bits.push("已生成 " + upscaled);
    ctx.metaEl.classList.remove("warn");
    ctx.metaEl.textContent = bits.join(" · ");
    if (ctx.meta) {
      const src = ctx.meta.source || {};
      const name = String(src.layers_json || "").split("/").pop();
      if (name) bits.push(name);
      if (src.synthetic) bits.push("从 PNG 合成（无 layers.json）");
      ctx.metaEl.textContent = bits.join(" · ");
      if ((ctx.meta.warnings || []).length) {
        ctx.metaEl.classList.add("warn");
        ctx.metaEl.title = ctx.meta.warnings.join("\n");
      } else {
        ctx.metaEl.title = "";
      }
    }
    updateScale(ctx);
    updateRunState(ctx);
    updateFocusHint(ctx);
  }

  function updateChipCounts(ctx) {
    if (!ctx.chipsEl) return;
    const chips = [{ id: "all", label: "全部" }]
      .concat(CATEGORIES.map((c) => ({ id: c.id, label: c.label })))
      .concat([{ id: "empty", label: "空层" }]);
    if (!ctx.chipButtons) ctx.chipButtons = new Map();
    ctx.chipsEl.innerHTML = "";
    for (const chip of chips) {
      let count = 0;
      for (const layer of ctx.layers) {
        if (chip.id === "all") count++;
        else if (chip.id === "empty") count += layer.empty ? 1 : 0;
        else if (categoryOf(layer.name).id === chip.id) count++;
      }
      const b = el("button", "stlu-chip" + (ctx.chip === chip.id ? " on" : ""), chip.label);
      b.appendChild(el("span", "n", String(count)));
      b.addEventListener("click", () => {
        ctx.chip = chip.id;
        renderList(ctx);
      });
      ctx.chipsEl.appendChild(b);
    }
  }

  function rowSubText(layer, entry) {
    const src = layer.w + "×" + layer.h;
    if (!entry.checked) return src + " · 不重绘，PSD 使用原图";
    const aspect = layer.w / Math.max(1, layer.h);
    let w;
    let h;
    if (aspect >= 1) {
      w = entry.longSide;
      h = round32(entry.longSide / aspect);
    } else {
      h = entry.longSide;
      w = round32(entry.longSide * aspect);
    }
    return src + " → " + w + "×" + h + " · " + (entry.auto ? "自动" : "固定");
  }

  function updateRowEl(ref, layer, entry) {
    if (!ref) return;
    ref.row.classList.toggle("off", !entry.checked);
    // 批量按钮（全选/仅非空/清空、分区全选）只改 entry，不会触发 change 事件，
    // 所以这里必须把原生复选框同步回去，否则会出现“置灰但仍是勾选态”。
    if (ref.cb && ref.cb.checked !== !!entry.checked) ref.cb.checked = !!entry.checked;
    if (ref.sub) ref.sub.textContent = rowSubText(layer, entry);
    if (ref.size) ref.size.value = String(entry.longSide);
    if (ref.autoBtn) {
      ref.autoBtn.classList.toggle("on", entry.auto);
      ref.autoBtn.textContent = entry.auto ? "自动" : "固定";
      ref.autoBtn.title = entry.auto
        ? "跟随 PSD 画布倍率（在右侧数字框里输入即固定）"
        : "已固定；点击恢复自动";
    }
    if (ref.doneTag) {
      ref.doneTag.style.display = ref.ctx && ref.ctx.done.has(layer.name) ? "" : "none";
    }
  }

  function makeRow(ctx, layer) {
    const entry = entryFor(ctx, layer);
    const row = el("div", "stlu-row");
    const ref = { row, layer, entry, ctx };
    ctx.rowEls.set(layer.name, ref);

    const cb = el("input");
    cb.type = "checkbox";
    cb.checked = entry.checked;
    cb.addEventListener("change", () => {
      entry.checked = cb.checked;
      updateRowEl(ref, layer, entry);
      updateSummary(ctx);
      updateChipCounts(ctx);
      sync(ctx);
    });
    ref.cb = cb;

    const thumb = el("img", "stlu-thumb");
    thumb.loading = "lazy";
    thumb.alt = "";
    if (layer.thumb) {
      thumb.src = apiUrl(layer.thumb);
      thumb.addEventListener("click", () => openLayerPreview(ctx, layer));
      thumb.addEventListener("error", () => { thumb.style.visibility = "hidden"; });
    }
    ref.thumb = thumb;

    const who = el("div", "stlu-who");
    const nameEl = el("div", "stlu-name");
    nameEl.appendChild(el("span", null, layer.name));
    if (layer.empty) nameEl.appendChild(el("span", "stlu-tag", "空层"));
    ref.doneTag = el("span", "stlu-tag ok", "✓");
    ref.doneTag.style.display = "none";
    nameEl.appendChild(ref.doneTag);
    if (layer.upscaled) nameEl.appendChild(el("span", "stlu-tag ok", "已生成"));
    who.appendChild(nameEl);
    ref.sub = el("div", "stlu-sub");
    who.appendChild(ref.sub);

    const sizebox = el("div", "stlu-sizebox");
    const size = el("input", "stlu-size");
    size.type = "number";
    size.min = "32";
    size.max = "8192";
    size.step = "32";
    size.title = "放大后的长边像素（按部件宽高比换算 W×H）";
    size.addEventListener("change", () => {
      let value = Number(size.value) || layer.suggested.long_side;
      value = Math.max(32, Math.min(8192, round32(value)));
      entry.longSide = value;
      entry.auto = false;
      updateRowEl(ref, layer, entry);
      sync(ctx);
    });
    ref.size = size;
    const autoBtn = el("button", "stlu-auto");
    autoBtn.addEventListener("click", () => {
      if (entry.auto) return;
      entry.auto = true;
      entry.longSide = layer.suggested.long_side;
      updateRowEl(ref, layer, entry);
      sync(ctx);
    });
    ref.autoBtn = autoBtn;
    sizebox.appendChild(size);
    sizebox.appendChild(autoBtn);

    row.appendChild(cb);
    row.appendChild(thumb);
    row.appendChild(who);
    row.appendChild(sizebox);
    updateRowEl(ref, layer, entry);
    return row;
  }

  function sectionHeader(ctx, category, layers) {
    const head = el("div", "stlu-sec");
    head.appendChild(el("span", null, category.label + " · " + layers.length));
    head.appendChild(el("span", "grow"));
    const btn = el("button", "stlu-mini", "全选");
    btn.addEventListener("click", () => {
      const allOn = layers.every((layer) => entryFor(ctx, layer).checked);
      for (const layer of layers) {
        const entry = entryFor(ctx, layer);
        entry.checked = !allOn;
        updateRowEl(ctx.rowEls.get(layer.name), layer, entry);
      }
      updateSummary(ctx);
      updateChipCounts(ctx);
      sync(ctx);
    });
    head.appendChild(btn);
    return head;
  }

  function renderList(ctx) {
    ctx.listEl.innerHTML = "";
    ctx.rowEls = new Map();
    const query = (ctx.query || "").trim().toLowerCase();
    const visible = ctx.layers
      .filter((layer) => {
        if (query && layer.name.toLowerCase().indexOf(query) === -1) return false;
        if (ctx.chip === "all") return true;
        if (ctx.chip === "empty") return !!layer.empty;
        return categoryOf(layer.name).id === ctx.chip;
      })
      // 空层统一排在非空之后；sort 是稳定的，同类部件保持原有相对顺序。
      .sort((a, b) => (a.empty ? 1 : 0) - (b.empty ? 1 : 0));

    if (ctx.chip === "all") {
      for (const category of CATEGORIES) {
        const rows = visible.filter((layer) => categoryOf(layer.name).id === category.id);
        if (!rows.length) continue;
        ctx.listEl.appendChild(sectionHeader(ctx, category, rows));
        for (const layer of rows) ctx.listEl.appendChild(makeRow(ctx, layer));
      }
    } else {
      for (const layer of visible) ctx.listEl.appendChild(makeRow(ctx, layer));
    }
    if (!visible.length) {
      ctx.listEl.appendChild(el("div", "stlu-item", "没有匹配的部件"));
    }
    updateChipCounts(ctx);
    updateSummary(ctx);
  }

  function setAll(ctx, predicate) {
    for (const layer of ctx.layers) {
      const entry = entryFor(ctx, layer);
      entry.checked = !!predicate(layer);
      updateRowEl(ctx.rowEls.get(layer.name), layer, entry);
    }
    updateSummary(ctx);
    updateChipCounts(ctx);
    sync(ctx);
  }

  function bulkApply(ctx, value) {
    if (!value) return;
    let count = 0;
    for (const layer of ctx.layers) {
      const entry = ctx.entries.get(layer.name);
      if (!entry || !entry.checked) continue;
      if (value === "auto") {
        entry.auto = true;
        entry.longSide = layer.suggested.long_side;
      } else {
        entry.auto = false;
        entry.longSide = Number(value);
      }
      updateRowEl(ctx.rowEls.get(layer.name), layer, entry);
      count++;
    }
    ctx.bulk.value = "";
    sync(ctx);
    if (count) {
      setStatus(ctx, "已把 " + count + " 个勾选部件的长边设为 " + (value === "auto" ? "自动" : value + " px"));
    } else {
      setStatus(ctx, "先勾选要修改的部件", "error");
    }
  }

  function readFocusWidgets(ctx) {
    if (!ctx.focusCb) return;
    ctx.focusCb.checked = widgetValue(ctx.node, "focus_crop", true) !== false;
    const mode = String(widgetValue(ctx.node, "focus_context", "backdrop") || "backdrop");
    ctx.focusCtxSel.value = mode === "reference" ? "reference" : "backdrop";
    let margin = Number(widgetValue(ctx.node, "focus_margin", 32));
    if (!Number.isFinite(margin)) margin = 32;
    margin = Math.max(0, Math.min(256, Math.round(margin / 8) * 8));
    ctx.focusMargin.value = String(margin);
    ctx.focusPrompt.value = String(widgetValue(ctx.node, "focus_prompt", DEFAULT_FOCUS_PROMPT) || "");
    updateFocusHint(ctx);
  }

  function writeFocusWidgets(ctx) {
    if (!ctx.focusCb || !ctx.node) return;
    let margin = Number(ctx.focusMargin.value);
    if (!Number.isFinite(margin)) margin = 32;
    margin = Math.max(0, Math.min(256, Math.round(margin / 8) * 8));
    ctx.focusMargin.value = String(margin);
    setWidget(ctx.node, "focus_crop", !!ctx.focusCb.checked);
    setWidget(ctx.node, "focus_context", ctx.focusCtxSel.value);
    setWidget(ctx.node, "focus_margin", margin);
    setWidget(ctx.node, "focus_prompt", ctx.focusPrompt.value);
    if (ctx.focusCtxSel.value === "reference") probeRefSize(ctx);
    else updateFocusHint(ctx);
    updateSummary(ctx);
  }

  const refSizeCache = new Map();

  async function probeRefSize(ctx) {
    const node = findReferenceNode();
    const widget = node ? widgetOf(node, "image") : null;
    const name = widget ? String(widget.value || "") : "";
    if (!name) {
      ctx.refSize = null;
      updateFocusHint(ctx);
      return;
    }
    if (refSizeCache.has(name)) {
      ctx.refSize = refSizeCache.get(name);
      updateFocusHint(ctx);
      return;
    }
    try {
      const data = await getJson("/stlu/imgsize?scope=input&path=" + encodeURIComponent(name));
      const size = [Number(data.width) || 0, Number(data.height) || 0];
      refSizeCache.set(name, size);
      ctx.refSize = size;
    } catch (e) {
      ctx.refSize = null;
    }
    updateFocusHint(ctx);
  }

  function updateFocusHint(ctx) {
    if (!ctx.focusHint) return;
    const enabled = ctx.focusCb ? !!ctx.focusCb.checked : true;
    const src = (ctx.meta && ctx.meta.source) || {};
    const cw = Number(src.canvas_w) || 1024;
    const ch = Number(src.canvas_h) || cw;
    const canvasArea = Math.max(1, cw * ch);
    let focus = 0;
    let full = 0;
    let overscale = 0;
    for (const layer of ctx.layers || []) {
      const entry = ctx.entries.get(layer.name);
      if (!entry || !entry.checked) continue;
      const box = layer.box || [0, 0, cw, ch];
      const bw = Math.max(0, box[2] - box[0]);
      const bh = Math.max(0, box[3] - box[1]);
      if (bw * bh >= 0.95 * canvasArea) full++;
      else focus++;
      const srcSide = Math.max(1, bw, bh);
      if ((Number(entry.longSide) || 0) / srcSide > 8) overscale++;
    }
    const bits = [];
    if (enabled) {
      bits.push("局部裁切 " + focus + " 层");
      if (full) bits.push("整帧 " + full + " 层");
    } else {
      bits.push("整帧编辑（关）");
    }
    let warn = overscale > 0;
    if (overscale) bits.push("⚠ " + overscale + " 层超过源尺寸 8×");
    if (enabled && ctx.focusCtxSel && ctx.focusCtxSel.value === "reference") {
      const size = ctx.refSize;
      if (size && size[0] > 0) {
        const kx = size[0] / cw;
        const ky = size[1] / ch;
        const k = (kx + ky) / 2;
        if (Math.abs(kx - ky) > 0.02 * Math.max(kx, ky)) {
          bits.push("⚠ 参考图 " + size[0] + "×" + size[1] + " 比例不符，会缩放回画布");
          warn = true;
        } else if (k > 1.02) {
          bits.push("参考图 " + size[0] + "×" + size[1] + " · 原生分辨率裁取 ×" + k.toFixed(2));
        } else {
          bits.push("参考图 " + size[0] + "×" + size[1] + "（与画布同尺寸）");
        }
      } else {
        bits.push("参考图");
      }
    }
    ctx.focusHint.textContent = bits.join(" · ");
    ctx.focusHint.classList.toggle("warn", warn);
    if (ctx.focusCtxSel) ctx.focusCtxSel.disabled = !enabled;
    if (ctx.focusMargin) ctx.focusMargin.disabled = !enabled;
    if (ctx.focusPrompt) ctx.focusPrompt.disabled = !enabled;
  }

  function sync(ctx) {
    const names = [];
    const sizes = templateRules(ctx.baseRulesObj);
    for (const layer of ctx.layers) {
      const entry = ctx.entries.get(layer.name);
      if (!entry || !entry.checked) continue;
      names.push(layer.name);
      if (!entry.auto) sizes[layer.name] = { long_side: entry.longSide };
    }
    setWidget(ctx.node, "select", names.length ? names.join(", ") : "!*");
    setWidget(ctx.node, "size_rules", JSON.stringify(sizes));
    setWidget(ctx.node, "layers_json", ctx.pathInput.value.trim());
    setWidget(ctx.node, "exclude", "");
    setWidget(ctx.node, "canvas_size", ctx.canvas);
    setWidget(ctx.node, "out_dir", ctx.outDir);
  }

  async function refresh(ctx) {
    if (!ctx.node || !ctx.listEl) return;
    const source = ctx.pathInput.value.trim();
    const rules = String(widgetValue(ctx.node, "size_rules", "") || "");
    const m0 = i18n();
    const keepStatus = ctx.statusEl ? (m0 ? m0.raw(ctx.statusEl) : ctx.statusEl.textContent) : "";
    if (!ctx.running) setStatus(ctx, "正在读取部件…");
    try {
      const data = await getJson(
        "/stlu/layers?dir=" + encodeURIComponent(source) +
        "&canvas=" + encodeURIComponent(ctx.canvas) +
        "&rules=" + encodeURIComponent(rules) +
        "&out_dir=" + encodeURIComponent(ctx.outDir)
      );
      ctx.layers = data.layers || [];
      ctx.meta = data;
      ctx.source = data.source || {};
      // 空 out_dir 时由后端按拆层文件夹算出实际目录（相对 output）。
      ctx.autoOutDir = String(data.auto_out_dir || "");
      ctx.outResolved = String(data.out_dir || outDirPath(ctx));
      const valid = new Set(ctx.layers.map((layer) => layer.name));
      for (const key of Array.from(ctx.entries.keys())) {
        if (!valid.has(key)) ctx.entries.delete(key);
      }
      if (ctx.restore) {
        // Rebuild every entry once so the saved select / pinned sizes apply to
        // layers that are not currently visible under the active filter.
        ctx.savedSelectNames = parseSelectNames(widgetValue(ctx.node, "select", ""));
        ctx.savedSizes = parseSizes(widgetValue(ctx.node, "size_rules", ""));
        for (const layer of ctx.layers) entryFor(ctx, layer);
        ctx.restore = false;
      }
      ctx.canvasInput.value = String(ctx.canvas);
      ctx.outBtn.textContent = outDirLabel(ctx);
      ctx.outBtn.title = "输出文件夹：" + ctx.outResolved + "（相对 output）\n"
        + "点击更改；当前为自动 = 拆层文件夹下的 UpScale。";
      renderList(ctx);
      sync(ctx);
      refreshRef(ctx);
      if (!ctx.running && keepStatus !== "正在读取部件…") {
        setStatus(ctx, keepStatus || "就绪");
      }
    } catch (e) {
      ctx.layers = ctx.layers || [];
      ctx.rowEls = new Map();
      setStatus(ctx, "读取失败：" + e.message, "error");
      if (ctx.metaEl) {
        ctx.metaEl.classList.add("warn");
        ctx.metaEl.textContent = "读取失败：" + e.message;
      }
    }
  }

  function refreshRef(ctx) {
    const node = findReferenceNode();
    ctx.refNode = node;
    const w = node ? widgetOf(node, "image") : null;
    if (w && !w.__stluHooked) {
      // Changing the image directly on the LoadImage node should update the
      // reference row too.
      const original = w.callback;
      w.callback = function () {
        let result;
        try {
          result = original ? original.apply(this, arguments) : undefined;
        } finally {
          setTimeout(() => forEachPlanState((other) => refreshRef(other)), 0);
        }
        return result;
      };
      w.__stluHooked = true;
    }
    const name = w ? String(w.value || "") : "";
    if (!node) {
      ctx.refName.textContent = "未找到 LoadImage 节点";
      ctx.refThumb.style.visibility = "hidden";
      ctx.refSize = null;
      updateFocusHint(ctx);
      return;
    }
    if (!name) {
      ctx.refName.textContent = "未选择（点击“选择…”）";
      ctx.refThumb.style.visibility = "hidden";
      ctx.refSize = null;
      updateFocusHint(ctx);
      return;
    }
    ctx.refName.textContent = name;
    ctx.refThumb.style.visibility = "";
    probeRefSize(ctx);
    ctx.refThumb.src = apiUrl(
      "/stlu/thumb?scope=input&path=" + encodeURIComponent(name) + "&size=64&v=" + encodeURIComponent(name)
    );
    ctx.refThumb.onclick = () => {
      openPicker({
        title: "选择全身参考图",
        mode: "image",
        scopes: [{ scope: "input", label: "输入目录" }],
        start: "",
        onPick: (rel) => {
          setWidget(node, "image", rel);
          refreshRef(ctx);
        },
      });
    };
  }

  async function runWorkflow(ctx) {
    if (ctx.running) return;
    const count = selectedCount(ctx);
    if (count > 0) {
      const ref = findReferenceNode();
      const refWidget = ref ? widgetOf(ref, "image") : null;
      if (refWidget && !String(refWidget.value || "")) {
        setStatus(ctx, "先选择参考图", "error");
        return;
      }
    }
    sync(ctx);
    ctx.running = true;
    resetDoneMarks(ctx);
    ctx.total = count;
    updateRunState(ctx);
    setStatus(ctx, count ? "已提交，等待执行…" : "已提交：直接用原图合成 PSD…");
    try {
      if (!app.queuePrompt) throw new Error("当前前端版本不支持程序化排队，请点右上角“运行”");
      await app.queuePrompt(0, 1);
    } catch (e) {
      ctx.running = false;
      updateRunState(ctx);
      setStatus(ctx, "提交失败：" + (e && e.message ? e.message : e), "error");
    }
  }

  function resetDoneMarks(ctx) {
    ctx.done.clear();
    for (const ref of ctx.rowEls.values()) {
      if (ref.doneTag) ref.doneTag.style.display = "none";
    }
  }

  function markRowDone(ctx, name) {
    // The upscaled file now exists; remember it so a later refresh shows it.
    const layer = ctx.layers.find((l) => l.name === name);
    if (layer) layer.upscaled = true;
    // 统一走 updateRowEl：让 “off” 只反映勾选状态，别把未勾选的行误清灰。
    const ref = ctx.rowEls.get(name);
    if (ref) updateRowEl(ref, ref.layer, ref.entry);
  }

  async function openCompositePreview(ctx) {
    const { dialog, close } = openOverlay("wide");
    const head = el("div", "stlu-dialog-head");
    head.appendChild(el("strong", "grow", "合成预览（未勾选部件使用原图）"));
    head.appendChild(closeButton(close));
    dialog.appendChild(head);
    const box = el("div", "stlu-prevbox");
    box.appendChild(el("div", "stlu-hint", "正在合成…"));
    dialog.appendChild(box);
    const meta = el("div", "stlu-prevmeta");
    dialog.appendChild(meta);

    sync(ctx);
    const query = "/stlu/preview" +
      "?dir=" + encodeURIComponent(ctx.pathInput.value.trim()) +
      "&canvas=" + encodeURIComponent(ctx.canvas) +
      "&rules=" + encodeURIComponent(String(widgetValue(ctx.node, "size_rules", "") || "")) +
      "&select=" + encodeURIComponent(String(widgetValue(ctx.node, "select", "all") || "all")) +
      "&exclude=" + encodeURIComponent(String(widgetValue(ctx.node, "exclude", "") || "")) +
      "&out_dir=" + encodeURIComponent(ctx.outDir) +
      "&skip_empty=" + (widgetValue(ctx.node, "skip_empty", true) ? "1" : "0") +
      "&coverage=" + encodeURIComponent(String(widgetValue(ctx.node, "empty_coverage", 0.0005))) +
      "&size=1024";
    try {
      const data = await getJson(query);
      box.innerHTML = "";
      const img = el("img");
      img.src = apiUrl(data.url);
      box.appendChild(img);
      meta.innerHTML = "";
      meta.appendChild(el("span", null, data.layers + " 个图层"));
      meta.appendChild(el("span", null, data.canvas[0] + "×" + data.canvas[1]));
      meta.appendChild(el("span", null, "已选重绘 " + data.selected + " 个"));
      const openBtn = el("button", "stlu-btn", "输出目录");
      openBtn.style.marginLeft = "auto";
      openBtn.addEventListener("click", () => openPath(outDirPath(ctx), false));
      meta.appendChild(openBtn);
      if ((data.notes || []).length || (data.warnings || []).length) {
        const list = el("ul", "stlu-notes");
        for (const note of (data.notes || []).slice(0, 6)) list.appendChild(el("li", null, note));
        for (const warning of (data.warnings || []).slice(0, 4)) list.appendChild(el("li", null, warning));
        meta.appendChild(list);
      }
    } catch (e) {
      box.innerHTML = "";
      box.appendChild(el("div", "stlu-hint error", "合成失败：" + e.message));
    }
  }

  function bigThumb(url) {
    return String(url || "").replace(/size=\d+/, "size=768");
  }

  function openLayerPreview(ctx, layer) {
    const { dialog, close } = openOverlay("wide");
    const head = el("div", "stlu-dialog-head");
    head.appendChild(el("strong", "grow", layer.name));
    head.appendChild(closeButton(close));
    dialog.appendChild(head);
    const box = el("div", "stlu-layerbox");
    const figure = (caption, src) => {
      const fig = el("figure");
      const img = el("img");
      img.src = apiUrl(src);
      img.addEventListener("error", () => { img.style.visibility = "hidden"; });
      fig.appendChild(img);
      fig.appendChild(el("figcaption", null, caption));
      return fig;
    };
    if (layer.thumb) box.appendChild(figure("原图 · " + layer.w + "×" + layer.h, bigThumb(layer.thumb)));
    if (layer.upscaled_thumb) {
      box.appendChild(figure("上次放大结果", bigThumb(layer.upscaled_thumb)));
    }
    dialog.appendChild(box);
    const foot = el("div", "stlu-dialog-foot");
    foot.appendChild(el("span", "path", layer.filename || ""));
    const openOut = el("button", "stlu-btn", "打开输出目录");
    openOut.addEventListener("click", () => openPath((ctx && outDirPath(ctx)) || "", false));
    foot.appendChild(openOut);
    dialog.appendChild(foot);
  }

  // -------------------------------------------------------------------------
  // Plan node wiring
  // -------------------------------------------------------------------------

  function setupPlan(node) {
    injectCss();
    const ctx = {
      node,
      layers: [],
      entries: new Map(),
      rowEls: new Map(),
      running: false,
      done: new Set(),
      total: 0,
    };
    for (const name of HIDDEN_PLAN) hideWidget(widgetOf(node, name));
    ctx.baseRulesObj = (() => {
      try {
        return JSON.parse(String(widgetValue(node, "size_rules", "") || "{}"));
      } catch (e) {
        return { "*": {} };
      }
    })();

    buildPlanPanel(node, ctx);
    planStates.add(ctx);

    const keep = {
      width: Math.max(node.size[0], 780),
      height: Math.max(node.size[1], 900),
    };
    node.setSize([keep.width, keep.height]);

    let timer = null;
    const schedule = (delay) => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = null;
        refresh(ctx);
      }, delay == null ? 120 : delay);
    };

    // Re-read every hidden widget after the node is configured (workflow load,
    // paste, undo): the saved folder / selection / sizes must win over the
    // defaults used when the node was created.
    const reloadFromWidgets = () => {
      ctx.pathInput.value = String(widgetValue(node, "layers_json", "") || "");
      ctx.outDir = readOutDir(node);
      ctx.canvas = Number(widgetValue(node, "canvas_size", 2048)) || 2048;
      ctx.canvasInput.value = String(ctx.canvas);
      ctx.outBtn.textContent = outDirLabel(ctx);
      try {
        ctx.baseRulesObj = JSON.parse(String(widgetValue(node, "size_rules", "") || "{}"));
      } catch (e) {
        ctx.baseRulesObj = { "*": {} };
      }
      ctx.entries = new Map();
      ctx.restore = true;
      readFocusWidgets(ctx);
      schedule(200);
    };

    const oldConfigured = node.onAfterGraphConfigured;
    node.onAfterGraphConfigured = function () {
      const result = oldConfigured ? oldConfigured.apply(this, arguments) : undefined;
      reloadFromWidgets();
      return result;
    };

    const oldConfigure = node.onConfigure;
    node.onConfigure = function () {
      const result = oldConfigure ? oldConfigure.apply(this, arguments) : undefined;
      try {
        reloadFromWidgets();
      } catch (e) {
        console.warn("[STLU] reload after configure failed", e);
      }
      return result;
    };

    const oldRemoved = node.onRemoved;
    node.onRemoved = function () {
      if (oldRemoved) oldRemoved.apply(this, arguments);
      planStates.delete(ctx);
    };

    setTimeout(() => schedule(250), 250);
  }

  // -------------------------------------------------------------------------
  // Batch node card
  // -------------------------------------------------------------------------

  function setupBatch(node) {
    injectCss();
    for (const name of HIDDEN_BATCH) hideWidget(widgetOf(node, name));

    const root = el("div", "stlu-panel");
    root.style.padding = "8px";
    const line = el("div", "stlu-line");
    const dot = el("span", "stlu-dot");
    line.appendChild(dot);
    const title = el("span", "stlu-cardtitle", "子图模板");
    line.appendChild(title);
    const nameEl = el("span", "stlu-hint");
    nameEl.style.flex = "1";
    line.appendChild(nameEl);
    root.appendChild(line);
    root.appendChild(el("div", "stlu-hint",
      "每一层都会克隆这个工作流里的 Qwen 编辑子图；改了 prompt / 采样器会自动生效。"));
    const actions = el("div", "stlu-line");
    const openBtn = el("button", "stlu-btn", "打开工作流");
    const pickBtn = el("button", "stlu-btn", "更换…");
    actions.appendChild(openBtn);
    actions.appendChild(pickBtn);
    root.appendChild(actions);

    const state = { path: "", exists: false };
    let checking = false;

    async function check() {
      if (checking) return;
      checking = true;
      try {
        const path = String(widgetValue(node, "workflow_json", "") || "");
        const data = await getJson("/stlu/wf?path=" + encodeURIComponent(path));
        state.path = data.path || "";
        state.exists = !!data.exists;
        nameEl.textContent = data.name || "?";
        nameEl.title = state.path;
        dot.className = "stlu-dot " + (state.exists ? "ok" : "bad");
        nameEl.className = "stlu-hint" + (state.exists ? "" : " warn");
        openBtn.disabled = !state.exists;
      } catch (e) {
        dot.className = "stlu-dot bad";
        nameEl.className = "stlu-hint warn";
        nameEl.textContent = "无法检查（" + e.message + "）";
      } finally {
        checking = false;
      }
    }

    openBtn.addEventListener("click", () => {
      if (state.path) openPath(state.path, true);
    });
    pickBtn.addEventListener("click", () => {
      openPicker({
        title: "选择子图模板工作流",
        mode: "workflow",
        scopes: [{ scope: "workflows", label: "工作流目录" }],
        start: "",
        onPick: (rel, abs) => {
          setWidget(node, "workflow_json", abs || rel);
          check();
        },
      });
    });

    addDOM(node, "stlu_batch_card", root, 150, 190);
    const oldConfigured = node.onAfterGraphConfigured;
    node.onAfterGraphConfigured = function () {
      const result = oldConfigured ? oldConfigured.apply(this, arguments) : undefined;
      check();
      return result;
    };
    const oldConfigure = node.onConfigure;
    node.onConfigure = function () {
      const result = oldConfigure ? oldConfigure.apply(this, arguments) : undefined;
      check();
      return result;
    };
    setTimeout(check, 300);
  }

  // -------------------------------------------------------------------------
  // Assemble node card
  // -------------------------------------------------------------------------

  function setupAssemble(node) {
    injectCss();
    for (const name of HIDDEN_ASSEMBLE) hideWidget(widgetOf(node, name));

    const root = el("div", "stlu-panel");
    root.style.padding = "8px";
    root.style.gap = "5px";

    const line = el("div", "stlu-line");
    const dot = el("span", "stlu-dot");
    line.appendChild(dot);
    const title = el("span", "stlu-cardtitle", "等待运行");
    line.appendChild(title);
    line.appendChild(el("span", "grow"));
    const tag = el("span", "stlu-tag", "自动");
    line.appendChild(tag);
    root.appendChild(line);

    const preview = el("img", "stlu-psd-prev");
    preview.style.display = "none";
    root.appendChild(preview);

    const sub = el("div", "stlu-hint", "全部部件完成后自动合成 PSD，无需手动操作。");
    root.appendChild(sub);

    const actions = el("div", "stlu-line");
    const openPsd = el("button", "stlu-btn", "打开 PSD");
    const reveal = el("button", "stlu-btn", "打开文件夹");
    openPsd.disabled = true;
    reveal.disabled = true;
    actions.appendChild(openPsd);
    actions.appendChild(reveal);
    root.appendChild(actions);

    let info = null;
    const targetPath = () => (info ? (info.subfolder ? info.subfolder + "/" : "") + info.filename : "");
    openPsd.addEventListener("click", () => { if (info) openPath(targetPath(), true); });
    reveal.addEventListener("click", () => { if (info) openPath(info.subfolder || ".", false); });

    function applyInfo(data) {
      if (!data) return;
      info = data;
      const isError = !!data.error;
      dot.className = "stlu-dot " + (isError ? "bad" : "ok");
      title.textContent = data.filename || "PSD";
      title.title = targetPath();
      tag.textContent = isError ? "PSD 写入失败" : "完成";
      const bits = [];
      if (data.layers) bits.push(data.layers + " 层");
      if (data.canvas && data.canvas.length === 2) bits.push(data.canvas[0] + "×" + data.canvas[1]);
      if (data.size) bits.push(fmtBytes(data.size));
      if (isError) bits.push(String(data.error).slice(0, 120));
      sub.textContent = bits.join(" · ");
      sub.className = "stlu-hint" + (isError ? " error" : "");
      if (data.preview && data.preview.filename) {
        const rel = (data.preview.subfolder ? data.preview.subfolder + "/" : "") + data.preview.filename;
        preview.src = apiUrl(
          "/stlu/thumb?scope=output&path=" + encodeURIComponent(rel) +
          "&size=384&v=" + encodeURIComponent(data.preview.v || "")
        );
        preview.style.display = "";
      }
      openPsd.disabled = isError;
      reveal.disabled = false;
    }

    addDOM(node, "stlu_psd_card", root, 210, 300);

    const oldExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      const result = oldExecuted ? oldExecuted.apply(this, arguments) : undefined;
      try {
        const payload = message && message.stlu_psd;
        if (payload && payload.length) applyInfo(payload[0]);
      } catch (e) {
        console.error("[STLU] cannot read PSD result", e);
      }
      return result;
    };
  }

  // -------------------------------------------------------------------------
  // Grid-split tiles card
  // -------------------------------------------------------------------------

  function setupTiles(node) {
    injectCss();
    syncTileParamsFromSplit(node);
    for (const name of ["tile_width", "tile_height", "overlap",
                        "merge_parts", "group_layers",
                        "write_depth", "include_original", "merged_psd",
                        "source", "grid", "tile_indices", "group", "tiles", "out_dir"]) {
      hideWidget(widgetOf(node, name));
    }

    const outWidget = widgetOf(node, "out_dir");
    const srcWidget = widgetOf(node, "source");
    const groupWidget = widgetOf(node, "group");
    const tilesWidget = widgetOf(node, "tiles");

    const root = el("div", "stlu-panel");
    root.style.padding = "8px";
    root.style.gap = "5px";

    const line = el("div", "stlu-line");
    const dot = el("span", "stlu-dot");
    line.appendChild(dot);
    line.appendChild(el("span", "stlu-cardtitle", "③ 图片 → 分格 PSD"));
    line.appendChild(el("span", "grow"));
    const tag = el("span", "stlu-tag", "未选");
    line.appendChild(tag);
    root.appendChild(line);

    const preview = el("img", "stlu-psd-prev");
    preview.style.display = "none";
    root.appendChild(preview);

    const sub = el("div", "stlu-hint",
      "图片来源二选一：① 本次结果（刚跑完 SeeThrough 的输出，跑 GPU）；② 历史结果（点“选历史结果…”挑一个以前的文件夹，不跑 GPU）。");
    root.appendChild(sub);

    // --- source: a graphical picker, not a dropdown ------------------------
    const srcLine = el("div", "stlu-line");
    srcLine.appendChild(el("span", "stlu-label", "来源"));
    const srcText = el("span", "stlu-hint grow", "本次结果（实时）");
    srcText.style.overflow = "hidden";
    srcText.style.textOverflow = "ellipsis";
    srcText.style.whiteSpace = "nowrap";
    srcLine.appendChild(srcText);
    const srcPick = el("button", "stlu-btn", "选历史结果…");
    const srcLive = el("button", "stlu-btn", "用本次结果");
    srcLine.appendChild(srcPick);
    srcLine.appendChild(srcLive);
    root.appendChild(srcLine);

    const subsetLine = el("div", "stlu-hint", "块：全部");
    root.appendChild(subsetLine);

    // No output switches here on purpose: depth data and the split sub
    // layers are always kept so step four can do whatever is asked.  The
    // only output choices live in step four.
    root.appendChild(el("div", "stlu-hint",
      "每块都保存 depth 数据和左右/前后子层；是否导出 depth 由 ④ 决定。"));

    // --- group prefix + output root --------------------------------------
    const groupLine = el("div", "stlu-line");
    groupLine.appendChild(el("span", "stlu-label", "分类前缀"));
    const groupInput = el("input", "stlu-input");
    groupInput.placeholder = "可留空 → SeeThrough/<本次 id>/";
    groupInput.value = groupWidget ? String(groupWidget.value || "") : "";
    groupInput.addEventListener("change", () => {
      if (groupWidget) setWidget(node, "group", groupInput.value.trim());
    });
    groupLine.appendChild(groupInput);
    root.appendChild(groupLine);

    const dirLine = el("div", "stlu-line");
    dirLine.appendChild(el("span", "stlu-label", "输出根目录"));
    const dirText = el("span", "stlu-hint grow");
    dirText.textContent = (outWidget && outWidget.value) || "SeeThrough";
    dirText.title = dirText.textContent;
    dirLine.appendChild(dirText);
    const pickBtn = el("button", "stlu-btn", "选择…");
    pickBtn.addEventListener("click", () => {
      openPicker({
        title: "选择输出根目录",
        mode: "folder",
        start: outWidget ? outWidget.value : "",
        onPick: (rel) => {
          if (!outWidget) return;
          setWidget(node, "out_dir", rel || "SeeThrough");
          dirText.textContent = outWidget.value;
          dirText.title = outWidget.value;
        },
      });
    });
    dirLine.appendChild(pickBtn);
    root.appendChild(dirLine);
    if (outWidget) {
      outWidget.callback = () => {
        dirText.textContent = outWidget.value || "SeeThrough";
        dirText.title = outWidget.value || "";
      };
    }

    const actions = el("div", "stlu-line");
    const offlineBtn = el("button", "stlu-btn primary", "生成分格 PSD");
    const openFirst = el("button", "stlu-btn", "打开 PSD");
    const reveal = el("button", "stlu-btn", "打开文件夹");
    openFirst.disabled = true;
    reveal.disabled = true;
    actions.appendChild(offlineBtn);
    actions.appendChild(openFirst);
    actions.appendChild(reveal);
    root.appendChild(actions);

    const statusLine = el("div", "stlu-hint", "");
    root.appendChild(statusLine);

    let info = null;
    let sourceInfo = null;
    let sourceInfoDir = "";
    let loadingInfo = false;

    function readSource() {
      return srcWidget ? String(srcWidget.value || "").trim() : "";
    }

    function linkedSource() {
      return upstreamNode(node, "source") ? lastImagesDir : "";
    }

    function hasSource() {
      return !!(readSource() || linkedSource());
    }

    function selectedTiles() {
      return parseIndexList(tilesWidget ? tilesWidget.value : "");
    }

    function sourceLabel() {
      const value = readSource();
      if (value) {
        const segments = value.split("/");
        let fallback = segments[segments.length - 1] || value;
        if (fallback === "images" && segments.length > 1) fallback = segments[segments.length - 2];
        const infoName = sourceInfo && sourceInfo.name ? sourceInfo.name : fallback;
        const count = sourceInfo && sourceInfo.count ? sourceInfo.count + " 块" : "";
        return [infoName, count].filter(Boolean).join(" · ");
      }
      const linked = linkedSource();
      if (linked) return "跟随 source 输入（" + linked + "）";
      if (upstreamNode(node, "parts")) return "本次结果（实时，来自 SeeThrough）";
      return "未选择";
    }

    function refreshSource(skipInfo) {
      const value = readSource();
      const linked = linkedSource();
      const live = !!upstreamNode(node, "parts");
      srcText.textContent = sourceLabel();
      srcText.title = value || linked || (live ? "本次结果（实时）" : "");
      tag.className = "stlu-tag" + (value || linked ? " ok" : "");
      tag.textContent = value ? "历史" : (linked ? "历史*" : (live ? "本次" : "未选"));
      const picked = selectedTiles();
      if (value && picked && picked.length) {
        const total = sourceInfo && sourceInfo.count ? sourceInfo.count : "?";
        subsetLine.textContent = "块：只做 " + picked.length + " / " + total + " 块（点“选历史结果…”可改）";
        subsetLine.className = "stlu-hint warn";
      } else if (value || linked) {
        subsetLine.textContent = "块：全部";
        subsetLine.className = "stlu-hint";
      } else {
        subsetLine.textContent = "块：跟随 ① 的选格";
        subsetLine.className = "stlu-hint";
      }
      if (value) {
        srcText.className = "stlu-hint grow ok";
      } else if (linked || live) {
        srcText.className = "stlu-hint grow";
      } else {
        srcText.className = "stlu-hint grow warn";
        srcText.textContent = "没有接 parts：请先“选历史结果…”挑一份图片";
      }
      offlineBtn.disabled = !hasSource();
      if (value) {
        if (!skipInfo) loadSourceInfo();
      } else {
        sourceInfo = null;
        sourceInfoDir = "";
      }
    }

    async function loadSourceInfo() {
      const value = readSource();
      if (!value || loadingInfo || value === sourceInfoDir) return;
      loadingInfo = true;
      try {
        const data = await getJson("/stlu/tiles/images?dir=" + encodeURIComponent(value));
        sourceInfo = (data && data.source) || null;
        sourceInfoDir = value;
      } catch (e) {
        sourceInfo = null;
        sourceInfoDir = value;
      } finally {
        loadingInfo = false;
      }
      if (readSource() === value) refreshSource(true);
    }

    async function build() {
      offlineBtn.disabled = true;
      statusLine.className = "stlu-hint";
      statusLine.textContent = "生成中…";
      try {
        const data = await runTilesFromImages(node);
        statusLine.className = "stlu-hint ok";
        statusLine.textContent = "✓ 完成：" + (data.count || 0) + " 块 → " +
          (data.wave || data.dir || "");
      } catch (error) {
        statusLine.className = "stlu-hint error";
        statusLine.textContent = "失败：" + error.message;
      } finally {
        offlineBtn.disabled = !hasSource();
      }
    }

    srcPick.addEventListener("click", () => {
      openPicker({
        title: "选择图片波次文件夹（③ 的图片来源）",
        mode: "folder",
        scopes: [{ scope: "output", label: "输出目录" }],
        start: String(widgetValue(node, "source", "") || ""),
        onPick: (rel) => {
          setWidget(node, "source", rel || "");
          setWidget(node, "tiles", "");
          refreshTilesPanels();
          refreshSource(true);
        },
      });
    });
    srcLive.addEventListener("click", () => {
      setWidget(node, "source", "");
      setWidget(node, "tiles", "");
      sourceInfo = null;
      refreshSource();
    });
    offlineBtn.addEventListener("click", build);

    if (srcWidget) {
      const oldCb = srcWidget.callback;
      srcWidget.callback = function () {
        const r = oldCb ? oldCb.apply(this, arguments) : undefined;
        refreshSource();
        return r;
      };
    }
    if (tilesWidget) {
      const oldCb = tilesWidget.callback;
      tilesWidget.callback = function () {
        const r = oldCb ? oldCb.apply(this, arguments) : undefined;
        refreshSource();
        return r;
      };
    }
    if (groupWidget) {
      const oldCb = groupWidget.callback;
      groupWidget.callback = function () {
        const r = oldCb ? oldCb.apply(this, arguments) : undefined;
        groupInput.value = String(groupWidget.value || "");
        return r;
      };
    }

    // The console (or another panel) changed the source: pick it up.
    const refreshAll = () => {
      try { refreshSource(); } catch (e) { /* not mounted */ }
    };
    tilesStates.add(refreshAll);
    const oldRemoved = node.onRemoved;
    node.onRemoved = function () {
      tilesStates.delete(refreshAll);
      return oldRemoved ? oldRemoved.apply(this, arguments) : undefined;
    };

    function pathOf(file, subfolder) {
      if (!file) return "";
      return (subfolder ? subfolder + "/" : "") + file;
    }

    openFirst.addEventListener("click", () => {
      if (!info) return;
      if (info.merged_psd && info.merged_psd.filename) {
        openPath(pathOf(info.merged_psd.filename, info.merged_psd.subfolder || info.dir), true);
      } else if (info.filename) {
        openPath(pathOf(info.filename, info.subfolder || info.dir), true);
      }
    });
    reveal.addEventListener("click", () => {
      if (info) openPath(info.dir || info.subfolder || ".", false);
    });

    function applyInfo(data) {
      if (!data) return;
      info = data;
      const isError = !!data.error;
      const merged = data.merged_psd;
      dot.className = "stlu-dot " + (isError ? "bad" : "ok");
      tag.className = "stlu-tag" + (isError ? "" : " ok");
      tag.textContent = isError ? "错误" : "完成";
      const bits = [];
      if (data.wave) bits.push("文件夹 " + data.wave);
      if (data.count) bits.push(data.count + " 块");
      if (data.layers) bits.push(data.layers + " 层/块");
      if (merged) {
        if (merged.layers) bits.push("合并 " + merged.layers + " 层");
        if (merged.canvas && merged.canvas.length === 2) bits.push(merged.canvas[0] + "×" + merged.canvas[1]);
      }
      if (data.size) bits.push(fmtBytes(data.size));
      if (data.merge_error) bits.push("合并失败：" + data.merge_error);
      sub.textContent = isError ? String(data.error).slice(0, 160) : bits.join(" · ");
      sub.className = "stlu-hint" + ((isError || data.merge_error) ? " error" : "");
      if (data.images_dir) {
        sub.textContent += " · 图片在 " + data.images_dir;
        sub.title = "图片目录：" + data.images_dir;
      }
      openFirst.textContent = merged ? "打开合并 PSD" : "打开第一个 PSD";
      if (data.dir) {
        dirText.textContent = data.dir;
        dirText.title = data.dir;
      }
      if (data.preview && data.preview.filename) {
        const rel = pathOf(data.preview.filename, data.preview.subfolder || "");
        preview.src = thumbUrl(rel, 420);
        preview.style.display = "";
      }
      openFirst.disabled = isError;
      reveal.disabled = isError;
      refreshSource();
    }

    const dom = addDOM(node, "stlu_tiles_card", root, 340, 520);
    registerPanel(node, "tiles", dom, root);

    const oldExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      const result = oldExecuted ? oldExecuted.apply(this, arguments) : undefined;
      try {
        const payload = message && message.stlu_tiles;
        if (payload && payload.length) {
          applyInfo(payload[0]);
          handleTilesInfo(payload[0]);
        }
      } catch (e) {
        console.error("[STLU] cannot read tile PSD result", e);
      }
      return result;
    };

    refreshSource();
  }

  // -------------------------------------------------------------------------
  // Tile selection (run only the checked tiles)
  // -------------------------------------------------------------------------

  function upstreamNode(node, inputName) {
    const input = (node.inputs || []).find((i) => i.name === inputName);
    const link = input && input.link != null ? linkById(input.link) : null;
    if (!link) return null;
    const originId = link.origin_id != null ? link.origin_id : link[1];
    return allNodes().find((n) => n.id === originId) || null;
  }

  function findSplitNode() {
    const nodes = allNodes();
    return nodes.find((n) => n.type === TILE_GRID)
      || nodes.find((n) => n.type === "SplitImageToTileList")
      || null;
  }

  // The grid node (or the legacy splitter) owns tile size/overlap: every later
  // stage copies them so the selection grid and the tile coordinates stored in
  // the manifests cannot drift apart.
  function gridSourceOf(node) {
    const viaGrid = upstreamNode(node, "grid");
    if (viaGrid && viaGrid.type === TILE_GRID) return viaGrid;
    const viaTiles = upstreamNode(node, "tiles");
    if (viaTiles && (viaTiles.type === TILE_GRID || viaTiles.type === "SplitImageToTileList")) {
      return viaTiles;
    }
    return findSplitNode();
  }

  function syncTileParamsFromSplit(node) {
    const split = gridSourceOf(node);
    if (!split) return;
    const targets = [node];
    for (const other of allNodes()) {
      if (other !== node && (other.type === TILES || other.type === TILE_SELECT
                             || other.type === SAVE_TILES)) {
        targets.push(other);
      }
    }
    for (const target of targets) {
      for (const name of ["tile_width", "tile_height", "overlap"]) {
        const src = widgetOf(split, name);
        const dst = widgetOf(target, name);
        if (!src || !dst) continue;
        if (Number(dst.value) !== Number(src.value)) {
          setWidget(target, name, Number(src.value));
        }
      }
    }
  }

  function findUpstreamImageNode(node) {
    // Walk back through image/tile links until a node that actually names an
    // image file is found: SelectTiles <- TileGrid <- LoadImage, or a grid
    // node fed directly by LoadImage.
    let current = node;
    const seen = new Set();
    for (let depth = 0; depth < 8 && current && !seen.has(current.id); depth++) {
      seen.add(current.id);
      const imageWidget = widgetOf(current, "image");
      if (imageWidget && String(imageWidget.value || "").trim()) return current;
      current = upstreamNode(current, "image")
        || upstreamNode(current, "tiles")
        || upstreamNode(current, "original")
        || null;
    }
    return allNodes().find((n) => n.type === "LoadImage") || null;
  }

  function imagePathOf(imageNode) {
    if (!imageNode) return "";
    const w = widgetOf(imageNode, "image");
    return w ? String(w.value || "").replace(/\\/g, "/") : "";
  }

  function imageURLOf(imageNode) {
    const value = imagePathOf(imageNode);
    if (!value) return null;
    const cut = value.lastIndexOf("/");
    const sub = cut >= 0 ? value.slice(0, cut) : "";
    const name = cut >= 0 ? value.slice(cut + 1) : value;
    return apiUrl(
      "/view?filename=" + encodeURIComponent(name) +
      "&subfolder=" + encodeURIComponent(sub) + "&type=input"
    );
  }

  // Mirrors the Python grid_coords.  offsetX/offsetY are the phase of the
  // interior split lines; the first and last tiles still touch the edges, so
  // every tile keeps the full block size (no letterboxed edge crops).
  function axisStartsJS(length, tile, stride, phase) {
    const len = Math.max(1, Math.round(length));
    const size = Math.max(1, Math.round(tile));
    if (size >= len) return [0];
    const edge = len - size;
    const step = Math.max(1, Math.round(stride));
    const p = ((Math.round(phase || 0) % step) + step) % step;
    const starts = new Set([0, edge]);
    for (let s = p; s < edge; s += step) {
      if (s > 0) starts.add(s);
    }
    return Array.from(starts).sort((a, b) => a - b);
  }

  function gridCoordsJS(width, height, tileW, tileH, overlap, offsetX = 0, offsetY = 0) {
    const coords = [];
    const sx = Math.round(Math.max(tileW * 0.25, tileW - overlap));
    const sy = Math.round(Math.max(tileH * 0.25, tileH - overlap));
    const xs = axisStartsJS(width, tileW, sx, offsetX);
    const ys = axisStartsJS(height, tileH, sy, offsetY);
    for (const y of ys) {
      const yEnd = Math.min(y + tileH, height);
      for (const x of xs) {
        coords.push([x, y, Math.min(x + tileW, width), yEnd]);
      }
    }
    return coords;
  }

  // --- user-placed square windows (mirrors region_plan.py) ---------------
  // The plan key must match Python's byte for byte: it is what select_plan
  // carries, so a mismatch would make the backend re-map a current selection.
  function customPlanKeyJS(regions, width, height) {
    const parts = (regions || []).map((r) =>
      [Math.round(Number(r.x) || 0), Math.round(Number(r.y) || 0),
       Math.round(Number(r.size) || 0), Math.round(Number(r.res) || 0)].join(","));
    return "custom:" + parts.join(";") + ":" + Math.round(width) + "x" + Math.round(height);
  }

  function customCoordsJS(regions) {
    return (regions || []).map((r) => {
      const x = Math.round(Number(r.x) || 0);
      const y = Math.round(Number(r.y) || 0);
      const size = Math.round(Number(r.size) || 0);
      return [x, y, x + size, y + size];
    });
  }

  function normalizeCustomRegion(region, width, height) {
    const limit = Math.max(CUSTOM_MIN, Math.min(width || 0, height || 0));
    let size = Math.round((Number(region && region.size) || 0) / CUSTOM_STEP) * CUSTOM_STEP;
    if (!Number.isFinite(size) || size < CUSTOM_MIN) size = CUSTOM_MIN;
    size = Math.min(size, limit);
    let x = Math.round(Number(region && region.x) || 0);
    let y = Math.round(Number(region && region.y) || 0);
    x = Math.max(0, Math.min(Math.max(0, (width || 0) - size), x));
    y = Math.max(0, Math.min(Math.max(0, (height || 0) - size), y));
    let res = Math.round(Number(region && region.res) || 0);
    if (res > 0) {
      res = Math.round(res / CUSTOM_STEP) * CUSTOM_STEP;
      res = Math.max(CUSTOM_RES_MIN, Math.min(CUSTOM_RES_MAX, res));
    }
    const name = String((region && region.name) || "").replace(/[\r\n\t]+/g, " ").trim().slice(0, 24);
    return { x, y, size, res, name };
  }

  // Mirror of region_plan.uniform_windows: the largest inscribed square,
  // centred, deduplicated.  Importing never invents an area the user did not
  // already have.
  function uniformWindowsJS(boxes, width, height) {
    const limit = Math.min(width || 0, height || 0);
    const seen = new Set();
    const out = [];
    for (const box of boxes || []) {
      if (!box || box.length < 4) continue;
      const x0 = Number(box[0]), y0 = Number(box[1]);
      const x1 = Number(box[2]), y1 = Number(box[3]);
      const bw = x1 - x0, bh = y1 - y0;
      if (!(bw > 0) || !(bh > 0)) continue;
      let size = Math.round(Math.min(bw, bh) / CUSTOM_STEP) * CUSTOM_STEP;
      const cap = limit || size;
      size = Math.max(CUSTOM_MIN, Math.min(Math.max(CUSTOM_MIN, cap), size));
      if (size > bw || size > bh) continue;
      let x = x0 + Math.floor((bw - size) / 2);
      let y = y0 + Math.floor((bh - size) / 2);
      if (width) x = Math.max(0, Math.min(width - size, x));
      if (height) y = Math.max(0, Math.min(height - size, y));
      const key = x + "," + y + "," + size;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ x, y, size, res: 0, name: "" });
    }
    return out;
  }

  function parseSelection(text, total) {
    const raw = String(text == null ? "all" : text).trim().toLowerCase();
    if (!raw || raw === "all") return new Set(Array.from({ length: total }, (_, i) => i));
    if (raw === "none") return new Set();
    const picked = new Set();
    for (const chunk of raw.replace(/\s+/g, "").split(",")) {
      if (!chunk) continue;
      if (chunk.includes("-")) {
        const [a, b] = chunk.split("-");
        const lo = parseInt(a, 10);
        const hi = parseInt(b, 10);
        if (Number.isFinite(lo) && Number.isFinite(hi)) {
          for (let i = Math.min(lo, hi); i <= Math.max(lo, hi); i++) {
            if (i >= 0 && i < total) picked.add(i);
          }
        }
      } else {
        const v = parseInt(chunk, 10);
        if (Number.isFinite(v) && v >= 0 && v < total) picked.add(v);
      }
    }
    return picked;
  }

  function encodeSelection(set, total) {
    if (total > 0 && set.size >= total) return "all";
    if (!set.size) return "none";
    const arr = Array.from(set).sort((a, b) => a - b);
    const parts = [];
    let start = arr[0];
    let prev = arr[0];
    for (let i = 1; i < arr.length; i++) {
      const v = arr[i];
      if (v === prev + 1) { prev = v; continue; }
      parts.push(start === prev ? String(start) : start + "-" + prev);
      start = prev = v;
    }
    parts.push(start === prev ? String(start) : start + "-" + prev);
    return parts.join(",");
  }

  function overlapArea(a, b) {
    if (!a || !b) return 0;
    const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
    const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
    return w > 0 && h > 0 ? w * h : 0;
  }

  // Map a selection from one geometry (uniform tiles or an older strip plan)
  // onto new region boxes by source area.  A new box is kept only when at
  // least half of it was already checked: strip boxes overlap by design
  // (519 px in a 1662 px strip), so a bare intersection test would turn a
  // one-strip choice into every neighbour that shares the overlap band —
  // exactly the "it re-checked tiles I never picked" behaviour.
  function mapSelectionByArea(oldCoords, oldSelected, newCoords) {
    const next = new Set();
    const oldTotal = oldCoords ? oldCoords.length : 0;
    if (!oldTotal || !newCoords || !newCoords.length) return next;
    if (oldSelected && oldSelected.size >= oldTotal) {
      for (let i = 0; i < newCoords.length; i++) next.add(i);
      return next;
    }
    const boxes = [];
    for (const index of oldSelected || []) {
      if (index >= 0 && index < oldTotal) boxes.push(oldCoords[index]);
    }
    newCoords.forEach((box, index) => {
      const area = Math.max(1, (box[2] - box[0]) * (box[3] - box[1]));
      const covered = boxes.reduce((sum, oldBox) => sum + overlapArea(box, oldBox), 0);
      if (Math.min(1, covered / area) >= 0.5) next.add(index);
    });
    return next;
  }

  function setupTileSelect(node) {
    injectCss();
    syncTileParamsFromSplit(node);
    const selectWidget = widgetOf(node, "select");
    if (selectWidget) hideWidget(selectWidget);
    for (const name of ["tile_width", "tile_height", "overlap"]) {
      hideWidget(widgetOf(node, name));
    }

    const root = el("div", "stlu-panel");
    root.style.padding = "6px";
    root.style.gap = "4px";

    const head = el("div", "stlu-line");
    head.appendChild(el("span", "stlu-cardtitle", "选择要生成的块"));
    const count = el("span", "stlu-tag", "…");
    head.appendChild(count);
    head.appendChild(el("span", "grow"));
    const reloadBtn = el("button", "stlu-mini", "刷新");
    head.appendChild(reloadBtn);
    root.appendChild(head);

    // The picker below is a button grid, not the canvas: it works even when
    // the source image cannot be shown (wrong path, huge file, remote node),
    // which used to leave the panel with nothing clickable at all.
    const grid = el("div", "stlu-tilegrid");
    root.appendChild(grid);

    const hint = el("div", "stlu-hint", "正在读取原图…");
    root.appendChild(hint);

    const wrap = el("div", "stlu-gridwrap");
    wrap.style.display = "none";
    const canvas = document.createElement("canvas");
    wrap.appendChild(canvas);
    root.appendChild(wrap);

    const actions = el("div", "stlu-line");
    const allBtn = el("button", "stlu-btn", "全选");
    const noneBtn = el("button", "stlu-btn", "清空");
    const invertBtn = el("button", "stlu-btn", "反选");
    actions.appendChild(allBtn);
    actions.appendChild(noneBtn);
    actions.appendChild(invertBtn);
    root.appendChild(actions);

    root.appendChild(el("div", "stlu-hint", "点格子切换；未勾选的块不会跑扩散。"));

    let image = null;
    let imageUrl = "";
    let srcW = 0;
    let srcH = 0;
    let coords = [];
    let total = 0;
    let selected = new Set();
    let pendingText = selectWidget ? String(selectWidget.value || "all") : "all";

    function tileParam(name, fallback) {
      const v = Number(widgetValue(node, name, fallback));
      return Number.isFinite(v) && v > 0 ? v : fallback;
    }

    function syncWidget() {
      if (!selectWidget) return;
      setWidget(node, "select", encodeSelection(selected, total));
    }

    function updateCount() {
      count.textContent = total ? selected.size + " / " + total + " 块" : "…";
      count.className = "stlu-tag" + (total && selected.size !== total ? " ok" : "");
    }

    function toggle(index) {
      if (index < 0 || index >= total) return;
      if (selected.has(index)) selected.delete(index);
      else selected.add(index);
      refreshSelection();
    }

    function refreshSelection() {
      grid.querySelectorAll(".stlu-tcell").forEach((cell) => {
        const i = Number(cell.dataset.index);
        cell.classList.toggle("on", selected.has(i));
      });
      drawCanvas();
      updateCount();
      syncWidget();
    }

    function buildGrid() {
      grid.innerHTML = "";
      if (!total) {
        grid.appendChild(el("div", "stlu-hint", "没有可选的块"));
        return;
      }
      const scaleX = 72 / Math.max(1, tileParam("tile_width", 1024));
      const scaleY = 72 / Math.max(1, tileParam("tile_height", 1024));
      coords.forEach((c, i) => {
        const [x0, y0] = c;
        const cell = el("div", "stlu-tcell" + (selected.has(i) ? " on" : ""));
        cell.dataset.index = String(i);
        cell.title = "第 " + i + " 块 · x" + x0 + " y" + y0 + " · 点击切换";
        if (imageUrl) {
          cell.style.backgroundImage = 'url("' + imageUrl + '")';
          cell.style.backgroundSize = (srcW * scaleX) + "px " + (srcH * scaleY) + "px";
          cell.style.backgroundPosition = (-x0 * scaleX) + "px " + (-y0 * scaleY) + "px";
        }
        cell.appendChild(el("span", null, String(i)));
        const hit = (event) => {
          event.preventDefault();
          event.stopPropagation();
          toggle(i);
        };
        cell.addEventListener("pointerdown", hit);
        grid.appendChild(cell);
      });
    }

    function drawCanvas() {
      if (!image || !image.naturalWidth || !total) return;
      const iw = image.naturalWidth;
      const ih = image.naturalHeight;
      if (canvas.width !== iw || canvas.height !== ih) {
        canvas.width = iw;
        canvas.height = ih;
      }
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, iw, ih);
      ctx.drawImage(image, 0, 0, iw, ih);
      const lw = Math.max(2, Math.round(iw / 500));
      coords.forEach((c, i) => {
        const [x0, y0, x1, y1] = c;
        const on = selected.has(i);
        if (on) {
          ctx.fillStyle = "rgba(45,108,223,.28)";
          ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
        }
        ctx.strokeStyle = on ? "#5b93f5" : "rgba(255,255,255,.55)";
        ctx.lineWidth = lw;
        ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
      });
    }

    function layout() {
      syncTileParamsFromSplit(node);
      const tw = tileParam("tile_width", 1024);
      const th = tileParam("tile_height", 1024);
      const ov = Math.max(0, Number(widgetValue(node, "overlap", 128)) || 0);
      if (srcW > 0 && srcH > 0) {
        coords = gridCoordsJS(srcW, srcH, tw, th, ov);
      } else {
        coords = [];
      }
      total = coords.length;
      if (pendingText != null) {
        selected = parseSelection(pendingText, total);
        pendingText = null;
      } else {
        selected = new Set(Array.from(selected).filter((i) => i < total));
      }
      buildGrid();
      drawCanvas();
      updateCount();
      syncWidget();
    }

    async function loadSource() {
      hint.className = "stlu-hint";
      hint.textContent = "正在读取原图…";
      const sourceNode = findUpstreamImageNode(node);
      const path = imagePathOf(sourceNode);
      imageUrl = imageURLOf(sourceNode) || "";
      image = null;
      srcW = 0;
      srcH = 0;
      wrap.style.display = "none";

      if (path) {
        // Ask the backend for the pixel size: the grid therefore works even
        // if the browser cannot decode/load the picture.
        try {
          const data = await getJson("/stlu/imgsize?scope=input&path=" + encodeURIComponent(path));
          srcW = Number(data.width) || 0;
          srcH = Number(data.height) || 0;
        } catch (e) {
          srcW = 0;
          srcH = 0;
        }
      }

      if (!srcW || !srcH) {
        hint.className = "stlu-hint warn";
        hint.textContent = path
          ? "读不到原图尺寸（" + path + "），检查 image 输入是否接好"
          : "接上原图（image 输入）才能算出分块网格";
        layout();
        if (imageUrl) {
          // Last resort: fall back to the browser-decoded size.
          const img = new Image();
          img.crossOrigin = "anonymous";
          img.onload = () => {
            if (srcW && srcH) return;
            srcW = img.naturalWidth;
            srcH = img.naturalHeight;
            image = img;
            hint.className = "stlu-hint";
            hint.textContent = "原图 " + srcW + "×" + srcH + "（后端读不到，用浏览器尺寸）";
            layout();
          };
          img.src = imageUrl + (imageUrl.includes("?") ? "&" : "?") + "t=" + Date.now();
        }
        return;
      }

      layout();
      hint.textContent = "原图 " + srcW + "×" + srcH + " · 共 " + total + " 块";
      if (imageUrl) {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
          image = img;
          wrap.style.display = "";
          drawCanvas();
        };
        img.src = imageUrl + (imageUrl.includes("?") ? "&" : "?") + "t=" + Date.now();
      }
    }

    canvas.addEventListener("pointerdown", (event) => {
      if (!image || !total) return;
      event.preventDefault();
      event.stopPropagation();
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const x = ((event.clientX - rect.left) / rect.width) * srcW;
      const y = ((event.clientY - rect.top) / rect.height) * srcH;
      let hit = -1;
      coords.forEach((c, i) => {
        if (x >= c[0] && x < c[2] && y >= c[1] && y < c[3]) hit = i;
      });
      toggle(hit);
    });

    allBtn.addEventListener("click", () => {
      selected = new Set(Array.from({ length: total }, (_, i) => i));
      refreshSelection();
    });
    noneBtn.addEventListener("click", () => {
      selected = new Set();
      refreshSelection();
    });
    invertBtn.addEventListener("click", () => {
      const next = new Set();
      for (let i = 0; i < total; i++) if (!selected.has(i)) next.add(i);
      selected = next;
      refreshSelection();
    });
    reloadBtn.addEventListener("click", loadSource);

    // Recompute the grid when the tile parameters change.
    for (const name of ["tile_width", "tile_height", "overlap"]) {
      const w = widgetOf(node, name);
      if (!w) continue;
      const old = w.callback;
      w.callback = function () {
        const r = old ? old.apply(this, arguments) : undefined;
        pendingText = selectWidget ? String(selectWidget.value || "all") : "all";
        layout();
        return r;
      };
    }

    addDOM(node, "stlu_tile_select_card", root, 320, 640);
    loadSource();
  }

  // -------------------------------------------------------------------------
  // Stage 1: the global tile grid
  // -------------------------------------------------------------------------

  async function probeImageSize(raw, scopes) {
    for (const scope of scopes) {
      try {
        const data = await getJson(
          "/stlu/imgsize?scope=" + scope + "&path=" + encodeURIComponent(raw));
        const width = Number(data.width) || 0;
        const height = Number(data.height) || 0;
        if (width && height) return { width, height, scope };
      } catch (e) { /* try the next scope */ }
    }
    return null;
  }

  // One shared listener per source node: several panels may watch the same
  // LoadImage, and each needs to hear when its ``image`` widget changes
  // (picking a file, or finishing an upload).  Keyed by node so wrappers
  // never pile up.
  const imageWatch = new WeakMap();

  function watchImageNode(source, listener) {
    if (!source) return function () {};
    let entry = imageWatch.get(source);
    if (!entry) {
      const widget = widgetOf(source, "image");
      const listeners = new Set();
      if (widget) {
        const prev = widget.callback;
        widget.callback = function () {
          const result = prev ? prev.apply(this, arguments) : undefined;
          for (const fn of Array.from(listeners)) {
            try { fn(); } catch (e) { console.warn("[STLU] image watch", e); }
          }
          return result;
        };
      }
      entry = { widget, listeners };
      imageWatch.set(source, entry);
    }
    entry.listeners.add(listener);
    return function () {
      entry.listeners.delete(listener);
    };
  }

  // Stage ① cuts the grid and picks the tiles in one place: the panel draws
  // the source image with the tile grid on top, and clicking a cell writes
  // the node's visible ``select`` widget.  Typing "0-2,5" works the same.
  function setupTileGrid(node) {
    injectCss();
    const selectWidget = widgetOf(node, "select");

    const root = el("div", "stlu-panel");
    root.style.padding = "6px";
    root.style.gap = "4px";

    const isPipeline = node.type === PIPELINE
      || node.type === PIPELINE_SETTINGS
      || node.type === SIMPLE;

    const head = el("div", "stlu-line");
    head.appendChild(el("span", "stlu-cardtitle",
      isPipeline ? "① 用户输入（只改这里）" : "① 切格 + 选块"));
    const count = el("span", "stlu-tag", "…");
    head.appendChild(count);
    head.appendChild(el("span", "grow"));
    const reloadBtn = el("button", "stlu-mini", "刷新");
    head.appendChild(reloadBtn);
    root.appendChild(head);

    const infoLine = el("div", "stlu-hint", "接上 LoadImage 就会显示切格");
    root.appendChild(infoLine);

    // Cache warning: shown when the folder's manifest was written with
    // different pixel-affecting settings.  Queue will clear and rerun all.
    const cacheHint = el("div", "stlu-hint", "");
    cacheHint.style.display = "none";
    cacheHint.style.whiteSpace = "normal";
    root.appendChild(cacheHint);

    // Subject-strip plan summary and warnings, filled by /stlu/tiles/strips.
    const planInfo = el("div", "stlu-hint", "");
    planInfo.style.display = "none";
    planInfo.style.whiteSpace = "normal";
    root.appendChild(planInfo);
    const planWarnings = el("div", "stlu-hint warn", "");
    planWarnings.style.display = "none";
    planWarnings.style.whiteSpace = "pre-wrap";
    root.appendChild(planWarnings);

    const grid = el("div", "stlu-tilegrid");
    root.appendChild(grid);

    const wrap = el("div", "stlu-gridwrap");
    wrap.style.display = "none";
    const stack = el("div", "stlu-stack");
    const canvas = document.createElement("canvas");
    stack.appendChild(canvas);
    // Vector overlay for the custom window editor: dragging repaints only the
    // boxes, never the (possibly 4K) source image underneath.
    const overlay = document.createElement("canvas");
    overlay.className = "stlu-vec";
    overlay.tabIndex = 0;
    overlay.title = "拖框移动；拖四角缩放；双击空白新建；选中后按 Delete 删除";
    stack.appendChild(overlay);
    wrap.appendChild(stack);
    root.appendChild(wrap);

    const actions = el("div", "stlu-line");
    const allBtn = el("button", "stlu-btn", "全选");
    const noneBtn = el("button", "stlu-btn", "清空");
    const invertBtn = el("button", "stlu-btn", "反选");
    actions.appendChild(allBtn);
    actions.appendChild(noneBtn);
    actions.appendChild(invertBtn);
    root.appendChild(actions);

    // An empty selection is not a cosmetic state: the node raises before any
    // generation.  Say so next to the count instead of letting Queue fail.
    const selectionHint = el("div", "stlu-hint error", "");
    selectionHint.style.display = "none";
    selectionHint.style.whiteSpace = "normal";
    root.appendChild(selectionHint);

    // "Click the subject" mode: the next canvas click moves the grid phase so
    // that point lands in the middle of a tile.  That is the practical fix
    // for a centred face being cut by a seam.
    let alignMode = false;
    let alignBtn = null;
    function setAlignMode(on) {
      alignMode = !!on;
      canvas.style.cursor = alignMode ? "crosshair" : "";
      if (alignBtn) {
        alignBtn.classList.toggle("active", alignMode);
        alignBtn.textContent = alignMode ? "点一下画面…" : "点图对齐主体";
        alignBtn.title = alignMode
          ? "在画面上点人物（例如脸的中心），网格会平移到这个点正好落在一块的中心"
          : "点一下这个按钮，再点画面上的脸/头部中心：自动平移内部切分线，让某一块完整包住人物";
      }
    }

    let image = null;
    let imageUrl = "";
    let srcW = 0;
    let srcH = 0;
    let coords = [];
    let total = 0;
    let selected = new Set();
    let writingSelect = false;
    // Subject-strip preview state.  planState is null until the CPU route has
    // answered; layout() then uses its square regions instead of the grid.
    let sourcePath = "";
    let sourceEpoch = 0;
    let planToken = 0;
    let planState = null;
    let planAppliedKey = "";
    // Custom square windows: the editor lives on the canvas, the list below
    // reads and writes the same array.  Array order is the generation order.
    let customRegions = [];
    let customActive = -1;
    let customPreview = null;
    let customDrag = null;
    let customRaf = 0;
    let lastCustomSize = 1024;
    // Windows produced by the last Queue, importable as a starting point.
    let lastExecutedWindows = null;
    // Assigned by buildPipelineControls (the list DOM lives in that card).
    let renderCustomList = function () {};
    let customCountTag = null;

    // Must come AFTER the state above: the factory builds the custom-window
    // card and initialises those bindings.  Declaring them later put them in
    // the temporal dead zone and aborted the whole ① panel on load.
    const pipelineControls = isPipeline ? buildPipelineControls() : null;
    if (pipelineControls && pipelineControls.modeCard) {
      // Mode is the first decision; keep it above the preview instead of
      // burying it below the grid and canvas.
      root.insertBefore(pipelineControls.modeCard, infoLine);
    }

    function isCustomMode() {
      return String(widgetValue(node, "partition_mode", "均匀网格") || "均匀网格") === CUSTOM_MODE;
    }

    function customCoords() {
      return customCoordsJS(customRegions);
    }

    function customPlanKey() {
      return customPlanKeyJS(customRegions, srcW || 0, srcH || 0);
    }

    function looseRegion(item) {
      return {
        x: Math.round(Number(item && item.x) || 0),
        y: Math.round(Number(item && item.y) || 0),
        size: Math.round(Number(item && item.size) || 0),
        res: Math.round(Number(item && item.res) || 0),
        name: String((item && item.name) || "").replace(/[\r\n\t]+/g, " ").trim().slice(0, 24),
      };
    }

    function readCustomRegionsFromWidget() {
      const raw = String(widgetValue(node, "custom_regions", "[]") || "[]").trim() || "[]";
      let data;
      try {
        data = JSON.parse(raw);
      } catch (e) {
        console.warn("[STLU] custom_regions is not valid JSON", e);
        return [];
      }
      if (!Array.isArray(data)) return [];
      if (!srcW || !srcH) return data.filter((item) => item && typeof item === "object").map(looseRegion);
      return data.filter((item) => item && typeof item === "object")
        .map((item) => normalizeCustomRegion(item, srcW, srcH));
    }

    // Swap in a new window array.  Selection follows the indices unless the
    // caller passes an explicit set (add/reorder) or asks for area mapping
    // (delete/import).  Every commit keeps select_plan in step with the key so
    // the backend never re-maps a selection that already matches this layout.
    function applyCustomRegions(next, options) {
      const opts = options || {};
      const oldCoords = customCoords();
      const oldSelected = new Set(selected);
      const oldTotal = total;
      customRegions = (next || []).map((region) => normalizeCustomRegion(region, srcW, srcH));
      const newCoords = customCoordsJS(customRegions);
      let nextSelected = oldSelected;
      if (opts.selection) {
        nextSelected = new Set(opts.selection);
      } else if (opts.remap && oldCoords.length) {
        const base = oldTotal > 0
          ? oldSelected : parseSelection(currentText(), oldCoords.length);
        const mapped = mapSelectionByArea(oldCoords, base, newCoords);
        nextSelected = mapped.length
          ? new Set(mapped) : parseSelection(currentText(), newCoords.length);
      }
      selected = new Set(Array.from(nextSelected)
        .filter((index) => index >= 0 && index < newCoords.length));
      total = newCoords.length;
      customPreview = null;
      customDrag = null;
      setWidget(node, "custom_regions", JSON.stringify(customRegions));
      setWidget(node, "select_plan", customPlanKey());
      if (renderCustomList) renderCustomList();
      // Write the selection before layout(): layout trusts a select string
      // whose select_plan matches, and must not reparse the stale one.
      writeSelect();
      layout();
      refreshCacheHint();
    }

    function addCustomWindow(cx, cy) {
      const limit = Math.max(CUSTOM_MIN, Math.min(srcW || CUSTOM_MIN, srcH || CUSTOM_MIN));
      let size = Math.round((lastCustomSize || 1024) / CUSTOM_STEP) * CUSTOM_STEP;
      size = Math.max(CUSTOM_MIN, Math.min(limit, size));
      const centerX = cx == null ? (srcW || size) / 2 : cx;
      const centerY = cy == null ? (srcH || size) / 2 : cy;
      const x = Math.max(0, Math.min((srcW || size) - size, Math.round(centerX - size / 2)));
      const y = Math.max(0, Math.min((srcH || size) - size, Math.round(centerY - size / 2)));
      const next = customRegions.concat([{ x, y, size, res: 0, name: "" }]);
      const selection = new Set(selected);
      selection.add(next.length - 1);
      customActive = next.length - 1;
      lastCustomSize = size;
      applyCustomRegions(next, { selection });
    }

    function deleteCustomWindow(index) {
      if (index < 0 || index >= customRegions.length) return;
      customActive = -1;
      applyCustomRegions(customRegions.filter((_, i) => i !== index), { remap: true });
    }

    function moveCustomWindow(index, delta) {
      const other = index + delta;
      if (other < 0 || other >= customRegions.length) return;
      const next = customRegions.slice();
      const swap = next[index];
      next[index] = next[other];
      next[other] = swap;
      // The selection is per index; swap the two flags so it stays with the
      // window the user checked, not the slot.
      const selection = new Set(selected);
      const a = selection.has(index);
      const b = selection.has(other);
      selection.delete(index);
      selection.delete(other);
      if (b) selection.add(index);
      if (a) selection.add(other);
      customActive = other;
      applyCustomRegions(next, { selection });
    }

    // A new source image may be smaller than the one the windows were placed
    // on; clamp before drawing/queueing instead of failing at run time.
    function clampCustomToCanvas() {
      if (!srcW || !srcH || !customRegions.length) return;
      const next = customRegions.map((region) => normalizeCustomRegion(region, srcW, srcH));
      const changed = next.some((region, index) => {
        const old = customRegions[index];
        return region.x !== old.x || region.y !== old.y ||
          region.size !== old.size || region.res !== old.res;
      });
      if (!changed) return;
      customRegions = next;
      total = next.length;
      setWidget(node, "custom_regions", JSON.stringify(customRegions));
      setWidget(node, "select_plan", customPlanKey());
      if (renderCustomList) renderCustomList();
    }

    customRegions = readCustomRegionsFromWidget();

    function isStripsMode() {
      return String(widgetValue(node, "partition_mode", "均匀网格") || "均匀网格") === STRIPS_MODE;
    }

    function stripsPlanOk() {
      return !!(planState && planState.plan && planState.plan.ok);
    }

    function planFailure() {
      return (planState && planState.plan && !planState.plan.ok) ? planState.plan : null;
    }

    function stripSettings() {
      return {
        overlap: Math.max(0, Number(widgetValue(node, "overlap", 128)) || 0),
        max_side: Math.max(512, Number(widgetValue(node, "strip_max_side", 2048)) || 2048),
        resolution: Math.max(512, Number(widgetValue(node, "resolution", 512)) || 512),
        margin: Math.max(0, Number(widgetValue(node, "strip_margin", 32)) || 0),
      };
    }

    function tileParam(name, fallback) {
      const v = Number(widgetValue(node, name, fallback));
      return Number.isFinite(v) && v > 0 ? v : fallback;
    }

    function currentText() {
      return selectWidget ? String(selectWidget.value || "all") : "all";
    }

    function writeSelect() {
      if (!selectWidget) return;
      // The node uses select_plan to know which geometry this selection
      // belongs to.  Without it, changing the strip plan would misread old
      // indices; with it, the backend can still map an untouched old string.
      if (isPipeline && isStripsMode() && planState && planState.plan && planState.plan.ok) {
        const planId = String(planState.plan.plan_id || "");
        if (planId && String(widgetValue(node, "select_plan", "") || "") !== planId) {
          setWidget(node, "select_plan", planId);
        }
      } else if (isPipeline && isCustomMode()) {
        const planId = customPlanKey();
        if (planId && String(widgetValue(node, "select_plan", "") || "") !== planId) {
          setWidget(node, "select_plan", planId);
        }
      }
      const encoded = encodeSelection(selected, total);
      if (String(selectWidget.value || "") === encoded) return;
      writingSelect = true;
      try { setWidget(node, "select", encoded); } finally { writingSelect = false; }
      // The strip hint names the checked count; refresh it now instead of
      // waiting for the one-second poll (it would show a stale 0/N).
      if (isPipeline && isStripsMode()) refreshCacheHint();
    }

    function syncFromWidget() {
      if (writingSelect) return;
      selected = parseSelection(currentText(), total);
      paintCells();
      drawCanvas();
      updateCount();
    }

    function toggle(index) {
      if (index < 0 || index >= total) return;
      if (selected.has(index)) selected.delete(index);
      else selected.add(index);
      paintCells();
      drawCanvas();
      updateCount();
      writeSelect();
    }

    function updateCount() {
      // In a failed strip plan the cells on screen are the uniform fallback.
      const strips = isStripsMode() && stripsPlanOk();
      const custom = isCustomMode();
      const unit = custom ? "个窗口" : (strips ? "条" : "块");
      const empty = total > 0 && selected.size === 0;
      count.textContent = total ? selected.size + " / " + total + " " + unit : "…";
      count.className = "stlu-tag" +
        (empty ? " error" : (total && selected.size !== total ? " ok" : ""));
      allBtn.classList.toggle("attention", empty);
      selectionHint.style.display = empty ? "" : "none";
      selectionHint.textContent = empty
        ? "未勾选任何" + unit + "：Queue 会在这里停下。点「全选」或点格子/画面勾选。"
        : "";
    }

    function showPlan(plan) {
      if (!isPipeline) return;
      if (!plan) {
        planInfo.style.display = "none";
        planWarnings.style.display = "none";
        return;
      }
      if (!plan.ok) {
        planInfo.style.display = "";
        planInfo.textContent = "分区未生效：Queue 会按均匀网格运行";
        const text = plan.error || "无法生成方形分区";
        planWarnings.className = "stlu-hint error";
        planWarnings.textContent = [text].concat(plan.warnings || []).filter(Boolean).join("\n");
        planWarnings.style.display = "";
        return;
      }
      planWarnings.className = "stlu-hint warn";
      const regions = plan.regions || [];
      const side = Number(plan.side) || 0;
      const resolution = Number(plan.resolution) || 0;
      const density = Number(plan.density) || (resolution ? side / resolution : 0);
      const bbox = (plan.subject && plan.subject.bbox) ||
        (plan.subject_detection && plan.subject_detection.bbox) || null;
      const boxText = bbox && bbox.length === 4
        ? "主体 " + (bbox[2] - bbox[0]) + "×" + (bbox[3] - bbox[1]) +
          " @ " + bbox[0] + "," + bbox[1]
        : "主体已检测";
      planInfo.style.display = "";
      planInfo.textContent = boxText + " · " + regions.length + " 条 · 边长 " + side +
        " · 模型 " + resolution + "（" + density.toFixed(2) + " 源px/遮罩px）";
      const warn = (plan.warnings || []).filter(Boolean);
      planWarnings.textContent = warn.join("\n");
      planWarnings.style.display = warn.length ? "" : "none";
    }

    function schedulePlan(force) {
      if (!isPipeline) return;
      if (!isStripsMode()) {
        planToken += 1;
        planState = null;
        showPlan(null);
        layout();
        return;
      }
      if (!sourcePath) {
        planState = null;
        showPlan({ ok: false, error: "接上 LoadImage 后才能自动分区。", warnings: [] });
        layout();
        return;
      }
      const settings = stripSettings();
      const key = [sourcePath, sourceEpoch, settings.overlap, settings.max_side,
        settings.resolution, settings.margin].join("|");
      if (!force && planState && planState.key === key && planState.plan) return;
      const token = ++planToken;
      planInfo.style.display = "";
      planInfo.textContent = "正在计算方形分区…";
      planWarnings.style.display = "none";
      const query = new URLSearchParams({
        path: sourcePath,
        overlap: String(settings.overlap),
        max_side: String(settings.max_side),
        resolution: String(settings.resolution),
        margin: String(settings.margin),
      });
      getJson("/stlu/tiles/strips?" + query.toString())
        .then((plan) => {
          if (token !== planToken) return;
          planState = plan && plan.ok ? {
            key,
            plan,
            coords: (plan.regions || []).map((box) => box.slice(0, 4)),
            side: Number(plan.side) || 0,
            resolution: Number(plan.resolution) || 0,
            plan_id: String(plan.plan_id || ""),
            warnings: plan.warnings || [],
          } : { key, plan: plan || { ok: false, error: "分区接口没有返回结果。" }, coords: [] };
          showPlan(planState.plan);
          layout();
          refreshCacheHint();
        })
        .catch((err) => {
          if (token !== planToken) return;
          console.warn("[STLU] strip plan failed", err);
          planState = {
            key,
            plan: { ok: false, error: "读取分区计划失败：" + (err && err.message ? err.message : err) },
            coords: [],
          };
          showPlan(planState.plan);
          layout();
          refreshCacheHint();
        });
    }

    // --- cache compatibility hint ----------------------------------------
    // The backend owns the real check; this mirrors the non-model part so a
    // changed setting is visible before Queue.  Model changes are caught by
    // the stlu.cache_reset event during execution.
    const GEN_LABELS = {
      source: "原图",
      source_size: "原图尺寸",
      tile_size: "块尺寸",
      overlap: "重叠",
      offset: "网格偏移",
      resolution: "分辨率",
      num_inference_steps: "步数",
      depth_resolution: "深度分辨率",
      random_seed: "种子",
      tblr_split: "左右拆分",
      use_lama: "LaMa",
      alpha_threshold: "透明阈值",
      context_pad: "块外上下文",
      guide: "引导参数",
      layerdiff_model: "拆层模型",
      depth_model: "深度模型",
      partition: "分区方式",
      plan_id: "分区位置",
    };

    function currentGeneration() {
      const gen = {
        version: 1,
        source_size: [srcW, srcH],
        tile_size: [
          Math.max(64, Number(widgetValue(node, "tile_width", 1024)) || 1024),
          Math.max(64, Number(widgetValue(node, "tile_height", 1024)) || 1024),
        ],
        overlap: Math.max(0, Number(widgetValue(node, "overlap", 128)) || 0),
        resolution: Number(widgetValue(node, "resolution", 512)),
        num_inference_steps: Number(widgetValue(node, "num_inference_steps", 30)),
        depth_resolution: Number(widgetValue(node, "depth_resolution", -1)),
        random_seed: Number(widgetValue(node, "random_seed", 0)),
        tblr_split: !!widgetValue(node, "tblr_split", true),
        use_lama: !!widgetValue(node, "use_lama", true),
        alpha_threshold: Number(widgetValue(node, "alpha_threshold", 8)),
        context_pad: Math.max(0, Number(widgetValue(node, "context_pad", 0)) || 0),
        guide: {
          // The simple node has no guide widgets; the backend fills in the
          // shared defaults, so these fallbacks must be those defaults.
          enabled: !!widgetValue(node, "guide_context", true),
          feather: Math.max(0, Number(widgetValue(node, "guide_feather", 32)) || 0),
          strength: Math.min(100, Math.max(0, Number(widgetValue(node, "guide_strength", 40)) || 0)),
          tolerance: Math.min(255, Math.max(0, Number(widgetValue(node, "guide_tolerance", 16)) || 0)),
        },
      };
      // Mirrors the backend: the key is absent at 0, so a folder written by
      // the previous version is not reported as a mismatch.
      const ox = Math.max(0, Number(widgetValue(node, "grid_offset_x", 0)) || 0);
      const oy = Math.max(0, Number(widgetValue(node, "grid_offset_y", 0)) || 0);
      if (ox || oy) gen.offset = [ox, oy];
      return gen;
    }

    function genDeepEqual(a, b) {
      if (a === b) return true;
      if (Array.isArray(a) || Array.isArray(b)) {
        if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) if (!genDeepEqual(a[i], b[i])) return false;
        return true;
      }
      if (a && b && typeof a === "object" && typeof b === "object") {
        const ka = Object.keys(a);
        const kb = Object.keys(b);
        if (ka.length !== kb.length) return false;
        for (const k of ka) {
          if (!Object.prototype.hasOwnProperty.call(b, k) || !genDeepEqual(a[k], b[k])) return false;
        }
        return true;
      }
      return false;
    }

    function outputDirRel() {
      const out = String(widgetValue(node, "out_dir", "SeeThrough") || "SeeThrough")
        .replace(/^[\\/]+|[\\/]+$/g, "");
      const grp = String(widgetValue(node, "group", "") || "")
        .replace(/^[\\/]+|[\\/]+$/g, "");
      return grp ? out + "/" + grp : out;
    }

    let lastCacheKey = "";
    let cacheResetNotice = "";
    let cacheResetAt = 0;
    function showCacheHint(text, kind) {
      if (!isPipeline) { cacheHint.style.display = "none"; return; }
      if (!text) { cacheHint.style.display = "none"; return; }
      cacheHint.style.display = "";
      cacheHint.className = "stlu-hint" + (kind ? " " + kind : "");
      cacheHint.textContent = text;
    }

    function resetNoticeActive() {
      return !!cacheResetNotice && (Date.now() - cacheResetAt) < 6 * 60 * 60 * 1000;
    }

    function showResetNotice() {
      if (!resetNoticeActive()) return false;
      showCacheHint(cacheResetNotice, "warn");
      return true;
    }

    async function refreshCacheHint() {
      if (!isPipeline) return;
      if (isStripsMode()) {
        // Strip plans are compared by the backend's generation/plan_id.  The
        // panel only needs to warn that a mode or plan change clears the
        // folder and that only checked strips run; a full JS mirror of the
        // plan object would report false diffs while the CPU route is loading.
        const plan = planState && planState.plan;
        lastCacheKey = "";
        if (!plan) {
          showCacheHint("方形全宽分区：正在计算分区计划…", "warn");
          return;
        }
        // planInfo/planWarnings already carry the plan and its error.  This
        // line only states the reset/selection rule, so nothing is repeated
        // and an empty selection stays a single red selectionHint.
        if (!plan.ok) {
          showCacheHint(
            "方形全宽分区未生效：Queue 会按均匀网格运行；" +
            "修改分区参数会清空该目录。", "warn");
          return;
        }
        const count = (plan.regions || []).length;
        showCacheHint(
          "方形全宽分区：自动计划 " + count + " 条。" +
          "切换分区方式或修改分区参数会清空该目录，只重跑勾选的条。", "warn");
        return;
      }
      if (isCustomMode()) {
        // The backend compares the ordered window list in the fingerprint;
        // the panel only needs to state the reset rule (a per-window resolution
        // or an order change is not derivable from the tab's other widgets).
        lastCacheKey = "";
        if (!customRegions.length) {
          showCacheHint(
            "自定义分区：还没有窗口。双击画面或点「＋ 新窗口」创建后再 Queue。", "warn");
          return;
        }
        showCacheHint(
          "自定义分区：" + customRegions.length + " 个窗口，顺序 = 生成顺序。" +
          "改位置/边长/分辨率/顺序会清空该目录，只重跑勾选的窗口。", "warn");
        return;
      }
      const current = currentGeneration();
      const dir = outputDirRel();
      const key = dir + "|" + JSON.stringify(current);
      if (key === lastCacheKey) return;
      lastCacheKey = key;
      let doc;
      try {
        doc = await getJson("/stlu/tiles/manifest?dir=" + encodeURIComponent(dir));
      } catch (e) {
        // The manifest is written at the end of a run; keep the reset notice
        // visible while the folder is being regenerated.
        if (!showResetNotice()) showCacheHint("", "");
        return;
      }
      const old = doc && doc.generation;
      // A mismatch clears the folder first; an explicit tile selection only
      // changes how much is regenerated, never whether the folder is reset.
      const partial = selected.size > 0 && selected.size < total;
      if (!old || typeof old !== "object") {
        if (!showResetNotice()) {
          showCacheHint(partial
            ? "该目录是旧版结果（没有参数指纹）。Queue 会清空该目录，只重跑选中的 " + selected.size + " 块。"
            : "该目录是旧版结果（没有参数指纹）。Queue 会清空并重跑全部块。", "warn");
        }
        return;
      }
      const keys = new Set([...Object.keys(current), ...Object.keys(old)]);
      const changed = Array.from(keys).filter((k) => {
        // Seed / context padding are targeted re-runs: the backend keeps the
        // folder and only regenerates the checked tiles.
        if (k === "random_seed" || k === "context_pad") return false;
        if (k === "source_size" && (!srcW || !srcH)) return false;
        return !genDeepEqual(old[k], current[k]);
      });
      if (changed.length) {
        const names = changed.map((k) => GEN_LABELS[k] || k).join("、");
        showCacheHint(partial
          ? "检测到生成参数已变：" + names + "。Queue 会清空该目录，只重跑选中的 " + selected.size + " 块。"
          : "检测到生成参数已变：" + names + "。Queue 会清空该目录并重跑全部块。", "warn");
      } else {
        const localChanged = ["random_seed", "context_pad"].filter((k) =>
          old[k] !== undefined && !genDeepEqual(old[k], current[k]));
        if (localChanged.length) {
          const names = localChanged.map((k) => GEN_LABELS[k] || k).join("、");
          showCacheHint(partial
            ? names + "已变：Queue 不清空目录，只重跑选中的 " + selected.size +
              " 块，未勾选的块保留旧结果。"
            : names + "已变：Queue 不清空目录，重跑全部块并覆盖。", "warn");
        } else {
          showCacheHint("", "");
        }
      }
    }

    function showCacheReset(payload) {
      const rel = (payload && payload.rel) || outputDirRel();
      const keys = (payload && payload.keys) || [];
      const names = keys.map((k) => GEN_LABELS[k] || k).join("、");
      const tiles = payload && payload.tiles ? payload.tiles : 0;
      if (payload && payload.partial) {
        cacheResetNotice = "参数变化" + (names ? "（" + names + "）" : "") +
          "：已清空 " + rel + "，只重跑选中的 " + tiles + " 块";
      } else {
        cacheResetNotice = "已清空 " + rel + "，正在重跑全部 " + tiles + " 块" +
          (names ? "（参数变化：" + names + "）" : "");
      }
      cacheResetAt = Date.now();
      showCacheHint(cacheResetNotice, "warn");
      lastCacheKey = "";
    }

    function clearCacheResetNotice() {
      cacheResetNotice = "";
      cacheResetAt = 0;
      lastCacheKey = "";
    }

    function paintCells() {
      grid.querySelectorAll(".stlu-tcell").forEach((cell) => {
        cell.classList.toggle("on", selected.has(Number(cell.dataset.index)));
      });
    }

    function buildGrid() {
      grid.innerHTML = "";
      const custom = isPipeline && isCustomMode();
      if (!total) {
        grid.appendChild(el("div", "stlu-hint",
          custom ? "还没有窗口：双击画面或点「＋ 新窗口」" : "没有可选的块"));
        return;
      }
      const strips = isPipeline && isStripsMode();
      const unit = custom ? "个窗口" : (strips ? "条" : "块");
      coords.forEach((c, i) => {
        const [x0, y0] = c;
        // Each custom window keeps its own scale so the thumbnail is not
        // stretched by a neighbour of a different side.
        const cellW = custom ? Math.max(1, c[2] - c[0]) : Math.max(64, tileParam("tile_width", 1024));
        const cellH = custom ? Math.max(1, c[3] - c[1]) : (strips && planState && planState.plan && planState.plan.ok
          ? Math.max(64, Number(planState.plan.side) || 1024)
          : Math.max(64, tileParam("tile_height", 1024)));
        const scaleX = 72 / cellW;
        const scaleY = 72 / cellH;
        const cell = el("div", "stlu-tcell" + (selected.has(i) ? " on" : ""));
        cell.dataset.index = String(i);
        const res = custom && customRegions[i] ? Number(customRegions[i].res) || 0 : 0;
        cell.title = "第 " + i + " " + unit + " · x" + x0 + " y" + y0 +
          " " + (c[2] - c[0]) + "×" + (c[3] - c[1]) +
          (custom ? (" · 模型 " + (res ? res : "跟随全局")) : "") + " · 点击切换";
        if (imageUrl) {
          cell.style.backgroundImage = 'url("' + imageUrl + '")';
          cell.style.backgroundSize = (srcW * scaleX) + "px " + (srcH * scaleY) + "px";
          cell.style.backgroundPosition = (-x0 * scaleX) + "px " + (-y0 * scaleY) + "px";
        }
        cell.appendChild(el("span", null, String(i)));
        const hit = (event) => {
          event.preventDefault();
          event.stopPropagation();
          if (isPipeline && isCustomMode()) customActive = i;
          toggle(i);
          if (isPipeline && isCustomMode()) drawCustomOverlay();
        };
        cell.addEventListener("pointerdown", hit);
        grid.appendChild(cell);
      });
    }

    function drawCustomOverlay() {
      if (!overlay) return;
      const custom = isPipeline && isCustomMode();
      wrap.classList.toggle("custom", custom);
      overlay.style.display = custom ? "" : "none";
      if (!custom || !image || !image.naturalWidth) return;
      const iw = image.naturalWidth;
      const ih = image.naturalHeight;
      if (overlay.width !== iw || overlay.height !== ih) {
        overlay.width = iw;
        overlay.height = ih;
      }
      const ctx = overlay.getContext("2d");
      ctx.clearRect(0, 0, iw, ih);
      const lw = Math.max(2, Math.round(iw / 500));
      const fontPx = Math.max(14, Math.round(iw / 55));
      const handle = Math.max(5, Math.round(iw / 180));
      const scaleX = iw / Math.max(1, srcW || iw);
      const scaleY = ih / Math.max(1, srcH || ih);
      const list = customRegions.map((region, index) =>
        (customPreview && index === customActive) ? customPreview : region);
      list.forEach((region, index) => {
        const x = region.x * scaleX;
        const y = region.y * scaleY;
        const size = region.size * scaleX;
        const on = selected.has(index);
        const active = index === customActive;
        if (on) {
          ctx.fillStyle = "rgba(45,108,223,.24)";
          ctx.fillRect(x, y, size, size);
        }
        ctx.strokeStyle = active ? "#ffd34d" : (on ? "#5b93f5" : "rgba(255,255,255,.55)");
        ctx.lineWidth = active ? lw * 2 : lw;
        ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
        // Index + model size, drawn on a dark chip so it stays readable over
        // light and dark artwork alike.
        const label = String(index) + (region.res ? " · " + region.res : "");
        ctx.font = "600 " + fontPx + "px system-ui,-apple-system,'Segoe UI',sans-serif";
        const textW = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(0,0,0,.66)";
        ctx.fillRect(x + lw, y + lw, textW + fontPx * 0.5, fontPx * 1.4);
        ctx.fillStyle = "#f0f0f0";
        ctx.fillText(label, x + lw + fontPx * 0.25, y + lw + fontPx * 1.12);
        if (active) {
          ctx.fillStyle = "#ffd34d";
          const corners = [[x, y], [x + size, y], [x, y + size], [x + size, y + size]];
          for (const [hx, hy] of corners) {
            ctx.fillRect(hx - handle, hy - handle, handle * 2, handle * 2);
          }
        }
      });
    }

    function drawCanvas() {
      if (!image || !image.naturalWidth) return;
      const custom = isPipeline && isCustomMode();
      if (!total && !custom) return;
      const iw = image.naturalWidth;
      const ih = image.naturalHeight;
      if (canvas.width !== iw || canvas.height !== ih) {
        canvas.width = iw;
        canvas.height = ih;
      }
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, iw, ih);
      ctx.drawImage(image, 0, 0, iw, ih);
      if (custom) {
        drawCustomOverlay();
        return;
      }
      drawCustomOverlay();  // hides the overlay outside custom mode
      const lw = Math.max(2, Math.round(iw / 500));
      const strips = isPipeline && isStripsMode();
      if (strips && planState && planState.plan && planState.plan.ok) {
        const plan = planState.plan;
        const bbox = (plan.subject && plan.subject.bbox) ||
          (plan.subject_detection && plan.subject_detection.bbox);
        if (bbox && bbox.length === 4) {
          ctx.save();
          ctx.setLineDash([Math.max(6, lw * 3), Math.max(6, lw * 3)]);
          ctx.strokeStyle = "rgba(255,190,40,.95)";
          ctx.lineWidth = lw;
          ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
          ctx.restore();
        }
      }
      coords.forEach((c, i) => {
        const on = selected.has(i);
        if (on) {
          ctx.fillStyle = "rgba(45,108,223,.28)";
          ctx.fillRect(c[0], c[1], c[2] - c[0], c[3] - c[1]);
        }
        ctx.strokeStyle = on ? "#5b93f5" : "rgba(255,255,255,.55)";
        ctx.lineWidth = lw;
        ctx.strokeRect(c[0] + 0.5, c[1] + 0.5, c[2] - c[0] - 1, c[3] - c[1] - 1);
      });
    }

    function layout() {
      const tw = Math.max(64, tileParam("tile_width", 1024));
      const th = Math.max(64, tileParam("tile_height", 1024));
      const ov = Math.max(0, Number(widgetValue(node, "overlap", 128)) || 0);
      const ox = Math.max(0, Number(widgetValue(node, "grid_offset_x", 0)) || 0);
      const oy = Math.max(0, Number(widgetValue(node, "grid_offset_y", 0)) || 0);
      const strips = isPipeline && isStripsMode();
      const custom = isPipeline && isCustomMode();
      const uniform = (srcW > 0 && srcH > 0)
        ? gridCoordsJS(srcW, srcH, tw, th, ov, ox, oy)
        : [];
      let nextCoords = uniform;
      let nextKey = ["uniform", srcW, srcH, tw, th, ov, ox, oy].join("|");
      let plan = null;
      if (custom) {
        nextCoords = customCoords();
        nextKey = "custom|" + customPlanKey();
      } else if (strips) {
        if (planState && planState.plan && planState.plan.ok) {
          nextCoords = planState.coords || [];
          nextKey = "strips|" + planState.key + "|" + String(planState.plan.plan_id || "");
          plan = planState.plan;
        } else {
          nextKey = "strips_pending|" + srcW + "|" + srcH + "|" + ov + "|" +
            Number(widgetValue(node, "strip_max_side", 2048)) + "|" +
            Number(widgetValue(node, "resolution", 512)) + "|" +
            Number(widgetValue(node, "strip_margin", 32));
        }
      }
      const oldCoords = coords.slice();
      const oldSelected = new Set(selected);
      const oldTotal = total;
      const keyChanged = nextKey !== planAppliedKey;
      total = nextCoords.length;
      const involvesStrips = strips || String(planAppliedKey).startsWith("strips");
      const involvesCustom = custom || String(planAppliedKey).startsWith("custom");
      // A selection already written for THIS plan is in the right index
      // space: parse it as-is.  Remapping it by area is only for a genuinely
      // different geometry, and even then it must never widen the choice.
      const planIdNow = custom ? customPlanKey()
        : (strips && plan ? String(plan.plan_id || "") : "");
      const storedPlan = String(widgetValue(node, "select_plan", "") || "").trim();
      const indexesTrusted = !!planIdNow && storedPlan === planIdNow;
      if (keyChanged && (involvesStrips || involvesCustom) && oldCoords.length &&
          !indexesTrusted) {
        const baseSelection = oldTotal > 0
          ? oldSelected
          : parseSelection(currentText(), oldCoords.length);
        const mapped = mapSelectionByArea(oldCoords, baseSelection, nextCoords);
        // An empty mapping means the old boxes barely overlap the new plan;
        // fall back to the literal indices so the cells on screen stay
        // checked (still never wider than what the user wrote).
        selected = mapped.length ? new Set(mapped)
          : parseSelection(currentText(), total);
      } else {
        selected = parseSelection(currentText(), total);
      }
      coords = nextCoords;
      planAppliedKey = nextKey;
      buildGrid();
      drawCanvas();
      updateCount();
      // A smaller new picture may leave indices past the end: write the
      // surviving selection back so the widget matches the cells on screen.
      if (total) writeSelect();
      if (total && custom) {
        infoLine.className = "stlu-hint";
        infoLine.textContent = "原图 " + srcW + "×" + srcH + " · " + total +
          " 个自定义窗口（拖框移动 · 拖角缩放 · 双击空白新建 · Delete 删除）";
      } else if (total) {
        if (strips && plan) {
          infoLine.className = "stlu-hint";
          infoLine.textContent = "原图 " + srcW + "×" + srcH + " · " +
            total + " 条（边长 " + plan.side + " · 模型 " + plan.resolution + "）";
        } else if (strips) {
          const failed = planFailure();
          infoLine.className = "stlu-hint" + (failed ? " warn" : "");
          infoLine.textContent = "原图 " + srcW + "×" + srcH + " · " +
            (failed
              ? "方形分区未生效，当前按均匀网格显示（Queue 会回退均匀网格）"
              : "正在计算方形分区…");
        } else {
          const rows = new Set(coords.map((c) => c[1])).size;
          const cols = new Set(coords.map((c) => c[0])).size;
          infoLine.className = "stlu-hint";
          infoLine.textContent = "原图 " + srcW + "×" + srcH + " · " +
            total + " 块（" + rows + "×" + cols + "）";
        }
      } else if (custom) {
        infoLine.className = "stlu-hint warn";
        infoLine.textContent = srcW && srcH
          ? "自定义分区：还没有窗口。双击画面或点「＋ 新窗口」创建。"
          : "接上 LoadImage 后才能摆放窗口。";
      }
    }

    async function loadSource() {
      const upstream = findUpstreamImageNode(node);
      const path = imagePathOf(upstream);
      if (String(path || "") !== sourcePath) planState = null;
      sourcePath = path || "";
      sourceEpoch += 1;
      image = null;
      imageUrl = "";
      srcW = 0;
      srcH = 0;
      wrap.style.display = "none";
      if (!path) {
        infoLine.className = "stlu-hint";
        infoLine.textContent = "接上 LoadImage 就会显示切格";
        if (isStripsMode()) schedulePlan(true);
        layout();
        return;
      }
      infoLine.className = "stlu-hint";
      infoLine.textContent = "正在读取 " + path + "…";
      const size = await probeImageSize(path, ["input", "output"]);
      if (size) {
        srcW = size.width;
        srcH = size.height;
      }
      if (!srcW || !srcH) {
        infoLine.className = "stlu-hint warn";
        infoLine.textContent = "读不到原图尺寸（" + path + "），检查 image 输入是否接好";
        layout();
        return;
      }
      imageUrl = imageURLOf(upstream) || "";
      clampCustomToCanvas();
      layout();
      if (isStripsMode()) schedulePlan(true);
      if (!imageUrl) return;
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        image = img;
        if (!srcW || !srcH) {
          srcW = img.naturalWidth;
          srcH = img.naturalHeight;
        }
        clampCustomToCanvas();
        wrap.style.display = "";
        layout();
        if (isStripsMode()) schedulePlan(true);
      };
      img.src = imageUrl + (imageUrl.includes("?") ? "&" : "?") + "t=" + Date.now();
    }

    canvas.addEventListener("pointerdown", (event) => {
      if (isPipeline && isCustomMode()) return;  // the overlay owns this mode
      if (!image || !total) return;
      event.preventDefault();
      event.stopPropagation();
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const x = ((event.clientX - rect.left) / rect.width) * srcW;
      const y = ((event.clientY - rect.top) / rect.height) * srcH;
      if (alignMode && !(isPipeline && isStripsMode())) {
        const tw = Math.max(64, tileParam("tile_width", 1024));
        const th = Math.max(64, tileParam("tile_height", 1024));
        const ov = Math.max(0, Number(widgetValue(node, "overlap", 128)) || 0);
        const sx = Math.round(Math.max(tw * 0.25, tw - ov));
        const sy = Math.round(Math.max(th * 0.25, th - ov));
        const wrapValue = (v, m) => ((Math.round(v) % m) + m) % m;
        setWidget(node, "grid_offset_x", wrapValue(x - tw / 2, sx));
        setWidget(node, "grid_offset_y", wrapValue(y - th / 2, sy));
        setAlignMode(false);
        if (pipelineControls) pipelineControls.sync();
        layout();
        return;
      }
      let hit = -1;
      coords.forEach((c, i) => {
        if (x >= c[0] && x < c[2] && y >= c[1] && y < c[3]) hit = i;
      });
      toggle(hit);
    });

    // --- custom window editor ---------------------------------------------
    function customPoint(event) {
      const rect = overlay.getBoundingClientRect();
      if (!rect.width || !rect.height || !srcW || !srcH) return null;
      return {
        x: ((event.clientX - rect.left) / rect.width) * srcW,
        y: ((event.clientY - rect.top) / rect.height) * srcH,
      };
    }

    function customHit(x, y) {
      for (let index = customRegions.length - 1; index >= 0; index--) {
        const region = customRegions[index];
        if (x >= region.x && x < region.x + region.size &&
            y >= region.y && y < region.y + region.size) return index;
      }
      return -1;
    }

    function customCornerAt(x, y, index, tolerance) {
      const region = customRegions[index];
      if (!region) return -1;
      const corners = [
        [region.x, region.y],
        [region.x + region.size, region.y],
        [region.x, region.y + region.size],
        [region.x + region.size, region.y + region.size],
      ];
      for (let corner = 0; corner < 4; corner++) {
        if (Math.abs(x - corners[corner][0]) <= tolerance &&
            Math.abs(y - corners[corner][1]) <= tolerance) return corner;
      }
      return -1;
    }

    overlay.addEventListener("pointerdown", (event) => {
      if (!isPipeline || !isCustomMode() || !image) return;
      event.preventDefault();
      event.stopPropagation();
      overlay.focus();
      const pt = customPoint(event);
      if (!pt) return;
      const rect = overlay.getBoundingClientRect();
      const tolerance = (srcW / Math.max(1, rect.width)) * 10;
      let index = customActive;
      let corner = index >= 0 ? customCornerAt(pt.x, pt.y, index, tolerance) : -1;
      if (corner < 0) {
        index = customHit(pt.x, pt.y);
        corner = index >= 0 ? customCornerAt(pt.x, pt.y, index, tolerance) : -1;
      }
      if (index < 0) {
        customActive = -1;
        drawCustomOverlay();
        return;
      }
      customActive = index;
      customPreview = null;
      const region = { ...customRegions[index] };
      if (corner >= 0) {
        // The diagonally opposite corner stays put; the square grows around it.
        const anchorX = (corner === 0 || corner === 2)
          ? region.x + region.size : region.x;
        const anchorY = (corner === 0 || corner === 1)
          ? region.y + region.size : region.y;
        customDrag = { kind: "resize", index, anchorX, anchorY, start: pt, region, moved: false };
      } else {
        customDrag = { kind: "move", index, start: pt, region, moved: false };
      }
      try { overlay.setPointerCapture(event.pointerId); } catch (e) { /* ignore */ }
      drawCustomOverlay();
    });

    overlay.addEventListener("pointermove", (event) => {
      if (!customDrag) return;
      event.preventDefault();
      const pt = customPoint(event);
      if (!pt) return;
      const drag = customDrag;
      const travel = Math.abs(pt.x - drag.start.x) + Math.abs(pt.y - drag.start.y);
      if (!drag.moved && travel < 3) return;
      drag.moved = true;
      const region = drag.region;
      if (drag.kind === "move") {
        const x = Math.max(0, Math.min(Math.max(0, srcW - region.size),
          region.x + (pt.x - drag.start.x)));
        const y = Math.max(0, Math.min(Math.max(0, srcH - region.size),
          region.y + (pt.y - drag.start.y)));
        customPreview = { ...region, x: Math.round(x), y: Math.round(y) };
      } else {
        const dx = pt.x - drag.anchorX;
        const dy = pt.y - drag.anchorY;
        const limit = Math.max(CUSTOM_MIN, Math.min(srcW || CUSTOM_MIN, srcH || CUSTOM_MIN));
        let size = Math.round(Math.max(Math.abs(dx), Math.abs(dy)) / CUSTOM_STEP) * CUSTOM_STEP;
        size = Math.max(CUSTOM_MIN, Math.min(limit, size));
        let x = dx < 0 ? drag.anchorX - size : drag.anchorX;
        let y = dy < 0 ? drag.anchorY - size : drag.anchorY;
        x = Math.max(0, Math.min(Math.max(0, srcW - size), x));
        y = Math.max(0, Math.min(Math.max(0, srcH - size), y));
        customPreview = { ...region, x: Math.round(x), y: Math.round(y), size };
      }
      if (!customRaf) {
        customRaf = requestAnimationFrame(() => { customRaf = 0; drawCustomOverlay(); });
      }
    });

    function finishCustomDrag(event, commit) {
      if (!customDrag) return;
      const drag = customDrag;
      const preview = customPreview;
      customDrag = null;
      customPreview = null;
      if (customRaf) { cancelAnimationFrame(customRaf); customRaf = 0; }
      try { overlay.releasePointerCapture(event.pointerId); } catch (e) { /* ignore */ }
      if (commit && drag.moved && preview) {
        lastCustomSize = preview.size;
        const next = customRegions.map((region, index) =>
          index === drag.index ? preview : region);
        applyCustomRegions(next);
        return;
      }
      if (commit && !drag.moved) {
        // A plain click toggles the run selection of that window.
        if (selected.has(drag.index)) selected.delete(drag.index);
        else selected.add(drag.index);
        if (renderCustomList) renderCustomList();
        drawCanvas();
        updateCount();
        writeSelect();
        refreshCacheHint();
        return;
      }
      drawCustomOverlay();
    }

    overlay.addEventListener("pointerup", (event) => finishCustomDrag(event, true));
    overlay.addEventListener("pointercancel", (event) => finishCustomDrag(event, false));

    overlay.addEventListener("dblclick", (event) => {
      if (!isPipeline || !isCustomMode() || !image) return;
      event.preventDefault();
      event.stopPropagation();
      const pt = customPoint(event);
      if (!pt || customHit(pt.x, pt.y) >= 0) return;
      addCustomWindow(pt.x, pt.y);
    });

    overlay.addEventListener("keydown", (event) => {
      if (!isPipeline || !isCustomMode()) return;
      if (event.key === "Delete" || event.key === "Backspace") {
        if (customActive >= 0 && customActive < customRegions.length) {
          event.preventDefault();
          event.stopPropagation();
          deleteCustomWindow(customActive);
        }
      }
    });

    allBtn.addEventListener("click", () => {
      selected = new Set(Array.from({ length: total }, (_, i) => i));
      paintCells();
      drawCanvas();
      updateCount();
      writeSelect();
    });
    noneBtn.addEventListener("click", () => {
      selected = new Set();
      paintCells();
      drawCanvas();
      updateCount();
      writeSelect();
    });
    invertBtn.addEventListener("click", () => {
      const next = new Set();
      for (let i = 0; i < total; i++) if (!selected.has(i)) next.add(i);
      selected = next;
      paintCells();
      drawCanvas();
      updateCount();
      writeSelect();
    });
    reloadBtn.addEventListener("click", () => { loadSource(); });

    // The tile sizes may be fed by the primitive nodes in the "用户输入"
    // group; links only exist after the workflow finishes loading, so rebind
    // on configure and from the poll below.
    function bindParamWatchers() {
      for (const name of ["tile_width", "tile_height", "overlap",
                          "grid_offset_x", "grid_offset_y"]) {
        watchValue(node, name, layout);
      }
      for (const name of ["partition_mode", "strip_max_side",
                          "resolution", "strip_margin"]) {
        watchValue(node, name, () => {
          if (pipelineControls) pipelineControls.applyMode();
          schedulePlan(true);
          layout();
          refreshCacheHint();
        });
      }
      // onConfigure / onExecuted may rewrite the window list; adopt it without
      // re-mapping (the accompanying select_plan says which geometry it is).
      watchValue(node, "custom_regions", () => {
        const loaded = readCustomRegionsFromWidget();
        if (JSON.stringify(loaded) === JSON.stringify(customRegions)) return;
        customRegions = loaded;
        total = customRegions.length;
        if (renderCustomList) renderCustomList();
        layout();
      });
    }
    bindParamWatchers();
    if (selectWidget) {
      const oldCb = selectWidget.callback;
      selectWidget.callback = function () {
        const r = oldCb ? oldCb.apply(this, arguments) : undefined;
        syncFromWidget();
        return r;
      };
    }

    // Every setting the fixed pipeline consumes, grouped in one panel.  The
    // node's own widgets are hidden: this panel is the only place to edit.
    function buildPipelineControls() {
      const fields = [];

      // 面板上只留控件；解释性文字统一收进这个折叠区，节点上不再堆注释。
      const detailsCard = el("details", "stlu-card stlu-hints");
      detailsCard.style.gap = "4px";
      detailsCard.appendChild(el("summary", "stlu-cardtitle", "细节说明（点开）"));
      function hint(text) {
        const box = el("div", "stlu-hint", text || "");
        detailsCard.appendChild(box);
        return box;
      }

      function makeInput(spec) {
        const input = document.createElement("input");
        input.className = "stlu-input";
        input.type = spec.kind === "text" ? "text" : "number";
        if (input.type === "number") input.step = "1";
        input.value = String(widgetValue(node, spec.name, spec.fallback));
        input.title = spec.label || spec.name;
        if (spec.placeholder) input.placeholder = spec.placeholder;
        input.addEventListener("change", () => {
          let value = input.value;
          if (input.type === "number") {
            const num = Number(value);
            if (!Number.isFinite(num)) {
              input.value = String(widgetValue(node, spec.name, spec.fallback));
              return;
            }
            value = Math.round(num);
          }
          setWidget(node, spec.name, value);
          if (spec.onChange) spec.onChange();
        });
        fields.push({ input, spec });
        return input;
      }

      function line(parent, specs, tail) {
        const rowEl = el("div", "stlu-line");
        for (const spec of specs) {
          rowEl.appendChild(el("span", "stlu-label", spec.label));
          rowEl.appendChild(makeInput(spec));
        }
        if (tail) rowEl.appendChild(tail);
        parent.appendChild(rowEl);
        return rowEl;
      }

      function card(title) {
        const box = el("div", "stlu-card");
        box.style.gap = "4px";
        box.appendChild(el("div", "stlu-cardtitle", title));
        root.appendChild(box);
        return box;
      }

      function check(parent, labelText, name, fallback, title) {
        const label = el("label", "stlu-check");
        if (title) label.title = title;
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = !!widgetValue(node, name, fallback);
        cb.addEventListener("change", () => setWidget(node, name, cb.checked));
        fields.push({ input: cb, spec: { name, fallback, kind: "bool" } });
        label.appendChild(cb);
        label.appendChild(el("span", null, labelText));
        parent.appendChild(label);
        return cb;
      }

      const modeCard = card("分区方式");
      const modeRow = el("div", "stlu-line");
      modeRow.appendChild(el("span", "stlu-label", "方式"));
      const modeSelect = document.createElement("select");
      modeSelect.className = "stlu-input";
      for (const label of ["均匀网格", STRIPS_MODE, CUSTOM_MODE]) {
        const option = document.createElement("option");
        option.value = label;
        option.textContent = label;
        modeSelect.appendChild(option);
      }
      modeSelect.value = String(widgetValue(node, "partition_mode", "均匀网格"));
      modeSelect.title = "方形全宽分区：自动检测主体，按主体宽度做方形条；每条一次推理看到完整左右，"
        + "不再有竖直缝穿过人物。自定义分区：自己摆放正方形窗口，每个窗口完整包住一个部位。";
      modeSelect.addEventListener("change", () => {
        setWidget(node, "partition_mode", modeSelect.value);
        applyMode();
        schedulePlan(true);
        layout();
        refreshCacheHint();
      });
      modeRow.appendChild(modeSelect);
      modeCard.appendChild(modeRow);
      fields.push({ input: modeSelect, spec: {
        name: "partition_mode", fallback: "均匀网格", kind: "select",
      } });
      const modeHint = hint(
        "「方形全宽分区」会忽略块宽/块高/偏移和点图对齐；区域最大边是目标值，"
        + "主体更宽时会自动放大到覆盖全宽，不会竖直切开。"
        + "「自定义分区」由你在画面上摆放正方形窗口。切换模式会清空输出目录。");

      const cut = card("切格");
      const tileRow = line(cut, [
        { label: "块宽", name: "tile_width", fallback: 1024, onChange: layout },
        { label: "块高", name: "tile_height", fallback: 1024, onChange: layout },
      ]);
      line(cut, [{ label: "重叠", name: "overlap", fallback: 128, onChange: () => {
        schedulePlan(true);
        layout();
      } }]);
      const offsetRow = line(cut, [
        { label: "偏移X", name: "grid_offset_x", fallback: 0, onChange: layout },
        { label: "偏移Y", name: "grid_offset_y", fallback: 0, onChange: layout },
      ]);
      const alignRow = el("div", "stlu-line");
      alignBtn = el("button", "stlu-btn", "点图对齐主体");
      alignBtn.addEventListener("click", () => setAlignMode(!alignMode));
      setAlignMode(false);
      alignRow.appendChild(alignBtn);
      alignBtn.title = "接缝切到脸时：点按钮，再点画面上的脸/头部中心。";
      hint("点图对齐主体：接缝切到脸时，先点「点图对齐主体」，再点画面上的脸/头部中心。");
      cut.appendChild(alignRow);
      const stripRow = line(cut, [
        { label: "区域最大边", name: "strip_max_side", fallback: 2048,
          onChange: () => { schedulePlan(true); layout(); refreshCacheHint(); } },
        { label: "主体边距", name: "strip_margin", fallback: 32,
          onChange: () => { schedulePlan(true); layout(); refreshCacheHint(); } },
      ]);
      if (widgetOf(node, "context_pad")) {
        line(cut, [
          { label: "块外上下文", name: "context_pad", fallback: 0,
            onChange: () => { layout(); refreshCacheHint(); } },
        ]);
        hint(
          "块外上下文 0 = 关闭；大于 0 时每块向四周多裁这么多像素送进模型，生成后裁回原条带。"
          + "让只有腿脚的块也能看到上半身，减少按错误部位补全；不改变分块位置。"
          + "改它会重新生成相关块。");
      }
      const stripHint = hint(
        "方形分区每条都是方形窗口，自动以主体宽度/位置为中心；头部在第一条内完整。"
        + "勾选按原图覆盖区域从旧网格迁移，不会自动扩大；如果迁移后为空，"
        + "面板会标红提示，点「全选」再 Queue。");
      const cutHint = hint(
        "让某一块完整包住人物后，合并时这个部位会整块采用那一块，不再被接缝切成两半。"
        + "也可以把重叠调到块宽的一半（块数会增加），让每块看到更多邻居。");

      // --- custom square windows ------------------------------------------
      const customCard = card("自定义正方形窗口（每个包住一个部位）");
      const customBar = el("div", "stlu-line");
      customBar.style.flexWrap = "wrap";
      const addWindowBtn = el("button", "stlu-btn", "＋ 新窗口");
      addWindowBtn.title = "在画面中心新建一个正方形窗口；也可以直接在画面上双击空白处";
      const importToggleBtn = el("button", "stlu-btn", "导入现有分区");
      customCountTag = el("span", "stlu-tag", "0 个");
      customBar.appendChild(addWindowBtn);
      customBar.appendChild(importToggleBtn);
      customBar.appendChild(customCountTag);
      customCard.appendChild(customBar);

      // A two-item chooser in the panel instead of a prompt(): the source of
      // the squares must be visible before it replaces the current layout.
      const importRow = el("div", "stlu-line");
      importRow.style.display = "none";
      const importSrc = document.createElement("select");
      importSrc.className = "stlu-input";
      for (const [value, text] of [["current", "当前分区"], ["last", "上次执行的分区"]]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = text;
        importSrc.appendChild(option);
      }
      const importApply = el("button", "stlu-btn", "导入");
      const importCancel = el("button", "stlu-mini", "取消");
      importRow.appendChild(el("span", "stlu-label", "来源"));
      importRow.appendChild(importSrc);
      importRow.appendChild(importApply);
      importRow.appendChild(importCancel);
      customCard.appendChild(importRow);

      const importNote = el("div", "stlu-hint", "");
      importNote.style.whiteSpace = "normal";
      importNote.style.display = "none";
      customCard.appendChild(importNote);

      const customList = el("div", "stlu-customlist");
      customCard.appendChild(customList);
      const customHint = hint(
        "自定义窗口顺序 = 生成顺序；后面的窗口会看到前面窗口在重叠区的结果。"
        + "改位置/边长/分辨率/顺序会清空该目录，只重跑勾选的窗口。"
        + "分辨率「跟随全局」使用「拆层质量」里的「分辨率」，每个窗口可单独覆盖。");

      function updateCustomField(index, patch) {
        const next = customRegions.map((region, row) =>
          row === index ? { ...region, ...patch } : region);
        applyCustomRegions(next);
      }

      renderCustomList = function () {
        customList.innerHTML = "";
        if (customCountTag) {
          customCountTag.textContent = customRegions.length + " 个窗口";
          customCountTag.className = "stlu-tag" + (customRegions.length ? "" : " error");
        }
        if (!customRegions.length) {
          customList.appendChild(el("div", "stlu-hint",
            "还没有窗口。点「＋ 新窗口」或在画面上双击空白处创建。"));
          return;
        }
        customRegions.forEach((region, index) => {
          const row = el("div", "stlu-crow" + (index === customActive ? " active" : ""));
          const check = document.createElement("input");
          check.type = "checkbox";
          check.checked = selected.has(index);
          check.title = "勾选：本次生成这个窗口";
          check.addEventListener("change", () => {
            if (check.checked) selected.add(index);
            else selected.delete(index);
            drawCustomOverlay();
            updateCount();
            writeSelect();
            refreshCacheHint();
          });
          row.appendChild(check);
          row.appendChild(el("span", "idx", String(index)));
          const nameInput = document.createElement("input");
          nameInput.type = "text";
          nameInput.className = "cname";
          nameInput.placeholder = "名称";
          nameInput.value = region.name || "";
          nameInput.title = "只用于识别，不参与缓存指纹";
          nameInput.addEventListener("change", () => {
            updateCustomField(index, { name: nameInput.value });
          });
          row.appendChild(nameInput);
          for (const [label, key] of [["X", "x"], ["Y", "y"], ["边", "size"]]) {
            row.appendChild(el("span", "stlu-label", label));
            const input = document.createElement("input");
            input.type = "number";
            input.step = String(CUSTOM_STEP);
            input.value = String(region[key]);
            input.title = key === "size"
              ? "正方形边长（源图像素，64 的倍数）"
              : "左上角坐标（源图像素）";
            input.addEventListener("change", () => {
              updateCustomField(index, { [key]: Number(input.value) });
            });
            row.appendChild(input);
          }
          row.appendChild(el("span", "stlu-label", "模型"));
          const resSelect = document.createElement("select");
          resSelect.title = "这个窗口的模型分辨率；跟随全局 = 上面的「分辨率」";
          for (const value of [0, 512, 768, 1024, 1280, 1536, 1792, 2048]) {
            const option = document.createElement("option");
            option.value = String(value);
            option.textContent = value ? String(value) : "跟随全局";
            resSelect.appendChild(option);
          }
          resSelect.value = String(region.res || 0);
          resSelect.addEventListener("change", () => {
            updateCustomField(index, { res: Number(resSelect.value) });
          });
          row.appendChild(resSelect);
          row.appendChild(el("span", "grow"));
          const upBtn = el("button", "move", "↑");
          const downBtn = el("button", "move", "↓");
          const delBtn = el("button", "del", "✕");
          upBtn.title = "上移：提前生成（引导顺序随之改变）";
          downBtn.title = "下移：延后生成";
          delBtn.title = "删除这个窗口";
          upBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            moveCustomWindow(index, -1);
          });
          downBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            moveCustomWindow(index, 1);
          });
          delBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            deleteCustomWindow(index);
          });
          row.appendChild(upBtn);
          row.appendChild(downBtn);
          row.appendChild(delBtn);
          row.addEventListener("pointerdown", (event) => {
            if (event.target.closest("input,select,button")) return;
            customActive = index;
            customList.querySelectorAll(".stlu-crow").forEach((cell, rowIndex) => {
              cell.classList.toggle("active", rowIndex === index);
            });
            drawCustomOverlay();
          });
          customList.appendChild(row);
        });
      };

      addWindowBtn.addEventListener("click", () => addCustomWindow(null, null));
      importToggleBtn.addEventListener("click", () => {
        const show = importRow.style.display === "none";
        importRow.style.display = show ? "" : "none";
        importNote.style.display = "none";
        const lastOption = Array.from(importSrc.options)
          .find((option) => option.value === "last");
        if (lastOption) {
          lastOption.disabled = !(lastExecutedWindows && lastExecutedWindows.length);
        }
        importSrc.value = "current";
      });
      importCancel.addEventListener("click", () => {
        importRow.style.display = "none";
        importNote.style.display = "none";
      });
      importApply.addEventListener("click", () => {
        let boxes = [];
        let label = "当前分区";
        if (importSrc.value === "last") {
          boxes = (lastExecutedWindows || []).map((item) =>
            item.box || [item.x, item.y, item.x + item.size, item.y + item.size]);
          label = "上次执行的分区";
        } else if (isCustomMode()) {
          boxes = customCoords();
          label = "当前自定义窗口";
        } else if (isStripsMode() && planState && planState.plan && planState.plan.ok) {
          boxes = (planState.coords || []).slice();
          label = "当前方形分区";
        } else {
          if (!srcW || !srcH) {
            importNote.textContent = "接上 LoadImage 后才能导入。";
            importNote.style.display = "";
            return;
          }
          boxes = gridCoordsJS(
            srcW, srcH,
            Math.max(64, tileParam("tile_width", 1024)),
            Math.max(64, tileParam("tile_height", 1024)),
            Math.max(0, Number(widgetValue(node, "overlap", 128)) || 0),
            Math.max(0, Number(widgetValue(node, "grid_offset_x", 0)) || 0),
            Math.max(0, Number(widgetValue(node, "grid_offset_y", 0)) || 0));
          label = "当前均匀网格";
        }
        const windows = uniformWindowsJS(boxes, srcW, srcH);
        if (!windows.length) {
          importNote.textContent = "没有可导入的分区（" + label + "）。";
          importNote.style.display = "";
          return;
        }
        const skipped = Math.max(0, boxes.length - windows.length);
        customActive = -1;
        applyCustomRegions(windows, { remap: true });
        importNote.textContent = "已从" + label + "导入 " + windows.length +
          " 个正方形窗口" +
          (skipped ? "；" + skipped + " 个非正方形/过小的块已取内接正方形或跳过" : "") + "。";
        importNote.style.display = "";
        importRow.style.display = "none";
      });

      const quality = card("拆层质量");
      line(quality, [
        { label: "分辨率", name: "resolution", fallback: 512 },
        { label: "步数", name: "num_inference_steps", fallback: 30 },
      ]);
      line(quality, [{ label: "深度分辨率（-1=跟随）", name: "depth_resolution", fallback: -1 }]);
      const seedRow = el("div", "stlu-line");
      seedRow.appendChild(el("span", "stlu-label", "种子"));
      const seedInput = makeInput({ label: "种子", name: "random_seed", fallback: 0 });
      seedRow.appendChild(seedInput);
      const seedBtn = el("button", "stlu-btn", "随机");
      seedBtn.addEventListener("click", () => {
        const value = Math.floor(Math.random() * 0xffffffff);
        setWidget(node, "random_seed", value);
        seedInput.value = String(value);
      });
      seedRow.appendChild(seedBtn);
      quality.appendChild(seedRow);
      line(quality, [{ label: "透明阈值", name: "alpha_threshold", fallback: 8 }]);
      const qualityHint = hint(
        "「分辨率」是模型推理分辨率，三种分区都用它：均匀网格直接按块裁到该分辨率，"
        + "方形分区按它推理（分区摘要里显示实际值），自定义窗口「跟随全局」也用它。"
        + "透明阈值同时决定每块 PSD 和合并时的透明判定。"
        + "换种子不会清空目录：只重跑勾选的块，未勾选的块保留旧种子结果。");

      const post = card("后处理");
      const postRow = el("div", "stlu-line");
      postRow.style.flexWrap = "wrap";
      check(postRow, "左右拆成两半", "tblr_split", true);
      check(postRow, "用 LaMa 去背景", "use_lama", true);
      post.appendChild(postRow);

      // Only the full node exposes the sequential-guide knobs; the simple node
      // runs the same four-step pipeline without STLU_GuidedLayers, so showing
      // them there would be dead controls.
      if (widgetOf(node, "guide_context")) {
        const guide = card("分块引导（只影响生成）");
        const guideRow = el("div", "stlu-line");
        guideRow.style.flexWrap = "wrap";
        check(guideRow, "用已生成相邻块的重叠区做引导", "guide_context", true,
          "顺序逐块生成时，把先完成的相邻块在重叠区的像素回填到当前块再拆层；"
          + "只有「引导版」工作流会读取，普通四步流水线忽略。");
        guide.appendChild(guideRow);
        line(guide, [
          { label: "羽化", name: "guide_feather", fallback: 32 },
          { label: "强度%", name: "guide_strength", fallback: 40 },
        ]);
        line(guide, [{ label: "与原图容差", name: "guide_tolerance", fallback: 16 }]);
        hint("分块引导：只回填生成结果与原图接近的像素；容差越小越保守，0 = 不过滤（会把补全内容也带过去）。"
          + "强度 100 会用上一块的补全结果覆盖原图，容易带出不明物体/丢眼睛，建议 30-50。");
      }

      const output = card("输出位置");
      const outRow = el("div", "stlu-line");
      outRow.appendChild(el("span", "stlu-label", "根目录"));
      const outInput = makeInput({
        label: "输出根目录", name: "out_dir", fallback: "SeeThrough/我的自设/",
        kind: "text", onChange: updateOut,
      });
      outRow.appendChild(outInput);
      const browseBtn = el("button", "stlu-btn", "浏览…");
      browseBtn.addEventListener("click", () => {
        openPicker({
          title: "选择输出根目录",
          mode: "folder",
          scopes: [{ scope: "output", label: "输出目录" }],
          start: String(widgetValue(node, "out_dir", "") || ""),
          onPick: (rel) => {
            setWidget(node, "out_dir", rel || "SeeThrough");
            if (outInput) outInput.value = String(widgetValue(node, "out_dir", "SeeThrough"));
            updateOut();
          },
        });
      });
      outRow.appendChild(browseBtn);
      output.appendChild(outRow);
      line(output, [
        { label: "分类（可空）", name: "group", fallback: "", kind: "text", onChange: updateOut },
        { label: "分块前缀", name: "filename_prefix", fallback: "tile", kind: "text" },
      ]);
      const outLine = el("div", "stlu-hint");
      outLine.style.wordBreak = "break-all";
      output.appendChild(outLine);

      const clearRow = el("div", "stlu-line");
      const clearBtn = el("button", "stlu-btn", "清空此目录");
      clearBtn.title = "删除该目录里本流水线生成的切块 PSD、预览、depth、manifest 和 images/；"
        + "你自己放进去的其他文件会保留。";
      clearBtn.addEventListener("click", async () => {
        const dir = outputDirRel();
        if (!dir) return;
        if (!window.confirm(tr("清空 " + dir + " 里已生成的切块结果？此操作不可撤销。"))) return;
        clearBtn.disabled = true;
        try {
          const data = await postJson("/stlu/tiles/clear", { dir });
          clearCacheResetNotice();
          lastCacheKey = "";
          showCacheHint("已清空 " + (data.dir || dir) + "（" +
            (data.removed || 0) + " 个文件，锁定快照保留）。"
            + "Queue 会重新生成勾选的块，锁定的图会找回。", "");
          refreshCacheHint();
        } catch (e) {
          showCacheHint("清空失败：" + (e && e.message ? e.message : e), "error");
        } finally {
          clearBtn.disabled = false;
        }
      });
      clearRow.appendChild(clearBtn);
      output.appendChild(clearRow);

      function updateOut() {
        outLine.textContent = "→ " + wavePreview(node) +
          "（直接写入所选文件夹；已有的块会复用，只重跑勾选的块）";
      }

      const mergeBox = card("合并");
      line(mergeBox, [{ label: "合并文件名", name: "merge_name", fallback: "merged", kind: "text" }]);
      const mergeModeRow = el("div", "stlu-line");
      mergeModeRow.style.flexWrap = "wrap";
      const groupCb = check(mergeModeRow, "同名图层单独成组（不裁剪）", "group_layers", true,
        "勾选：不裁剪版——各块同名部件保留完整原件，收进同名文件夹，"
        + "接缝裁切、幻影清理、完整部件淘汰都不做，便于在 PS 里修；"
        + "不勾：直出版——按接缝归属裁切后合成一层。只影响合并，不会重跑生成。");
      mergeBox.appendChild(mergeModeRow);
      const seamRow = line(mergeBox, [
        { label: "接缝过渡（仅直出）", name: "seam_feather", fallback: 32 },
      ]);
      const seamInput = seamRow.querySelector("input");
      const extraRow = el("div", "stlu-line");
      extraRow.style.flexWrap = "wrap";
      check(extraRow, "depth 另存 PSD", "include_depth", false);
      check(extraRow, "原图作底图", "include_original", false);
      mergeBox.appendChild(extraRow);
      const mergeHint = hint("");
      function syncMergeMode() {
        const grouped = !!widgetValue(node, "group_layers", true);
        if (seamInput) {
          seamInput.disabled = grouped;
          seamInput.title = grouped
            ? "不裁剪模式不使用接缝过渡"
            : "直出合并时落败块在接缝外侧的淡出宽度（像素）；0 = 全部硬切";
        }
      }
      function paintMergeHint() {
        const grouped = !!widgetValue(node, "group_layers", true);
        mergeHint.textContent = grouped
          ? "不裁剪版：同名副本收进同名组，组内是各块自己的完整副本（层名带块号），"
            + "只有一个副本的部位保持普通图层；重叠处会重复显示——"
            + "修完在 PS 里隐藏/删除不要的副本即可。"
          : "直出版：每个像素只归离自己块边更远的一块，另一块在接缝处被裁掉；"
            + "落败块在接缝外侧按「接缝过渡」淡出。某一块完整看到某部位时整块采用它。";
        syncMergeMode();
      }
      groupCb.addEventListener("change", paintMergeHint);
      paintMergeHint();

      // 说明文字统一收在最下面的折叠区。
      root.appendChild(detailsCard);

      for (const w of node.widgets || []) hideWidget(w);

      function applyMode() {
        const strips = isStripsMode();
        const custom = isCustomMode();
        modeSelect.value = String(widgetValue(node, "partition_mode", "均匀网格"));
        tileRow.style.display = (strips || custom) ? "none" : "";
        offsetRow.style.display = (strips || custom) ? "none" : "";
        alignRow.style.display = (strips || custom) ? "none" : "";
        stripRow.style.display = strips ? "" : "none";
        stripHint.style.display = strips ? "" : "none";
        customCard.style.display = custom ? "" : "none";
        qualityHint.textContent = custom
          ? "自定义窗口的「跟随全局」分辨率使用「拆层质量」里的「分辨率」；每个窗口可在列表里单独覆盖。"
            + "透明阈值同时决定每块 PSD 和合并时的透明判定。"
            + "换种子不会清空目录：只重跑勾选的窗口，未勾选的保留旧种子结果。"
          : ("「分辨率」是模型推理分辨率，三种分区都用它；方形分区的实际值见分区摘要。"
            + "透明阈值同时决定每块 PSD 和合并时的透明判定。"
            + "换种子不会清空目录：只重跑勾选的块，未勾选的块保留旧种子结果。");
        cutHint.textContent = strips
          ? "方形全宽分区：每条一次推理看到完整左右，水平缝落在条之间；"
            + "合并仍按完整部件规则优先采用整条看到该部位的副本。"
          : (custom
            ? "自定义分区：每个正方形窗口一次推理，只看到框内完整部位；"
              + "重叠区按运行顺序把前面窗口的结果回填给后面的窗口。"
            : ("让某一块完整包住人物后，合并时这个部位会整块采用那一块，不再被接缝切成两半。"
              + "也可以把重叠调到块宽的一半（块数会增加），让每块看到更多邻居。"));
        if (strips || custom) setAlignMode(false);
        if (typeof paintMergeHint === "function") paintMergeHint();
        drawCustomOverlay();
      }

      function sync() {
        for (const { input, spec } of fields) {
          if (document.activeElement === input) continue;
          const value = widgetValue(node, spec.name, spec.fallback);
          if (spec.kind === "bool") input.checked = !!value;
          else input.value = String(value == null ? "" : value);
        }
        // Another panel / onConfigure may have rewritten the widget.
        const loaded = readCustomRegionsFromWidget();
        if (JSON.stringify(loaded) !== JSON.stringify(customRegions)) {
          customRegions = loaded;
          total = customRegions.length;
          if (renderCustomList) renderCustomList();
        }
        applyMode();
        updateOut();
      }
      sync();
      return { sync, applyMode, modeCard };
    }

    const dom = addDOM(node, "stlu_tile_grid_card", root,
      isPipeline ? 760 : 300, isPipeline ? 1800 : 620);
    if (isPipeline) root.style.overflowY = "auto";
    registerPanel(node, "grid", dom, root);
    const gridState = {
      node, refresh: loadSource, showCacheReset,
      clearCacheReset: clearCacheResetNotice,
    };
    gridStates.add(gridState);

    // Picking another file in LoadImage (or finishing an upload) reloads the
    // preview immediately; a slow poll also catches link rewiring and edits
    // made outside this panel.
    let watchedNode = null;
    let watchedPath = "";
    let unwatch = null;

    function bindWatch(reload) {
      const upstream = findUpstreamImageNode(node);
      const path = imagePathOf(upstream);
      let changed = false;
      if (upstream !== watchedNode) {
        if (unwatch) { unwatch(); unwatch = null; }
        watchedNode = upstream;
        if (upstream) {
          unwatch = watchImageNode(upstream, () => {
            watchedPath = imagePathOf(upstream);
            loadSource();
          });
        }
        changed = true;
      }
      if (path !== watchedPath) {
        watchedPath = path;
        changed = true;
      }
      if (changed && reload) loadSource();
    }

    bindWatch(false);
    const watchTimer = setInterval(() => {
      bindParamWatchers();
      bindWatch(true);
      refreshCacheHint();
    }, 1000);

    const oldRemoved = node.onRemoved;
    node.onRemoved = function () {
      clearInterval(watchTimer);
      if (unwatch) { unwatch(); unwatch = null; }
      gridStates.delete(gridState);
      return oldRemoved ? oldRemoved.apply(this, arguments) : undefined;
    };

    const oldExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      const result = oldExecuted ? oldExecuted.apply(this, arguments) : undefined;
      try {
        const payload = message && message.stlu_grid;
        if (payload && payload.length) {
          const data = payload[0];
          const src = data.source || [];
          if (src.length === 2 && Number(src[0]) > 0 && Number(src[1]) > 0) {
            srcW = Number(src[0]);
            srcH = Number(src[1]);
          }
          const mode = String(data.mode || "uniform");
          if (mode === "subject_strips" && data.partition) {
            const p = data.partition;
            const regions = (p.regions || []).map((region) => (region.box || region).slice(0, 4));
            lastExecutedWindows = regions.map((box) => ({ box }));
            const plan = {
              ok: true,
              regions,
              side: Number(p.side) || 0,
              resolution: Number(p.resolution) || 0,
              density: Number(p.density) || 0,
              overlap: Number(p.overlap) || 0,
              plan_id: String(p.plan_id || ""),
              warnings: p.warnings || data.warnings || [],
              subject: p.subject || {},
              subject_detection: p.subject_detection || {},
            };
            const key = ["executed", srcW, srcH, plan.plan_id, regions.length].join("|");
            planState = {
              key,
              plan,
              coords: regions,
              side: plan.side,
              resolution: plan.resolution,
              plan_id: plan.plan_id,
              warnings: plan.warnings,
            };
            planAppliedKey = "strips|" + key + "|" + plan.plan_id;
            if (data.selection != null) setWidget(node, "select", String(data.selection));
            if (plan.plan_id) setWidget(node, "select_plan", plan.plan_id);
            showPlan(plan);
          } else if (mode === "uniform_fallback") {
            const fallback = data.fallback || {};
            const failurePlan = {
              ok: false,
              error: fallback.message || "方形分区不可用，已回退均匀网格。",
              warnings: data.warnings || [],
            };
            planState = { key: "executed_fallback", plan: failurePlan, coords: [] };
            showPlan(failurePlan);
            if (data.selection != null) setWidget(node, "select", String(data.selection));
            // Match layout()'s pending key so the executed uniform selection
            // is parsed as-is instead of being geometrically remapped.
            planAppliedKey = "strips_pending|" + srcW + "|" + srcH + "|" +
              Math.max(0, Number(widgetValue(node, "overlap", 128)) || 0) + "|" +
              Number(widgetValue(node, "strip_max_side", 2048)) + "|" +
              Number(widgetValue(node, "resolution", 512)) + "|" +
              Number(widgetValue(node, "strip_margin", 32));
          } else if (mode === "custom_regions" && data.partition) {
            const p = data.partition;
            const regions = (p.regions || []).filter((region) => region && typeof region === "object");
            lastExecutedWindows = regions.map((region) => ({
              box: (region.box || []).slice(0, 4),
            }));
            customRegions = regions.map((region) => {
              const box = region.box || [0, 0, 0, 0];
              const side = Number(region.side) ||
                (Number(box[2]) - Number(box[0])) || 0;
              return normalizeCustomRegion({
                x: Number(box[0]) || 0,
                y: Number(box[1]) || 0,
                size: side,
                // Follow-global keeps 0 so a later resolution change still
                // applies to this window.
                res: region.res_follow ? 0 : (Number(region.resolution) || 0),
                name: region.name || "",
              }, srcW, srcH);
            });
            const key = customPlanKey();
            customActive = -1;
            setWidget(node, "custom_regions", JSON.stringify(customRegions));
            if (data.selection != null) setWidget(node, "select", String(data.selection));
            setWidget(node, "select_plan", String(p.plan_id || key));
            planAppliedKey = "custom|" + key;
            total = customRegions.length;
            if (renderCustomList) renderCustomList();
          }
          layout();
        }
      } catch (e) {
        console.error("[STLU] cannot read grid result", e);
      }
      return result;
    };

    const oldConfigure = node.onConfigure;
    node.onConfigure = function () {
      const result = oldConfigure ? oldConfigure.apply(this, arguments) : undefined;
      try {
        if (pipelineControls) pipelineControls.sync();
        bindParamWatchers();
        bindWatch(false);
        loadSource();
      } catch (e) {
        console.warn("[STLU] grid reload failed", e);
      }
      return result;
    };

    loadSource();
  }

  // Stage ①b: click cells to fill the node's visible ``select`` widget.  The
  // widget itself stays on the node, so typing "0-2,5" works exactly the same.
  function setupSelectTiles(node) {
    injectCss();
    const selectWidget = widgetOf(node, "select");

    const root = el("div", "stlu-panel");
    root.style.padding = "6px";
    root.style.gap = "4px";

    const head = el("div", "stlu-line");
    head.appendChild(el("span", "stlu-cardtitle", "① 选块（点格子）"));
    const count = el("span", "stlu-tag", "…");
    head.appendChild(count);
    head.appendChild(el("span", "grow"));
    const reloadBtn = el("button", "stlu-mini", "刷新");
    head.appendChild(reloadBtn);
    root.appendChild(head);

    const hint = el("div", "stlu-hint",
      "点格子切换；也可以直接改节点上的 select 文字（all / 0-2,5）。");
    root.appendChild(hint);

    const grid = el("div", "stlu-tilegrid");
    root.appendChild(grid);

    const wrap = el("div", "stlu-gridwrap");
    wrap.style.display = "none";
    const canvas = document.createElement("canvas");
    wrap.appendChild(canvas);
    root.appendChild(wrap);

    const actions = el("div", "stlu-line");
    const allBtn = el("button", "stlu-btn", "全选");
    const noneBtn = el("button", "stlu-btn", "清空");
    const invertBtn = el("button", "stlu-btn", "反选");
    actions.appendChild(allBtn);
    actions.appendChild(noneBtn);
    actions.appendChild(invertBtn);
    root.appendChild(actions);

    let image = null;
    let imageUrl = "";
    let srcW = 0;
    let srcH = 0;
    let coords = [];
    let total = 0;
    let selected = new Set();

    function splitNode() {
      const viaTiles = upstreamNode(node, "tiles");
      if (viaTiles && viaTiles.type === TILE_GRID) return viaTiles;
      return findSplitNode();
    }

    function splitValue(name, fallback) {
      const split = splitNode();
      const raw = split ? widgetValue(split, name, fallback) : fallback;
      const value = Number(raw);
      return Number.isFinite(value) ? value : fallback;
    }

    function currentText() {
      return selectWidget ? String(selectWidget.value || "all") : "all";
    }

    function paintCells() {
      grid.querySelectorAll(".stlu-tcell").forEach((cell) => {
        cell.classList.toggle("on", selected.has(Number(cell.dataset.index)));
      });
    }

    function updateCount() {
      count.textContent = total ? selected.size + " / " + total + " 块" : "…";
      count.className = "stlu-tag" + (total && selected.size !== total ? " ok" : "");
    }

    function writeSelect() {
      if (!selectWidget) return;
      const encoded = encodeSelection(selected, total);
      if (String(selectWidget.value || "") !== encoded) {
        setWidget(node, "select", encoded);
      }
    }

    function syncFromWidget() {
      selected = parseSelection(currentText(), total);
      paintCells();
      drawCanvas();
      updateCount();
    }

    function toggle(index) {
      if (index < 0 || index >= total) return;
      if (selected.has(index)) selected.delete(index);
      else selected.add(index);
      paintCells();
      drawCanvas();
      updateCount();
      writeSelect();
    }

    function buildGrid() {
      grid.innerHTML = "";
      if (!total) {
        grid.appendChild(el("div", "stlu-hint", "没有可选的块"));
        return;
      }
      const scaleX = 72 / Math.max(64, splitValue("tile_width", 1024));
      const scaleY = 72 / Math.max(64, splitValue("tile_height", 1024));
      coords.forEach((c, i) => {
        const [x0, y0] = c;
        const cell = el("div", "stlu-tcell" + (selected.has(i) ? " on" : ""));
        cell.dataset.index = String(i);
        cell.title = "第 " + i + " 块 · x" + x0 + " y" + y0 + " · 点击切换";
        if (imageUrl) {
          cell.style.backgroundImage = 'url("' + imageUrl + '")';
          cell.style.backgroundSize = (srcW * scaleX) + "px " + (srcH * scaleY) + "px";
          cell.style.backgroundPosition = (-x0 * scaleX) + "px " + (-y0 * scaleY) + "px";
        }
        cell.appendChild(el("span", null, String(i)));
        const hit = (event) => {
          event.preventDefault();
          event.stopPropagation();
          toggle(i);
        };
        cell.addEventListener("pointerdown", hit);
        grid.appendChild(cell);
      });
    }

    function drawCanvas() {
      if (!image || !image.naturalWidth || !total) return;
      const iw = image.naturalWidth;
      const ih = image.naturalHeight;
      if (canvas.width !== iw || canvas.height !== ih) {
        canvas.width = iw;
        canvas.height = ih;
      }
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, iw, ih);
      ctx.drawImage(image, 0, 0, iw, ih);
      const lw = Math.max(2, Math.round(iw / 500));
      coords.forEach((c, i) => {
        const on = selected.has(i);
        if (on) {
          ctx.fillStyle = "rgba(45,108,223,.28)";
          ctx.fillRect(c[0], c[1], c[2] - c[0], c[3] - c[1]);
        }
        ctx.strokeStyle = on ? "#5b93f5" : "rgba(255,255,255,.55)";
        ctx.lineWidth = lw;
        ctx.strokeRect(c[0] + 0.5, c[1] + 0.5, c[2] - c[0] - 1, c[3] - c[1] - 1);
      });
    }

    function layout() {
      const tw = Math.max(64, splitValue("tile_width", 1024));
      const th = Math.max(64, splitValue("tile_height", 1024));
      const ov = Math.max(0, splitValue("overlap", 128));
      const ox = Math.max(0, splitValue("grid_offset_x", 0) || 0);
      const oy = Math.max(0, splitValue("grid_offset_y", 0) || 0);
      coords = (srcW > 0 && srcH > 0)
        ? gridCoordsJS(srcW, srcH, tw, th, ov, ox, oy)
        : [];
      total = coords.length;
      selected = parseSelection(currentText(), total);
      buildGrid();
      drawCanvas();
      updateCount();
    }

    async function loadSource() {
      const upstream = findUpstreamImageNode(node);
      const path = imagePathOf(upstream);
      image = null;
      imageUrl = "";
      srcW = 0;
      srcH = 0;
      wrap.style.display = "none";
      if (path) {
        const size = await probeImageSize(path, ["input", "output"]);
        if (size) {
          srcW = size.width;
          srcH = size.height;
        }
        imageUrl = imageURLOf(upstream) || "";
      }
      layout();
      if (!imageUrl) return;
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        image = img;
        if (!srcW || !srcH) {
          srcW = img.naturalWidth;
          srcH = img.naturalHeight;
        }
        wrap.style.display = "";
        layout();
      };
      img.src = imageUrl + (imageUrl.includes("?") ? "&" : "?") + "t=" + Date.now();
    }

    canvas.addEventListener("pointerdown", (event) => {
      if (!image || !total) return;
      event.preventDefault();
      event.stopPropagation();
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const x = ((event.clientX - rect.left) / rect.width) * srcW;
      const y = ((event.clientY - rect.top) / rect.height) * srcH;
      let hit = -1;
      coords.forEach((c, i) => {
        if (x >= c[0] && x < c[2] && y >= c[1] && y < c[3]) hit = i;
      });
      toggle(hit);
    });

    allBtn.addEventListener("click", () => {
      selected = new Set(Array.from({ length: total }, (_, i) => i));
      buildGrid();
      drawCanvas();
      updateCount();
      writeSelect();
    });
    noneBtn.addEventListener("click", () => {
      selected = new Set();
      buildGrid();
      drawCanvas();
      updateCount();
      writeSelect();
    });
    invertBtn.addEventListener("click", () => {
      const next = new Set();
      for (let i = 0; i < total; i++) if (!selected.has(i)) next.add(i);
      selected = next;
      buildGrid();
      drawCanvas();
      updateCount();
      writeSelect();
    });
    reloadBtn.addEventListener("click", () => { loadSource(); });

    if (selectWidget) {
      const oldCb = selectWidget.callback;
      selectWidget.callback = function () {
        const r = oldCb ? oldCb.apply(this, arguments) : undefined;
        syncFromWidget();
        return r;
      };
    }

    const dom = addDOM(node, "stlu_tile_grid_card", root, 300, 620);
    registerPanel(node, "grid", dom, root);
    const gridState = { node, refresh: loadSource };
    gridStates.add(gridState);
    const oldRemoved = node.onRemoved;
    node.onRemoved = function () {
      gridStates.delete(gridState);
      return oldRemoved ? oldRemoved.apply(this, arguments) : undefined;
    };

    const oldConfigure = node.onConfigure;
    node.onConfigure = function () {
      const result = oldConfigure ? oldConfigure.apply(this, arguments) : undefined;
      try { loadSource(); } catch (e) { console.warn("[STLU] select reload failed", e); }
      return result;
    };

    loadSource();
  }

  // Offline stage ③: the source path stays a visible widget; this panel only
  // adds the folder browser, so nothing is hidden.
  function setupRebuildPSD(node) {
    injectCss();

    const root = el("div", "stlu-panel");
    root.style.padding = "6px";
    root.style.gap = "4px";

    const head = el("div", "stlu-line");
    head.appendChild(el("span", "stlu-cardtitle", "③ 从图片重建（CPU）"));
    head.appendChild(el("span", "grow"));
    const pickBtn = el("button", "stlu-btn", "浏览…");
    head.appendChild(pickBtn);
    root.appendChild(head);

    const info = el("div", "stlu-hint",
      "选一个波次文件夹或 *_layers.json；不跑 GPU。");
    root.appendChild(info);

    pickBtn.addEventListener("click", () => {
      openPicker({
        title: "选择图片波次文件夹",
        mode: "folder",
        scopes: [{ scope: "output", label: "输出目录" }],
        start: String(widgetValue(node, "source", "") || ""),
        onPick: (rel) => {
          setWidget(node, "source", rel || "");
          info.textContent = rel
            ? "来源：" + rel
            : "选一个波次文件夹或 *_layers.json；不跑 GPU。";
        },
      });
    });

    const dom = addDOM(node, "stlu_tile_grid_card", root, 110, 240);
    registerPanel(node, "tiles", dom, root);
  }

  // -------------------------------------------------------------------------
  // Stage ③: one plain-language panel per node
  // -------------------------------------------------------------------------

  // ③a names this run's folder.  The three widgets stay visible; the panel
  // spells out the resulting path and adds a folder browser.
  function setupWave(node) {
    injectCss();

    const root = el("div", "stlu-panel");
    root.style.padding = "6px";
    root.style.gap = "4px";

    const head = el("div", "stlu-line");
    head.appendChild(el("span", "stlu-cardtitle", "③ 输出文件夹（每次运行新建）"));
    head.appendChild(el("span", "grow"));
    const pickBtn = el("button", "stlu-btn", "浏览…");
    head.appendChild(pickBtn);
    root.appendChild(head);

    root.appendChild(el("div", "stlu-hint",
      "所有结果都存在这里。每按一次 Queue 都新建一个波次文件夹，上一次的结果不会被覆盖。"));

    const pathLine = el("div", "stlu-hint");
    pathLine.style.wordBreak = "break-all";
    root.appendChild(pathLine);

    function refresh() {
      pathLine.textContent = "本次会写到 → " + wavePreview(node);
    }

    pickBtn.addEventListener("click", () => {
      openPicker({
        title: "选择输出根目录",
        mode: "folder",
        scopes: [{ scope: "output", label: "输出目录" }],
        start: String(widgetValue(node, "out_dir", "") || ""),
        onPick: (rel) => {
          setWidget(node, "out_dir", rel || "SeeThrough");
          refresh();
        },
      });
    });

    // These may be fed by the primitive nodes in the "用户输入" group; watchers
    // are rebound once the workflow's links exist (after load).
    function bindWatchers() {
      watchValue(node, "out_dir", refresh);
      watchValue(node, "group", refresh);
    }
    bindWatchers();

    const oldConfigure = node.onConfigure;
    node.onConfigure = function () {
      const result = oldConfigure ? oldConfigure.apply(this, arguments) : undefined;
      refresh();
      setTimeout(() => { bindWatchers(); refresh(); }, 400);
      return result;
    };

    const dom = addDOM(node, "stlu_wave_card", root, 130, 280);
    registerPanel(node, "wave", dom, root);
    refresh();
    setTimeout(() => { bindWatchers(); refresh(); }, 400);
  }

  // ③b runs once per selected tile (ComfyUI expands the list automatically).
  // The panel explains that and shows the last written preview.
  function setupTilePSD(node) {
    injectCss();

    const root = el("div", "stlu-panel");
    root.style.padding = "6px";
    root.style.gap = "4px";

    const head = el("div", "stlu-line");
    head.appendChild(el("span", "stlu-cardtitle", "③ 每块存一份 PSD（自动逐块）"));
    const tag = el("span", "stlu-tag", "等待运行");
    head.appendChild(tag);
    head.appendChild(el("span", "grow"));
    root.appendChild(head);

    const hint = el("div", "stlu-hint",
      "② 送来几块，本节点就自动执行几次：每块写一份 PSD、预览、depth 和 images/ 里的图层图片。不用手点。");
    root.appendChild(hint);

    const dirLine = el("div", "stlu-hint");
    dirLine.style.wordBreak = "break-all";
    dirLine.textContent = "运行完成后这里会显示输出文件夹。";
    root.appendChild(dirLine);

    const wrap = el("div", "stlu-gridwrap");
    wrap.style.display = "none";
    const thumb = document.createElement("img");
    thumb.style.width = "100%";
    thumb.style.maxHeight = "150px";
    thumb.style.objectFit = "contain";
    thumb.style.borderRadius = "4px";
    thumb.alt = "";
    wrap.appendChild(thumb);
    root.appendChild(wrap);

    const actions = el("div", "stlu-line");
    const openBtn = el("button", "stlu-btn", "打开波次文件夹");
    openBtn.disabled = true;
    actions.appendChild(openBtn);
    root.appendChild(actions);

    let runs = 0;
    let waveDir = "";

    function update(data) {
      if (!data) return;
      waveDir = data.dir || data.subfolder || waveDir;
      const count = Number(data.count) || 0;
      if (count) {
        tag.className = "stlu-tag ok";
        tag.textContent = count + " 块";
      }
      if (data.dir) {
        dirLine.textContent = "→ output/" + data.dir + "/";
        dirLine.title = data.dir;
      }
      openBtn.disabled = !waveDir;
      const src = previewSrc(data.preview);
      if (src) {
        wrap.style.display = "";
        thumb.src = src;
      }
    }

    openBtn.addEventListener("click", () => {
      if (waveDir) openPath(waveDir, false);
    });

    const onStart = () => {
      runs = 0;
      tag.className = "stlu-tag";
      tag.textContent = "正在逐块写入…";
    };
    try { api.addEventListener("execution_start", onStart); } catch (e) { /* old frontend */ }

    const oldExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      const result = oldExecuted ? oldExecuted.apply(this, arguments) : undefined;
      try {
        const payload = message && message.stlu_tile;
        if (payload && payload.length) {
          const info = payload[0] || {};
          if (!info.empty) {
            runs += 1;
            tag.className = "stlu-tag ok";
            tag.textContent = "已存 " + runs + " 块";
          }
        }
      } catch (e) {
        console.warn("[STLU] tile panel", e);
      }
      return result;
    };

    stage3States.add(update);
    const oldRemoved = node.onRemoved;
    node.onRemoved = function () {
      stage3States.delete(update);
      try { api.removeEventListener("execution_start", onStart); } catch (e) { /* ignore */ }
      return oldRemoved ? oldRemoved.apply(this, arguments) : undefined;
    };

    const dom = addDOM(node, "stlu_tile_psd_card", root, 210, 380);
    registerPanel(node, "tile", dom, root);
  }

  // ③c runs once, after every tile was stored: it writes the wave-wide
  // manifest and hands the folder to stage ④.
  function setupWaveIndex(node) {
    injectCss();

    const root = el("div", "stlu-panel");
    root.style.padding = "6px";
    root.style.gap = "4px";

    const head = el("div", "stlu-line");
    head.appendChild(el("span", "stlu-cardtitle", "③ 写清单（全部块完成后）"));
    const tag = el("span", "stlu-tag", "等待运行");
    head.appendChild(tag);
    head.appendChild(el("span", "grow"));
    root.appendChild(head);

    root.appendChild(el("div", "stlu-hint",
      "全部块写完后，这里统一写 manifest.json / images.json / original.png，" +
      "并把波次文件夹交给 ④ 合并。"));

    const dirLine = el("div", "stlu-hint");
    dirLine.style.wordBreak = "break-all";
    dirLine.textContent = "运行完成后这里会显示结果文件夹。";
    root.appendChild(dirLine);

    const actions = el("div", "stlu-line");
    const openBtn = el("button", "stlu-btn", "打开结果文件夹");
    openBtn.disabled = true;
    actions.appendChild(openBtn);
    root.appendChild(actions);

    let waveDir = "";

    function update(data) {
      if (!data || !data.dir) return;
      waveDir = data.dir;
      const count = Number(data.count) || 0;
      tag.className = "stlu-tag ok";
      tag.textContent = count ? count + " 块" : "已写入";
      dirLine.textContent = "✓ output/" + waveDir + "/";
      dirLine.title = waveDir;
      openBtn.disabled = false;
    }

    openBtn.addEventListener("click", () => {
      if (waveDir) openPath(waveDir, false);
    });

    stage3States.add(update);
    const oldRemoved = node.onRemoved;
    node.onRemoved = function () {
      stage3States.delete(update);
      return oldRemoved ? oldRemoved.apply(this, arguments) : undefined;
    };

    const dom = addDOM(node, "stlu_wave_index_card", root, 180, 320);
    registerPanel(node, "wave_index", dom, root);
  }

  // -------------------------------------------------------------------------
  // Stage 2: save the generated tile images
  // -------------------------------------------------------------------------

  function setupSaveTiles(node) {
    injectCss();
    syncTileParamsFromSplit(node);
    for (const name of ["grid", "tile_indices", "out_dir", "filename_prefix"]) {
      hideWidget(widgetOf(node, name));
    }

    const root = el("div", "stlu-panel");
    root.style.padding = "8px";
    root.style.gap = "5px";

    const head = el("div", "stlu-line");
    head.appendChild(el("span", "stlu-cardtitle", "② 保存分块图片"));
    const tag = el("span", "stlu-tag", "等待运行");
    head.appendChild(tag);
    head.appendChild(el("span", "grow"));
    root.appendChild(head);

    const hint = el("div", "stlu-hint", "把每块的部件图片按块存成文件夹，第三步可以直接读取。");
    root.appendChild(hint);

    const dirLine = el("div", "stlu-line");
    dirLine.appendChild(el("span", "stlu-label", "目录"));
    const dirText = el("span", "stlu-hint grow", "—");
    dirText.style.overflow = "hidden";
    dirText.style.textOverflow = "ellipsis";
    dirLine.appendChild(dirText);
    root.appendChild(dirLine);

    const actions = el("div", "stlu-line");
    const openDir = el("button", "stlu-btn", "打开文件夹");
    const feedBtn = el("button", "stlu-btn", "填到分块 PSD 节点");
    openDir.disabled = true;
    feedBtn.disabled = true;
    actions.appendChild(openDir);
    actions.appendChild(feedBtn);
    root.appendChild(actions);

    let info = null;

    function applyInfo(data) {
      if (!data) return;
      info = data;
      const isError = !!data.error;
      tag.className = "stlu-tag" + (isError ? "" : " ok");
      tag.textContent = isError ? "失败" : (data.count || 0) + " 块";
      dirText.textContent = data.dir || "—";
      dirText.title = data.dir || "";
      hint.className = "stlu-hint" + (isError ? " error" : "");
      hint.textContent = isError
        ? String(data.error).slice(0, 160)
        : (data.count || 0) + " 块 · " + ((data.canvas || []).join("×") || "");
      openDir.disabled = !data.dir;
      feedBtn.disabled = !data.dir;
    }

    openDir.addEventListener("click", () => { if (info && info.dir) openPath(info.dir, false); });
    feedBtn.addEventListener("click", () => {
      if (!info || !info.dir) return;
      const target = allNodes().find((n) => n.type === TILES);
      if (!target) {
        feedBtn.textContent = "图里没有分块 PSD 节点";
        setTimeout(() => { feedBtn.textContent = "填到分块 PSD 节点"; }, 1800);
        return;
      }
      setWidget(target, "source", info.dir);
      feedBtn.textContent = "已填入 ✓";
      setTimeout(() => { feedBtn.textContent = "填到分块 PSD 节点"; }, 1500);
    });

    addDOM(node, "stlu_save_tiles_card", root, 160, 260);

    const oldExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      const result = oldExecuted ? oldExecuted.apply(this, arguments) : undefined;
      try {
        const payload = message && message.stlu_tile_images;
        if (payload && payload.length) applyInfo(payload[0]);
      } catch (e) {
        console.error("[STLU] cannot read tile image result", e);
      }
      return result;
    };
  }

  // -------------------------------------------------------------------------
  // Tile merge panel (reuse existing tile PSDs, no GPU)
  // -------------------------------------------------------------------------

  function setupTileMerge(node) {
    injectCss();
    const outW = widgetOf(node, "out_dir");
    const nameW = widgetOf(node, "name");
    const groupW = widgetOf(node, "group_layers");
    const depthW = widgetOf(node, "include_depth");
    const origW = widgetOf(node, "include_original");
    const runW = widgetOf(node, "run_dir");
    const seamW = widgetOf(node, "seam_feather");
    for (const w of [outW, nameW, groupW, depthW, origW, runW, seamW]) hideWidget(w);
    // When the fixed pipeline feeds merge_doc in, this panel is a result
    // preview: the folder and options are already decided upstream.
    let docMode = false;

    const root = el("div", "stlu-panel");
    root.style.padding = "8px";
    root.style.gap = "6px";

    const head = el("div", "stlu-line");
    head.appendChild(el("span", "stlu-cardtitle", "分块 PSD 合并"));
    head.appendChild(el("span", "grow"));
    const refreshBtn = el("button", "stlu-btn", "刷新");
    head.appendChild(refreshBtn);
    root.appendChild(head);

    const dirSelect = document.createElement("select");
    dirSelect.className = "stlu-input";
    root.appendChild(dirSelect);

    const ctrl = el("div", "stlu-line");
    const nameInput = el("input", "stlu-input");
    nameInput.value = String(widgetValue(node, "name", "merged") || "merged");
    nameInput.placeholder = "合并文件名";
    nameInput.style.flex = "0 1 120px";
    ctrl.appendChild(nameInput);
    const allBtn = el("button", "stlu-btn", "全选");
    const noneBtn = el("button", "stlu-btn", "清空");
    const invBtn = el("button", "stlu-btn", "反选");
    ctrl.appendChild(allBtn);
    ctrl.appendChild(noneBtn);
    ctrl.appendChild(invBtn);
    ctrl.appendChild(el("span", "grow"));
    const count = el("span", "stlu-tag", "—");
    ctrl.appendChild(count);
    root.appendChild(ctrl);

    const opts = el("div", "stlu-panel");
    opts.style.gap = "4px";
    function addToggle(parent, name, label, title, fallback) {
      const box = el("label", "stlu-check");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      box.title = title || "";
      cb.checked = !!widgetValue(node, name, fallback);
      cb.addEventListener("change", () => setWidget(node, name, cb.checked));
      box.appendChild(cb);
      box.appendChild(el("span", null, label));
      parent.appendChild(box);
      // Initial state may be read before the workflow's links exist; re-sync
      // once loading finishes so a linked primitive wins.
      setTimeout(() => { cb.checked = !!widgetValue(node, name, fallback); }, 400);
      return cb;
    }

    opts.appendChild(el("div", "stlu-cardtitle", "合并方式"));
    const modeRow = el("div", "stlu-chips");
    opts.appendChild(modeRow);
    const groupCb = addToggle(modeRow, "group_layers", "同名图层单独成组（不裁剪）",
      "勾选：不裁剪版——两格里同名部件各自保留完整原件并收进同名文件夹，"
      + "接缝裁切/幻影清理/完整部件淘汰都不做；不勾：直出版——按接缝归属裁切后合成一层", true);
    const seamBox = el("label", "stlu-check");
    seamBox.appendChild(el("span", null, "接缝过渡"));
    const seamInput = el("input", "stlu-input");
    seamInput.type = "number";
    seamInput.min = "0";
    seamInput.max = "512";
    seamInput.style.width = "56px";
    seamInput.value = String(widgetValue(node, "seam_feather", 32));
    seamInput.addEventListener("change", () => {
      const value = Math.max(0, Math.round(Number(seamInput.value) || 0));
      seamInput.value = String(value);
      setWidget(node, "seam_feather", value);
    });
    seamBox.appendChild(seamInput);
    seamBox.appendChild(el("span", null, "px"));
    modeRow.appendChild(seamBox);

    opts.appendChild(el("div", "stlu-cardtitle", "附加输出"));
    const extraRow = el("div", "stlu-chips");
    opts.appendChild(extraRow);
    const depthCb = addToggle(extraRow, "include_depth", "depth 另存 PSD",
      "导出独立的 <名字>_depth.psd；不勾则不导出。depth 永远不会进入彩色 PSD", false);
    const origCb = addToggle(extraRow, "include_original", "原图作底图",
      "合并 PSD 底部放入原图图层和 original.png", false);

    const modeHint = el("div", "stlu-hint", "");
    modeHint.style.whiteSpace = "normal";
    opts.appendChild(modeHint);
    function syncMode() {
      const grouped = !!widgetValue(node, "group_layers", true);
      seamInput.disabled = grouped;
      seamBox.title = grouped
        ? "不裁剪模式不使用接缝过渡"
        : "直出合并时落败块在接缝外侧的淡出宽度（像素）；0 = 全部硬切";
      modeHint.textContent = grouped
        ? "不裁剪版：同名副本收进同名组，组内是各块自己的完整副本；"
          + "只有一个副本的部位保持普通图层。重叠处会重复显示，"
          + "修完在 PS 里隐藏/删除不要的副本即可。"
        : "直出版：每个像素只归离自己块边更远的一块，落败块在接缝外侧按该宽度淡出；"
          + "depth 永远单独一个 PSD，和彩色 PSD 互不相干。";
    }
    groupCb.addEventListener("change", syncMode);
    setTimeout(syncMode, 450);
    syncMode();
    root.appendChild(opts);
    root.appendChild(el("div", "stlu-hint",
      "同一文件夹里已有的块会自动参与合并——只重跑一块也能出完整成品。"));

    const wall = el("div", "stlu-tilewall");
    root.appendChild(wall);

    const foot = el("div", "stlu-line");
    const mergeBtn = el("button", "stlu-btn primary", "合并所选");
    const status = el("span", "stlu-hint grow", "选择目录后勾选要拼的块");
    foot.appendChild(mergeBtn);
    foot.appendChild(status);
    root.appendChild(foot);

    const preview = el("img", "stlu-merge-prev");
    preview.style.display = "none";
    preview.style.cursor = "zoom-in";
    preview.addEventListener("click", () => {
      if (!result || !result.preview || !result.preview.filename) return;
      const rel = (result.preview.subfolder ? result.preview.subfolder + "/" : "") + result.preview.filename;
      showBigImage(rel, "合并预览");
    });
    root.appendChild(preview);

    // Depth is merged on its own so it is offered on its own: a small
    // thumbnail plus a zoom, never blended into the colour preview.
    const depthPrev = el("img", "stlu-merge-prev");
    depthPrev.style.display = "none";
    depthPrev.style.maxHeight = "140px";
    depthPrev.style.cursor = "zoom-in";
    depthPrev.title = "depth 图（单独合并）";
    depthPrev.addEventListener("click", () => {
      if (!result || !result.depth || !result.depth.filename) return;
      const rel = (result.depth.subfolder ? result.depth.subfolder + "/" : "") + result.depth.filename;
      showBigImage(rel, "depth 图");
    });
    root.appendChild(depthPrev);

    const actions = el("div", "stlu-line");
    const openPsd = el("button", "stlu-btn", "打开合并 PSD");
    const openDepthPsd = el("button", "stlu-btn", "打开 depth PSD");
    const openDir = el("button", "stlu-btn", "打开文件夹");
    openPsd.disabled = true;
    openDepthPsd.disabled = true;
    openDir.disabled = true;
    actions.appendChild(openPsd);
    actions.appendChild(openDepthPsd);
    actions.appendChild(openDir);
    root.appendChild(actions);

    let dirs = [];
    let currentDir = "";
    let manifest = null;
    let selected = new Set();
    let result = null;

    function setStatus(text, kind) {
      status.textContent = text || "";
      status.className = "stlu-hint grow" + (kind ? " " + kind : "");
    }

    function updateCount() {
      const total = manifest ? (manifest.tiles || []).length : 0;
      count.textContent = selected.size + " / " + total + " 块";
    }

    function renderWall() {
      wall.innerHTML = "";
      if (!manifest || !manifest.tiles || !manifest.tiles.length) {
        wall.appendChild(el("div", "stlu-hint", "这个目录里没有分块"));
        return;
      }
      for (const tile of manifest.tiles) {
        const idx = Number(tile.index);
        const cell = el("div", "stlu-tile" + (selected.has(idx) ? " on" : ""));
        if (tile.preview) {
          const img = document.createElement("img");
          img.loading = "lazy";
          const rel = (manifest.dir ? manifest.dir + "/" : "") + tile.preview;
          img.src = apiUrl(
            "/stlu/thumb?scope=output&path=" + encodeURIComponent(rel) + "&size=180"
          );
          cell.appendChild(img);
        } else {
          cell.appendChild(el("div", "empty", "PSD"));
        }
        const badge = el("span", "badge");
        if (tile.tile) badge.textContent = "x" + tile.tile[0] + " y" + tile.tile[1];
        else badge.textContent = String(idx);
        cell.appendChild(badge);
        cell.appendChild(el("span", "chk"));
        cell.title = (tile.file || ("块 " + idx)) + " · 单击勾选，双击放大";
        let clickTimer = null;
        const toggle = () => {
          if (selected.has(idx)) selected.delete(idx);
          else selected.add(idx);
          cell.classList.toggle("on", selected.has(idx));
          updateCount();
        };
        cell.addEventListener("click", () => {
          if (clickTimer) {
            clearTimeout(clickTimer);
            clickTimer = null;
            if (tile.preview) {
              showBigImage((manifest.dir ? manifest.dir + "/" : "") + tile.preview,
                           "块 " + idx + " 预览");
            }
            return;
          }
          clickTimer = setTimeout(() => {
            clickTimer = null;
            toggle();
          }, 220);
        });
        wall.appendChild(cell);
      }
      updateCount();
    }

    async function loadManifest(dir) {
      setStatus("读取分块…");
      try {
        manifest = await getJson("/stlu/tiles/manifest?dir=" + encodeURIComponent(dir));
        currentDir = dir;
        // Keep the node's own input in sync so queueing it merges this run.
        if (runW) setWidget(node, "run_dir", dir);
        selected = new Set((manifest.tiles || []).map((t) => Number(t.index)));
        renderWall();
        const reused = Array.isArray(manifest.reused_tiles) ? manifest.reused_tiles.length : 0;
        const suffix = reused ? "（其中 " + reused + " 块复用自之前的运行）" : "";
        if (manifest.group_layers === false) {
          setStatus("这些分块已经把同部件合成一层了；要按部件分文件夹，请在 ③ 里保留子层再生成", "warn");
        } else if (manifest.merged && manifest.merged.file) {
          setStatus("上次合并：" + (manifest.merged.file || "") + suffix);
        } else {
          setStatus("选择要拼进成品的块" + suffix);
        }
      } catch (error) {
        manifest = null;
        currentDir = "";
        wall.innerHTML = "";
        setStatus("读取失败：" + error.message, "error");
      }
    }

    async function loadDirs() {
      if (docMode) return;
      refreshBtn.disabled = true;
      try {
        const data = await getJson("/stlu/tiles/dirs");
        if (docMode) return;
        dirs = data.dirs || [];
        dirSelect.innerHTML = "";
        if (!dirs.length) {
          const opt = el("option", null, "还没有分块结果");
          opt.value = "";
          dirSelect.appendChild(opt);
          wall.innerHTML = "";
          setStatus("先跑一次分块工作流", "warn");
          return;
        }
        for (const entry of dirs) {
          const opt = el("option", null, entry.name + " · " + entry.count + " 块");
          opt.value = entry.dir;
          dirSelect.appendChild(opt);
        }
        const keep = dirs.find((d) => d.dir === currentDir) || dirs[0];
        dirSelect.value = keep.dir;
        await loadManifest(keep.dir);
      } catch (error) {
        setStatus("列目录失败：" + error.message, "error");
      } finally {
        refreshBtn.disabled = false;
      }
    }

    dirSelect.addEventListener("change", () => {
      if (dirSelect.value) loadManifest(dirSelect.value);
    });
    refreshBtn.addEventListener("click", loadDirs);
    nameInput.addEventListener("input", () => {
      setWidget(node, "name", nameInput.value || "merged");
    });

    allBtn.addEventListener("click", () => {
      if (!manifest) return;
      selected = new Set((manifest.tiles || []).map((t) => Number(t.index)));
      renderWall();
    });
    noneBtn.addEventListener("click", () => {
      selected = new Set();
      renderWall();
    });
    invBtn.addEventListener("click", () => {
      if (!manifest) return;
      const all = new Set((manifest.tiles || []).map((t) => Number(t.index)));
      const next = new Set();
      for (const i of all) if (!selected.has(i)) next.add(i);
      selected = next;
      renderWall();
    });

    openPsd.addEventListener("click", () => {
      if (!result) return;
      let rel = "";
      if (result.psd && result.psd.filename) {
        rel = (result.psd.subfolder ? result.psd.subfolder + "/" : "") + result.psd.filename;
      } else if (result.file) {
        rel = (result.dir || currentDir || "") + (result.dir ? "/" : "") + result.file;
      }
      if (rel) openPath(rel, true);
    });
    openDepthPsd.addEventListener("click", () => {
      if (!result || !result.depth_psd || !result.depth_psd.filename) return;
      const rel = (result.depth_psd.subfolder ? result.depth_psd.subfolder + "/" : "") +
        result.depth_psd.filename;
      openPath(rel, true);
    });
    openDir.addEventListener("click", () => {
      if (result) openPath(result.dir || currentDir, false);
    });

    function previewRel(data, key) {
      const value = data && data[key];
      if (!value) return "";
      if (typeof value === "string") {
        return (data.dir ? data.dir + "/" : "") + value;
      }
      if (value.filename) {
        return (value.subfolder ? value.subfolder + "/" : "") + value.filename;
      }
      return "";
    }

    // Shared by the panel button and by a queued run of this node, so both
    // paths render the same card.
    function applyMergeResult(data) {
      if (!data) return;
      result = data;
      if (data.dir && runW) setWidget(node, "run_dir", data.dir);
      const prevRel = previewRel(data, "preview");
      if (prevRel) {
        preview.src = apiUrl(
          "/stlu/thumb?scope=output&path=" + encodeURIComponent(prevRel) +
          "&size=640&v=" + Date.now()
        );
        preview.style.display = "";
      }
      const depthRel = previewRel(data, "depth");
      if (depthRel) {
        depthPrev.src = apiUrl(
          "/stlu/thumb?scope=output&path=" + encodeURIComponent(depthRel) +
          "&size=420&v=" + Date.now()
        );
        depthPrev.style.display = "";
      } else {
        depthPrev.style.display = "none";
      }
      const layers = Array.isArray(data.layers) ? data.layers.length : 0;
      const groups = Array.isArray(data.groups) ? data.groups.length : 0;
      const canvas = (data.canvas || []).join("×");
      setStatus("完成：" + (data.file || "合并") + " · " + layers + " 层" +
                (groups ? " · " + groups + " 个文件夹" : "") + " · " + canvas +
                (data.depth_psd && data.depth_psd.filename
                  ? " · depth 另存 " + data.depth_psd.filename : ""), "ok");
      openPsd.disabled = !(data.file || (data.psd && data.psd.filename));
      openDepthPsd.disabled = !(data.depth_psd && data.depth_psd.filename);
      openDir.disabled = false;
      if (manifest) manifest.merged = { file: data.file };
    }
    mergeStates.add(applyMergeResult);
    const oldRemovedMerge = node.onRemoved;
    node.onRemoved = function () {
      mergeStates.delete(applyMergeResult);
      return oldRemovedMerge ? oldRemovedMerge.apply(this, arguments) : undefined;
    };

    mergeBtn.addEventListener("click", async () => {
      if (!currentDir) return;
      if (!selected.size) {
        setStatus("先勾选至少一块", "warn");
        return;
      }
      mergeBtn.disabled = true;
      setStatus("合并中…");
      try {
        const data = await postJson("/stlu/tiles/merge", {
          dir: currentDir,
          indices: Array.from(selected).sort((a, b) => a - b),
          name: nameInput.value || "merged",
          group_layers: !!groupCb.checked,
          include_depth: !!depthCb.checked,
          include_original: !!origCb.checked,
          merge_parts: manifest ? !!manifest.merge_parts : true,
          seam_feather: Math.max(0, Math.round(Number(seamInput.value) || 0)),
        });
        applyMergeResult(data);
      } catch (error) {
        setStatus("合并失败：" + error.message, "error");
      } finally {
        mergeBtn.disabled = false;
      }
    });

    function applyDocMode() {
      const input = (node.inputs || []).find((i) => i.name === "merge_doc");
      if (!input || input.link == null) return false;
      docMode = true;
      dirSelect.style.display = "none";
      ctrl.style.display = "none";
      opts.style.display = "none";
      wall.style.display = "none";
      mergeBtn.style.display = "none";
      setStatus("自动合并：目录和选项来自 ① 用户输入", "ok");
      return true;
    }

    const mergeDom = addDOM(node, "stlu_tile_merge_card", root, 320, 640);
    registerPanel(node, "merge", mergeDom, root);
    const oldConfigureMerge = node.onConfigure;
    node.onConfigure = function () {
      const result = oldConfigureMerge ? oldConfigureMerge.apply(this, arguments) : undefined;
      // Links only exist once the graph finishes loading.
      setTimeout(applyDocMode, 0);
      return result;
    };
    applyDocMode();
    loadDirs();
  }

  // -------------------------------------------------------------------------
  // Edit-parts batch card
  // -------------------------------------------------------------------------

  // The panel writes the same comma-separated string the backend understands;
  // commas matter because several part names contain a space ("back hair").
  // Name this one differently from the tile-index parser above: two module
  // level `parseSelection` declarations would silently shadow each other and
  // break the tile picker.
  function parsePartSelection(text, names) {
    const all = new Set(names);
    const value = String(text == null ? "" : text).trim();
    if (!value || value.toLowerCase() === "all") return new Set(all);
    if (value.toLowerCase() === "none") return new Set();
    const tokens = value.includes(",") ? value.split(",") : value.split(/\s+/);
    let selected = null;
    const removed = new Set();
    for (let token of tokens) {
      token = token.trim();
      if (!token) continue;
      if (token.startsWith("-")) {
        removed.add(token.slice(1).trim().toLowerCase());
        continue;
      }
      if (token.toLowerCase() === "all") {
        selected = new Set(all);
        continue;
      }
      if (!selected) selected = new Set();
      const lower = token.toLowerCase();
      const hit = names.find((n) => n.toLowerCase() === lower);
      if (hit) selected.add(hit);
    }
    if (!selected) selected = new Set(all);
    for (const name of names) {
      if (removed.has(name.toLowerCase())) selected.delete(name);
    }
    return selected;
  }

  function setupEditBatch(node) {
    injectCss();
    const selectW = widgetOf(node, "select");

    const root = el("div", "stlu-panel");
    root.style.padding = "8px";
    root.style.gap = "6px";

    const head = el("div", "stlu-line");
    head.appendChild(el("span", "stlu-cardtitle", "部件选择"));
    const count = el("span", "stlu-tag", "…");
    head.appendChild(count);
    head.appendChild(el("span", "grow"));
    const allBtn = el("button", "stlu-btn", "全选");
    const noneBtn = el("button", "stlu-btn", "清空");
    head.appendChild(allBtn);
    head.appendChild(noneBtn);
    root.appendChild(head);

    const search = el("input", "stlu-input stlu-search");
    search.placeholder = "搜索部件…";
    root.appendChild(search);

    const listEl = el("div", "stlu-list");
    listEl.style.minHeight = "120px";
    root.appendChild(listEl);

    const sub = el("div", "stlu-hint", "正在读取部件表…");
    root.appendChild(sub);

    const actions = el("div", "stlu-line");
    const openPsd = el("button", "stlu-btn", "打开 PSD");
    const reveal = el("button", "stlu-btn", "打开文件夹");
    openPsd.disabled = true;
    reveal.disabled = true;
    actions.appendChild(openPsd);
    actions.appendChild(reveal);
    root.appendChild(actions);

    const ctx = {
      node, sub, openPsd, reveal, count,
      parts: [], rows: [], groups: [], selected: new Set(),
      nameToRow: new Map(), groupEls: new Map(),
      total: 0, done: new Set(), psd: "", out: "",
    };
    editStates.add(ctx);

    openPsd.addEventListener("click", () => { if (ctx.psd) openPath(ctx.psd, true); });
    reveal.addEventListener("click", () => { if (ctx.out) openPath(ctx.out, false); });

    function updateCount() {
      const n = ctx.selected.size;
      count.textContent = n + "/" + ctx.parts.length;
      count.className = "stlu-tag" + (n === ctx.parts.length ? " ok" : "");
    }

    function writeSelect() {
      if (!ctx.parts.length) return;
      const names = ctx.parts.map((p) => p.name).filter((n) => ctx.selected.has(n));
      let text;
      if (names.length === ctx.parts.length) text = "all";
      else if (!names.length) text = "none";
      else text = names.join(", ");
      setWidget(node, "select", text);
      updateCount();
    }

    function applyRow(name, on) {
      if (on) ctx.selected.add(name);
      else ctx.selected.delete(name);
      const row = ctx.nameToRow.get(name);
      if (row) {
        row.row.classList.toggle("off", !on);
        row.cb.checked = on;
      }
    }

    function setAll(on) {
      for (const part of ctx.parts) applyRow(part.name, on);
      writeSelect();
    }

    allBtn.addEventListener("click", () => setAll(true));
    noneBtn.addEventListener("click", () => setAll(false));

    function filter() {
      const q = search.value.trim().toLowerCase();
      for (const row of ctx.rows) {
        const hit = !q
          || row.name.toLowerCase().includes(q)
          || String(row.label || "").toLowerCase().includes(q);
        row.row.classList.toggle("hidden", !hit);
      }
      for (const group of ctx.groupEls.values()) {
        const any = group.rows.some((r) => !r.row.classList.contains("hidden"));
        group.sec.style.display = any ? "" : "none";
      }
    }
    search.addEventListener("input", filter);

    function render(parts, groups) {
      ctx.parts = parts;
      ctx.groups = groups;
      listEl.innerHTML = "";
      ctx.rows = [];
      ctx.nameToRow = new Map();
      ctx.groupEls = new Map();
      const names = parts.map((p) => p.name);
      ctx.selected = parsePartSelection(selectW ? selectW.value : "all", names);

      for (const group of groups) {
        const items = parts.filter((p) => p.group === group.id);
        if (!items.length) continue;
        const sec = el("div", "stlu-sec");
        sec.appendChild(el("span", null, group.label));
        sec.appendChild(el("span", "grow"));
        sec.appendChild(el("span", "stlu-hint", String(items.length)));
        listEl.appendChild(sec);
        const groupRows = [];
        ctx.groupEls.set(group.id, { sec, rows: groupRows });

        for (const part of items) {
          const row = el("label", "stlu-prow");
          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.checked = ctx.selected.has(part.name);
          const who = el("div", "who");
          who.appendChild(el("span", "n", part.label || part.name));
          who.appendChild(el("span", "d",
            part.name + (part.desc ? " · " + part.desc : "")));
          row.appendChild(cb);
          row.appendChild(who);
          if (!cb.checked) row.classList.add("off");
          cb.addEventListener("change", () => {
            applyRow(part.name, cb.checked);
            writeSelect();
          });
          listEl.appendChild(row);
          const entry = { ...part, row, cb };
          ctx.rows.push(entry);
          ctx.nameToRow.set(part.name, entry);
          groupRows.push(entry);
        }
      }
      updateCount();
      filter();
    }

    function reload() {
      if (!ctx.parts.length) return;
      const names = ctx.parts.map((p) => p.name);
      ctx.selected = parsePartSelection(selectW ? selectW.value : "all", names);
      for (const part of ctx.parts) {
        const row = ctx.nameToRow.get(part.name);
        const on = ctx.selected.has(part.name);
        if (row) {
          row.cb.checked = on;
          row.row.classList.toggle("off", !on);
        }
      }
      updateCount();
    }

    const oldConfigured = node.onAfterGraphConfigured;
    node.onAfterGraphConfigured = function () {
      const result = oldConfigured ? oldConfigured.apply(this, arguments) : undefined;
      reload();
      return result;
    };
    const oldConfigure = node.onConfigure;
    node.onConfigure = function () {
      const result = oldConfigure ? oldConfigure.apply(this, arguments) : undefined;
      try { reload(); } catch (e) { console.warn("[STLU] edit parts reload failed", e); }
      return result;
    };

    addDOM(node, "stlu_edit_card", root, 330, 430);

    // If the route is missing (old backend still loaded) the raw `select`
    // widget stays visible, so the workflow can still be driven by hand.
    getJson("/stlu/edit_parts").then((data) => {
      hideWidget(selectW);
      render(data.parts || [], data.groups || []);
      sub.textContent = "勾选这次要抠的部件；勾得越多跑得越久。";
    }).catch((err) => {
      console.error("[STLU] edit parts catalogue failed", err);
      sub.textContent = "读取部件表失败：请重启 ComfyUI 后刷新（仍可手写 select）。";
      sub.className = "stlu-hint error";
    });

    const oldRemoved = node.onRemoved;
    node.onRemoved = function () {
      editStates.delete(ctx);
      return oldRemoved ? oldRemoved.apply(this, arguments) : undefined;
    };
  }

  // -------------------------------------------------------------------------
  // websocket events
  // -------------------------------------------------------------------------

  function forEachPlanState(fn) {
    for (const ctx of Array.from(planStates)) {
      try { fn(ctx); } catch (e) { console.error("[STLU]", e); }
    }
  }

  function forEachEditState(fn) {
    for (const ctx of Array.from(editStates)) {
      try { fn(ctx); } catch (e) { console.error("[STLU]", e); }
    }
  }

  function bindEvents() {
    if (bindEvents.done) return;
    bindEvents.done = true;
    const listen = (name, fn) => {
      try {
        api.addEventListener(name, (event) => {
          try { fn((event && event.detail) || {}); } catch (e) { console.error("[STLU]", name, e); }
        });
      } catch (e) {
        console.warn("[STLU] cannot listen to", name, e);
      }
    };

    listen("execution_start", () => forEachPlanState((ctx) => {
      if (ctx.running) return;
      ctx.running = true;
      resetDoneMarks(ctx);
      ctx.total = selectedCount(ctx);
      updateRunState(ctx);
      setStatus(ctx, "正在运行…");
    }));

    listen("stlu.start", (data) => forEachPlanState((ctx) => {
      ctx.running = true;
      resetDoneMarks(ctx);
      ctx.total = Number(data.total) || selectedCount(ctx);
      updateRunState(ctx);
      const skipped = Array.isArray(data.skipped) ? data.skipped.length : 0;
      if (ctx.total) {
        setStatus(ctx, "开始放大 " + ctx.total + " 个部件…" +
          (skipped ? "（跳过 " + skipped + " 个已生成）" : ""));
      } else if (skipped) {
        setStatus(ctx, "全部已生成，无需重跑（删掉 UpScale 里的 PNG 可重做）", "ok");
      }
    }));

    // Fires when a layer starts, before its (slow) model reload.  Without this
    // the status line sits unchanged for minutes and looks like a hang.
    listen("stlu.layer_start", (data) => forEachPlanState((ctx) => {
      if (!ctx.running || !data || !data.name) return;
      const pos = ctx.done.size + 1;
      const total = Math.max(ctx.total || 0, pos);
      setStatus(ctx, "正在放大 " + pos + "/" + total + " · " + data.name + "（单层约数分钟，模型加载中）");
    }));

    listen("stlu.progress", (data) => {
      if (!data || !data.name) return;
      forEachPlanState((ctx) => {
        if (!ctx.running) return;
        if (!ctx.done.has(data.name)) ctx.done.add(data.name);
        ctx.total = Math.max(ctx.total, ctx.done.size);
        markRowDone(ctx, data.name);
        setStatus(ctx, "正在放大 " + ctx.done.size + "/" + Math.max(ctx.total, ctx.done.size) + " · " + data.name);
      });
    });

    listen("stlu.done", (data) => forEachPlanState((ctx) => {
      if (data && data.error) {
        setStatus(ctx, "✗ PSD 写入失败：" + String(data.error).split("\n")[0].slice(0, 120), "error");
      } else if (data && data.filename) {
        setStatus(ctx, "✓ 已生成 " + data.filename, "ok");
      } else {
        setStatus(ctx, "✓ PSD 已生成", "ok");
      }
    }));

    listen("execution_interrupted", () => forEachPlanState((ctx) => {
      ctx.running = false;
      updateRunState(ctx);
      setStatus(ctx, "已中断", "error");
    }));

    listen("execution_success", () => forEachPlanState((ctx) => {
      const wasRunning = ctx.running;
      ctx.running = false;
      updateRunState(ctx);
      if (wasRunning) {
        const m = i18n();
        const shown = (m && m.raw(ctx.statusEl)) || (ctx.statusEl && ctx.statusEl.textContent) || "";
        if (!ctx.statusEl || !/已生成|运行完成|Generated|Run finished/.test(shown)) {
          setStatus(ctx, "✓ 运行完成", "ok");
        }
        refresh(ctx);
      }
    }));

    listen("execution_error", (data) => forEachPlanState((ctx) => {
      ctx.running = false;
      updateRunState(ctx);
      const msg = (data && (data.exception_message || data.exception_type)) || "未知错误";
      setStatus(ctx, "✗ 运行失败：" + String(msg).split("\n")[0].slice(0, 160), "error");
    }));

    listen("execution_success", () => {
      // The run may have changed the reference image or the split: re-read
      // the grid so the tile cells are fresh without pressing 刷新.
      for (const entry of Array.from(gridStates)) {
        try {
          if (entry.clearCacheReset) entry.clearCacheReset();
          entry.refresh();
        } catch (e) { console.warn("[STLU] grid refresh", e); }
      }
      // The grid payload sometimes lands just after the success event; one
      // more pass makes the cells and the thumbnail match what was written.
      setTimeout(() => {
        for (const entry of Array.from(gridStates)) {
          try { entry.refresh(); } catch (e) { /* keep the first layout */ }
        }
      }, 400);
    });

    listen("stlu.cache_reset", (data) => {
      for (const entry of Array.from(gridStates)) {
        try {
          if (entry.showCacheReset) entry.showCacheReset(data);
        } catch (e) { console.warn("[STLU] cache hint", e); }
      }
    });

    listen("stlu.grid", (data) => {
      // Stage one keeps the split in memory and hands it to the select node
      // over the grid_doc link; nothing is written to a grids/ folder.
      const ref = data && data.grid ? String(data.grid) : "";
      lastGridPath = ref && !ref.startsWith("{") ? ref : "";
      lastGridDoc = data && data.grid_doc ? String(data.grid_doc) : "";
      const planId = data && data.partition && data.partition.plan_id;
      for (const entry of Array.from(gridStates)) {
        try {
          // The transparent workflow plans inside a subgraph, so this event
          // (not the node's own onExecuted) is what carries the selection
          // the backend actually used back to the panel's widgets.
          if (data && data.selection != null && entry.node) {
            setWidget(entry.node, "select", String(data.selection));
          }
          if (planId && entry.node) {
            setWidget(entry.node, "select_plan", String(planId));
          }
          entry.refresh();
        } catch (e) { console.warn("[STLU] grid refresh", e); }
      }
    });
    listen("stlu.tiles", (data) => {
      if (!data) return;
      if (data.images_dir) lastImagesDir = String(data.images_dir);
      handleTilesInfo(data);
    });
    listen("stlu.tile_images", (data) => {
      if (data && data.dir) lastImagesDir = String(data.dir);
      for (const fn of Array.from(tilesStates)) {
        try { fn(); } catch (e) { console.error("[STLU] tiles panel", e); }
      }
    });

    // --- stage four: the merge node ran in the queue ---------------------
    listen("stlu.tile_merge", (data) => {
      handleMergeInfo(data);
    });

    // --- edit-parts batch ------------------------------------------------
    listen("stlu.parts.start", (data) => forEachEditState((ctx) => {
      ctx.total = Number(data.total) || 0;
      ctx.done.clear();
      ctx.sub.textContent = "开始抠 " + ctx.total + " 个部件…";
      ctx.sub.className = "stlu-hint";
      ctx.openPsd.disabled = true;
      ctx.reveal.disabled = true;
    }));

    // The generic layer-progress event fires once per saved part.
    listen("stlu.progress", (data) => {
      if (!data || !data.name) return;
      forEachEditState((ctx) => {
        if (!ctx.done.has(data.name)) ctx.done.add(data.name);
        const total = Math.max(ctx.total, ctx.done.size);
        ctx.sub.textContent = "正在抠 " + ctx.done.size + "/" + total + " · " + data.name;
      });
    });

    listen("stlu.parts.psd", (data) => forEachEditState((ctx) => {
      const rel = (data.subfolder ? data.subfolder + "/" : "") + (data.filename || "");
      ctx.psd = rel;
      ctx.out = data.subfolder || ".";
      ctx.sub.textContent = "✓ PSD：" + (data.filename || "") +
        "（" + (data.layers || 0) + " 层）";
      ctx.sub.className = "stlu-hint";
      ctx.openPsd.disabled = false;
      ctx.reveal.disabled = false;
    }));

    // The reference image may be changed from the LoadImage node itself;
    // refresh that one row when the window regains focus.
    window.addEventListener("focus", () => forEachPlanState((ctx) => refreshRef(ctx)));
  }

  // -------------------------------------------------------------------------
  // Final preview: one labeled thumbnail per PSD layer (part)
  // -------------------------------------------------------------------------

  function setupPartsPreview(node) {
    injectCss();
    const root = el("div", "stlu-panel");
    root.style.padding = "8px";
    root.style.gap = "6px";
    // Let the content decide the height; scheduleLayout() grows the node to
    // match, so a zoomed-in list is never clipped by a fixed widget box.
    root.style.height = "auto";

    const head = el("div", "stlu-line");
    head.appendChild(el("span", "stlu-cardtitle", "结果预览"));
    const count = el("span", "stlu-tag", "…");
    head.appendChild(count);
    head.appendChild(el("span", "grow"));
    const hint = el("span", "stlu-hint", "当前输出目录为空");
    head.appendChild(hint);
    root.appendChild(head);

    const tabs = el("div", "stlu-tabs");
    tabs.style.borderBottom = "none";
    tabs.style.padding = "0";
    const MODES = [
      { key: "tiles", label: "按块" },
      { key: "parts", label: "按部位" },
    ];
    const tabButtons = {};
    for (const mode of MODES) {
      const button = el("button", "stlu-tab", mode.label);
      button.title = mode.key === "tiles"
        ? "每块一张卡：合并图 + 这块的各个部位"
        : "每个部位一张卡：合并图 + 组成它的各块副本";
      button.addEventListener("click", () => {
        state.mode = mode.key;
        syncTabs();
        if (!state.data[mode.key]) {
          groups.innerHTML = "";
          count.textContent = "…";
        }
        ensure(mode.key);
      });
      tabButtons[mode.key] = button;
      tabs.appendChild(button);
    }
    root.appendChild(tabs);

    // Lock toolbar: the row button fills a whole block / part in one go and
    // the two page buttons do every thumbnail on screen.  All three send a
    // single batch request instead of one request per picture.
    const lockBar = el("div", "stlu-line");
    lockBar.style.gap = "4px";
    lockBar.style.flexWrap = "wrap";
    const lockAll = el("button", "stlu-btn", "全锁");
    lockAll.title = "锁定当前页显示的所有拆分图";
    const unlockAll = el("button", "stlu-btn", "全解");
    unlockAll.title = "解除当前页显示的所有锁";
    for (const button of [lockAll, unlockAll]) {
      button.style.fontSize = "10px";
      button.style.padding = "2px 7px";
      lockBar.appendChild(button);
    }
    const lockHint = el("span", "stlu-hint");
    lockHint.style.whiteSpace = "normal";
    lockHint.style.fontSize = "9px";
    lockHint.style.lineHeight = "1.35";
    lockHint.style.flex = "1 1 140px";
    lockHint.style.minWidth = "0";
    lockHint.textContent = "行尾「锁整行」批量锁这一块/部位；"
      + "左上角小锁单独切换。参数变化清空目录后仍会找回。";
    lockBar.appendChild(lockHint);
    root.appendChild(lockBar);

    // Whole-canvas overview on the left, always visible (no click, no fold),
    // and a wall of small cards on the right that are all expanded at once.
    // Nothing scrolls inside the panel: the wheel does not reliably scroll a
    // DOM widget in ComfyUI, so every card stays small enough to fit.
    const body = el("div");
    body.style.display = "flex";
    body.style.gap = "7px";
    body.style.alignItems = "flex-start";
    body.style.minHeight = "0";
    root.appendChild(body);

    const overview = el("div");
    overview.style.flex = "0 0 104px";
    overview.style.display = "flex";
    overview.style.flexDirection = "column";
    overview.style.gap = "3px";
    overview.style.minWidth = "0";
    body.appendChild(overview);

    const groups = el("div");
    groups.style.flex = "1 1 auto";
    groups.style.minWidth = "0";
    groups.style.display = "flex";
    groups.style.flexDirection = "column";
    groups.style.gap = "0";
    body.appendChild(groups);

    const state = {
      mode: "tiles", path: "", maxSide: 768, data: {}, loading: {},
      // Per-tile model inputs handed over by ② (``stlu_guide`` in the node's
      // own execution payload).  Not on disk: only the run that produced
      // them can pin them to a tile row.
      guide: {},
      // {tileIndex: Set(lowercase part names)} kept in the run folder's
      // ``_locks/`` store, outside everything a parameter-change reset wipes.
      locks: {},
    };

    // Thumbnail scale, remembered in the browser so it survives reloads.
    // 100% = 56px merged picture / 40px split pictures.
    const ZOOM_KEY = "stlu_psd_preview_zoom_v1";
    const ZOOM_MIN = 0.4;
    const ZOOM_MAX = 3.0;
    try {
      const saved = parseFloat(window.localStorage.getItem(ZOOM_KEY));
      if (Number.isFinite(saved)) {
        state.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, saved));
      }
    } catch (e) { /* storage unavailable */ }
    if (!state.zoom) state.zoom = 1;

    const spacer = el("span");
    spacer.style.flex = "1";
    tabs.appendChild(spacer);
    // Bordered buttons so they read as controls, not as text links.
    const zoomOut = el("button", "stlu-btn", "−");
    zoomOut.title = "缩小合并图 / 拆分图（下限 40%）";
    const zoomLabel = el("span", "");
    zoomLabel.style.fontSize = "11px";
    zoomLabel.style.minWidth = "46px";
    zoomLabel.style.textAlign = "center";
    zoomLabel.style.padding = "2px 4px";
    zoomLabel.style.background = "#1d1d1f";
    zoomLabel.style.border = "1px solid #3a3a3c";
    zoomLabel.style.borderRadius = "4px";
    zoomLabel.style.color = "#ddd";
    const zoomIn = el("button", "stlu-btn", "+");
    zoomIn.title = "放大合并图 / 拆分图（上限 300%）";
    for (const button of [zoomOut, zoomIn]) {
      button.style.width = "26px";
      button.style.height = "24px";
      button.style.padding = "0";
      button.style.fontSize = "15px";
      button.style.fontWeight = "600";
      button.style.lineHeight = "1";
    }
    function setZoom(next) {
      state.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, Math.round(next * 10) / 10));
      zoomLabel.textContent = Math.round(state.zoom * 100) + "%";
      try { window.localStorage.setItem(ZOOM_KEY, String(state.zoom)); } catch (e) { /* ignore */ }
      render();
    }
    zoomOut.addEventListener("click", () => setZoom(state.zoom - 0.2));
    zoomIn.addEventListener("click", () => setZoom(state.zoom + 0.2));
    zoomLabel.textContent = Math.round(state.zoom * 100) + "%";
    tabs.appendChild(zoomOut);
    tabs.appendChild(zoomLabel);
    tabs.appendChild(zoomIn);

    function mergePx() { return Math.round(56 * state.zoom); }
    function splitW() { return Math.round(44 * state.zoom); }
    function splitH() { return Math.round(40 * state.zoom); }

    // ---- per-tile, per-part locks -----------------------------------------
    // The backend keeps a copy of each locked layer in <run>/_locks/, so a
    // lock survives the automatic clear that a parameter change performs.

    function normalizeLocks(raw) {
      const out = {};
      if (!raw || typeof raw !== "object") return out;
      for (const [tile, names] of Object.entries(raw)) {
        const index = Number(tile);
        if (!Number.isFinite(index)) continue;
        const set = new Set();
        for (const name of names || []) {
          const key = String(name || "").trim().toLowerCase();
          if (key) set.add(key);
        }
        if (set.size) out[String(index)] = set;
      }
      return out;
    }

    function isLocked(tile, name) {
      if (tile == null) return false;
      const set = state.locks[String(Number(tile))];
      return !!set && set.has(String(name || "").trim().toLowerCase());
    }

    function lockedInSplit(split) {
      let count = 0;
      for (const part of split || []) {
        if (isLocked(part.tile, part.name || part.layer)) count += 1;
      }
      return count;
    }

    // Every lock write -- one thumbnail, one row, or the whole page -- goes
    // through this: one optimistic DOM update, one batch POST, one render.
    async function setLocks(pairs, locked) {
      const path = state.path || declaredPath();
      if (!path) return false;
      const items = [];
      const seen = new Set();
      for (const pair of pairs || []) {
        if (!pair || pair.tile == null) continue;
        const name = String(pair.name || pair.layer || "").trim();
        if (!name) continue;
        const key = String(Number(pair.tile)) + "|" + name.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        items.push({ tile: Number(pair.tile), name, locked });
      }
      if (!items.length) return false;
      const before = {};
      for (const key of Object.keys(state.locks)) {
        before[key] = new Set(state.locks[key]);
      }
      for (const item of items) {
        const key = String(item.tile);
        const set = new Set(state.locks[key] || []);
        if (locked) set.add(item.name.toLowerCase());
        else set.delete(item.name.toLowerCase());
        if (set.size) state.locks[key] = set;
        else delete state.locks[key];
      }
      render();
      let message = "";
      let attention = false;
      try {
        const doc = await postJson("/stlu/locks", { path, items });
        if (doc && doc.tiles) state.locks = normalizeLocks(doc.tiles);
        const failed = (doc && doc.failed) || [];
        attention = failed.length > 0;
        message = failed.length
          ? "锁 " + ((doc && doc.done) || 0) + " 张，" + failed.length
            + " 张失败：" + ((failed[0] && failed[0].error) || "")
          : (locked ? "已锁 " : "已解锁 ") + items.length + " 张";
      } catch (error) {
        state.locks = before;
        attention = true;
        message = "批量锁定失败：" + ((error && error.message) || error);
      }
      render();
      // Written after render so the run summary does not overwrite it; the
      // next render puts the lock total back.
      const total = Object.values(state.locks)
        .reduce((sum, set) => sum + set.size, 0);
      lockHint.className = attention ? "stlu-hint attention" : "stlu-hint";
      lockHint.textContent = message;
      lockHint.style.color = attention || !total ? "" : "#e2b33c";
      return true;
    }

    async function toggleLock(tile, name) {
      const was = isLocked(tile, name);
      const ok = await setLocks([{ tile, name }], !was);
      if (ok) {
        lockHint.className = "stlu-hint";
        lockHint.style.color = "#e2b33c";
        lockHint.textContent = (was ? "已解锁 " : "已锁定 ") + name
          + "（块 " + String(tile).padStart(2, "0") + "）";
      }
    }

    // Every lockable thumbnail on the current page / tab.
    function visibleParts() {
      const out = [];
      for (const item of listOf(state.mode)) {
        for (const part of item.split || []) {
          if (part.tile != null && (part.name || part.layer)) out.push(part);
        }
      }
      return out;
    }

    lockAll.addEventListener("click", () => setLocks(visibleParts(), true));
    unlockAll.addEventListener("click", () => setLocks(visibleParts(), false));

    function imageUrl(meta, bust) {
      // Thumbnail filenames already hash the source mtime/size, so a stable
      // ``v`` keeps the browser cache useful; the dialog forces a fresh copy.
      const version = bust ? "&v=" + Date.now()
        : (meta.v ? "&v=" + encodeURIComponent(meta.v) : "");
      return apiUrl("/view?filename=" + encodeURIComponent(meta.filename || "") +
        "&subfolder=" + encodeURIComponent(meta.subfolder || "") +
        "&type=" + encodeURIComponent(meta.type || "temp") + version);
    }

    function showBig(meta, title) {
      const { dialog, close } = openOverlay();
      const top = el("div", "stlu-dialog-head");
      top.appendChild(el("strong", "grow", title || "部位"));
      top.appendChild(closeButton(close));
      dialog.appendChild(top);
      const box = el("div", "stlu-prevbox");
      const img = document.createElement("img");
      img.src = imageUrl(meta, true);
      box.appendChild(img);
      dialog.appendChild(box);
    }

    // The queue result carries the resolved path; remember it on the node so
    // a fresh page load (workflow reopen) can ask for the same merge again.
    function declaredPath() {
      const saved = node.properties && node.properties.stlu_last_psd;
      if (saved) return String(saved);
      return String(widgetValue(node, "psd_path", "") || "").trim();
    }

    // ①-0 参数包里的输出根目录就是 ③ 实际写盘的位置；挑结果文件夹时从
    // 这里出发，免得每个人自己再从 output/ 一层层点下去。
    function settingsOutDir() {
      const graph = app.graph;
      const nodes = (graph && graph._nodes) || [];
      const settings = nodes.find(
        (n) => n && (n.type === PIPELINE_SETTINGS || n.type === SIMPLE));
      if (!settings) return "";
      // 后端的运行目录是 out_dir/group（空 group 就落在 out_dir 本身）。
      const dir = readOutDir(settings);
      const group = String(widgetValue(settings, "group", "") || "")
        .trim().replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
      if (!dir) return "";
      return group ? dir + "/" + group : dir;
    }

    function syncTabs() {
      for (const mode of MODES) {
        tabButtons[mode.key].classList.toggle("on", mode.key === state.mode);
      }
    }
    syncTabs();
    renderOverview({});

    function tileTag(tile) {
      return tile == null ? "合并" : "块 " + String(tile).padStart(2, "0");
    }

    // One split thumbnail in the row's picture list.  Fixed 42px wide so the
    // list wraps predictably; the tiny name line stays under every picture.
    function splitCell(part, mode) {
      const meta = part.image || {};
      const title = part.name || part.layer || "";
      // The name under each split picture grows a little with the zoom but
      // stops at 12px so a 300% view does not turn into giant captions.
      const namePx = Math.min(12, Math.max(8, Math.round(8 * state.zoom)));
      const width = splitW();
      const cell = el("div");
      cell.style.position = "relative";
      cell.style.flex = "0 0 " + width + "px";
      cell.style.width = width + "px";
      cell.style.cursor = "zoom-in";

      const img = document.createElement("img");
      img.src = imageUrl(meta);
      img.alt = title;
      img.title = title + (part.size ? " · " + part.size.join("×") : "")
        + (mode === "parts" ? " · " + tileTag(part.tile) : "");
      img.style.display = "block";
      img.style.width = "100%";
      img.style.height = splitH() + "px";
      img.style.objectFit = "contain";
      img.style.borderRadius = "3px";
      img.style.border = "1px solid #2a2a2c";
      img.style.background =
        "repeating-conic-gradient(#242426 0% 25%,#1a1a1c 0% 50%) 50%/10px 10px";
      cell.appendChild(img);

      // Lock toggle: keep exactly this tile's copy of this part on the next
      // run.  The backend snapshots it into _locks/, which a reset does not
      // clear, so the lock still works after a parameter change.
      const locked = isLocked(part.tile, title);
      if (locked) img.style.border = "1px solid #e2b33c";
      const lockSize = Math.max(14, Math.min(22, Math.round(14 * state.zoom)));
      const lock = el("button", "", "锁");
      lock.title = part.tile == null
        ? "这一份是跨块合并出来的，不能单独锁定；请切到「按块」锁定"
        : (locked ? "已锁定：重跑这一块时保留这一张（清空后也会找回）"
                  : "锁定：重跑这一块时保留这一张（清空后也会找回）");
      lock.disabled = part.tile == null;
      lock.style.position = "absolute";
      lock.style.left = "1px";
      lock.style.top = "1px";
      lock.style.width = lockSize + "px";
      lock.style.height = lockSize + "px";
      lock.style.padding = "0";
      lock.style.fontSize = Math.max(8, Math.round(lockSize * 0.55)) + "px";
      lock.style.lineHeight = "1";
      lock.style.borderRadius = "3px";
      lock.style.cursor = part.tile == null ? "default" : "pointer";
      lock.style.border = "1px solid " + (locked ? "#e2b33c" : "#3a3a3c");
      lock.style.background = locked ? "rgba(226,179,60,.92)" : "rgba(20,20,22,.78)";
      lock.style.color = locked ? "#161616" : (part.tile == null ? "#777" : "#cfcfd3");
      lock.style.opacity = part.tile == null ? ".55" : "1";
      lock.addEventListener("click", (event) => {
        event.stopPropagation();
        if (part.tile == null) return;
        toggleLock(part.tile, title);
      });
      cell.appendChild(lock);

      if (mode === "parts") {
        const badge = el("span", "",
          part.tile == null ? "合" : String(part.tile).padStart(2, "0"));
        badge.title = part.tile == null ? "跨块合并成的一层" : "来自块 " + part.tile;
        badge.style.position = "absolute";
        badge.style.right = "1px";
        badge.style.bottom = Math.round(namePx * 1.2) + "px";
        badge.style.padding = "0 2px";
        badge.style.borderRadius = "2px";
        badge.style.fontSize = "8px";
        badge.style.lineHeight = "1.5";
        badge.style.background = part.tile == null ? "rgba(0,0,0,.7)" : "rgba(45,108,223,.85)";
        badge.style.color = part.tile == null ? "#bbb" : "#fff";
        badge.style.pointerEvents = "none";
        cell.appendChild(badge);
      }

      const name = el("span", "stlu-hint", title);
      name.style.display = "block";
      name.style.fontSize = namePx + "px";
      name.style.lineHeight = "1.2";
      name.style.width = "100%";
      name.style.textAlign = "center";
      name.title = title;
      cell.appendChild(name);

      cell.addEventListener("click", () => showBig(meta, title));
      return cell;
    }

    // One list row: name column, then the merged picture, then the picture
    // list of this block's parts.  Compact enough to read like a table.
    function groupRow(item, mode) {
      const title = item.label || item.name || "";
      const split = item.split || [];
      const row = el("div");
      row.style.display = "flex";
      row.style.alignItems = "flex-start";
      row.style.gap = "7px";
      row.style.padding = "4px 4px 5px";
      row.style.borderBottom = "1px solid #262628";
      row.style.minWidth = "0";

      const who = el("div");
      who.style.flex = "0 0 84px";
      who.style.minWidth = "0";
      const name = el("div", "stlu-cardtitle", title);
      name.style.fontSize = "11px";
      name.style.whiteSpace = "nowrap";
      name.style.overflow = "hidden";
      name.style.textOverflow = "ellipsis";
      name.title = title;
      who.appendChild(name);
      const locksHere = lockedInSplit(split);
      const sub = el("div", "stlu-hint", (mode === "tiles"
        ? split.length + " 部位"
        : (item.count || 1) + " 份" + (item.size ? " · " + item.size.join("×") : ""))
        + (locksHere ? " · 锁 " + locksHere : ""));
      sub.style.fontSize = "9px";
      if (locksHere) sub.style.color = "#e2b33c";
      who.appendChild(sub);

      // Row-level batch lock: one click fills every thumbnail in this row.
      // A partly locked row fills the rest; a fully locked row releases all.
      const lockable = split.filter(
        (part) => part.tile != null && (part.name || part.layer));
      const lockedHere = lockable.filter(
        (part) => isLocked(part.tile, part.name || part.layer)).length;
      const rowLock = el("button", "stlu-btn",
        !lockable.length ? "锁整行"
          : lockedHere === lockable.length ? "解锁整行"
          : lockedHere ? "锁 " + lockedHere + "/" + lockable.length
          : "锁整行");
      rowLock.title = mode === "tiles"
        ? "一次锁定 / 解锁这一块的所有拆分图"
        : "一次锁定 / 解锁这个部位在所有块里的副本";
      rowLock.disabled = !lockable.length;
      rowLock.style.marginTop = "2px";
      rowLock.style.fontSize = "9px";
      rowLock.style.padding = "1px 5px";
      rowLock.style.whiteSpace = "nowrap";
      if (lockable.length && lockedHere === lockable.length) {
        rowLock.style.borderColor = "#e2b33c";
        rowLock.style.color = "#e2b33c";
      }
      rowLock.addEventListener("click", (event) => {
        event.stopPropagation();
        setLocks(lockable, lockedHere !== lockable.length);
      });
      who.appendChild(rowLock);
      row.appendChild(who);

      // Merged group picture: blue frame + "合" badge, distinct from splits.
      // The model input (green frame) stacks underneath it, not beside it --
      // both are square so a column keeps the row one picture wide.
      const merge = mergePx();
      const pair = el("div");
      pair.style.display = "flex";
      pair.style.flexDirection = "column";
      pair.style.alignItems = "center";
      pair.style.gap = "3px";
      pair.style.flex = "0 0 auto";
      const frame = el("div");
      frame.style.position = "relative";
      frame.style.flex = "0 0 " + merge + "px";
      frame.style.width = merge + "px";
      frame.style.height = merge + "px";
      frame.style.border = "1px solid rgba(45,108,223,.55)";
      frame.style.borderRadius = "4px";
      frame.style.background = "#101013";
      frame.style.padding = "2px";
      frame.style.boxSizing = "border-box";
      frame.style.cursor = "zoom-in";
      const big = document.createElement("img");
      big.src = item.thumb ? apiUrl(item.thumb) : imageUrl(item.image || {});
      big.alt = title;
      big.title = title + " · 合并";
      big.style.display = "block";
      big.style.width = "100%";
      big.style.height = "100%";
      big.style.objectFit = "contain";
      frame.appendChild(big);
      const chip = el("span", "", "合");
      chip.style.position = "absolute";
      chip.style.right = "1px";
      chip.style.bottom = "1px";
      chip.style.fontSize = "8px";
      chip.style.lineHeight = "1.5";
      chip.style.padding = "0 2px";
      chip.style.borderRadius = "2px";
      chip.style.background = "rgba(45,108,223,.85)";
      chip.style.color = "#fff";
      chip.style.pointerEvents = "none";
      frame.appendChild(chip);
      frame.addEventListener("click", () => showBig(item.image || {}, title + " · 合并"));
      pair.appendChild(frame);

      // What the models actually saw for this tile: the crop after guide
      // backfill and context padding.  It rides in on the node's own
      // execution payload (temp thumbnails), so it only exists for tiles the
      // current run really generated -- an empty dashed cell means "did not
      // run this tile" (unchecked, or reused from an earlier run).
      // Per-tile only: the "by part" tab has no tile to pin it to.
      if (mode === "tiles") {
        // This run's own payload wins (it is pinned to the tile by ②); after a
        // browser refresh the payload is gone, so fall back to the copy the
        // backend keeps next to the run in ``_guide/``.
        const guided = state.guide[String(Number(item.index))];
        const live = guided && guided.image ? guided.image : null;
        const disk = !live && item.guide ? item.guide : null;
        const side = mergePx();
        const guide = el("div");
        guide.style.position = "relative";
        guide.style.flex = "0 0 " + side + "px";
        guide.style.width = side + "px";
        guide.style.height = side + "px";
        guide.style.boxSizing = "border-box";
        guide.style.borderRadius = "4px";
        if (live || disk) {
          guide.style.border = "1px solid rgba(86,178,107,.65)";
          guide.style.background = "#101013";
          guide.style.padding = "2px";
          guide.style.cursor = "zoom-in";
          const img = document.createElement("img");
          img.src = live ? imageUrl(live)
            : (item.guide_thumb ? apiUrl(item.guide_thumb) : imageUrl(disk));
          img.alt = title + " · 实际输入";
          img.title = title + " · 引导回填后送进模型的输入 · 点击放大";
          img.style.display = "block";
          img.style.width = "100%";
          img.style.height = "100%";
          img.style.objectFit = "contain";
          guide.appendChild(img);
          const chip = el("span", "", "入");
          chip.style.position = "absolute";
          chip.style.right = "1px";
          chip.style.bottom = "1px";
          chip.style.fontSize = "8px";
          chip.style.lineHeight = "1.5";
          chip.style.padding = "0 2px";
          chip.style.borderRadius = "2px";
          chip.style.background = "rgba(86,178,107,.9)";
          chip.style.color = "#fff";
          chip.style.pointerEvents = "none";
          guide.appendChild(chip);
          guide.addEventListener("click", () => showBig(live || disk,
            title + " · 实际输入（引导回填后）"));
        } else {
          guide.style.border = "1px dashed #3a3a3c";
          guide.style.display = "flex";
          guide.style.alignItems = "center";
          guide.style.justifyContent = "center";
          guide.style.color = "#555";
          guide.style.fontSize = "10px";
          guide.title = "这次没有生成这块（未勾选，或用了上次的结果）";
          guide.textContent = "—";
        }
        pair.appendChild(guide);
      }
      row.appendChild(pair);

      const strip = el("div");
      strip.style.flex = "1 1 auto";
      strip.style.minWidth = "0";
      strip.style.display = "flex";
      strip.style.flexWrap = "wrap";
      strip.style.gap = "3px";
      if (!split.length) {
        strip.appendChild(el("span", "stlu-hint", "无可拆分图层"));
      } else {
        for (const part of split) strip.appendChild(splitCell(part, mode));
      }
      row.appendChild(strip);
      return row;
    }

    // Always-on overview: the merged canvas sits in its own left column so it
    // never competes with the cards and never costs a click to see.
    function renderOverview(data) {
      overview.innerHTML = "";
      const main = data.main;
      const can = !!(main && main.filename);
      overview.appendChild(el("span", "stlu-cardtitle", "总览"));
      if (!can) {
        // 面板会自动读取 ① 的输出目录；没有结果就只是空着，不提供手动挑目录。
        overview.appendChild(el("span", "stlu-hint", "当前输出目录为空"));
        return;
      }
      const frame = el("div");
      frame.style.border = "1px solid rgba(45,108,223,.55)";
      frame.style.borderRadius = "5px";
      frame.style.background = "#101013";
      frame.style.padding = "2px";
      frame.style.cursor = "zoom-in";
      const img = document.createElement("img");
      img.src = data.main_thumb ? apiUrl(data.main_thumb) : imageUrl(main);
      img.alt = "合并 PSD";
      img.title = "合并 PSD · 点击放大";
      img.style.display = "block";
      img.style.width = "100%";
      img.style.maxHeight = "248px";
      img.style.objectFit = "contain";
      frame.appendChild(img);
      frame.addEventListener("click", () => showBig(main, "合并 PSD · 总览"));
      overview.appendChild(frame);
      if (data.canvas) {
        const size = el("span", "stlu-hint", data.canvas.join("×"));
        size.style.fontSize = "10px";
        overview.appendChild(size);
      }
    }

    function listOf(mode) {
      const data = state.data[mode] || {};
      return mode === "tiles" ? (data.tiles || []) : (data.parts || []);
    }

    function syncCounts() {
      for (const mode of MODES) {
        const n = listOf(mode.key).length;
        tabButtons[mode.key].textContent = mode.label + (n ? " " + n : "");
      }
    }

    // Re-measure after a render so the node height follows the list.  The
    // panel itself has no scrollbars, so this is what keeps zoomed content
    // visible instead of cutting it off at the widget's minimum height.
    let layoutFrame = null;
    function scheduleLayout() {
      if (layoutFrame) return;
      layoutFrame = requestAnimationFrame(() => {
        layoutFrame = null;
        if (!node.graph || !node.size) return;
        try {
          const want = node.computeSize()[1];
          if (Math.abs(want - node.size[1]) > 4) {
            node.setSize([node.size[0], want]);
            node.setDirtyCanvas(true, true);
          }
        } catch (e) { /* layout not ready yet */ }
      });
    }

    function render() {
      scheduleLayout();
      const data = state.data[state.mode] || {};
      const items = listOf(state.mode);
      const noun = state.mode === "tiles" ? "块" : "部位";
      count.textContent = items.length ? items.length + " 个" + noun
        : (state.data[state.mode] ? "0 个" + noun : "…");
      count.className = "stlu-tag" + (items.length ? " ok" : "");
      hint.className = "stlu-hint";
      hint.textContent = data.file
        ? "合并 PSD：" + data.file + (data.canvas
            ? " · " + data.canvas.join("×") : "")
        : "当前输出目录为空";
      syncCounts();
      const lockTotal = Object.values(state.locks)
        .reduce((total, set) => total + set.size, 0);
      const lockableCount = visibleParts().length;
      lockAll.disabled = !lockableCount;
      unlockAll.disabled = !lockableCount;
      lockHint.className = "stlu-hint";
      lockHint.textContent = lockTotal
        ? "已锁定 " + lockTotal + " 张：重跑保留，参数变化清空目录后也会找回。"
          + "行尾「锁整行」可批量切换。"
        : "行尾「锁整行」批量锁这一块/部位；左上角小锁单独切换。"
          + "参数变化清空目录后仍会找回。";
      lockHint.style.color = lockTotal ? "#e2b33c" : "";
      renderOverview(data);
      groups.innerHTML = "";
      if (!items.length) {
        groups.appendChild(el("div", "stlu-hint",
          state.mode === "tiles"
            ? "没有分块（这不是分块运行文件夹）"
            : "没有可预览的部位"));
        return;
      }
      // Column header: the row reads as name | merged | split parts.
      const headRow = el("div");
      headRow.style.display = "flex";
      headRow.style.gap = "7px";
      headRow.style.padding = "0 4px 3px";
      headRow.style.borderBottom = "1px solid #2f2f31";
      const hName = el("span", "stlu-hint", "名称");
      hName.style.flex = "0 0 84px";
      hName.style.fontSize = "9px";
      headRow.appendChild(hName);
      const hMerge = el("span", "stlu-hint", "合并");
      hMerge.style.flex = "0 0 " + mergePx() + "px";
      hMerge.style.fontSize = "9px";
      hMerge.style.textAlign = "center";
      headRow.appendChild(hMerge);
      if (state.mode === "tiles") {
        const hGuide = el("span", "stlu-hint", "输入");
        hGuide.style.flex = "0 0 " + mergePx() + "px";
        hGuide.style.fontSize = "9px";
        hGuide.style.textAlign = "center";
        hGuide.title = "引导回填后真正送进模型的那张图";
        headRow.appendChild(hGuide);
      }
      const hSplit = el("span", "stlu-hint", "拆分图层");
      hSplit.style.flex = "1";
      hSplit.style.fontSize = "9px";
      headRow.appendChild(hSplit);
      groups.appendChild(headRow);
      for (const item of items) groups.appendChild(groupRow(item, state.mode));
    }

    async function ensure(mode, force) {
      const recorded = state.path || declaredPath();
      if (!recorded && state.missed) return;
      // ⑤ 没有自己的目录参数：没记过路径时先去 ① 的输出目录里找（运行目录
      // 就是 out_dir/group），找到就直接显示，找不到才回到空状态让用户手动挑。
      const path = recorded || settingsOutDir();
      const guessed = !recorded && !!path;
      if (!path) {
        hint.textContent = "等待 ④ 的结果（连上并运行一次即可）";
        render();
        return;
      }
      if (!force && state.data[mode]) {
        render();
        return;
      }
      if (state.loading[mode]) return;
      state.loading[mode] = true;
      hint.className = "stlu-hint";
      hint.textContent = "读取" + (mode === "tiles" ? "分块" : "部位") + "…";
      try {
        const query = new URLSearchParams({
          path: path,
          mode: mode,
          max_side: String(state.maxSide || 768),
        });
        const [data, lockDoc] = await Promise.all([
          getJson("/stlu/psd_preview?" + query.toString()),
          getJson("/stlu/locks?path=" + encodeURIComponent(path)).catch(() => null),
        ]);
        state.path = path;
        state.missed = false;
        state.data[mode] = data;
        if (lockDoc) state.locks = normalizeLocks(lockDoc.tiles);
        syncCounts();
        if (mode === state.mode) render();
      } catch (error) {
        if (guessed) {
          // ① 的目录里没有合并结果：保持空状态，露出“浏览其他文件夹”按钮。
          state.path = "";
          state.missed = true;
          delete state.data[mode];
          render();
        } else {
          hint.textContent = String((error && error.message) || error);
          hint.className = "stlu-hint attention";
        }
      } finally {
        state.loading[mode] = false;
      }
    }

    const oldConfigure = node.onConfigure;
    node.onConfigure = function () {
      const result = oldConfigure ? oldConfigure.apply(this, arguments) : undefined;
      try {
        // ComfyUI restores node.properties before this runs, so the last
        // merged PSD is shown immediately when a saved workflow is opened.
        state.path = declaredPath();
        state.data = {};
        state.locks = {};
        state.guide = {};
        state.missed = false;
        if (state.path) ensure(state.mode, true);
        // 没有记录路径时延后一帧再猜：① 的 widget 值可能还没恢复完。
        else setTimeout(() => ensure(state.mode, true), 0);
      } catch (e) {
        console.warn("[STLU] PSD preview restore", e);
      }
      return result;
    };

    const oldExecuted = node.onExecuted;
    node.onExecuted = function (message) {
      const result = oldExecuted ? oldExecuted.apply(this, arguments) : undefined;
      try {
        // ②'s per-tile model inputs arrive as temp thumbnails keyed by grid
        // index.  Replace (never merge) so a run that generated nothing can
        // not leave the previous run's pictures pinned to the rows.
        const guided = message && message.stlu_guide;
        if (Array.isArray(guided)) {
          state.guide = {};
          for (const entry of guided) {
            if (!entry || entry.index == null || !entry.image) continue;
            state.guide[String(Number(entry.index))] = entry;
          }
        }
        const payload = message && message.stlu_parts;
        const path = payload && payload.length && payload[0].path;
        if (path) {
          if (!node.properties) node.properties = {};
          node.properties.stlu_last_psd = String(path);
          state.path = String(path);
          state.missed = false;
          state.data = {};
          state.locks = {};
          ensure(state.mode, true);
        }
      } catch (e) {
        console.error("[STLU] PSD preview", e);
      }
      return result;
    };

    // Dynamic minimum height: computeSize() asks for this after every
    // render, so the widget box matches the rendered list.
    const measureHeight = () => Math.max(360, root.scrollHeight + 2);
    addDOM(node, "stlu_parts_card", root, measureHeight, 3000);

    // The card wall only stays short if it gets horizontal room.  Widen a
    // fresh/undersized node, never shrink one the user already sized.
    if (node.size && node.size[0] < 560) node.setSize([560, node.size[1]]);

    const initial = declaredPath();
    if (initial) {
      state.path = initial;
      ensure(state.mode, true);
    } else {
      // 新建节点时 ① 可能还没进图，延后一帧再按 ① 的输出目录猜一次。
      setTimeout(() => ensure(state.mode, true), 0);
    }
  }

  // -------------------------------------------------------------------------
  // extension registration
  // -------------------------------------------------------------------------

  app.registerExtension({
    name: "STLU.PlanUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
      const LOCKABLE = new Set([
        "SeeThrough_LoadLayerDiffModel", "SeeThrough_LoadDepthModel",
        "SeeThrough_GenerateLayers", "SeeThrough_GenerateDepth",
        "SeeThrough_PostProcess", TILE_PSD, WAVE_INDEX, TILE_MERGE,
        "STLU_GuidedLayers",
      ]);
      if (LOCKABLE.has(nodeData.name)) {
        const lockCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
          const result = lockCreated ? lockCreated.apply(this, arguments) : undefined;
          applyLockedWidgets(this);
          return result;
        };
        const lockConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
          const result = lockConfigure ? lockConfigure.apply(this, arguments) : undefined;
          applyLockedWidgets(this);
          setTimeout(() => applyLockedWidgets(this), 0);
          return result;
        };
      }

      if (nodeData.name === PIPELINE || nodeData.name === PIPELINE_SETTINGS
          || nodeData.name === SIMPLE) {
        const hideCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
          const result = hideCreated ? hideCreated.apply(this, arguments) : undefined;
          applyHiddenWidgets(this);
          return result;
        };
        const hideConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
          const result = hideConfigure ? hideConfigure.apply(this, arguments) : undefined;
          applyHiddenWidgets(this);
          setTimeout(() => applyHiddenWidgets(this), 0);
          return result;
        };
      }

      if (nodeData.name !== PLAN && nodeData.name !== BATCH
          && nodeData.name !== ASSEMBLE && nodeData.name !== EDIT_BATCH
          && nodeData.name !== TILES && nodeData.name !== TILE_SELECT
          && nodeData.name !== TILE_MERGE && nodeData.name !== TILE_GRID
          && nodeData.name !== PIPELINE
          && nodeData.name !== PIPELINE_SETTINGS
          && nodeData.name !== SIMPLE
          && nodeData.name !== SELECT_TILES && nodeData.name !== REBUILD_PSD
          && nodeData.name !== SAVE_TILES && nodeData.name !== WAVE
          && nodeData.name !== TILE_PSD && nodeData.name !== WAVE_INDEX
          && nodeData.name !== PSD_PREVIEW) return;
      bindEvents();
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = onCreated ? onCreated.apply(this, arguments) : undefined;
        try {
          const m = i18n();
          if (m && m.enabled) m.labelCombos(this);
          if (nodeData.name === PLAN) setupPlan(this);
          else if (nodeData.name === BATCH) setupBatch(this);
          else if (nodeData.name === EDIT_BATCH) setupEditBatch(this);
          else if (nodeData.name === TILES) setupTiles(this);
          else if (nodeData.name === TILE_SELECT) setupTileSelect(this);
          else if (nodeData.name === TILE_MERGE) setupTileMerge(this);
          else if (nodeData.name === TILE_GRID) setupTileGrid(this);
          else if (nodeData.name === PIPELINE
            || nodeData.name === PIPELINE_SETTINGS
            || nodeData.name === SIMPLE) setupTileGrid(this);
          else if (nodeData.name === SELECT_TILES) setupSelectTiles(this);
          else if (nodeData.name === REBUILD_PSD) setupRebuildPSD(this);
          else if (nodeData.name === SAVE_TILES) setupSaveTiles(this);
          else if (nodeData.name === WAVE) setupWave(this);
          else if (nodeData.name === TILE_PSD) setupTilePSD(this);
          else if (nodeData.name === WAVE_INDEX) setupWaveIndex(this);
          else if (nodeData.name === PSD_PREVIEW) setupPartsPreview(this);
          else setupAssemble(this);
        } catch (e) {
          console.error("[STLU] failed to build UI for " + nodeData.name, e);
        }
        return result;
      };
    },
  });
})();
