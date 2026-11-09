// One-off simulation of the per-tile tag editor logic.  It pulls the real
// functions out of web/stlu.js and runs them in a vm with DOM stubs, so the
// selection / store / hidden-tag paths are exercised without a browser.
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import assert from "node:assert/strict";

const root = path.resolve(import.meta.dirname, "..");
const src = fs.readFileSync(path.join(root, "web/stlu.js"), "utf8");

function grab(start, end) {
  const a = src.indexOf(start);
  const b = a < 0 ? -1 : src.indexOf(end, a + start.length);
  if (a < 0 || b < 0) throw new Error("cannot slice: " + start);
  return src.slice(a, b);
}

const parts = [
  grab("const TILE_TAG_HEAD = [", "const BAND_MIN_OVERLAP = 0.12;")
    + "const BAND_MIN_OVERLAP = 0.12;",
  grab("function readTileTagsFromWidget()", "\n    // --- proportional auto tags"),
  grab("function autoBands()", "    function buildTileTagEditor(container) {"),
  grab("      function applySelection(next, anchor) {", "      const pickAll ="),
  grab("      function applyToTiles(sel, mutate) {", "      function groupBlock(card) {"),
  grab("  function overlapArea(a, b) {", "\n  // Map a selection from one geometry"),
  grab("    function migrateTileTags(", "    function layout() {"),
];

const ctx = vm.createContext({
  console,
  writes: [],
  node: {},
  widgetValue: () => "{}",
  setWidget: (n, k, v) => { ctx.writes.push(v); return true; },
  syncTileTagUi() {},
  drawCanvas() {},
  total: 6,
  srcW: 200,
  srcH: 300,
  planState: null,
  coords: [
    [0, 0, 100, 100], [100, 0, 200, 100],
    [0, 100, 100, 200], [100, 100, 200, 200],
    [0, 200, 100, 300], [100, 200, 200, 300],
  ],
});

vm.runInContext(`
  let tileTags = {};
  let tagEditSel = new Set([0]);
  let tagEditIndex = 0;
  let tileTagKey = "";
  let planAppliedKey = "planA";
  let tagWriteHint = { style: {}, textContent: "" };
`, ctx);
vm.runInContext(parts.join("\n"), ctx);

const run = (code) => vm.runInContext(code, ctx);
let passed = 0;
function ok(cond, msg) {
  assert.ok(cond, msg);
  passed += 1;
}
const sel = () => Array.from(run("tagEditSel")).sort((a, b) => a - b);
const anchor = () => run("tagEditIndex");
const entry = (i) => run(`tileTags[${JSON.stringify(String(i))}]`);
const setsOf = (i) => run(`(() => {
  const s = tileTagSets(${i});
  return { body: Array.from(s.body).sort(), head: Array.from(s.head).sort() };
})()`);

// ---- selection / anchor -------------------------------------------------
run("applySelection([0,1,2,3,4,5], 0)");
ok(sel().join() === "0,1,2,3,4,5" && anchor() === 0, "select all keeps anchor 0");

run("applySelection([3], 3)");
ok(sel().join() === "3" && anchor() === 3, "pick one moves anchor");

run("applySelection([1,3,4], 3)");
ok(anchor() === 3, "anchor inside selection is kept");

// Ctrl-click on the anchor itself: it leaves the set, anchor must follow.
run("applySelection([1,4], 3)");
ok(anchor() === 1, "anchor falls back to min when removed");

// Shift-range always keeps a visible anchor.
run("applySelection([3,4,5], 2)");
ok(anchor() === 3 && sel().join() === "3,4,5", "range anchors on a selected tile");

run("applySelection([], null)");
ok(sel().length === 1 && run("tagEditSel.has(tagEditIndex)"), "empty selection keeps anchor");

run("applySelection([-1, 99], null)");
ok(run("tagEditSel.has(tagEditIndex)"), "out-of-range indexes are dropped");

// ---- single tile editing ------------------------------------------------
run("tileTags = {}; applyToTiles([5], (body) => body.add('head'))");
ok(entry(5) && entry(5).body.indexOf("head") >= 0, "adding a non-auto tag stores entry");
ok(run("tileIsCustomized(5)"), "tile with an override is customized");

run("applyToTiles([5], (body) => body.delete('head'))");
ok(entry(5) === undefined, "matching auto again removes the entry");
ok(!run("tileIsCustomized(5)"), "auto-matching tile is not customized");

// A tag auto already has must not create an entry.
run("tileTags = {}; applyToTiles([5], (body) => body.add('bottomwear'))");
ok(entry(5) === undefined, "auto tag click does not write an entry");

// ---- group buttons ------------------------------------------------------
run("tileTags = {}; applyToTiles([5], (body) => body.clear())");
ok(entry(5) && entry(5).body.length === 0, "group clear stores an empty whitelist");
ok(setsOf(5).body.length === 0, "cleared group shows empty");

run("applyToTiles([5], (body, head, i) => { body.clear(); autoSets(i).body.forEach((t) => body.add(t)); })");
ok(entry(5) === undefined, "restore-auto drops the group again");

// ---- all groups on/off --------------------------------------------------
run("tileTags = {}; applyToTiles([0,1], (body, head) => { TILE_TAG_BODY.forEach((t) => body.add(t)); TILE_TAG_HEAD.forEach((t) => head.add(t)); })");
ok(entry(0).body.length === 13 && setsOf(0).head.length === 11,
  "all-groups-on fills both groups (head omitted when it equals auto)");
ok(run("tileIsCustomized(0)"), "tile 0 (head/upper bands) counts as customized after fill");
run("applyToTiles([0,1], (body, head) => { body.clear(); head.clear(); })");
ok(entry(0).body.length === 0 && entry(0).head.length === 0, "all-groups-off clears both");

// ---- hidden tags --------------------------------------------------------
run("tileTags = { '5': { body: ['hair', 'head'] } }");
run("applyToTiles([5], (body) => body.clear())");
ok(entry(5).body.indexOf("hair") >= 0, "group clear keeps hidden v2 tag");
ok(entry(5).body.indexOf("head") < 0, "group clear still removes visible tag");

run("tileTags = { '5': { body: ['hair', 'head'], head: ['eyes'] } }");
run("autoFillAll()");
ok(entry(5).body.indexOf("hair") >= 0, "auto-fill keeps hidden body tag");
ok(entry(5).head.indexOf("eyes") >= 0, "auto-fill keeps hidden head tag");
ok(entry(5).body.indexOf("bottomwear") >= 0, "auto-fill writes proportional body tags");

// ---- one card must not touch the other cards sharing its group ------------
// hair / upper / lower / other all live in the "body" store, so a card
// button may only walk its own tag list.
run("tileTags = { '5': { body: ['back hair', 'topwear', 'bottomwear'] } }");
run("applyToTiles([5], (body) => cardTagChange(body, 5, 'body', ['front hair', 'back hair', 'head'], 'none'))");
ok(entry(5).body.join() === "topwear,bottomwear",
  "clearing the hair card leaves the upper/lower cards' tags alone");

run("tileTags = { '5': { body: ['wings'], head: ['face'] } }");
run("applyToTiles([5], (body) => cardTagChange(body, 5, 'body', ['neck', 'neckwear', 'topwear', 'handwear'], 'all'))");
ok(entry(5).body.indexOf("neck") >= 0 && entry(5).body.indexOf("wings") >= 0,
  "selecting all of the upper card keeps other cards' tags");
ok(entry(5).head.join() === "face", "touching a body card leaves the head group alone");

run("tileTags = { '5': { body: ['topwear', 'wings'], head: ['face', 'mouth'] } }");
run("applyToTiles([5], (body, head) => cardTagChange(head, 5, 'head', TILE_TAG_HEAD, 'none'))");
ok(entry(5).body.join() === "topwear,wings", "clearing the head card keeps the body tags");
ok(!(entry(5).head && entry(5).head.length),
  "clearing the head card clears only its own tags");

run("tileTags = {}; applyToTiles([5], (body) => cardTagChange(body, 5, 'body', ['neck', 'neckwear', 'topwear', 'handwear'], 'auto'))");
ok(entry(5) === undefined,
  "restoring the upper card to auto writes nothing when it already matches auto");

// Tile 5 sits in the lower band: auto body = bottomwear/legwear/footwear.
run("tileTags = { '5': { body: ['topwear', 'wings'] } }");
run("applyToTiles([5], (body) => cardTagChange(body, 5, 'body', ['bottomwear', 'legwear', 'footwear'], 'auto'))");
ok(entry(5).body.join() === "topwear,bottomwear,legwear,footwear,wings",
  "restoring one card to auto adds only that card's automatic tags");

// ---- tags survive a rebuilt grid ----------------------------------------
const BOX = [
  [0, 0, 100, 100], [100, 0, 200, 100],
  [0, 100, 100, 200], [100, 100, 200, 200],
  [0, 200, 100, 300], [100, 200, 200, 300],
];
const migrate = (a, b, tags) => run(
  `migrateTileTags(${JSON.stringify(a)}, ${JSON.stringify(b)}, ${JSON.stringify(tags)})`);
const TUNED = { "3": { body: ["topwear"], head: ["face", "mouth"] } };

ok(JSON.stringify(migrate(BOX, BOX, TUNED)) === JSON.stringify(TUNED),
  "identical boxes keep the tuned tags verbatim");
ok(Object.keys(migrate(BOX, BOX, TUNED)).length === 1,
  "auto tiles stay automatic after a rebuild");
const nudged = BOX.map(([x0, y0, x1, y1]) => [x0 + 10, y0 + 10, x1 + 10, y1 + 10]);
ok(migrate(BOX, nudged, TUNED)["3"].head.join() === "face,mouth",
  "a nudged grid carries the tuned tags by overlap");
ok(Object.keys(migrate(BOX, [[400, 400, 500, 500]], TUNED)).length === 0,
  "a grid that barely overlaps carries nothing (falls back to auto)");
ok(JSON.stringify(migrate([], BOX, TUNED)) === "{}",
  "no old geometry means no migration");

// ---- widget read-back ---------------------------------------------------
ctx.widgetValue = () => JSON.stringify({
  _key: "planA",
  tiles: { "2": { body: ["hair", "topwear", "wings"], head: ["eyes", "face"] } },
});
const parsed = run("readTileTagsFromWidget()");
ok(parsed.tiles["2"].body.join() === "topwear,wings,hair", "known tags first, hidden tags kept");
ok(parsed.tiles["2"].head.join() === "face,eyes", "head group keeps hidden tag too");

// ---- round trip ---------------------------------------------------------
ctx.writes.length = 0;
run("tileTags = {}; applyToTiles([4], (body) => body.add('topwear'))");
ok(ctx.writes.length > 0, "an edit writes the widget");
const saved = JSON.parse(ctx.writes[ctx.writes.length - 1]);
ok(saved._key === "planA" && saved.tiles["4"].body.indexOf("topwear") >= 0, "payload carries the change");

console.log("tag editor simulation: " + passed + " checks passed");
