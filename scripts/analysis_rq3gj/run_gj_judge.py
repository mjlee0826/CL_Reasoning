"""
run_gj_judge.py — RQ3-GJ 的 Judge 呼叫、預測與開跑前檢查（result/analysis/rq3gj/rq3gj_criteria.md）

判定標準確認前一律拒跑。五個步驟依序執行，每一步都要上一步的結果檔存在、寫於同一份判定標準之下且通過；
呼叫的步驟都可以中斷後用同一個指令續跑（已寫入的題目會跳過，不重複計費；API 失敗的題目不寫入，重跑時補上）：

    check    §10 流程核對：重跑 RQ1-KJ 的 qwen、gpt4omini × truthfulqa × M3S（舊紀錄的順序）-> precheck/、precheck.json
    pilot    §11 試跑：gpt4omini × mmlu × GS-105 的前 100 題（7 個版本）與 × EN+ZH 的前 100 題（5 個版本 × 2 種順序），
             直接寫進正式的 judge_outputs/、judge_outputs_k2/（每筆標 pilot: true）-> pilot.json
    predict  §7、§8.12、§8.13 的預測（離線；只用 RQ1-KJ 的紀錄、答案與事先決定的順序）-> 預測檔與 manifest；已存在就拒跑
    prerun   §12 正式跑之前的五項檢查（離線）-> prerun.json
    full     正式跑：8 個區塊 × 39 份菜單 × 7 個版本，加上 K = 2 的 3 個配對 × 5 個版本 × 2 種順序 -> judge_outputs/、judge_outputs_k2/

每次呼叫用 Strategy/SubstitutionJudge.py：宿主自己當裁判（T = 0、max_tokens 8192、不傳 seed；Qwen 關 thinking），
K = 3 用 choice-k-v1（Aggregator/MenuJudgeAggregator.py），K = 2 用 choice-v1（PairChoiceJudgeAggregator）；
被換進來的那條候選讀供體的 arm 檔；每筆紀錄記下順序、供體的位置、供應商回傳的模型版本、呼叫時間（UTC）、API token 與 prompt 的 sha256。

用法（從 repo root，conda 環境 clreasoning）：
    python scripts/analysis_rq3gj/run_gj_judge.py check
    python scripts/analysis_rq3gj/run_gj_judge.py pilot
    python scripts/analysis_rq3gj/run_gj_judge.py predict
    python scripts/analysis_rq3gj/run_gj_judge.py prerun
    python scripts/analysis_rq3gj/run_gj_judge.py full -w 16            # 同時跑的任務數
    python scripts/analysis_rq3gj/run_gj_judge.py full -m qwen          # 只跑某個宿主
"""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import numpy as np
import pandas as pd

from Aggregator.MenuJudgeAggregator import MenuJudgeAggregator, PairChoiceJudgeAggregator
from Arm.ArmSpec import ArmSpec
from Arm.PromptBuilder import PromptBuilder
from File.File import File
from File.ResultStore import ResultStore
from Log.NoLog import NoLog
from Log.OneAgentLog import OneAgentLog
from Strategy.SubstitutionJudge import SubstitutionJudge
from Strategy.StrategyConfig import StrategyConfig
from Runner.paths import armPath
from Runner.tasks import runTasks
from Runner.builders import buildModel, buildCandidateDatasets, buildEnglishDataset, runStrategy
from Analysis.alignment import loadRecords
from Analysis.blockStats import summarize
from Analysis.menuJudgeStats import blocksWithSplits
from Analysis.menuVote import MODELS, DATASETS
from Analysis.pathImprove import M12, HOSTS, DONORS, encodeBlocks
from Analysis.pathImproveK import HostDonorK
from Analysis.pathImproveGS import GROUPS, menuSets, agreementRates, menuGrouping, assignGroups, hostSplits, substitutionSplits, \
    onehot, GroupSums
from Analysis.preregistration import sha256, confirmationLine, loadStepFile
from Analysis.judgeSubstitution import (OUT_DIR, CRITERIA_FILE, JUDGE_DIR, JUDGE_DIR_K2, PRECHECK_DIR, PRECHECK_FILE, PILOT_FILE,
                                        PRERUN_FILE, STAGE0_CALLS, RUNNER, NUMS, AGGREGATOR_SEED, MAX_COST, PILOT_MAX_INVALID,
                                        RQ3GS_BLOCKS, RQ3GS_MENU_BLOCKS, MENU_ORDER, PAIRS, TOL, disagree, checkMenus, buildPlan,
                                        pilotVersions, checkVersions, precheckPath, checkStage0, tokenMeans, stage0Cost,
                                        evaluatePrecheck, evaluatePilot)
from Analysis.judgeSubstitutionPredict import (PRED_FILE, TABLE_FILE, EFFECT_FILE, LODO_FILE, CROSS_FILE, MANIFEST, trainingRecords,
                                               computePredictions)

STEPS = ["check", "pilot", "predict", "prerun", "full"]
STEP_FILES = {"check": PRECHECK_FILE, "pilot": PILOT_FILE, "predict": MANIFEST, "prerun": PRERUN_FILE}
DEFAULT_WORKERS = 16
RQ3GS_SUMMARY = ("+1.75", "+1.03", "+2.46", 8)


def parseArgs():
    parser = ArgumentParser(description="RQ3-GJ (rq3gj_criteria.md): check -> pilot -> predict -> prerun -> full")
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("-m", "--hosts", nargs="+", choices=HOSTS, default=HOSTS, help="Hosts to call in `full`")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS, help="Max concurrent tasks")
    parser.add_argument("--log", action="store_true", help="Enable terminal logging")
    return parser.parse_args()


def writeJson(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def stamp(criteria: str) -> dict:
    return {"criteria_sha256": sha256(criteria), "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")}


def requirePassed(args, criteria: str, step: str) -> dict:
    previous = STEPS[STEPS.index(step) - 1]
    path = os.path.join(args.out_dir, STEP_FILES[previous])
    data = loadStepFile(path, previous, sha256(criteria), RUNNER)
    if not data.get("passed"):
        raise SystemExit(f"❌ {path}: the `{previous}` step did not pass; report before going on")
    return data


def finish(path: str, data: dict, label: str):
    writeJson(path, data)
    print(f"\n{'✅ 通過' if data['passed'] else '❌ 未通過：停下來回報，不要跑下一步'}（{label}）-> 💾 {path}")
    if not data["passed"]:
        raise SystemExit(1)


# ------------------------------------------------------------------
# 載入與計畫
# ------------------------------------------------------------------
def loadAll(args) -> dict:
    raw, splits = defaultdict(dict), {}
    for block, sp in blocksWithSplits(args.armdir, args.aggdir):
        raw[block.dataset][block.model] = block
        splits[block.dataset] = sp
    coded, gold_text = {}, {}
    for dataset in DATASETS:
        blocks = [raw[dataset][model] for model in MODELS]
        for pb, cb in zip(blocks, encodeBlocks(blocks)):
            coded[(cb.model, cb.dataset)] = cb
            gold_text[(cb.model, cb.dataset)] = list(pb.gold)
    menus = checkMenus(coded)
    plan = buildPlan(coded, menus)
    stage0 = checkStage0(plan, os.path.join(args.out_dir, STAGE0_CALLS))
    if not stage0["ok"]:
        raise SystemExit(f"❌ The plan's calls differ from {STAGE0_CALLS}: {stage0['mismatches']}")
    print(f"計畫：{len(plan.versions)} 個（版本 × 順序），共 {stage0['total']:,} 次呼叫，與第零階段的表相同")
    return {"coded": coded, "gold_text": gold_text, "splits": splits, "menus": menus, "plan": plan}


# ------------------------------------------------------------------
# 一個版本的 Judge 檔
# ------------------------------------------------------------------
def runVersion(v, items: list, orders: dict, path: str, pilot: bool, args):
    arms = [ArmSpec.from_arm_id(a) for a in v.arm_ids]
    model = buildModel(v.host, 0.0)
    dataset, armDatasets = buildCandidateDatasets(v.dataset, arms, NUMS)
    aggregator = (MenuJudgeAggregator(3, model, dataset, seed=AGGREGATOR_SEED) if v.K == 3
                  else PairChoiceJudgeAggregator(model, dataset, seed=AGGREGATOR_SEED))
    files = [File(armPath(args.armdir, v.donor if j == v.slot else v.host, v.dataset, arm)) for j, arm in enumerate(arms)]
    strategy = SubstitutionJudge(
        StrategyConfig.from_dict({"strategyType": "aggregate", "languages": [arm.language for arm in arms]}),
        model, dataset, OneAgentLog() if args.log else NoLog(), aggregator, arms, files, armDatasets, ResultStore(path),
        menu=v.menu, onlyItems=set(items), version=v.version, donorSlot=v.slot, donorModel=v.donor, orders=orders,
        orderLabel=v.order, pilot=pilot, plannedItems=len(v.items))
    status = runStrategy(strategy)
    print(f"{status}: {v.host} | {v.dataset} | {v.menu} | {v.version}{' | ' + v.order if v.order else ''} -> {path}")


def runAll(tasks: list, args):
    workers = max(1, min(args.workers, len(tasks)))
    print(f"Total tasks: {len(tasks)} | Concurrent workers: {workers}\n")
    runTasks(runVersion, tasks, workers, args)


def missingItems(v, items: list, path: str) -> int:
    done = set(loadRecords(path)[1]) if os.path.exists(path) else set()
    return len(set(items) - done)


# ------------------------------------------------------------------
# 五個步驟
# ------------------------------------------------------------------
def stepCheck(args, criteria: str):
    precheck_dir = os.path.join(args.out_dir, PRECHECK_DIR)
    tasks = [(v, v.items, orders, precheckPath(precheck_dir, v), False) for v, orders in checkVersions()]
    runAll(tasks, args)
    result = evaluatePrecheck(precheck_dir)
    print(f"\n§10 流程核對（truthfulqa × M3S，原本的版本，舊紀錄的順序；最終答案相同的比例 Qwen 低於 90%、GPT-4o mini 低於 80% 就停）")
    for host, r in result["cells"].items():
        print(f"  {host:10s} {r['same_final']:4d} / {r['n_done']:4d} = {r['agreement']:.3f}（門檻 {r['min_agreement']:.2f}）"
              f"| 選到同一候選 {r['choice_agreement']:.3f} | 順序與舊紀錄相同 {r['same_orders']} | 版本 {list(r['returned_versions'])}"
              f"（RQ1-KJ {list(r['rq1kj_versions'])}）{'  ⚠️ 停' if r['stop'] else ''}")
    if not result["complete"]:
        raise SystemExit("\n⚠️ Not every cell is complete; precheck.json was not written. Rerun the same command")
    finish(os.path.join(args.out_dir, PRECHECK_FILE), {**stamp(criteria), **result}, "§10")


def stepPilot(args, criteria: str, ctx: dict):
    requirePassed(args, criteria, "pilot")
    plan = ctx["plan"]
    tasks = [(v, chosen, plan.ordersOf(v), v.path(args.out_dir), True) for v, chosen in pilotVersions(plan)]
    runAll(tasks, args)
    means = tokenMeans(args.aggdir)
    stage0 = stage0Cost(plan, means)
    pilot = evaluatePilot(args.out_dir, plan, means, stage0)
    print(f"\n§11 試跑（無效比例門檻 {PILOT_MAX_INVALID:.0%}，推估總費用門檻 ${MAX_COST:.0f}；第零階段估計 ${stage0['total']:.2f}）")
    for name, s in pilot["segments"].items():
        print(f"  {name}: {s['calls']} 次 | 找不到選擇 {s['no_choice']}、超出範圍 {s['out_of_range']}、被擋下 {s['refused']} -> 無效 "
              f"{s['invalid_rate']:.2%} | API in 平均 {s['tokens_in_api'].get('mean', float('nan')):.0f} 最大 "
              f"{s['tokens_in_api'].get('max', 0)}、out 平均 {s['tokens_out_api'].get('mean', float('nan')):.0f} 最大 "
              f"{s['tokens_out_api'].get('max', 0)} | 每次呼叫 ${s['cost_per_call_actual']:.6f}（第零階段 ${s['cost_per_call_stage0']:.6f}，"
              f"比值 {s['ratio']:.3f}）| 位置 {s['position_share']}")
    print(f"  推估總費用 ${pilot['projected_total_usd']:.2f}")
    if not pilot["complete"]:
        raise SystemExit("\n⚠️ The pilot is incomplete; rerun the same command. pilot.json was not written")
    finish(os.path.join(args.out_dir, PILOT_FILE), {**stamp(criteria), **pilot, "stage0_cost": stage0}, "§11")


def fileSha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def stepPredict(args, criteria: str, ctx: dict):
    requirePassed(args, criteria, "predict")
    manifest = os.path.join(args.out_dir, MANIFEST)
    if os.path.exists(manifest):
        raise SystemExit(f"❌ {manifest} already exists: predictions are written once, before the full run, and never recomputed")
    train = trainingRecords(ctx["coded"], ctx["gold_text"])
    out = computePredictions(ctx["plan"], ctx["coded"], ctx["splits"], train)
    files = {PRED_FILE: out["pred"], LODO_FILE: out["lodo"], CROSS_FILE: out["cross"], EFFECT_FILE: out["effects"], TABLE_FILE: out["tables"]}
    for name, frame in files.items():
        path = os.path.join(args.out_dir, name)
        if name.endswith(".gz"):
            import gzip
            with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                gz.write(frame.to_csv(index=False, float_format="%.12g").encode())
        else:
            frame.to_csv(path, index=False, float_format="%.12g")
    data = {**stamp(criteria), "files": {name: fileSha(os.path.join(args.out_dir, name)) for name in files},
            "rows": {name: len(frame) for name, frame in files.items()}, "training_records": len(train),
            "training_cells_used": int((train.cell >= 0).sum()), "passed": True}
    print(f"§7 預測：訓練紀錄 {len(train)} 筆（有格子的 {data['training_cells_used']} 筆）；" +
          "、".join(f"{n} {r:,} 列" for n, r in data["rows"].items()))
    finish(manifest, data, "§7 預測已存檔")


def stepPrerun(args, criteria: str, ctx: dict):
    requirePassed(args, criteria, "prerun")
    coded, splits, plan = ctx["coded"], ctx["splits"], ctx["plan"]
    weak = [(h, d) for h in HOSTS for d in DATASETS]
    result = {}

    # §12.1 重現 RQ3-GS 判定二（明顯落單組 62 份）與 39 份菜單的 E2_menu
    conf, _ = menuSets()
    rates = agreementRates([coded[k] for k in weak])
    gro = [menuGrouping(codes, rates) for _, codes in conf]
    groups, _, _ = assignGroups(gro)
    codes_of = dict(conf)
    clear = [mid for (mid, _), g in zip(conf, groups) if g == GROUPS[0]]
    rs = pd.read_csv(RQ3GS_BLOCKS).set_index(["model", "dataset"])
    rs_menu = pd.read_csv(RQ3GS_MENU_BLOCKS)
    rs_menu = rs_menu[rs_menu.menu_set == "conf"].drop_duplicates("menu").set_index("menu")
    e2, diff1, menu_sums = {}, 0.0, defaultdict(lambda: np.zeros(4))
    for h, d in weak:
        gs = GroupSums()
        for mid in sorted(set(clear) | set(MENU_ORDER)):
            codes = codes_of[mid]
            lone = onehot(hostSplits(coded[(h, d)], codes, splits[d], ("C",))["lone"])
            for g in DONORS:
                s = substitutionSplits(HostDonorK(coded[(h, d)], coded[(g, d)], splits[d]), codes, ("C",))
                num, den, inc = s["num"]["C"], s["den"], s["included"]
                if mid in clear:
                    gs.add("pair", num, den, ~lone, inc); gs.add("lone", num, den, lone, inc)
                if mid in MENU_ORDER:
                    lm, pm = lone & inc[:, None], ~lone & inc[:, None]
                    menu_sums[mid] += [num[pm].sum(), den[pm].sum(), num[lm].sum(), den[lm].sum()]
        e2[(h, d)] = gs.diff("pair", "lone")
        diff1 = max(diff1, abs(e2[(h, d)] - rs.loc[(h, d), "E2"]))
    s = summarize(list(e2.values()))
    shown = (f"{s['mean']:+.2f}", f"{s['ci_low']:+.2f}", f"{s['ci_high']:+.2f}", s["n_positive"])
    menu_diff = max(abs(10 * (v[0] / v[1] - v[2] / v[3]) - rs_menu.loc[m, "E2_menu"]) for m, v in menu_sums.items())
    result["1_rq3gs"] = {"ok": diff1 <= TOL and shown == RQ3GS_SUMMARY and menu_diff <= TOL, "summary": shown,
                         "max_block_diff": diff1, "max_menu_diff": menu_diff, "n_menus": len(menu_sums)}

    # §12.2 同一份菜單、同一題，所有版本的順序相同（計畫）；已寫入的紀錄（試跑）用的是計畫的順序
    plan_ok = all(plan.ordersOf(v) is plan.ordersOf(plan.find(v.K, v.host, v.dataset, v.menu, "orig", v.order)) for v in plan.versions)
    written, wrong = 0, 0
    for v in plan.versions:
        path = v.path(args.out_dir)
        if os.path.exists(path):
            _, recs = loadRecords(path)
            orders = plan.ordersOf(v)
            for i, r in recs.items():
                written += 1
                wrong += int(r["presentation_order"] != orders[int(i)])
    result["2_orders"] = {"ok": plan_ok and wrong == 0, "plan_shared": plan_ok, "records_checked": written, "records_wrong": wrong}

    # §12.3 預測檔已存檔，sha256 相符
    with open(os.path.join(args.out_dir, MANIFEST), encoding="utf-8") as f:
        manifest = json.load(f)
    shas = {name: fileSha(os.path.join(args.out_dir, name)) for name in manifest["files"]}
    result["3_predictions"] = {"ok": shas == manifest["files"] and manifest.get("criteria_sha256") == sha256(criteria),
                               "manifest": MANIFEST, "files": shas}

    # §12.4 換成自己（供體 = 宿主）：逐題的候選檔、解析答案、要呼叫的題目（子集二 (h, h) = 子集一）、順序都和原本的版本相同
    bad4, count4 = 0, 0
    for h, d in weak:
        H = coded[(h, d)]
        for K, menu, codes in [(3, m, plan.menus[m]["codes"]) for m in MENU_ORDER] + [(2, p, c) for p, c in PAIRS.items()]:
            rows = [M12.index(c) for c in codes]
            base = H.codes[rows]
            orig = plan.find(K, h, d, menu, "orig", "A" if K == 2 else None)
            for j, p in enumerate(codes):
                cc = base.copy()
                cc[j] = coded[(h, d)].codes[M12.index(p)]                      # 供體 = 宿主
                arms = [ArmSpec.from_arm_id(a) for a in orig.arm_ids]
                files_self = [armPath(args.armdir, h, d, a) for a in arms]     # slot j 讀「供體」= 宿主自己的檔
                files_orig = [armPath(args.armdir, h, d, a) for a in arms]
                same = (np.array_equal(cc, base) and files_self == files_orig
                        and np.array_equal(disagree(cc) & H.sub, disagree(base) & H.sub))
                count4 += 1
                bad4 += int(not same)
    result["4_self"] = {"ok": bad4 == 0, "substitutions": count4, "mismatches": bad4}

    # §12.5 K = 2：兩種順序互為對調；已寫入的紀錄（試跑）的 prompt 用的是對的檔案與順序（K = 3 也一起核對）
    swap_ok = all(plan.orders[(h, d, p, "B")][i] == list(reversed(plan.orders[(h, d, p, "A")][i]))
                  for h, d in weak for p in PAIRS for i in plan.orders[(h, d, p, "A")])
    prompt_checked, prompt_wrong = 0, 0
    datasets = {}
    for v in plan.versions:
        path = v.path(args.out_dir)
        if not os.path.exists(path):
            continue
        _, recs = loadRecords(path)
        if v.dataset not in datasets:
            datasets[v.dataset] = {e["id"]: e["question"] for e in buildEnglishDataset(v.dataset, NUMS).getData()}
        arms = [ArmSpec.from_arm_id(a) for a in v.arm_ids]
        files = {a.arm_id: File(armPath(args.armdir, v.donor if j == v.slot else v.host, v.dataset, a)) for j, a in enumerate(arms)}
        agg = MenuJudgeAggregator(3, None, None) if v.K == 3 else PairChoiceJudgeAggregator(None, None)
        for i, r in recs.items():
            texts = [files[a].getRecordById(int(i))["raw_text"] for a in r["presentation_order"]]
            prompt = agg.getJudgePrompt(datasets[v.dataset][int(i)], texts)
            prompt_checked += 1
            prompt_wrong += int(PromptBuilder.promptSha256([{"role": "user", "content": prompt}]) != r["trace"].get("prompt_sha256"))
    result["5_k2_swap"] = {"ok": swap_ok and prompt_wrong == 0 and prompt_checked > 0, "orders_swapped": swap_ok,
                           "records_checked": prompt_checked, "prompt_mismatches": prompt_wrong}
    for name, r in result.items():
        print(f"§12.{name}: {'通過' if r['ok'] else '❌ 不符'} {r}")
    finish(os.path.join(args.out_dir, PRERUN_FILE), {**stamp(criteria), "checks": result,
                                                     "passed": all(r["ok"] for r in result.values())}, "§12")


def stepFull(args, criteria: str, ctx: dict):
    requirePassed(args, criteria, "full")
    plan = ctx["plan"]
    todo = [v for v in plan.versions if v.participates and v.items and v.host in args.hosts]
    by_host = {h: [v for v in todo if v.host == h] for h in args.hosts}
    ordered = [v for group in zip(*by_host.values()) for v in group] + \
              [v for h in args.hosts for v in by_host[h][min(len(x) for x in by_host.values()):]]
    tasks = [(v, v.items, plan.ordersOf(v), v.path(args.out_dir), False) for v in ordered
             if missingItems(v, v.items, v.path(args.out_dir))]
    print(f"還沒完成的檔：{len(tasks)} / {len(ordered)}")
    if tasks:
        runAll(tasks, args)
    incomplete = [f"  {v.host} | {v.dataset} | {v.menu} | {v.version} | {v.order}: {missingItems(v, v.items, v.path(args.out_dir))} missing"
                  for v in ordered if missingItems(v, v.items, v.path(args.out_dir))]
    if incomplete:
        print(f"\n⚠️ {len(incomplete)} incomplete files (rerun the same command):\n" + "\n".join(incomplete[:30]))
    else:
        print(f"\n✅ All {len(ordered)} files complete. Next: python scripts/analysis_rq3gj/path_improve_gj.py")


def main():
    args = parseArgs()
    criteria = os.path.join(args.out_dir, CRITERIA_FILE)
    if confirmationLine(criteria) is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet (the 確認 line is empty); nothing is run")
    print(f"🚀 RQ3-GJ {args.step} | criteria sha256 {sha256(criteria)[:16]}\n")
    if args.step == "check":
        return stepCheck(args, criteria)
    ctx = loadAll(args)
    {"pilot": stepPilot, "predict": stepPredict, "prerun": stepPrerun, "full": stepFull}[args.step](args, criteria, ctx)


if __name__ == "__main__":
    main()
