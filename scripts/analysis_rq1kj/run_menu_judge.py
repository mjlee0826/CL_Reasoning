"""
run_menu_judge.py — RQ1-KJ 的 Judge 呼叫與開跑前檢查（result/analysis/rq1kj/rq1kj_criteria.md）

判定標準確認前一律拒跑。五個步驟依序執行，每一步都要上一步的結果檔存在、寫於同一份判定標準之下且通過；
呼叫的步驟都可以中斷後用同一個指令續跑（已寫入的題目會跳過，不重複計費；API 失敗的題目不寫入，重跑時補上）：

    reproduce  §6.1 離線重現：這次的程式算出的 A_V、S_in、Excess_V 要和 rq1k_blocks.csv 一致（1e-9）-> reproduce.json
    template   §6.2 模板核對：choice-k-v1 在 K = 2 時渲染 mmlu × EN+S1，重算 tokens 要等於主網格的 tokens_in
               -> template_check.json（Gemini 的 tokens 走免費的 count_tokens API，不產生文字）
    check      §6.3 流程核對：四個模型各重跑 mmlu × EN+S1 自己裁決自己（主網格記錄的順序）-> precheck/、precheck.json
    pilot      §6.4 試跑：每個模型 M12 100 題，各跑兩次（第一次寫進 judge_outputs/，全量沿用；第二次 pilot_rep2/）-> pilot.json
    full       全量：4 模型 × 4 資料集 × 4 菜單（M12、M3L、M3S、M3P）-> judge_outputs/

每次呼叫用 Strategy/MenuJudge.py + Aggregator/MenuJudgeAggregator.py：自己裁決自己、prompt choice-k-v1、
只裁決菜單內每條 path 都有答案且答案不完全一致的題目、候選順序依 §3.3；每筆紀錄記下供應商回傳的模型版本、
呼叫時間（UTC）、API token 與 prompt 的 sha256。

用法（從 repo root，conda 環境 clreasoning）：
    python scripts/analysis_rq1kj/run_menu_judge.py reproduce
    python scripts/analysis_rq1kj/run_menu_judge.py template
    python scripts/analysis_rq1kj/run_menu_judge.py check
    python scripts/analysis_rq1kj/run_menu_judge.py pilot
    python scripts/analysis_rq1kj/run_menu_judge.py full                         # 四個模型
    python scripts/analysis_rq1kj/run_menu_judge.py full -m deepseek4.1flash     # 只跑某些模型（例如 DeepSeek 排在離峰）
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

import pandas as pd

from Aggregator.MenuJudgeAggregator import MenuJudgeAggregator
from Arm.ArmSpec import ArmSpec
from File.File import File
from File.ResultStore import ResultStore
from Log.NoLog import NoLog
from Log.OneAgentLog import OneAgentLog
from Strategy.MenuJudge import MenuJudge
from Strategy.StrategyConfig import StrategyConfig
from Runner.paths import armPath, menuJudgePath
from Runner.tasks import defaultWorkers, interleaveByModel, runTasks, warnDeepseekPeak
from Runner.builders import buildModel, buildCandidateDatasets, buildEnglishDataset, runStrategy
from Analysis.alignment import CellData, loadRecords
from Analysis.preregistration import sha256, confirmationLine, loadStepFile
from Analysis.menuJudge import (MODELS, DATASETS, MENU_NAMES, MAIN_MENU, NUMS, AGGREGATOR_SEED, CHECK_DATASET, CHECK_MENU,
                                PILOT_MAX_INVALID, MAX_COST, OUT_DIR, CRITERIA_FILE, JUDGE_DIR, PRECHECK_DIR, REP2_DIR,
                                REPRODUCE_FILE, TEMPLATE_FILE, PRECHECK_FILE, PILOT_FILE, RUNNER, armIdsOf, itemPools,
                                pilotSample, mainGridJudgePath, evaluateReproduce, templateRows, evaluateTemplate,
                                evaluatePrecheck, evaluatePilot)
from Analysis.menuJudgeStats import blocksWithSplits, evaluateBlockMenu

STEPS = ["reproduce", "template", "check", "pilot", "full"]
STEP_FILES = {"reproduce": REPRODUCE_FILE, "template": TEMPLATE_FILE, "check": PRECHECK_FILE, "pilot": PILOT_FILE}
TEMPLATE_SOURCES = ["Strategy/PromptAbstractFactory/PromptJudgeChoiceFactory.py", "Aggregator/JudgeAggregator.py"]


def parseArgs():
    parser = ArgumentParser(description="RQ1-KJ judge calls (rq1kj_criteria.md): reproduce -> template -> check -> pilot -> full")
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("-m", "--models", nargs="+", choices=MODELS, default=MODELS,
                        help="Models to call (check / pilot / full); the step files always cover all four")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations", help="Main-grid judge files (§6.2, §6.3)")
    parser.add_argument("--rq1k-blocks", default="result/analysis/rq1k/rq1k_blocks.csv", help="§6.1 reference")
    parser.add_argument("--rq2-dir", default="result/analysis/rq2", help="RQ2 judge outputs (recorded model versions, §6.3)")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("-w", "--workers", type=int, default=None, help="Max concurrent threads (default: one per task)")
    parser.add_argument("--log", action="store_true", help="Enable terminal logging")
    return parser.parse_args()


def runCell(model: str, dataset_name: str, menu: str, outdir: str, onlyItems: set | None, referencePath: str | None, args):
    """一個模型 × 資料集 × 菜單的 Judge 檔（onlyItems：只跑這些題目；referencePath：§6.3 用主網格記錄的順序）。"""
    arms = [ArmSpec.from_arm_id(arm_id) for arm_id in armIdsOf(menu)]
    model_obj = buildModel(model, 0.0)
    dataset, armDatasets = buildCandidateDatasets(dataset_name, arms, NUMS)
    aggregator = MenuJudgeAggregator(len(arms), model_obj, dataset, seed=AGGREGATOR_SEED)
    store = ResultStore(menuJudgePath(outdir, model, dataset_name, menu))
    strategy = MenuJudge(
        StrategyConfig.from_dict({"strategyType": "aggregate", "languages": [arm.language for arm in arms]}),
        model_obj, dataset, OneAgentLog() if args.log else NoLog(), aggregator, arms,
        [File(armPath(args.armdir, model, dataset_name, arm)) for arm in arms], armDatasets, store,
        menu=menu, referencePath=referencePath, onlyItems=onlyItems,
    )
    status = runStrategy(strategy)
    print(f"{status}: {model} | {dataset_name} | {menu} -> {store.path} ({len(store.records)} records)")


def runAll(tasks: list[tuple], args):
    workers = args.workers or defaultWorkers(len(tasks))
    print(f"Total tasks: {len(tasks)} | Concurrent workers: {workers}\n")
    runTasks(runCell, tasks, workers, args)


def writeJson(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def stamp(criteria: str) -> dict:
    return {"criteria_sha256": sha256(criteria), "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")}


def requirePassed(args, criteria: str, step: str) -> dict:
    """上一步的結果檔：存在、寫於同一份判定標準之下、而且通過。"""
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
# 五個步驟
# ------------------------------------------------------------------
def stepReproduce(args, criteria: str):
    rows = [evaluateBlockMenu(block, menu, splits)["row"]
            for block, splits in blocksWithSplits(args.armdir, args.aggdir) for menu in MENU_NAMES]
    result = evaluateReproduce(pd.DataFrame(rows), pd.read_csv(args.rq1k_blocks))
    print(f"§6.1 離線重現：{result['n_rows']} 列（區塊 × 菜單），對 {args.rq1k_blocks}")
    for column, value in result["max_abs_diff"].items():
        print(f"  {column:9s} 最大誤差 {value:.2e}")
    for m in result["mismatches"][:20]:
        print(f"  ❌ {m['model']} | {m['dataset']} | {m['menu']} | {m['column']}: {m['abs_diff']:.2e}")
    finish(os.path.join(args.out_dir, REPRODUCE_FILE), {**stamp(criteria), "reference": args.rq1k_blocks, **result}, "§6.1")


def gitLog() -> str:
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h %ci %s", "--", *TEMPLATE_SOURCES],
                              capture_output=True, text=True, check=True).stdout.strip() or "(no git history)"
    except (OSError, subprocess.CalledProcessError) as e:
        return f"(git unavailable: {e})"


def stepTemplate(args, criteria: str):
    requirePassed(args, criteria, "template")
    dataset = buildEnglishDataset(CHECK_DATASET, NUMS)
    questions = {data["id"]: data["question"] for data in dataset.getData()}
    per_model = {}
    for model in MODELS:
        model_obj = buildModel(model, 0.0)
        aggregator = MenuJudgeAggregator(2, model_obj, dataset, seed=AGGREGATOR_SEED)
        render = lambda question, texts, a=aggregator: [{"role": "user", "content": a.getJudgePrompt(question, texts)}]
        _, main = loadRecords(mainGridJudgePath(args.aggdir, model, CHECK_DATASET, CHECK_MENU))
        per_model[model] = templateRows(model, CellData(args.armdir, args.aggdir, model, CHECK_DATASET), main, questions,
                                        render, model_obj.countTokens)
        print(f"  {model:20s} {sum(r['equal'] for r in per_model[model])} / {len(per_model[model])} 題重算 tokens 等於主網格的 tokens_in")
    result = evaluateTemplate(per_model, gitLog())
    print(f"  git（模板檔最後一次變動）：{result['git_log']}")
    finish(os.path.join(args.out_dir, TEMPLATE_FILE), {**stamp(criteria), **result}, "§6.2")


def stepCheck(args, criteria: str):
    requirePassed(args, criteria, "check")
    warnDeepseekPeak(args.models)
    precheck_dir = os.path.join(args.out_dir, PRECHECK_DIR)
    runAll([(m, CHECK_DATASET, CHECK_MENU, precheck_dir, None, mainGridJudgePath(args.aggdir, m, CHECK_DATASET, CHECK_MENU))
            for m in args.models], args)
    result = evaluatePrecheck(precheck_dir, args.armdir, args.aggdir, args.rq2_dir)
    print(f"\n§6.3 流程核對（{CHECK_DATASET} × {CHECK_MENU}，自己裁決自己 vs 主網格；比 RQ2 核對時低 10pp 以上就停）")
    for model, r in result["models"].items():
        print(f"  {model:20s} {r['n_agree']:4d} / {r['n']:4d} = {r['agreement']:.3f}（RQ2 {r['rq2_agreement']:.3f}，"
              f"差 {-r['drop']:+.3f}）完成 {r['n_done']} | tokens_in 相同 {r['n_same_tokens_in']} | "
              f"版本 {list(r['returned_versions'])} 與 arm 紀錄相同：{r['version_matches_arms']}"
              f"{'  ⚠️ 停' if r['stop'] else ''}")
    if not result["complete"]:
        raise SystemExit("\n⚠️ Not every model is complete; precheck.json was not written. Rerun the same command")
    finish(os.path.join(args.out_dir, PRECHECK_FILE), {**stamp(criteria), **result}, "§6.3")


def stepPilot(args, criteria: str):
    requirePassed(args, criteria, "pilot")
    warnDeepseekPeak(args.models)
    pools, estimates = itemPools(args.armdir, args.aggdir)
    judge_dir, rep2_dir = os.path.join(args.out_dir, JUDGE_DIR), os.path.join(args.out_dir, REP2_DIR)
    tasks = [(model, dataset, MAIN_MENU, directory, set(ids), None)
             for model in args.models for dataset, ids in pilotSample(pools, model).items() for directory in (judge_dir, rep2_dir)]
    runAll(tasks, args)

    pilot = evaluatePilot(judge_dir, rep2_dir, pools, estimates)
    print(f"\n§6.4 試跑（每個模型 M12 100 題 × 2 次；第一次的無效比例門檻 {PILOT_MAX_INVALID:.0%}，成本門檻 ${MAX_COST:.0f}）")
    for model, r in pilot["models"].items():
        print(f"  {model:20s} 完成 {r['n_done_run1']}/{r['n_done_run2']} | 第一次：找不到 choice {r['run1']['no_choice']}、"
              f"超出範圍 {r['run1']['out_of_range']}、被擋下 {r['run1']['refused']} | 兩次選到同一候選 {r['same_choice_rate']:.2f}、"
              f"最終答案相同 {r['same_final_rate']:.2f} | API in 平均 {r['tokens_in_api']['mean']:.0f} 最大 {r['tokens_in_api']['max']:.0f}、"
              f"out 平均 {r['tokens_out_api']['mean']:.0f} 最大 {r['tokens_out_api']['max']:.0f} | "
              f"全量最大 input 估計 {r['full_max_input_estimate']:.0f}{'（超過上限）' if r['full_exceeds'] else ''} | "
              f"全量 ${r['full_cost_usd']:.2f}")
        for menu, m in r["menus"].items():
            print(f"      {menu:4s} {m['calls']:5d} 次 × (in {m['input_per_call']:.0f} / out {m['output_per_call']:.0f}) = ${m['cost_usd']:.2f}")
    print(f"  全量成本估算（四份菜單合計，含試跑第二次）：${pilot['full_cost_usd']:.2f}")
    if not pilot["complete"]:
        raise SystemExit("\n⚠️ The pilot is incomplete (or calls lack API usage); rerun the same command. pilot.json was not written")
    finish(os.path.join(args.out_dir, PILOT_FILE), {**stamp(criteria), **pilot}, "§6.4")


def stepFull(args, criteria: str):
    requirePassed(args, criteria, "full")
    warnDeepseekPeak(args.models)
    judge_dir = os.path.join(args.out_dir, JUDGE_DIR)
    tasks = [(model, dataset, menu, judge_dir, None, None)
             for model, dataset, menu in interleaveByModel(args.models, DATASETS, MENU_NAMES)]
    runAll(tasks, args)

    pools, _ = itemPools(args.armdir, args.aggdir, args.models)
    incomplete = []
    for model, dataset, menu, *_ in tasks:
        path = menuJudgePath(judge_dir, model, dataset, menu)
        done = set(loadRecords(path)[1]) if os.path.exists(path) else set()
        if done != set(pools[(model, dataset, menu)]):
            incomplete.append(f"  {model} | {dataset} | {menu}: {len(done)}/{len(pools[(model, dataset, menu)])}")
    if incomplete:
        print("\n⚠️ Incomplete files (rerun the same command):\n" + "\n".join(incomplete))
    else:
        print(f"\n✅ All {len(tasks)} files complete. Next: python scripts/analysis_rq1kj/menu_judge.py")


def main():
    args = parseArgs()
    criteria = os.path.join(args.out_dir, CRITERIA_FILE)
    if confirmationLine(criteria) is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet (the 確認 line is empty); nothing is run")
    print(f"🚀 RQ1-KJ {args.step} | criteria sha256 {sha256(criteria)[:16]} | models {args.models}\n")
    {"reproduce": stepReproduce, "template": stepTemplate, "check": stepCheck, "pilot": stepPilot, "full": stepFull}[args.step](args, criteria)


if __name__ == "__main__":
    main()
