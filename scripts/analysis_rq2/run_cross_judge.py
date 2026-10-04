"""
run_cross_judge.py — RQ2 交叉實驗的 Judge 呼叫（result/analysis/rq2/rq2_criteria.md）

判定標準確認前一律拒跑。三個步驟依序執行；每一步都可以中斷後用同一個指令續跑
（已寫入的題目會跳過，不重複計費；API 失敗的題目不寫入，重跑時補上）：

    check   §4.1 流程核對：四個模型各重跑 mmlu × EN+S1「自己裁決自己」-> precheck/，
            逐題和主網格的選擇比對 -> precheck/precheck.json（一致率 < 95% 的模型，full 會重跑它的對角線）
    pilot   §4.2 試跑：每個 Judge 從它要裁決的題目抽 100 題（seed 0），寫進 judge_outputs/（full 時沿用）
            -> pilot.json（解析失敗、編號超出範圍、被擋下、off-menu、每次呼叫的 token、全量成本估算）
    full    全量：12 種非對角組合 + §4.1 判定要重跑的對角線。需要 precheck.json 與通過的 pilot.json

每次呼叫用 Strategy/CrossJudge.py：候選、prompt、呈現順序與主網格相同，只換 Judge 模型；
只裁決兩條 path 都有答案的分歧題；每筆紀錄記下供應商回傳的模型版本、呼叫時間（UTC）與 API token。

用法（從 repo root，conda 環境 clreasoning）：
    python scripts/analysis_rq2/run_cross_judge.py check
    python scripts/analysis_rq2/run_cross_judge.py pilot
    python scripts/analysis_rq2/run_cross_judge.py full                        # 四個 Judge
    python scripts/analysis_rq2/run_cross_judge.py full -j deepseek4.1flash    # 只跑某些 Judge（例如 DeepSeek 排在離峰）
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

from Aggregator.AggregatorConfig import AggregatorConfig
from Aggregator.AggregatorFactory import AggregatorFactory
from Aggregator.AggregatorType import AggregatorType
from Arm.ArmSpec import ArmSpec
from File.File import File
from File.ResultStore import ResultStore
from Log.NoLog import NoLog
from Log.OneAgentLog import OneAgentLog
from Strategy.CrossJudge import CrossJudge
from Strategy.StrategyConfig import StrategyConfig
from Runner.paths import armPath
from Runner.tasks import defaultWorkers, interleaveByModel, runTasks
from Runner.builders import buildModel, buildCandidateDatasets, runStrategy
from Analysis.alignment import loadRecords
from Analysis.preregistration import sha256, confirmationLine
from Analysis.crossJudge import (MODELS, DATASETS, PAIRS, PAIR_BY_LABEL, NUMS, CHECK_DATASET, CHECK_PAIR, AGREEMENT_MIN,
                                 PILOT_MAX_RATE, PILOT_MAX_COST, OUT_DIR, CRITERIA_FILE, JUDGE_DIR, PRECHECK_DIR, PRECHECK_FILE,
                                 PILOT_FILE, loadStepFile, itemPools, pilotSample, mainGridPath, judgeOutputPath, evaluatePrecheck, evaluatePilot)

# 與主網格的 run_aggregate.py 預設值相同（metadata 會核對 seed）
AGGREGATOR_SEED, DEBATE_THRESHOLD = 0, 3
# DeepSeek 官方 API 的尖峰時段（UTC，週一至週五），價格加倍
DEEPSEEK_PEAK_HOURS = set(range(1, 4)) | set(range(6, 10))


def parseArgs():
    parser = ArgumentParser(description="RQ2 cross-judge calls (rq2_criteria.md): check -> pilot -> full")
    parser.add_argument("step", choices=["check", "pilot", "full"])
    parser.add_argument("-j", "--judges", nargs="+", choices=MODELS, default=MODELS,
                        help="Judge models to run (check: the models whose diagonal is rechecked)")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations", help="Main-grid judge files (recorded orders, diagonal)")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("-w", "--workers", type=int, default=None, help="Max concurrent threads (default: one per task)")
    parser.add_argument("--log", action="store_true", help="Enable terminal logging")
    return parser.parse_args()


def runCell(judge: str, generator: str, dataset_name: str, pair, outdir: str, onlyItems: set | None, args):
    """One Judge × generator × dataset × pair file (onlyItems: restrict to these items, the pilot sample)."""
    arms = [ArmSpec.from_arm_id(arm_id) for arm_id in pair.arms]
    model = buildModel(judge, 0.0)
    dataset, armDatasets = buildCandidateDatasets(dataset_name, arms, NUMS)
    aggregator = AggregatorFactory().buildAggregator(
        AggregatorType.JUDGE, AggregatorConfig.from_dict({"seed": AGGREGATOR_SEED, "threshold": DEBATE_THRESHOLD}), model, dataset)
    store = ResultStore(judgeOutputPath(outdir, judge, generator, dataset_name, pair))
    strategy = CrossJudge(
        StrategyConfig.from_dict({"strategyType": "aggregate", "languages": [arm.language for arm in arms]}),
        model, dataset, OneAgentLog() if args.log else NoLog(), aggregator, arms,
        [File(armPath(args.armdir, generator, dataset_name, arm)) for arm in arms], armDatasets, store,
        generator=generator, referencePath=mainGridPath(args.aggdir, generator, dataset_name, pair), onlyItems=onlyItems,
    )
    status = runStrategy(strategy)
    print(f"{status}: judge {judge} | generator {generator} | {dataset_name} | {pair.label} -> {store.path} "
          f"({len(store.records)} records)")


def runAll(tasks: list[tuple], args):
    workers = args.workers or defaultWorkers(len(tasks))
    print(f"Total tasks: {len(tasks)} | Concurrent workers: {workers}\n")
    runTasks(runCell, tasks, workers, args)


def warnDeepseekPeak(judges: list[str]):
    now = datetime.now(timezone.utc)
    if "deepseek4.1flash" in judges and now.weekday() < 5 and now.hour in DEEPSEEK_PEAK_HOURS:
        print(f"⚠️ {now:%H:%M} UTC is DeepSeek's peak period (Mon–Fri UTC 01–04, 06–10, double price); "
              f"consider running -j deepseek4.1flash off-peak\n")


def writeJson(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def stamp(criteria: str) -> dict:
    return {"criteria_sha256": sha256(criteria), "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")}


def stepCheck(args, criteria: str, precheck_dir: str):
    runAll([(m, m, CHECK_DATASET, CHECK_PAIR, precheck_dir, None) for m in args.judges], args)
    result = evaluatePrecheck(precheck_dir, args.armdir, args.aggdir)
    print(f"\n§4.1 流程核對（{CHECK_DATASET} × {CHECK_PAIR.label}，自己裁決自己 vs 主網格；門檻 {AGREEMENT_MIN:.0%}）")
    for model, r in result.items():
        state = ("未完成，請用同一個指令續跑" if not r["complete"] else
                 "→ 對角線改用交叉流程重跑" if r["rerun_diagonal"] else "→ 沿用主網格舊檔")
        print(f"  {model:20s} {r['n_agree']:4d} / {r['n']:4d} 一致 = {r['agreement']:.3f}（已完成 {r['n_done']}）{state}")
    if not all(r["complete"] for r in result.values()):
        raise SystemExit("\n⚠️ Not every model is complete; precheck.json was not written")
    path = os.path.join(precheck_dir, PRECHECK_FILE)
    writeJson(path, {**stamp(criteria), "dataset": CHECK_DATASET, "pair": CHECK_PAIR.label, "models": result})
    print(f"\n💾 {path}")


def stepPilot(args, criteria: str, precheck_dir: str, judge_dir: str):
    precheck = loadStepFile(os.path.join(precheck_dir, PRECHECK_FILE), "check", sha256(criteria))
    warnDeepseekPeak(args.judges)
    pools = itemPools(args.armdir, args.aggdir)
    tasks = [(judge, generator, dataset, PAIR_BY_LABEL[label], judge_dir, items)
             for judge in args.judges for (generator, dataset, label), items in pilotSample(pools, judge).items()]
    runAll(tasks, args)

    pilot = evaluatePilot(judge_dir, pools, precheck["models"])
    print(f"\n§4.2 試跑（每個 Judge {next(iter(pilot['judges'].values()))['n']} 題；比例門檻 {PILOT_MAX_RATE:.0%}，"
          f"成本門檻 ${PILOT_MAX_COST:.0f}）")
    for judge, r in pilot["judges"].items():
        print(f"  {judge:20s} 完成 {r['n_done']}/{r['n']} | 找不到 choice {r['n_no_choice']} | 超出範圍 {r['n_out_of_range']} | "
              f"被擋下 {r['n_refused']} | off-menu {r['n_off_menu']} | API in/out {r['api_in_per_call']:.0f}/{r['api_out_per_call']:.0f} | "
              f"${r['cost_per_call_usd']:.5f}/次 × {r['full_calls']} = ${r['full_cost_usd']:.2f}")
    print(f"  全量成本估算：${pilot['full_cost_usd']:.2f}")
    if not pilot["complete"]:
        raise SystemExit("\n⚠️ The pilot is incomplete (or calls lack API usage); rerun the same command. pilot.json was not written")
    path = os.path.join(args.out_dir, PILOT_FILE)
    writeJson(path, {**stamp(criteria), "pilot": pilot})
    print(f"\n{'✅ 通過' if pilot['passed'] else '❌ 未通過：停下來回報，不要跑 full'} -> 💾 {path}")


def stepFull(args, criteria: str, precheck_dir: str, judge_dir: str):
    pilot_path = os.path.join(args.out_dir, PILOT_FILE)
    precheck = loadStepFile(os.path.join(precheck_dir, PRECHECK_FILE), "check", sha256(criteria))
    pilot = loadStepFile(pilot_path, "pilot", sha256(criteria))
    if not pilot["pilot"]["passed"]:
        raise SystemExit(f"❌ {pilot_path}: the pilot did not pass §4.2; report before running the full experiment")
    warnDeepseekPeak(args.judges)

    rerun = {model for model, r in precheck["models"].items() if r["rerun_diagonal"]}
    tasks = [(judge, generator, dataset, pair, judge_dir, None)
             for judge, generator, dataset, pair in interleaveByModel(args.judges, MODELS, DATASETS, PAIRS)
             if judge != generator or judge in rerun]
    for model in rerun & set(args.judges):
        # §4.1：重跑對角線時，流程核對已重跑的那一格直接沿用
        target = judgeOutputPath(judge_dir, model, model, CHECK_DATASET, CHECK_PAIR)
        if not os.path.exists(target):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(judgeOutputPath(precheck_dir, model, model, CHECK_DATASET, CHECK_PAIR), target)
    if rerun:
        print(f"Diagonal rerun (§4.1 agreement < {AGREEMENT_MIN:.0%}): {sorted(rerun)}")
    runAll(tasks, args)

    pools = itemPools(args.armdir, args.aggdir)
    incomplete = []
    for judge, generator, dataset, pair, *_ in tasks:
        path = judgeOutputPath(judge_dir, judge, generator, dataset, pair)
        done = len(loadRecords(path)[1]) if os.path.exists(path) else 0
        if done != len(pools[(generator, dataset, pair.label)]):
            incomplete.append(f"  judge {judge} | generator {generator} | {dataset} | {pair.label}: "
                              f"{done}/{len(pools[(generator, dataset, pair.label)])}")
    if incomplete:
        print("\n⚠️ Incomplete cells (rerun the same command):\n" + "\n".join(incomplete))
    else:
        print(f"\n✅ All {len(tasks)} cells complete. Next: python scripts/analysis_rq2/cross_judge.py")


def main():
    args = parseArgs()
    criteria = os.path.join(args.out_dir, CRITERIA_FILE)
    if confirmationLine(criteria) is None:
        raise SystemExit(f"❌ {criteria} has not been confirmed yet (the 確認 line is empty); no API call is allowed")
    precheck_dir, judge_dir = os.path.join(args.out_dir, PRECHECK_DIR), os.path.join(args.out_dir, JUDGE_DIR)
    print(f"🚀 RQ2 {args.step} | criteria sha256 {sha256(criteria)[:16]} | judges {args.judges}\n")
    if args.step == "check":
        stepCheck(args, criteria, precheck_dir)
    elif args.step == "pilot":
        stepPilot(args, criteria, precheck_dir, judge_dir)
    else:
        stepFull(args, criteria, precheck_dir, judge_dir)


if __name__ == "__main__":
    main()
