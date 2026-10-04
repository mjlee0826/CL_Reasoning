"""
run_analysis.py — K=2 aggregation cells (paper_status v7 Stage 2): result/arms + result/aggregations
-> result/analysis/aggregation_cells.csv

One row per model × dataset × pair × aggregator × subset:
    pairs        the 19 pairs of Analysis/experimentPlan.py (todo.png), whenever both arm files exist
    aggregators  blind (always, computed here) + judge / debate (whenever the aggregation file exists)
    subsets      all / both_answered (see Analysis/metrics.py)
Columns: Analysis/metrics.py VALUE_COLUMNS. Split-half: H1 picks the stronger candidate, H2 measures (seed, reps).

Per-item export (--items-dir, see Analysis/itemExport.py): paths.csv.gz (every path's answer per item) and
aggregations.csv.gz (every judge / debate final answer per item), from the same files and checks as the cells.

0A-1 cross-check: the legacy cross-lingual debate rows must reproduce the RecoveryBlind metadata that
Test/TestRecoveryBlind.py wrote into result/challenge exactly (same splitHalfStats, different data path).

Usage:
    conda run -n clreasoning python run_analysis.py
    conda run -n clreasoning python run_analysis.py -m gpt4omini qwen -d mathqa --out /tmp/cells.csv --challenge-dir ""
"""

from argparse import ArgumentParser
from dataclasses import dataclass, field
import glob
import json
import math
import os

import pandas as pd

from Arm.ArmSpec import ArmSpec
from Runner.paths import ACTIVE_DATASETS
from Analysis.experimentPlan import PAIRS, FILE_AGGREGATORS
from Analysis.alignment import CellData, alignPair
from Analysis.metrics import SUBSETS, SPLIT_COLUMNS, VALUE_COLUMNS, subsetMask, computeRow
from Analysis.itemExport import pathRows, aggregationRows, writeItems

ID_COLUMNS = ["model", "dataset", "pair", "arm_a", "arm_b", "axis", "symmetric", "n_english", "aggregator", "subset"]
# RecoveryBlind metadata field -> aggregation_cells column
LEGACY_FIELDS = {"N": "n", "D": "n_dis", "d": "d", "c": "c", "m": "m", **{key: key for key in SPLIT_COLUMNS}}


def parseArgs():
    parser = ArgumentParser(description="K=2 aggregation cells -> result/analysis/aggregation_cells.csv")
    parser.add_argument("--armdir", default="result/arms", help="Root directory of the arm files")
    parser.add_argument("--aggdir", default="result/aggregations", help="Root directory of the aggregation files")
    parser.add_argument("-m", "--model", nargs="+", default=None, help="Models (default: every model under --armdir)")
    parser.add_argument("-d", "--dataset", choices=ACTIVE_DATASETS, nargs="+", default=ACTIVE_DATASETS)
    parser.add_argument("--seed", type=int, default=0, help="Split-half seed (0 = the 0A-1 splits)")
    parser.add_argument("--reps", type=int, default=200, help="Split-half repetitions")
    parser.add_argument("--out", default="result/analysis/aggregation_cells.csv")
    parser.add_argument("--items-dir", dest="items_dir", default="result/analysis/items",
                        help="Directory of the per-item export (\"\" to skip)")
    parser.add_argument("--challenge-dir", dest="challenge_dir", default="result/challenge",
                        help="Legacy challenge files for the 0A-1 cross-check (\"\" to skip)")
    return parser.parse_args()


@dataclass
class CellResult:
    """One model × dataset: cell rows, pairs skipped for missing arm files, unused aggregation files, per-item rows."""
    rows: list[dict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    unused: list[str] = field(default_factory=list)
    path_rows: list[dict] = field(default_factory=list)
    aggregation_rows: list[dict] = field(default_factory=list)


def analyzeCell(model_name: str, dataset_name: str, args) -> CellResult:
    cell = CellData(args.armdir, args.aggdir, model_name, dataset_name)
    result = CellResult(unused=cell.unusedFiles)
    if args.items_dir:
        result.path_rows = pathRows(cell)
    for pair in PAIRS:
        files = {agg: cell.aggregationFiles[(pair, agg)] for agg in FILE_AGGREGATORS if (pair, agg) in cell.aggregationFiles}
        if not cell.hasArms(pair):
            if files:
                raise FileNotFoundError(f"{model_name} | {dataset_name} | {pair.label}: aggregation files {list(files.values())} "
                                        f"exist but an arm file is missing")
            result.skipped.append(pair.label)
            continue
        for aggregator_id, path in [("blind", None), *files.items()]:
            arrays = alignPair(cell, pair, path)
            if path and args.items_dir:
                result.aggregation_rows += aggregationRows(cell, pair, aggregator_id, path)
            for subset in SUBSETS:
                ids = {"model": model_name, "dataset": dataset_name, "pair": pair.label,
                       "arm_a": pair.arm_a, "arm_b": pair.arm_b, "axis": pair.axis, "symmetric": pair.symmetric,
                       "n_english": pair.n_english, "aggregator": aggregator_id, "subset": subset}
                result.rows.append({**ids, **computeRow(arrays.subset(subsetMask(arrays, subset)), pair, aggregator_id,
                                                        args.seed, args.reps)})
    return result


def legacyChallengeFiles(challenge_dir: str) -> dict:
    """{(model, dataset, frozenset(languages)): path} of result/challenge/{model}_{dataset}_challenge_{l1}_vs_{l2}.json"""
    files = {}
    for path in glob.glob(os.path.join(challenge_dir, "*_challenge_*_vs_*.json")):
        prefix, langs = os.path.basename(path)[: -len(".json")].split("_challenge_")
        model_name, dataset_name = prefix.split("_", 1)
        files[(model_name, dataset_name, frozenset(langs.split("_vs_")))] = path
    return files


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def crossCheckLegacy(df: pd.DataFrame, challenge_dir: str):
    """0A-1: every legacy cross-lingual debate row must equal the challenge file's RecoveryBlind metadata exactly."""
    files = legacyChallengeFiles(challenge_dir)
    # Only models with legacy challenge files; newer models' language debates were run by run_aggregate.py and have no legacy file
    legacy_models = {model for model, _, _ in files}
    rows = df[(df["aggregator"] == "debate") & (df["subset"] == "all") & (df["axis"] == "L") & df["model"].isin(legacy_models)]
    mismatches = []
    for row in rows.itertuples(index=False):
        languages = {arm_id: ArmSpec.from_arm_id(arm_id).language for arm_id in (row.arm_a, row.arm_b)}
        path = files.get((row.model, row.dataset, frozenset(languages.values())))
        if path is None:
            mismatches.append(f"{row.model} | {row.dataset} | {row.pair}: no legacy challenge file")
            continue
        with open(path, encoding="utf-8") as f:
            legacy = json.load(f)[0].get("RecoveryBlind")
        if legacy is None:
            mismatches.append(f"{path}: no RecoveryBlind metadata (run scripts/legacy_eval/test_em.py -t testrecoveryblind first)")
            continue
        values = row._asdict()
        values["anchor"] = languages[values["anchor"]]   # metadata stores the language name
        for field, column in LEGACY_FIELDS.items():
            if not same(legacy[field], values[column]):
                mismatches.append(f"{path}: {field} = {legacy[field]!r} vs {column} = {values[column]!r}")
    if mismatches:
        raise SystemExit("❌ 0A-1 cross-check failed:\n  " + "\n  ".join(mismatches[:50]))
    print(f"✅ 0A-1 cross-check: {len(rows)} legacy debate rows reproduce the RecoveryBlind metadata exactly")


def main():
    args = parseArgs()
    models = args.model or sorted(name for name in os.listdir(args.armdir) if os.path.isdir(os.path.join(args.armdir, name)))

    rows, path_rows, aggregation_rows, notes = [], [], [], []
    for model_name in models:
        for dataset_name in args.dataset:
            result = analyzeCell(model_name, dataset_name, args)
            rows += result.rows
            path_rows += result.path_rows
            aggregation_rows += result.aggregation_rows
            if result.skipped:
                notes.append(f"  {model_name} | {dataset_name}: no arm files for {', '.join(result.skipped)}")
            notes += [f"  unused aggregation file: {path}" for path in result.unused]
            print(f"📊 {model_name} | {dataset_name}: {len(result.rows)} rows")

    df = pd.DataFrame(rows, columns=ID_COLUMNS + VALUE_COLUMNS)
    if notes:
        print("\nℹ️ Not analyzed:\n" + "\n".join(notes))
    print("\nRows per model × aggregator (both subsets):")
    print(df.groupby(["model", "aggregator"]).size().unstack(fill_value=0).to_string())

    if args.challenge_dir:
        crossCheckLegacy(df, args.challenge_dir)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n💾 {len(df)} rows -> {args.out}")

    if args.items_dir:
        paths_file, aggregations_file = writeItems(args.items_dir, path_rows, aggregation_rows)
        print(f"💾 per-item export: {len(path_rows)} path rows -> {paths_file}, "
              f"{len(aggregation_rows)} aggregation rows -> {aggregations_file}")


if __name__ == "__main__":
    main()
