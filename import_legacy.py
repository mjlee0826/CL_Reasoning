"""
Adapter: legacy IMSR results -> generation / aggregation schema (result/arms, result/aggregations).

    result/baseline/{m}_{d}_onelanguage_{lang}.json          -> result/arms/{m}/{d}/L_{code}.json
    result/challenge/{m}_{d}_challenge_{l1}_vs_{l2}.json      -> result/aggregations/{m}/{d}/debate__L_{c1}__L_{c2}.json
    (optional) result/english_cot/{m}_{d}_onelanguage_english.json -> result/arms/{m}/{d}/W_rewrite1.json

seed = null, model_version_string = metadata Model.modelName, tokens recounted with Model.countTokens
(Gemini counts through the count_tokens API, so its cells take hours). Outputs are resumable per record.

    conda run -n clreasoning python import_legacy.py -m gpt4omini -d mathqa -w 1
"""
from argparse import ArgumentParser
import glob
import os
import re

from tqdm import tqdm

from Model.ModelConfig import ModelConfig
from Model.ModelFactory import ModelFactory
from Model.ModelType import ModelType
from Dataset.DatasetType import DatasetType, get_dataset_map
from Arm.ArmSpec import ArmSpec
from Arm.AxisType import LANGUAGE_TO_LANG_CODE
from Arm.GenerationRecord import GenerationRecord
from Arm.PromptBuilder import PromptBuilder
from Aggregator.AggregationRecord import AggregationRecord
from Aggregator.DebateAggregator import DebateAggregator
from File.File import File
from File.ResultStore import ResultStore
from Strategy.Generate import SCHEMA_VERSION as GENERATION_SCHEMA
from Strategy.Aggregate import SCHEMA_VERSION as AGGREGATION_SCHEMA
from Runner.paths import ACTIVE_DATASETS, armPath, aggregationPath
from Runner.tasks import runTasks
from Runner.builders import buildArmDataset

LEGACY_MODELS = ["deepseek", "gemini", "gpt4omini", "qwen"]
LEGACY_THRESHOLD = 3

# What a legacy API alias actually served when the legacy results were generated (2026-03).
# deepseek-chat pointed to DeepSeek-V3.2 from 2025-12-01 until 2026-04-24 (see Model/Deepseek.py).
LEGACY_VERSION_LABELS = {"deepseek-chat": "DeepSeek-V3.2"}


def legacyVersionString(model_meta: dict) -> str:
    name = model_meta.get("modelName", "")
    return LEGACY_VERSION_LABELS.get(name, name)


def importArm(model, source: File, arm: ArmSpec, out_path: str) -> ResultStore:
    """Legacy onelanguage file -> GenerationRecords of `arm`. The prompt is rebuilt from the stored Question."""
    model_meta, dataset_meta = source.metadata["Model"], source.metadata["Dataset"]
    dataset_type = dataset_meta["datasetType"]

    # Question text the current Dataset would produce; a mismatch means the prompt can no longer be rebuilt
    dataset = buildArmDataset(dataset_type, arm, dataset_meta["nums"])
    current = {data["id"]: data["question"] for data in dataset.getData()}
    records = source.records_map

    store = ResultStore(out_path)
    store.metadata = {
        "Model": model_meta,
        "Dataset": dataset_meta,
        "Strategy": {"strategyType": "generate", "displayName": f"Generate ({arm.arm_id})",
                     "languages": [arm.language], "promptStyle": arm.promptStyle},
        "Arm": arm.to_dict(),
        "schema_version": GENERATION_SCHEMA,
        "source": "legacy_import",
        "legacy_path": source.file_path,
        "question_text_mismatches": sum(r.get("Question") != current.get(q_id) for q_id, r in records.items()),
        "error_string_records": sum(str(r.get("Result", "")).startswith("Error") for r in records.values()),
    }

    builder = PromptBuilder(arm)
    for q_id in tqdm(sorted(records), desc=f"{os.path.basename(out_path)}"):
        if store.has(q_id):
            continue
        record = records[q_id]
        messages = builder.messages(record.get("Question", ""))
        raw_text, parsed = record.get("Result") or "", record.get("MyAnswer") or ""
        store.add(GenerationRecord(
            item_id=q_id,
            arm_id=arm.arm_id,
            axis=arm.axis,
            is_anchor=arm.is_anchor,
            lang=arm.lang,
            raw_text=raw_text,
            parsed_answer=parsed,
            parse_ok=GenerationRecord.isParseOk(parsed),
            tokens_in=sum(model.countTokens(m["content"]) for m in messages),
            tokens_out=model.countTokens(raw_text),
            model=model_meta["modelType"],
            model_version_string=legacyVersionString(model_meta),
            temperature=model_meta.get("temperature", 0.0),
            seed=None,
            prompt_hash=PromptBuilder.promptHash(messages),
            gold=str(record.get("Answer", "")),
        ).to_dict())

    store.save()
    return store


def importRefineArm(model, source: File, arm: ArmSpec, baseStore: ResultStore, baseFile: File, out_path: str) -> ResultStore:
    """
    Legacy self_reflection file -> GenerationRecords of the derived arm F:{lang} (base = L:{lang}).
    The initial generation is the base arm's record (SR stores it again as `Response`), so only the
    revision becomes a new record. tokens_in counts the reflection prompt, which contains that output.
    """
    model_meta, dataset_meta = source.metadata["Model"], source.metadata["Dataset"]
    records = source.records_map
    checks = {"question_mismatch": 0, "base_output_mismatch": 0, "prompt_rebuild_mismatch": 0}
    builder = PromptBuilder(arm)
    store = ResultStore(out_path)

    pending = []
    for q_id in sorted(records):
        record = records[q_id]
        base = baseStore.records.get(q_id)
        if base is None:
            raise ValueError(f"{out_path}: base arm {arm.base_arm_id} has no item {q_id}")
        checks["question_mismatch"] += record.get("Question") != baseFile.getRecordById(q_id).get("Question")
        checks["base_output_mismatch"] += record.get("Response") != base["raw_text"]
        rebuilt = builder.text(record.get("Question", ""), base["raw_text"])
        checks["prompt_rebuild_mismatch"] += rebuilt != record.get("Reflection")
        if not store.has(q_id):
            pending.append(q_id)

    store.metadata = {
        "Model": model_meta,
        "Dataset": dataset_meta,
        "Strategy": {"strategyType": "generate", "displayName": f"Generate ({arm.arm_id})",
                     "languages": [arm.language], "promptStyle": arm.promptStyle},
        "Arm": arm.to_dict(),
        "base_arm_file": baseStore.path,
        "schema_version": GENERATION_SCHEMA,
        "source": "legacy_import",
        "legacy_path": source.file_path,
        "legacy_checks": checks,
        "error_string_records": sum(str(r.get("Result", "")).startswith("Error") for r in records.values()),
    }

    for q_id in tqdm(pending, desc=f"{os.path.basename(out_path)}"):
        record = records[q_id]
        # The stored Reflection is what was actually sent; legacy_checks says whether it rebuilds exactly
        prompt = record.get("Reflection") or ""
        raw_text, parsed = record.get("Result") or "", record.get("MyAnswer") or ""
        store.add(GenerationRecord(
            item_id=q_id,
            arm_id=arm.arm_id,
            axis=arm.axis,
            is_anchor=arm.is_anchor,
            lang=arm.lang,
            raw_text=raw_text,
            parsed_answer=parsed,
            parse_ok=GenerationRecord.isParseOk(parsed),
            tokens_in=model.countTokens(prompt),
            tokens_out=model.countTokens(raw_text),
            model=model_meta["modelType"],
            model_version_string=legacyVersionString(model_meta),
            temperature=model_meta.get("temperature", 0.0),
            seed=None,
            prompt_hash=PromptBuilder.promptHash([{"role": "user", "content": prompt}]),
            gold=str(record.get("Answer", "")),
        ).to_dict())

    store.save()
    return store


def importDebate(model, source: File, arms: tuple, armStores: tuple, baselines: tuple, out_path: str, compare) -> dict:
    """Legacy challenge file -> AggregationRecords (aggregator_id = debate). Returns the consistency checks."""
    meta = source.metadata
    lang1, lang2 = meta["Strategy"]["languages"]
    arm_ids = [arm.arm_id for arm in arms]
    checks = {"record_prefix_mismatch": 0, "debate_prompt_mismatch": 0, "initial_answer_mismatch": 0}
    store = ResultStore(out_path)
    pending = []

    for q_id in sorted(source.records_map):
        r = source.records_map[q_id]
        R1, R2, AR1, AR2 = r["Record1"], r["Record2"], r["AnswerRecord1"], r["AnswerRecord2"]
        gen1, gen2 = armStores[0].records[q_id], armStores[1].records[q_id]

        checks["initial_answer_mismatch"] += str(AR1[0]) != gen1["parsed_answer"] or str(AR2[0]) != gen2["parsed_answer"]
        checks["record_prefix_mismatch"] += (
            R1[0]["content"] != baselines[0].getRecordById(q_id)["Question"] or R1[1]["content"] != gen1["raw_text"]
            or R2[0]["content"] != baselines[1].getRecordById(q_id)["Question"] or R2[1]["content"] != gen2["raw_text"]
        )
        times = int(r.get("Times", 0))
        checks["debate_prompt_mismatch"] += any(
            R1[2 * k]["content"] != DebateAggregator.getDebatePrompt(lang1, R2[2 * k - 1]["content"])
            or R2[2 * k]["content"] != DebateAggregator.getDebatePrompt(lang2, R1[2 * k - 1]["content"])
            for k in range(1, times + 1)
        )
        if not store.has(q_id):
            pending.append(q_id)

    n_disagreement = sum(not compare(str(r["AnswerRecord1"][0]), str(r["AnswerRecord2"][0])) for r in source.records_map.values())
    store.metadata = {
        "Model": meta["Model"],
        "Dataset": meta["Dataset"],
        "Strategy": {"strategyType": "aggregate", "displayName": f"Debate ({' vs '.join(arm_ids)})", "languages": [lang1, lang2]},
        "Aggregator": {"aggregatorType": "debate", "displayName": "Debate", "k": 2, "seed": 0, "threshold": LEGACY_THRESHOLD},
        "candidate_arms": arm_ids,
        "candidate_files": [store_.path for store_ in armStores],
        "n_items": len(source.records_map),
        "n_disagreement": n_disagreement,
        "schema_version": AGGREGATION_SCHEMA,
        "source": "legacy_import",
        "legacy_path": source.file_path,
        "legacy_checks": checks,
    }

    for q_id in tqdm(pending, desc=f"{os.path.basename(out_path)}"):
        r = source.records_map[q_id]
        R1, R2, AR1, AR2 = r["Record1"], r["Record2"], r["AnswerRecord1"], r["AnswerRecord2"]
        init1, init2, final = str(AR1[0]), str(AR2[0]), str(r.get("MyAnswer") or "")
        times = int(r.get("Times", 0))

        # Round k: each agent sends its history up to the k-th debate prompt and generates R[2k+1]
        tokens_in = tokens_out = 0
        for k in range(1, times + 1):
            for R in (R1, R2):
                tokens_in += sum(model.countTokens(m["content"]) for m in R[:2 * k + 1])
                tokens_out += model.countTokens(R[2 * k + 1]["content"])

        disagreement = not compare(init1, init2)
        if disagreement and not compare(str(AR1[-1]), str(AR2[-1])):
            judge_prompt = DebateAggregator.getJudgePrompt(lang1, r["Question1"], R1[-1]["content"], R2[-1]["content"], lang1, lang2)
            tokens_in += model.countTokens(judge_prompt)
            tokens_out += model.countTokens(r.get("Result3") or "")

        store.add(AggregationRecord(
            item_id=q_id,
            aggregator_id="debate",
            candidate_arms=arm_ids,
            final_answer=final,
            off_menu=not any(compare(final, answer) for answer in (init1, init2)),
            n_rounds=times,
            presentation_order=arm_ids if disagreement else None,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            trace=None,  # the transcript stays in legacy_path
        ).to_dict())

    store.save()
    return checks


def importCell(model_name: str, dataset_name: str, args):
    pattern = os.path.join(args.baseline_dir, f"{model_name}_{dataset_name}_onelanguage_*.json")
    baseline_paths = sorted(glob.glob(pattern))
    if not baseline_paths:
        print(f"⏭️ Skip {model_name} | {dataset_name}: no baseline files")
        return

    model = None
    armStores, baselines = {}, {}
    for path in baseline_paths:
        language = re.search(r"_onelanguage_(\w+)\.json$", path).group(1)
        source = File(path)
        if model is None:
            model = ModelFactory().buildModel(ModelType(model_name), ModelConfig.from_dict(source.metadata["Model"]))
        arm = ArmSpec("L", lang=LANGUAGE_TO_LANG_CODE[language])
        armStores[language] = importArm(model, source, arm, armPath(args.armdir, model_name, dataset_name, arm))
        baselines[language] = source
        print(f"✅ {path} -> {armStores[language].path} (question mismatches: {armStores[language].metadata['question_text_mismatches']})")

    if args.sr_dir:
        # Self-reflection exists for english / chinese only; its file name matches the baseline file
        for language, baseStore in armStores.items():
            path = os.path.join(args.sr_dir, f"{model_name}_{dataset_name}_onelanguage_{language}.json")
            if not os.path.exists(path):
                continue
            arm = ArmSpec("F", lang=LANGUAGE_TO_LANG_CODE[language])
            store = importRefineArm(model, File(path), arm, baseStore, baselines[language],
                                    armPath(args.armdir, model_name, dataset_name, arm))
            print(f"✅ {path} -> {store.path} {store.metadata['legacy_checks']}")

    if args.english_cot_dir:
        path = os.path.join(args.english_cot_dir, f"{model_name}_{dataset_name}_onelanguage_english.json")
        if os.path.exists(path):
            arm = ArmSpec("W", questionSource="rewrite1")
            store = importArm(model, File(path), arm, armPath(args.armdir, model_name, dataset_name, arm))
            print(f"✅ {path} -> {store.path} (question mismatches: {store.metadata['question_text_mismatches']})")

    if args.skip_challenge:
        return

    compare = get_dataset_map()[DatasetType(dataset_name)].compareTwoAnswer
    pattern = os.path.join(args.challenge_dir, f"{model_name}_{dataset_name}_challenge_*_vs_*.json")
    for path in sorted(glob.glob(pattern)):
        lang1, lang2 = re.search(r"_challenge_(\w+?)_vs_(\w+)\.json$", path).groups()
        arms = (ArmSpec("L", lang=LANGUAGE_TO_LANG_CODE[lang1]), ArmSpec("L", lang=LANGUAGE_TO_LANG_CODE[lang2]))
        out_path = aggregationPath(args.aggdir, model_name, dataset_name, "debate", list(arms))
        checks = importDebate(model, File(path), arms, (armStores[lang1], armStores[lang2]),
                              (baselines[lang1], baselines[lang2]), out_path, compare)
        print(f"✅ {path} -> {out_path} {checks}")


def parseArgs():
    parser = ArgumentParser(description="Import legacy baseline / challenge results into result/arms and result/aggregations")
    parser.add_argument("-m", "--model", choices=LEGACY_MODELS, nargs="+", default=LEGACY_MODELS)
    parser.add_argument("-d", "--dataset", choices=ACTIVE_DATASETS, nargs="+", default=ACTIVE_DATASETS)
    parser.add_argument("--baseline-dir", dest="baseline_dir", default="result/baseline")
    parser.add_argument("--challenge-dir", dest="challenge_dir", default="result/challenge")
    parser.add_argument("--sr-dir", dest="sr_dir", default=None,
                        help="Also import result/self_reflection as the derived arms F:en / F:zh")
    parser.add_argument("--english-cot-dir", dest="english_cot_dir", default=None,
                        help="Also import result/english_cot as W:rewrite1 (rewrite version 1 + CoT + T0)")
    parser.add_argument("--skip-challenge", dest="skip_challenge", action="store_true")
    parser.add_argument("--armdir", default="result/arms")
    parser.add_argument("--aggdir", default="result/aggregations")
    parser.add_argument("-w", "--workers", type=int, default=2, help="Cells imported in parallel")
    return parser.parse_args()


def main():
    args = parseArgs()
    tasks = [(m, d) for m in args.model for d in args.dataset]
    print(f"🚀 Importing {len(tasks)} cells with {args.workers} workers")

    runTasks(importCell, tasks, args.workers, args)

    print("\n✅ Legacy import finished!")


if __name__ == "__main__":
    main()
