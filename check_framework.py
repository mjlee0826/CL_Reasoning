"""
Offline checks for the generation / aggregation framework. No API calls: a deterministic FakeModel
answers from a hash of the messages it receives.

    conda run -n clreasoning python check_framework.py
"""
import hashlib
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import httpx
import numpy as np
import openai

from Model.Model import Model
from Model.ModelConfig import ModelConfig
from Model.ModelType import ModelType
from Model.ModelFactory import ModelFactory
from Dataset.Dataset import Dataset
from Dataset.DatasetConfig import DatasetConfig
from Strategy.Strategy import Strategy
from Strategy.StrategyConfig import StrategyConfig
from Strategy.Generate import Generate
from Strategy.Aggregate import Aggregate
from Strategy.Rewrite import Rewrite
from Dataset.path import rewriteFileName
from Runner.tasks import defaultWorkers, interleaveByModel
from Arm.ArmSpec import ArmSpec
from Arm.PromptBuilder import PromptBuilder
from Arm.GenerationRecord import GenerationRecord
from Strategy.PromptAbstractFactory.PromptSelfReflectionCOTFactory import PromptSelfReflectionCOTFactory
from Strategy.PromptAbstractFactory.PromptFormatFactory import PromptFormatFactory
from Aggregator.Aggregator import AggregationItem, Candidate
from Aggregator.AggregatorConfig import AggregatorConfig
from Aggregator.AggregatorFactory import AggregatorFactory
from Aggregator.JudgeAggregator import JudgeAggregator
from File.File import File
from File.ResultStore import ResultStore
from Log.NoLog import NoLog
from Analysis.splitHalf import recoveryStats, makeSplits
import pandas as pd
from Strategy.StrategyType import StrategyType
from Test.TestTokenNums import TOKEN_COUNTERS
from Analysis.experimentPlan import ARMS_TO_PAIR
from Analysis.alignment import CellData, alignPair
from Analysis.metrics import computeRow, subsetMask
from Analysis.itemExport import pathRows, aggregationRows
from Analysis.itemMatrix import Block, Cell, PAIR_BY_LABEL
from Analysis.experimentPlan import PATHS
from Analysis import probe as rq1
from Analysis import crossJudge as rq2
from Analysis import crossJudgeStats as rq2stats
from Analysis import blockStats
from Analysis import menuVote as rq1k
from Strategy.CrossJudge import CrossJudge
from Strategy.MenuJudge import MenuJudge, ORDER_GROUPED
from Aggregator.MenuJudgeAggregator import MenuJudgeAggregator
from Analysis import menuJudge as rq1kj
from Analysis import menuJudgeStats as rq1kjstats

# Hashes of the prompts produced by the legacy OnlyOneLanguage.getPrompt (deleted 2026-10) BEFORE it was refactored onto PromptBuilder
GOLDEN_QUESTION = 'There is a Problem: \nWhat is 2+2?.\nAnd there are 5 choices\na ) 1 , b ) 2 , c ) 3 , d ) 4 , e ) 5\n'
GOLDEN_PROMPT_HASHES = {
    "cot|english": "160f1ba2026c97bb", "cot|chinese": "a4ea73cd7c234d14", "cot|japanese": "2df8509d86f0c774",
    "cot|russian": "84b8ea137eb74ef1", "cot|spanish": "5851547b30a600a0",
    "short_cot|english": "49db993fa6501d02", "short_cot|chinese": "69bac09a68788b3d", "short_cot|japanese": "56fd32aa3e484053",
    "short_cot|russian": "c40bc84dd1ab4656", "short_cot|spanish": "e4a224f6b1b2049e",
    "direct|english": "8d3f8621bccd6f51", "direct|chinese": "8ad765b552510cda", "direct|japanese": "7b56d270722222bb",
    "direct|russian": "333cff9f9829cf8c", "direct|spanish": "10568bf1989bc3d6",
}
# Hashes of the legacy SelfReflection.getPrompt prompts (Strategy/SelfReflection.py, deleted 2026-10) for GOLDEN_QUESTION
GOLDEN_REFLECTION_PREVIOUS = 'Step 1: 2+2=4.\n{"answer":"d"}'
GOLDEN_REFLECTION_HASHES = {"english": "39cc5dde24fa3346", "chinese": "c69b28888f158191", "japanese": "db7ea6cde0169086",
                            "russian": "f1a45ecd8ad135dd", "spanish": "314bc8189b6dcbb9"}
# Digest of the legacy Challenge strategy (Strategy/Challenge.py, deleted 2026-10) on the 300 scripted items of check 5:
# per item [Record1[2:], Record2[2:], AnswerRecord1, AnswerRecord2, judge output, rounds, final answer]
GOLDEN_DEBATE_DIGEST = "48a38e8f2d4b9d3b"
RECORD_FIELDS = {"item_id", "arm_id", "axis", "is_anchor", "lang", "raw_text", "parsed_answer", "parse_ok", "tokens_in",
                 "tokens_out", "model", "model_version_string", "temperature", "seed", "prompt_hash", "gold"}

failures = []


def check(name: str, condition: bool):
    print(("✅ " if condition else "❌ ") + name)
    if not condition:
        failures.append(name)


class FakeModel(Model):
    """
    Replies depend only on (messages, temperature, seed); markers in the last message inject failures.
    A judge prompt (it asks for {"choice": ...}) gets a candidate number instead of an option.
    """
    def __init__(self):
        super().__init__(ModelConfig(modelType="gpt4omini"))
        self.calls = 0
        self.failOn, self.noJsonOn = set(), set()
        self.lastMessages = None

    def reply(self, messages, temperature, seed):
        key = json.dumps(messages, ensure_ascii=False) + f"|{float(temperature)}|{seed}"
        h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)
        if any(marker in messages[-1]["content"] for marker in self.noJsonOn):
            return f"no json {h % 997}"
        if '{"choice":' in messages[-1]["content"]:
            return f'reasoning {h % 997}\n{{"choice":"{1 + h % 2}"}}'
        return f'reasoning {h % 997}\n{{"answer":"{"abc"[h % 3]}"}}'

    def _complete(self, messages, temperature, seed):
        self.calls += 1
        self.lastMessages = json.loads(json.dumps(messages))
        if any(marker in messages[-1]["content"] for marker in self.failOn):
            raise ValueError("fake API failure")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply(messages, temperature, seed)))],
                               model="fake-model-v1", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    def getListRes(self, promptList):
        self.calls += 1
        return self.reply(promptList, self.temperature, None)

    def getRes(self, prompt):
        return self.getListRes([{"role": "user", "content": prompt}])

    def getTokenLens(self, text):
        return len(text.split())


class FixedReplyModel(FakeModel):
    """Always replies with the same text."""
    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def reply(self, messages, temperature, seed):
        return self.text


def makeDataset(n: int) -> Dataset:
    dataset = Dataset(DatasetConfig(datasetType="mathqa", nums=n))
    dataset.data = [{"id": i, "question": f"Question {i}: pick a letter", "answer": "abc"[i % 3]} for i in range(n)]
    return dataset


PARSER = Strategy(StrategyConfig(strategyType="generate")).parseAnswer


def buildAggregator(aggregator_id, model, dataset, parser=PARSER):
    return AggregatorFactory().buildAggregator(aggregator_id, AggregatorConfig(seed=0), model, dataset, parser)


def candidate(arm, answer, raw=None, question="Q"):
    return Candidate(arm=arm, record={"parsed_answer": answer, "raw_text": raw or f'{arm.arm_id} {{"answer":"{answer}"}}'},
                     question=question)


def checkPrompts():
    ok = True
    for key, expected in GOLDEN_PROMPT_HASHES.items():
        style, lang = key.split("|")
        built = PromptBuilder.buildText(lang, GOLDEN_QUESTION, style)
        ok &= PromptBuilder.promptHash([{"role": "user", "content": built}]) == expected
    check("1. PromptBuilder reproduces the legacy OnlyOneLanguage prompts (3 styles × 5 languages)", ok)


def checkArmIds():
    good = ["L:en", "L:zh", "L:ja", "L:ru", "L:es", "S:T0.7:seed3", "S:T1.0:seed0", "S:T1.3:seed12",
            "R:short_cot", "R:direct", "P:expert", "W:rewrite1", "W:rewrite2", "F:en", "F:zh"]
    check("2a. arm_id round trip", all(ArmSpec.from_arm_id(a).arm_id == a for a in good))
    check("2b. only L:en is the anchor", [a for a in good if ArmSpec.from_arm_id(a).is_anchor] == ["L:en"])
    rejected = 0
    for bad in ["R:cot", "S:T0.75:seed1", "S:T0.0:seed1", "L:de", "K:2", "W:original", "S:T0.7", "P:Expert", "L:EN", "F:de", "W:rewrite", "W:rewrite3"]:
        try:
            ArmSpec.from_arm_id(bad)
        except ValueError:
            rejected += 1
    for spec in [dict(axis="L", lang="ja", temperature=0.7, seed=1), dict(axis="F", lang="zh", temperature=0.7, seed=1),
                 dict(axis="F", lang="en", promptStyle="direct")]:
        try:
            ArmSpec(**spec)
        except ValueError:
            rejected += 1
    check("2c. invalid / non-canonical / multi-factor arms are rejected (15)", rejected == 15)

    derived = ArmSpec.from_arm_id("F:zh")
    check("2d. F arms are derived from L:{lang}",
          derived.is_derived and derived.base_arm_id == "L:zh" and derived.file_stem == "F_zh"
          and derived.language == "chinese" and not ArmSpec.from_arm_id("L:zh").is_derived
          and ArmSpec.from_arm_id("L:zh").base_arm_id is None)

    v1, v2, plain = (ArmSpec.from_arm_id(a).to_dataset_config("mathqa", 10) for a in ("W:rewrite1", "W:rewrite2", "L:en"))
    check("2f. rewrite versions map to the right dataset config and file name",
          v1.useRewrite and v1.rewriteVersion == 1 and v2.useRewrite and v2.rewriteVersion == 2 and not plain.useRewrite
          and rewriteFileName("mathqa", 1) == "mathqa_english.json" and rewriteFileName("mathqa", 2) == "mathqa_english_v2.json")


def checkRefinePrompt():
    """A derived arm's prompt must match the legacy SelfReflection prompts so imported SR prompts rebuild exactly."""
    arm = ArmSpec.from_arm_id("F:zh")
    question, previous = "問題：選一個字母", "先前輸出 {\"answer\":\"a\"}"
    expected = PromptSelfReflectionCOTFactory().getPrompt("chinese", question, previous) + PromptFormatFactory().getPrompt("chinese")
    built = PromptBuilder(arm).text(question, previous)
    raised = False
    try:
        PromptBuilder(arm).text(question)
    except ValueError:
        raised = True
    golden = all(PromptBuilder.promptHash([{"role": "user", "content": PromptBuilder.buildReflectionText(
        lang, GOLDEN_QUESTION, GOLDEN_REFLECTION_PREVIOUS)}]) == expected_hash for lang, expected_hash in GOLDEN_REFLECTION_HASHES.items())
    check("2e. derived prompt == legacy SelfReflection prompt (5 languages), and fails loudly without the base output",
          built == expected and raised and golden)

    cot = PromptBuilder.buildText("english", question, "cot")
    expert = PromptBuilder(ArmSpec.from_arm_id("P:expert")).text(question)
    skeptic = PromptBuilder(ArmSpec.from_arm_id("P:skeptic")).text(question)
    raised = False
    try:
        PromptBuilder(ArmSpec.from_arm_id("P:novice")).text(question)
    except ValueError:
        raised = True
    check("2g. persona prompts = persona sentence + the unchanged anchor prompt; unknown personas fail",
          expert.startswith("You are a seasoned expert") and expert.endswith(cot)
          and skeptic.startswith("You are a careful skeptic") and skeptic.endswith(cot) and raised)


def checkRunners():
    tasks = interleaveByModel(["gpt4omini", "qwen", "gemini"], ["mathqa", "mmlu"], ["S:T1.0:seed1", "R:short_cot"])
    check(f"10. default workers = one per task and tasks rotate through the models (first: {[t[0] for t in tasks[:3]]})",
          defaultWorkers(48) == 48 and defaultWorkers(2) == 2 and defaultWorkers(0) == 1 and len(tasks) == 12
          and [t[0] for t in tasks[:6]] == ["gpt4omini", "qwen", "gemini"] * 2
          and tasks[0] == ("gpt4omini", "mathqa", "S:T1.0:seed1"))


def checkAggregators():
    model, dataset = FakeModel(), makeDataset(10)
    A, B, C = (ArmSpec.from_arm_id(a) for a in ("L:en", "S:T0.7:seed1", "L:ja"))
    judge = buildAggregator("judge", model, dataset)

    calls = model.calls
    record = judge.aggregate(AggregationItem(1, [candidate(A, "a"), candidate(B, "a")], "Q", [A.arm_id, B.arm_id]))
    check("3a. unanimous items are a no-op (no call, 0 tokens, no order)",
          model.calls == calls and record.final_answer == "a" and not record.off_menu
          and record.tokens_in == record.tokens_out == 0 and record.presentation_order is None)

    blind = buildAggregator("blind", model, dataset)
    ok = blind.aggregate(AggregationItem(2, [candidate(B, "b"), candidate(A, "a")], "Q")).final_answer == "a"
    try:
        blind.validateCandidates([B, C])
        ok = False
    except ValueError:
        pass
    check("3b. Blind keeps the anchor and rejects candidate sets without one", ok)

    vote3 = buildAggregator("vote3", model, dataset)
    check("3c. Vote@3 majority", vote3.aggregate(AggregationItem(3, [candidate(A, "a"), candidate(B, "b"), candidate(C, "b")], "Q")).final_answer == "b")

    v2 = buildAggregator("v2", model, dataset)
    picks = [v2.aggregate(AggregationItem(i, [candidate(A, "a"), candidate(B, "b")], "Q")).final_answer for i in range(2000)]
    again = [v2.aggregate(AggregationItem(i, [candidate(A, "a"), candidate(B, "b")], "Q")).final_answer for i in range(2000)]
    check(f"3d. Vote@2 tie-break is reproducible and fair (first picked {picks.count('a') / 2000:.3f})",
          picks == again and 0.45 < picks.count("a") / 2000 < 0.55)

    # Synthetic population: V2 recovery ≈ 0 and Blind recovery == recovery_blind exactly
    rng = np.random.default_rng(1)
    n = 20000
    cA, cB = rng.random(n) < 0.6, rng.random(n) < 0.6
    wrong = lambda: "bc"[int(rng.integers(2))]
    answersA = ["a" if x else wrong() for x in cA]
    answersB = ["a" if x else wrong() for x in cB]
    dis = np.array([x != y for x, y in zip(answersA, answersB)])
    results = {}
    for aggregator in (v2, blind):
        final = [aggregator.aggregate(AggregationItem(i, [candidate(A, x), candidate(B, y)], "Q")).final_answer
                 for i, (x, y) in enumerate(zip(answersA, answersB))]
        results[aggregator.config.aggregatorType] = recoveryStats(cA, cB, np.array([f == "a" for f in final]), dis, np.ones(n, dtype=bool))
    check(f"3e. V2 recovery ≈ 0 ({results['v2']['recovery']:+.4f})", abs(results["v2"]["recovery"]) < 0.05)
    check("3f. Blind recovery == recovery_blind", abs(results["blind"]["recovery"] - results["blind"]["recovery_blind"]) < 1e-12)

    # Revise: mirror of Blind, always keeps the derived candidate
    revise = buildAggregator("revise", model, dataset)
    D = ArmSpec.from_arm_id("F:en")
    final = [revise.aggregate(AggregationItem(i, [candidate(A, x), candidate(D, y)], "Q")).final_answer
             for i, (x, y) in enumerate(zip(answersA, answersB))]
    stats = recoveryStats(cA, cB, np.array([f == "a" for f in final]), dis, np.ones(n, dtype=bool))
    ok = abs(stats["recovery"] + results["blind"]["recovery"]) < 1e-12
    try:
        revise.validateCandidates([A, B])
        ok = False
    except ValueError:
        pass
    check(f"3i. Revise recovery == −recovery_blind ({stats['recovery']:+.4f}) and needs a derived candidate", ok)

    item = AggregationItem(4, [candidate(A, "d", raw="RAW-ANCHOR"), candidate(B, "e", raw="RAW-OTHER")], "Q", [B.arm_id, A.arm_id])
    record = judge.aggregate(item)
    prompt = model.lastMessages[0]["content"]
    check("3g. Judge shows candidates in presentation_order with neutral labels",
          "Answer 1\n```\nRAW-OTHER" in prompt and "Answer 2\n```\nRAW-ANCHOR" in prompt and "english" not in prompt.lower().split("```")[0]
          and record.presentation_order == [B.arm_id, A.arm_id])
    check("3h. the judge prompt asks for a candidate number, not an option",
          '{"choice":"answer number"}' in prompt and '{"answer"' not in prompt and "an integer from 1 to 2" in prompt)

    # The choice refers to the presentation order: order [B, A] -> choice 1 = B's answer, choice 2 = A's
    outcomes = {}
    for name, reply in [("1", 'r\n{"choice":"1"}'), ("2", 'r\n{"choice": "Answer 2"}'), ("letter", 'r\n{"answer":"d"}'),
                        ("range", 'r\n{"choice":"3"}'), ("last", '{"choice":"2"} then\n{"choice":"1"}')]:
        outcomes[name] = buildAggregator("judge", FixedReplyModel(reply), dataset).aggregate(item)
    check("3j. the final answer is the chosen candidate's answer (in presentation order); a missing / out-of-range "
          "choice gives no answer and is the only off-menu case",
          outcomes["1"].final_answer == "e" and outcomes["1"].trace["chosen_arm"] == B.arm_id and not outcomes["1"].off_menu
          and outcomes["2"].final_answer == "d" and outcomes["2"].trace["choice"] == 2
          and outcomes["letter"].final_answer == "" and outcomes["letter"].off_menu and outcomes["letter"].trace["choice"] is None
          and outcomes["range"].final_answer == "" and outcomes["range"].off_menu
          and outcomes["last"].final_answer == "e"
          and JudgeAggregator.parseChoice('{"choice":"2"}', 5) == 2 and JudgeAggregator.parseChoice('{"choice":"0"}', 5) is None)


def checkBalancing():
    ids = list(range(101))
    two = Aggregate.balancedOrders(["L:en", "L:ja"], ids, 0)
    first = sum(order[0] == "L:en" for order in two.values())
    five_arms = ["L:en", "L:zh", "L:ja", "L:ru", "L:es"]
    five = Aggregate.balancedOrders(five_arms, ids, 0)
    counts = [sum(order[p] == arm for order in five.values()) for arm in five_arms for p in range(5)]
    check(f"4a. K=2: anchor first on {first}/101", first in (50, 51))
    check(f"4b. K=5: (arm, position) counts within ±1 ({min(counts)}–{max(counts)})", max(counts) - min(counts) <= 1)
    check("4c. balancing is deterministic", Aggregate.balancedOrders(["L:en", "L:ja"], ids, 0) == two)


def checkDebateEquivalence():
    model, dataset = FakeModel(), makeDataset(10)
    debate = buildAggregator("debate", model, dataset)
    arms = (ArmSpec.from_arm_id("L:en"), ArmSpec.from_arm_id("L:ja"))

    rows, rounds, judged = [], set(), 0
    for i in range(300):
        q1, q2 = f"Q{i} en", f"Q{i} ja"
        r1, r2 = f'init1 {i} {{"answer":"a"}}', f'init2 {i} {{"answer":"b"}}'
        record = debate.aggregate(AggregationItem(i, [candidate(arms[0], "a", r1, q1), candidate(arms[1], "b", r2, q2)], "Q"))
        trace = record.trace
        rows.append([trace["Record1"], trace["Record2"], trace["AnswerRecord1"], trace["AnswerRecord2"], trace["Result3"],
                     record.n_rounds, record.final_answer])
        rounds.add(record.n_rounds)
        judged += bool(trace["Result3"])
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    check(f"5. DebateAggregator == legacy Challenge on 300 scripted items (rounds {sorted(rounds)}, judge called {judged})",
          digest == GOLDEN_DEBATE_DIGEST)


def makeDerivedArmFile(path, base_file, dataset, arm, model):
    """Writes an F-arm file whose prompts are built from the base arm's outputs (what import_legacy produces)."""
    store = ResultStore(path)
    store.metadata = {
        "Model": model.config.to_dict(),
        "Dataset": dataset.config.to_dict(),
        "Strategy": {"strategyType": "generate", "languages": [arm.language], "promptStyle": arm.promptStyle},
        "Arm": arm.to_dict(), "schema_version": "generation/v1", "source": "test",
    }
    questions = {d["id"]: d["question"] for d in dataset.getData()}
    for item_id, base in sorted(base_file.records_map.items()):
        prompt = PromptBuilder(arm).text(questions[item_id], base["raw_text"])
        raw_text = f'revised {item_id}\n{{"answer":"{"abc"[(item_id + 1) % 3]}"}}'
        store.add(GenerationRecord(
            item_id=item_id, arm_id=arm.arm_id, axis=arm.axis, is_anchor=arm.is_anchor, lang=arm.lang,
            raw_text=raw_text, parsed_answer=PARSER(raw_text), parse_ok=True,
            tokens_in=len(prompt.split()), tokens_out=len(raw_text.split()),
            model=model.config.modelType, model_version_string="fake-model-v1", temperature=0.0, seed=None,
            prompt_hash=PromptBuilder.promptHash([{"role": "user", "content": prompt}]),
            gold=str(dataset.data[item_id]["answer"]),
        ).to_dict())
    store.save()


def checkGenerateAndAggregate(tmp: str):
    model, dataset = FakeModel(), makeDataset(30)
    anchor, sampled = ArmSpec.from_arm_id("L:en"), ArmSpec.from_arm_id("S:T0.7:seed1")
    paths = {arm.arm_id: os.path.join(tmp, "arms", f"{arm.file_stem}.json") for arm in (anchor, sampled)}

    def generate(arm):
        strategy = Generate(StrategyConfig(strategyType="generate", languages=[arm.language]), model, dataset, NoLog(),
                            arm, ResultStore(paths[arm.arm_id], checkpoint_every=7))
        return strategy.getRes()

    model.failOn, model.noJsonOn = {"Question 3:", "Question 7:"}, {"Question 5:"}
    failed = generate(anchor)
    check("6a. API failures are not written", sorted(failed) == [3, 7] and len(ResultStore(paths["L:en"]).records) == 28)

    model.failOn = set()
    calls = model.calls
    failed = generate(anchor)
    file = File(paths["L:en"])
    check("6b. rerun only sends the missing items", failed == [] and model.calls - calls == 2 and len(file.records_map) == 30)

    record = file.getRecordById(0)
    messages = PromptBuilder(anchor).messages(dataset.data[0]["question"])
    check("6c. GenerationRecord fields",
          set(record) == RECORD_FIELDS and record["prompt_hash"] == PromptBuilder.promptHash(messages)
          and record["tokens_out"] == len(record["raw_text"].split()) and record["model_version_string"] == "fake-model-v1"
          and record["is_anchor"] and record["parse_ok"] and record["lang"] == "en" and record["seed"] is None)
    check("6d. parse failures are kept with parse_ok = False",
          not file.getRecordById(5)["parse_ok"] and file.getRecordById(5)["parsed_answer"] == "")
    usage = file.metadata["api_usage"]
    check(f"6e. api_usage counts every call ({usage})", usage["calls"] == 32 and usage["calls_without_usage"] == 2)
    check("6f. no temp file left behind", not os.path.exists(paths["L:en"] + ".tmp"))

    model.noJsonOn = set()
    generate(sampled)
    arm_files = [File(paths["L:en"]), File(paths["S:T0.7:seed1"])]

    def aggregate(aggregator_id, files=arm_files):
        out = os.path.join(tmp, "aggregations", f"{aggregator_id}.json")
        strategy = Aggregate(StrategyConfig(strategyType="aggregate", languages=["english", "english"]), model, dataset, NoLog(),
                             buildAggregator(aggregator_id, model, dataset, None), [anchor, sampled], files, [dataset, dataset],
                             ResultStore(out))
        return strategy.getRes(), out

    for aggregator_id in ("v2", "blind", "judge", "debate"):
        failed, out = aggregate(aggregator_id)
        check(f"7a. Aggregate {aggregator_id}: all 30 items written", failed == [] and len(File(out).records_map) == 30)

    judge_file = File(os.path.join(tmp, "aggregations", "judge.json"))
    records = list(judge_file.records_map.values())
    disagreements = [r for r in records if r["presentation_order"] is not None]
    first = sum(r["presentation_order"][0] == "L:en" for r in disagreements)
    check(f"7b. judge orders balanced on the {len(disagreements)} disagreements (anchor first {first})",
          len(disagreements) == judge_file.metadata["n_disagreement"] and first in (len(disagreements) // 2, (len(disagreements) + 1) // 2))

    calls = model.calls
    aggregate("judge")
    check("7c. resumed aggregation makes no new calls", model.calls == calls)

    # A judge file written with the earlier prompt (no prompt_version) must not be resumed
    old_path = os.path.join(tmp, "aggregations", "judge_old_prompt.json")
    old = ResultStore(os.path.join(tmp, "aggregations", "judge.json"))
    old.metadata.pop("prompt_version")
    old.path = old_path
    old.save()
    try:
        Aggregate(StrategyConfig(strategyType="aggregate", languages=["english", "english"]), model, dataset, NoLog(),
                  buildAggregator("judge", model, dataset, None), [anchor, sampled], arm_files, [dataset, dataset],
                  ResultStore(old_path))
        raised = False
    except ValueError:
        raised = True
    check("7e. judge files record their prompt_version; a file written with another version is not resumed",
          judge_file.metadata["prompt_version"] == "choice-v1" and raised)

    tampered = File(paths["S:T0.7:seed1"])
    tampered.records_map[0]["prompt_hash"] = "0" * 16
    try:
        aggregate("v2", [arm_files[0], tampered])
        check("7d. prompt_hash mismatch is detected", False)
    except ValueError:
        check("7d. prompt_hash mismatch is detected", True)

    # Derived arm (F:en): its prompt contains the base arm's output, like the imported self-reflection arms
    derived = ArmSpec.from_arm_id("F:en")
    derived_path = os.path.join(tmp, "arms", "F_en.json")
    makeDerivedArmFile(derived_path, File(paths["L:en"]), dataset, derived, model)

    def generateDerived(name, baseStore):
        return Generate(StrategyConfig(strategyType="generate", languages=[derived.language]), model, dataset, NoLog(),
                        derived, ResultStore(os.path.join(tmp, "arms", name)), baseStore)

    refused = []
    for baseStore, error in ((None, FileNotFoundError), (ResultStore(os.path.join(tmp, "arms", "missing.json")), FileNotFoundError)):
        try:
            generateDerived("F_refused.json", baseStore)
            refused.append(False)
        except error:
            refused.append(True)
    partial = ResultStore(paths["L:en"])
    partial.records.pop(29)
    try:
        generateDerived("F_refused.json", partial)
        refused.append(False)
    except ValueError:
        refused.append(True)

    base = ResultStore(paths["L:en"])
    base.records[4] = {**base.records[4], "raw_text": ""}   # in memory only: an item whose base output is empty
    calls = model.calls
    failed = generateDerived("F_generated.json", base).getRes()
    generated = File(os.path.join(tmp, "arms", "F_generated.json"))
    questions = {data["id"]: data["question"] for data in dataset.getData()}
    prompts_ok = all(record["prompt_hash"] == PromptBuilder.promptHash(PromptBuilder(derived).messages(questions[i], base.records[i]["raw_text"]))
                     for i, record in generated.records_map.items())
    empty = generated.getRecordById(4)
    check("8a. Generate builds an F arm from its complete base arm; an empty base output is a no-answer record without a call",
          all(refused) and failed == [] and len(generated.records_map) == 30 and model.calls - calls == 29 and prompts_ok
          and not empty["parse_ok"] and empty["tokens_out"] == 0 and generated.metadata["no_answer_fill"]["ids"] == [4]
          and generated.metadata["base_arm_file"] == paths["L:en"])

    def aggregateArms(aggregator_id, arms, files, name):
        out = os.path.join(tmp, "aggregations", f"{name}.json")
        strategy = Aggregate(StrategyConfig(strategyType="aggregate", languages=["english", "english"]), model, dataset,
                             NoLog(), buildAggregator(aggregator_id, model, dataset, None), arms, files,
                             [dataset, dataset], ResultStore(out))
        return strategy.getRes(), File(out)

    failed, out_file = aggregateArms("revise", [anchor, derived], [File(paths["L:en"]), File(derived_path)], "revise")
    derived_records = File(derived_path).records_map
    check("8b. Revise reproduces the derived arm's answers and verifies its prompt_hash",
          failed == [] and out_file.metadata["prompt_hash_unverified_arms"] == []
          and all(r["final_answer"] == derived_records[r["item_id"]]["parsed_answer"] for r in out_file.records_map.values()))

    _, out_file = aggregateArms("v2", [derived, sampled], [File(derived_path), File(paths["S:T0.7:seed1"])], "v2_derived")
    check("8c. a derived arm without its base is reported as unverified",
          out_file.metadata["prompt_hash_unverified_arms"] == ["F:en"])


def checkRewrite(tmp: str):
    """A later rewrite version sees the earlier ones, never stores failed calls, and resumes."""
    model, dataset = FakeModel(), makeDataset(10)
    previous = {i: [f"first rewrite {i}"] for i in range(10)}
    path = os.path.join(tmp, "rewrite", "mathqa_english_v2.json")

    def rewrite():
        config = StrategyConfig(strategyType="rewrite", languages=["english"], rewriteVersion=2)
        return Rewrite(config, model, dataset, NoLog(), ResultStore(path, key="id"), previous).getRes()

    model.failOn = {"Question 4:"}
    failed = rewrite()
    prompt = model.lastMessages[0]["content"]
    check("9a. rewrite v2 sees the earlier version and never writes failed calls",
          failed == [4] and len(ResultStore(path, key="id").records) == 9
          and "Existing rewrite 1" in prompt and "first rewrite 9" in prompt)

    model.failOn = set()
    calls = model.calls
    failed = rewrite()
    check("9b. rerunning a rewrite only fills the missing items",
          failed == [] and model.calls - calls == 1 and len(File(path).records_map) == 10)

    # A rewrite that drops the answer options gets one follow-up turn; if it still drops them, it is not written
    question = 'There is a Question: \nWhich is red?\nAnd there are multiple choices:\nA: sky\nB: apple\n' \
               'At the end of your response, provide your answer in this exact JSON format: \n{"answer": "your_letter_choice"}\n'
    full = 'Which item is red?\nA: sky\nB: apple\n{"answer": "your_letter_choice"}'

    class ScriptedModel(FakeModel):
        def __init__(self, replies):
            super().__init__()
            self.replies = list(replies)
        def _complete(self, messages, temperature, seed):
            self.calls += 1
            self.lastMessages = json.loads(json.dumps(messages))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.replies.pop(0)))],
                                   model="fake-model-v1", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    one_item = makeDataset(1)
    one_item.data[0]["question"] = question
    outcomes = []
    for name, replies in [("repaired", ["Which item is red?", full]), ("rejected", ["Which item is red?", "Which item is red?"])]:
        store_path = os.path.join(tmp, "rewrite", f"{name}.json")
        scripted = ScriptedModel(replies)
        config = StrategyConfig(strategyType="rewrite", languages=["english"], rewriteVersion=2)
        failed = Rewrite(config, scripted, one_item, NoLog(), ResultStore(store_path, key="id"), {0: ["v1"]}).getRes()
        store = ResultStore(store_path, key="id")
        outcomes.append((failed, len(store.records), store.metadata["validation"], scripted.lastMessages[-1]["content"]))
    (f1, n1, v1, last1), (f2, n2, v2, _) = outcomes
    check("9c. dropped options trigger one repair turn; a rewrite that still drops them is not written",
          Rewrite.preservesOptions(question, full) and not Rewrite.preservesOptions(question, "Which item is red?")
          and f1 == [] and n1 == 1 and v1 == {"repair_turns": 1, "rejected": 0, "fence_stripped": 0} and "dropped" in last1
          and f2 == [0] and n2 == 0 and v2 == {"repair_turns": 1, "rejected": 1, "fence_stripped": 0})

    # A fence around the whole output is removed before checking and storing; other fences are kept
    store_path = os.path.join(tmp, "rewrite", "fenced.json")
    scripted = ScriptedModel(["```\n" + full + "\n```"])
    config = StrategyConfig(strategyType="rewrite", languages=["english"], rewriteVersion=2)
    failed = Rewrite(config, scripted, one_item, NoLog(), ResultStore(store_path, key="id"), {0: ["v1"]}).getRes()
    store = ResultStore(store_path, key="id")
    check("9d. a fence wrapping the whole rewrite is removed; fences elsewhere are left alone",
          failed == [] and store.records[0]["Rewritten"] == full and store.metadata["validation"]["fence_stripped"] == 1
          and Rewrite.stripOuterFence("```text\nabc\n```") == "abc"
          and Rewrite.stripOuterFence("intro\n```\nabc\n```") == "intro\n```\nabc\n```"
          and Rewrite.stripOuterFence("```\na\n```\nb\n```") == "```\na\n```\nb\n```")


class RefusingModel(FakeModel):
    """Blocks prompts containing a marker: Gemini-style (no message) or DashScope-style (400 data_inspection_failed)."""
    def __init__(self, blockOn=(), moderateOn=()):
        super().__init__()
        self.blockOn, self.moderateOn = set(blockOn), set(moderateOn)

    def _complete(self, messages, temperature, seed):
        content = messages[-1]["content"]
        if any(marker in content for marker in self.blockOn):
            self.calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=None, finish_reason="content_filter")],
                                   model="fake-model-v1", usage=None)
        if any(marker in content for marker in self.moderateOn):
            self.calls += 1
            raise openai.BadRequestError(
                "Output data may contain inappropriate content",
                response=httpx.Response(400, request=httpx.Request("POST", "https://example.invalid")),
                body={"code": "data_inspection_failed", "type": "data_inspection_failed"})
        return super()._complete(messages, temperature, seed)


def checkRefusals(tmp: str):
    model = RefusingModel(blockOn={"Question 2:"}, moderateOn={"Question 4:"})
    blocked = model.generate([{"role": "user", "content": "Question 2: x"}])
    moderated = model.generate([{"role": "user", "content": "Question 4: x"}])
    check("11a. safety blocks and moderation errors are refusals (ok, empty text), not API failures",
          blocked.ok and blocked.refused and blocked.text == "" and moderated.ok and moderated.refused
          and "data_inspection_failed" in moderated.refusal_reason
          and not model.generate([{"role": "user", "content": "Question 1: x"}]).refused)

    dataset, anchor = makeDataset(6), ArmSpec.from_arm_id("L:en")
    path = os.path.join(tmp, "refusals", "L_en.json")
    failed = Generate(StrategyConfig(strategyType="generate", languages=["english"]), model, dataset, NoLog(), anchor,
                      ResultStore(path)).getRes()
    file = File(path)
    refused = sorted(r["item_id"] for r in file.metadata.get("refusals", []))
    check("11b. a refused item is written as an output without an answer and listed in metadata",
          failed == [] and len(file.records_map) == 6 and refused == [2, 4]
          and file.getRecordById(2)["raw_text"] == "" and not file.getRecordById(2)["parse_ok"]
          and file.metadata["api_usage"]["refusals"] == 2)

    judge = buildAggregator("judge", RefusingModel(blockOn={"Answer 1"}), dataset)
    A, B = ArmSpec.from_arm_id("L:en"), ArmSpec.from_arm_id("S:T0.7:seed1")
    record = judge.aggregate(AggregationItem(0, [candidate(A, "a"), candidate(B, "b")], "Q", [A.arm_id, B.arm_id]))
    check("11c. a refused judge call gives an empty final answer instead of failing the item",
          record.final_answer == "" and record.trace.get("refused_calls") == 1)


def writeResultFile(path: str, metadata: dict, records: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([metadata, *records], f)


def checkAnalysis(tmp: str):
    """Analysis package on synthetic files: alignment, the blind row, the both_answered subset, unplanned files."""
    armdir, aggdir = os.path.join(tmp, "analysis", "arms"), os.path.join(tmp, "analysis", "aggregations")
    rng = np.random.default_rng(2)
    n = 400
    answers = {arm_id: [str(rng.choice(["a", "a", "b", "c", ""])) for _ in range(n)] for arm_id in ("L:en", "L:zh")}
    for arm_id, arm_answers in answers.items():
        writeResultFile(os.path.join(armdir, "m", "mathqa", f"{ArmSpec.from_arm_id(arm_id).file_stem}.json"), {},
                        [{"item_id": i, "parsed_answer": a, "parse_ok": a != "", "gold": "a", "tokens_out": 10} for i, a in enumerate(arm_answers)])
    finals = [x if x == y else str(rng.choice([x, y, "c", ""])) for x, y in zip(answers["L:en"], answers["L:zh"])]
    judge = [{"item_id": i, "final_answer": f, "off_menu": f not in (x, y), "tokens_out": 0 if x == y else 5}
             for i, (f, x, y) in enumerate(zip(finals, answers["L:en"], answers["L:zh"]))]
    cellDir = os.path.join(aggdir, "m", "mathqa")
    writeResultFile(os.path.join(cellDir, "judge__L_en__L_zh.json"), {"candidate_arms": ["L:en", "L:zh"]}, judge)
    writeResultFile(os.path.join(cellDir, "vote3__L_en__L_zh__L_ja.json"), {}, [])

    cell = CellData(armdir, aggdir, "m", "mathqa")
    pair = ARMS_TO_PAIR[frozenset(["L:en", "L:zh"])]
    arrays = alignPair(cell, pair, cell.aggregationFiles[(pair, "judge")])
    cA, cB = np.array([a == "a" for a in answers["L:en"]]), np.array([a == "a" for a in answers["L:zh"]])
    dis = np.array([x != y for x, y in zip(answers["L:en"], answers["L:zh"])])
    expected = recoveryStats(cA, cB, np.array([f == "a" for f in finals]), dis, np.ones(n, dtype=bool))
    judgeRow = computeRow(arrays, pair, "judge")
    blindRow = computeRow(arrays, pair, "blind")
    check("12a. analysis rows: judge recovery matches recoveryStats; Blind row = stronger side, recovery_H2 = recovery_blind_H2",
          judgeRow["recovery"] == expected["recovery"] and judgeRow["tok_out_agg"] == 5 * dis.mean()
          and blindRow["acc_final"] == max(cA.mean(), cB.mean()) and blindRow["tok_out_agg"] == 0
          and abs(blindRow["recovery_H2"] - blindRow["recovery_blind_H2"]) < 1e-12)

    both = subsetMask(arrays, "both_answered")
    bothRow = computeRow(arrays.subset(both), pair, "judge")
    noAnswer = np.mean([f == "" for f, d, b in zip(finals, dis, both) if d and b])
    check(f"12b. both_answered keeps {bothRow['n']}/{n} items, counts the judge's no-answers as wrong ({bothRow['agg_no_answer']:.3f})",
          bothRow["n"] == sum(x != "" and y != "" for x, y in zip(answers["L:en"], answers["L:zh"]))
          and bothRow["parse_fail_a"] == 0 and bothRow["agg_no_answer"] == noAnswer)

    paths = pathRows(cell)
    aggregations = aggregationRows(cell, pair, "judge", cell.aggregationFiles[(pair, "judge")])
    en = [r for r in paths if r["path"] == "EN"]
    check(f"12d. per-item export: {len(paths)} path rows, {len(aggregations)} judge rows, same correctness as the cell arrays",
          len(paths) == 2 * n and {r["path"] for r in paths} == {"EN", "ZH"}
          and [r["correct"] for r in en] == list(arrays.correct_a) and [r["answered"] for r in en] == list(arrays.answered_a)
          and [r["correct"] for r in aggregations] == list(arrays.final_correct)
          and [r["dis"] for r in aggregations] == list(arrays.dis)
          and [r["answered"] for r in aggregations] == list(arrays.final_answered))

    agree = int(np.flatnonzero(~dis)[0])
    judge[agree]["final_answer"] = "c" if answers["L:en"][agree] != "c" else "b"
    writeResultFile(os.path.join(cellDir, "judge__L_en__L_zh.json"), {"candidate_arms": ["L:en", "L:zh"]}, judge)
    try:
        alignPair(cell, pair, cell.aggregationFiles[(pair, "judge")])
        raised = False
    except ValueError:
        raised = True
    check("12c. unplanned aggregation files are reported; a changed answer on an agreement item fails loudly",
          cell.unusedFiles == [os.path.join(cellDir, "vote3__L_en__L_zh__L_ja.json")] and raised)


class RecordingClient:
    """Stands in for an OpenAI client: records the request kwargs and returns a fixed response."""
    def __init__(self, model_id: str):
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.model_id = model_id

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"a"}'), finish_reason="stop")],
                               model=self.model_id, usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


def checkModelRequests():
    """The request each model sends (no API call: the client is replaced)."""
    for key in ("GEMINI_API_KEY", "DEEPSEEK_API_KEY"):
        os.environ.setdefault(key, "offline-check")
    sent = {}
    for model_type in (ModelType.GEMINI, ModelType.GEMINI31FLASHLITE, ModelType.DEEPSEEK41FLASH):
        model = ModelFactory().buildModel(model_type, ModelConfig(modelType=model_type.value))
        model.client = RecordingClient(model.modelName)
        response = model.generate([{"role": "user", "content": "q"}], temperature=0.0, seed=1)
        sent[model_type] = (model, model.client.kwargs, response)

    gemini, kwargs25, _ = sent[ModelType.GEMINI]
    lite, kwargs31, _ = sent[ModelType.GEMINI31FLASHLITE]
    flash, kwargsDs, responseDs = sent[ModelType.DEEPSEEK41FLASH]
    check("14. new models: Gemini 3.1 Flash-Lite = minimal thinking, 8192 tokens, no seed; DeepSeek V4.1 Flash = thinking "
          "disabled, 8192 tokens, seed; Gemini 2.5 request unchanged",
          kwargs31["model"] == "gemini-3.1-flash-lite" and kwargs31["reasoning_effort"] == "minimal"
          and kwargs31["max_tokens"] == 8192 and "seed" not in kwargs31 and not lite.SUPPORTS_SEED
          and kwargsDs["model"] == "deepseek-flash" and kwargsDs["extra_body"] == {"thinking": {"type": "disabled"}}
          and kwargsDs["max_tokens"] == 8192 and kwargsDs["seed"] == 1 and kwargsDs["temperature"] == 0.0
          and responseDs.model_version == "deepseek-flash@DeepSeek-V4.1-Flash"
          and kwargs25 == dict(model="gemini-2.5-flash-lite", messages=[{"role": "user", "content": "q"}], max_tokens=4096,
                               temperature=0.0, stream=False))


def syntheticBlocks() -> dict:
    """Two models × two datasets with all 14 paths and judge cells EN+ZH, ZH+JA, EN+S1, P1+P2."""
    blocks = {}
    for model in ("m1", "m2"):
        for dataset, n in (("mathqa", 200), ("mmlu", 120)):
            rng = np.random.default_rng([len(model), n])
            correct = {code: rng.random(n) < 0.6 for code in PATHS}
            answered = {code: np.ones(n, dtype=bool) for code in PATHS}
            answered["ZH"][:5] = False                                  # both_answered drops these for pairs with ZH
            if (model, dataset) == ("m1", "mathqa"):
                correct["ZH"], correct["EN"] = np.ones(n, dtype=bool), np.zeros(n, dtype=bool)
            cells = {}
            for label in ("EN+ZH", "ZH+JA", "EN+S1", "P1+P2"):
                pair = PAIR_BY_LABEL[label]
                cell = Cell(pair=pair, aggregator="judge", correct_agg=None, dis=None)
                a, b = cell.codes
                dis = (correct[a] != correct[b]) | (rng.random(n) < 0.1)
                cell.dis = dis
                cell.correct_agg = np.where(dis, rng.random(n) < 0.5, correct[a])
                if (model, dataset, label) == ("m1", "mathqa", "EN+ZH"):
                    cell.correct_agg = np.zeros(n, dtype=bool)          # aggregating never pays here
                cells[(label, "judge")] = cell
            blocks[(model, dataset)] = Block(model=model, dataset=dataset, item_ids=np.arange(n), correct=correct,
                                             answered=answered, cells=cells)
    return blocks


def checkProbe():
    """RQ1 (Analysis/probe.py) on synthetic blocks: H1-only decisions, transfers by path name, flags, the §5 verdict."""
    check("15a. tie order: EN first, then path codes alphabetically; Excess = 0 is not aggregated (main) / aggregated (sensitivity a)",
          rq1.strongest({"ZH": 5, "EN": 5}) == "EN" and rq1.strongest({"P2": 3, "P1": 3}) == "P1"
          and rq1.strongest({"ZH": 3, "SR-ZH": 3}) == "SR-ZH"
          and rq1.choose(5, {"EN": 5, "ZH": 4}) == rq1.Choice(aggregate=False, aggregate_if_tie=True, tie=True, path="EN"))
    values = dict(zip(rq1.METRICS, rq1.score(rq1.Choice(False, False, False, "ZH"), 6, {"EN": 7, "ZH": 5}, 10)))
    check("15b. scores on H2: D / A / S / Oracle and the H2-true path of sensitivity (b)",
          (values["acc_D"], values["acc_A"], values["acc_S"], values["acc_O"], values["acc_D_truepath"], values["acc_S_truepath"])
          == (0.5, 0.6, 0.5, 0.7, 0.7, 0.7))

    blocks = syntheticBlocks()
    first = rq1.runProbes(blocks, ks=(25, 150), reps=3, seed=0)
    check("15c. same seed, same table", first.equals(rq1.runProbes(syntheticBlocks(), ks=(25, 150), reps=3, seed=0)))

    # Flip every label on H2 of rep 0: decisions (aggregate / tie / short) must not move, scores must
    one = rq1.runProbes(blocks, ks=(25, 150), reps=1, seed=0)
    flipped = syntheticBlocks()
    for block in flipped.values():
        h2 = ~makeSplits(len(block.item_ids), 1, 0)[0]
        for code in block.correct:
            block.correct[code] = np.where(h2, ~block.correct[code], block.correct[code])
        for cell in block.cells.values():
            cell.correct_agg = np.where(h2, ~cell.correct_agg, cell.correct_agg)
    other = rq1.runProbes(flipped, ks=(25, 150), reps=1, seed=0)
    decisions = ["aggregate", "tie", "short", "reps", "skipped_reps"]
    check("15d. decisions use only H1: flipping every H2 label changes the scores but no decision",
          one[decisions].equals(other[decisions]) and not one["acc_A"].equals(other["acc_A"]))

    def row(setting, model, dataset, label, source, baseline="pair", k=rq1.NO_K):
        found = one[(one["setting"] == setting) & (one["model"] == model) & (one["dataset"] == dataset) & (one["pair"] == label)
                    & (one["source"] == source) & (one["baseline"] == baseline) & (one["k"] == k)]
        return found.iloc[0]

    def h2Accuracy(model, dataset, code, label):
        block = blocks[(model, dataset)]
        both = block.bothAnswered(block.cells[(label, "judge")])
        h2 = ~makeSplits(len(block.item_ids), 1, 0)[0] & both
        return block.correct[code][h2].mean()

    to_model = row(rq1.TRANSFER_MODEL, "m2", "mathqa", "EN+ZH", "m1")
    to_dataset = row(rq1.TRANSFER_DATASET, "m1", "mmlu", "EN+ZH", "mathqa")
    to_global = row(rq1.TRANSFER_MODEL, "m2", "mathqa", "EN+ZH", "m1", baseline="global")
    to_source = row(rq1.TRANSFER_SOURCE, "m1", "mathqa", "EN+S1", "EN+ZH")
    check("15e. transfers: model / dataset reuse the source's stronger path by name (ZH), the global p* too; "
          "a source transfer takes the decision from the other pair but the target pair's own H1 path (S1)",
          to_model["aggregate"] == 0 and to_model["acc_S"] == h2Accuracy("m2", "mathqa", "ZH", "EN+ZH")
          and to_dataset["aggregate"] == 0 and to_dataset["acc_S"] == h2Accuracy("m1", "mmlu", "ZH", "EN+ZH")
          and to_global["acc_S"] == h2Accuracy("m2", "mathqa", "ZH", "EN+ZH")
          and to_source["aggregate"] == 0 and to_source["acc_S"] == h2Accuracy("m1", "mathqa", "S1", "EN+S1"))
    same_axis = one[(one["setting"] == rq1.TRANSFER_SOURCE_SAME_AXIS) & (one["pair"] == "EN+ZH")]["source"].unique().tolist()
    check(f"15f. same-axis sources are kept apart ({same_axis}); no transfer crosses model and dataset at once",
          same_axis == ["ZH+JA"] and set(one[one["setting"] == rq1.TRANSFER_MODEL]["source"]) == {"m1", "m2"}
          and set(one[one["setting"] == rq1.TRANSFER_DATASET]["source"]) == {"mathqa", "mmlu"})
    skipped = row(rq1.PROBE_RANDOM, "m1", "mathqa", "EN+ZH", "", k=150)
    short = row(rq1.PROBE_DIS, "m1", "mathqa", "P1+P2", "", k=150)
    check("15g. a random probe larger than H1 is skipped and flagged; a short disagreement probe uses all of them and is flagged",
          skipped["reps"] == 0 and skipped["skipped_reps"] == 1 and short["short"] == 1
          and (scored := one[one["reps"] > 0])["acc_O"].ge(scored[["acc_D", "acc_A", "acc_S"]].max(axis=1) - 1e-12).all())

    def blocksFor(probe_diff, transfer_diffs, regret_A=1.0):
        rows = []
        for setting, k, diffs in [(rq1.PROBE_RANDOM, 200, probe_diff)] + [(t, rq1.NO_K, d) for t, d in zip(rq1.TRANSFERS, transfer_diffs)]:
            for i, diff in enumerate(diffs):
                rows.append(dict(setting=setting, baseline="pair", aggregator="judge", k=k, model=f"m{i}", dataset="d",
                                 regret_A=regret_A, regret_S=2.0, diff_D_A=diff, diff_D_S=diff + 1.0))
        return pd.DataFrame(rows)
    good, bad = [0.6, 0.7, 0.8, 0.9], [0.1, -0.2, 0.3, 0.0]
    success = rq1.criteriaTable(blocksFor(good, [bad, good, bad]))
    failed = rq1.criteriaTable(blocksFor(good, [bad, bad, bad]))
    no_room = rq1.criteriaTable(blocksFor(good, [good, good, good], regret_A=0.1))
    check("15h. §5: space = smaller dumb regret, margin vs the better dumb method ≥ half the space with CI > 0, "
          "success needs the probe and one transfer; space < 0.2pp means no room",
          success["better"].tolist() == ["A"] * 4 and success["met"].tolist() == [True, False, True, False]
          and rq1.verdict(success).startswith("成功") and rq1.verdict(failed).startswith("不成功")
          and rq1.verdict(no_room).startswith("沒有空間"))


class RecordingModel(FakeModel):
    """FakeModel of a given modelType that keeps every message list it was sent, in call order."""
    def __init__(self, model_type: str):
        super().__init__()
        self.config.modelType = model_type
        self.sent = []

    def _complete(self, messages, temperature, seed):
        self.sent.append(json.loads(json.dumps(messages)))
        return super()._complete(messages, temperature, seed)


def checkCrossJudge(tmp: str):
    """RQ2 CrossJudge on the arm / judge files of check 6–7 (generator gpt4omini; L:en item 5 has no answer)."""
    dataset = makeDataset(30)
    arms = [ArmSpec.from_arm_id("L:en"), ArmSpec.from_arm_id("S:T0.7:seed1")]
    files = [File(os.path.join(tmp, "arms", f"{arm.file_stem}.json")) for arm in arms]
    reference = os.path.join(tmp, "aggregations", "judge.json")

    def strategy(model, name, onlyItems=None, ref=reference, cls=CrossJudge):
        extra = {"generator": "gpt4omini", "referencePath": ref, "onlyItems": onlyItems} if cls is CrossJudge else {}
        return cls(StrategyConfig(strategyType="aggregate", languages=["english", "english"]), model, dataset, NoLog(),
                   buildAggregator("judge", model, dataset, None), arms, files, [dataset, dataset],
                   ResultStore(os.path.join(tmp, "cross", f"{name}.json")), **extra)

    main = RecordingModel("gpt4omini")
    strategy(main, "main_grid", cls=Aggregate).getRes()
    ids = sorted(files[0].records_map)
    dis = [i for i in ids if files[0].getRecordById(i)["parsed_answer"] != files[1].getRecordById(i)["parsed_answer"]]
    judged = [i for i in dis if all(f.getRecordById(i)["parse_ok"] for f in files)]

    judge = RecordingModel("qwen")
    failed = strategy(judge, "full").getRes()
    out = File(os.path.join(tmp, "cross", "full.json"))
    ref = File(reference)
    meta = out.metadata
    check(f"16a. CrossJudge judges only the {len(judged)}/{len(dis)} both-answered disagreements, in the recorded orders; "
          "metadata names generator and judge; each record keeps its call",
          failed == [] and sorted(out.records_map) == judged and len(judged) < len(dis)
          and all(out.records_map[i]["presentation_order"] == ref.records_map[i]["presentation_order"] for i in judged)
          and (meta["generator"], meta["judge"], meta["n_judged_items"], meta["model_versions"]) == ("gpt4omini", "qwen", len(judged), {"fake-model-v1": len(judged)})
          and meta["Model"]["modelType"] == "qwen"
          and all(r["call"]["usage_in"] == 1 and r["call"]["called_at"] for r in out.records_map.values()))
    sent_main, sent_cross = dict(zip(dis, main.sent)), dict(zip(judged, judge.sent))
    check("16b. the cross judge receives exactly the main-grid judge prompt for every item",
          len(main.sent) == len(dis) and len(judge.sent) == len(judged) and all(sent_cross[i] == sent_main[i] for i in judged))

    pilot = RecordingModel("qwen")
    strategy(pilot, "pilot", onlyItems=set(judged[:3])).getRes()
    after_pilot = len(File(os.path.join(tmp, "cross", "pilot.json")).records_map)
    strategy(pilot, "pilot").getRes()
    calls = len(pilot.sent)
    strategy(pilot, "pilot").getRes()
    check("16c. a pilot subset is written first; the full run resumes the same file and never repeats a call",
          after_pilot == 3 and calls == len(judged) and len(pilot.sent) == calls)

    raised = []
    other = ResultStore(os.path.join(tmp, "cross", "full.json"))
    other.metadata["generator"] = "qwen"
    other.path = os.path.join(tmp, "cross", "other_generator.json")
    other.save()
    no_order = ResultStore(reference)
    no_order.records[dis[0]]["presentation_order"] = None
    no_order.path = os.path.join(tmp, "cross", "reference_without_order.json")
    no_order.save()
    for name, ref_path, reason in (("other_generator", reference, "holds generator"),
                                   ("missing_order", no_order.path, "no recorded presentation order")):
        try:
            strategy(RecordingModel("qwen"), name, ref=ref_path).getRes()
            raised.append(False)
        except ValueError as e:
            raised.append(reason in str(e))
    check("16d. CrossJudge refuses a file of another generator and stops when a recorded order is missing", all(raised))


def checkCrossAnalysis(tmp: str):
    """RQ2 analysis: the cross path (crossArrays + cellRow) reproduces computeRow on the both_answered subset exactly."""
    armdir, aggdir = os.path.join(tmp, "rq2", "arms"), os.path.join(tmp, "rq2", "aggregations")
    rng = np.random.default_rng(5)
    n = 300
    answers = {arm_id: [str(rng.choice(["a", "a", "b", "c", ""])) for _ in range(n)] for arm_id in ("L:en", "L:zh")}
    for arm_id, arm_answers in answers.items():
        writeResultFile(os.path.join(armdir, "m", "mathqa", f"{ArmSpec.from_arm_id(arm_id).file_stem}.json"), {},
                        [{"item_id": i, "parsed_answer": a, "parse_ok": a != "", "gold": "a", "tokens_out": 10} for i, a in enumerate(arm_answers)])
    records = []
    for i, (x, y) in enumerate(zip(answers["L:en"], answers["L:zh"])):
        choice = int(rng.choice([1, 2, 2, 0])) if x != y else None
        order = ["L:en", "L:zh"] if i % 2 else ["L:zh", "L:en"]
        final = x if x == y else ({"L:en": x, "L:zh": y}[order[choice - 1]] if choice else "")
        records.append({"item_id": i, "final_answer": final, "off_menu": final not in (x, y), "tokens_out": 0 if x == y else 7,
                        "presentation_order": order if x != y else None,
                        "trace": {"judge_output": f'{{"choice":"{choice}"}}' if choice else "none", "choice": choice} if x != y else None})
    path = os.path.join(aggdir, "m", "mathqa", "judge__L_en__L_zh.json")
    writeResultFile(path, {"candidate_arms": ["L:en", "L:zh"]}, records)

    cell = CellData(armdir, aggdir, "m", "mathqa")
    pair = rq2.PAIR_BY_LABEL["EN+ZH"]
    full = alignPair(cell, pair, path)
    judged = rq2.judgedItems(cell, pair)
    subset = {i: records[i] for i in judged}
    expected = rq2.cellRow(full.subset(subsetMask(full, "both_answered")), pair, subset)
    found = rq2.cellRow(rq2.crossArrays(cell, pair, subset), pair, subset)
    plain = computeRow(full.subset(subsetMask(full, "both_answered")), pair, "judge")
    try:
        rq2.crossArrays(cell, pair, {i: records[i] for i in judged[1:]})
        raised = False
    except ValueError:
        raised = True
    check(f"16e. cross path == main path on the both_answered subset ({len(judged)} judged items, recovery_H2 "
          f"{found['recovery_H2']:.4f}); pick_right = (1 + recovery) / 2; a missing record fails loudly",
          all(rq2._same(found[k], expected[k]) for k in expected) and all(rq2._same(found[k], plain[k]) for k in plain)
          and abs(found["pick_right"] - (1 + found["recovery"]) / 2) < 1e-12 and found["n_invalid_choice"] > 0 and raised)


def checkCrossJudgeStats():
    """RQ2 statistics on synthetic cells: effects, the three states, outcomes, residuals (0.75δ), McNemar, Δacc."""
    judge_bonus = {"gpt4omini": 0.1, "qwen": 0.2, "deepseek4.1flash": 0.4, "gemini3.1flashlite": 0.5}

    def cells(diagonal_bump=0.0):
        rows = []
        for g in rq2.MODELS:
            for j in rq2.MODELS:
                for d in rq2.DATASETS:
                    for k, p in enumerate(rq2.PAIRS):
                        rows.append({"generator": g, "judge": j, "dataset": d, "pair": p.label, "used_in_analysis": True,
                                     "recovery_H2": judge_bonus[j] + (diagonal_bump if g == j else 0) + (k - 1) * 0.01,
                                     "acc_final": 0.6 if j in rq2.STRONG else 0.5})
        return pd.DataFrame(rows)

    R = rq2stats.blockMeans(cells())
    judge_fx, candidate_fx = rq2stats.blockEffects(R, "judge"), rq2stats.blockEffects(R, "candidate")
    js, cs = blockStats.summarize(judge_fx["delta"]), blockStats.summarize(candidate_fx["delta"])
    res = rq2stats.residuals(rq2stats.blockMeans(cells(0.08)))
    diag = blockStats.summarize(rq2stats.diagonalResiduals(res)["residual"])
    undetermined = {"mean": 0.06, "ci_low": -0.01, "ci_high": 0.13}
    check(f"16f. judge-only data: judge effect {js['mean']:.3f} exists, candidate effect {cs['mean']:.3f} absent → (a); "
          f"a diagonal bump of 0.08 gives residual {diag['mean']:.3f} (0.75δ); undetermined → (e)",
          abs(js["mean"] - 0.3) < 1e-12 and rq2stats.effectState(js) == rq2stats.EXISTS
          and abs(cs["mean"]) < 1e-12 and rq2stats.effectState(cs) == rq2stats.ABSENT
          and rq2stats.outcome(rq2stats.EXISTS, rq2stats.ABSENT) == "a"
          and rq2stats.outcome(rq2stats.effectState(undetermined), rq2stats.EXISTS) == "e"
          and abs(diag["mean"] - 0.06) < 1e-12 and rq2stats.selfPreference(diag)
          and rq2stats.residuals(R)["residual"].abs().max() < 1e-12)

    items = pd.DataFrame([{"generator": g, "judge": j, "dataset": "mmlu", "pair": "EN+ZH", "item_id": i,
                           "correct_a": True, "correct_b": False, "correct": j in rq2.STRONG}
                          for g in rq2.MODELS for j in rq2.MODELS for i in range(10)])
    tests = rq2stats.mcnemar(items)
    impact = rq2stats.practicalImpact(cells())
    check("16g. McNemar uses the two generators that are neither judge (b = 20, c = 0); Δacc = strong judges − self, in pp",
          len(tests) == 4 and (tests["b"] == 20).all() and (tests["c"] == 0).all() and (tests["n_R"] == 20).all()
          and np.allclose(impact["delta_pp"], 10.0))


def syntheticPathBlock(n: int = 80, agree: bool = False) -> "rq1k.PathBlock":
    """14 paths on n items (gold "a"); ZH has no answer on item 0. agree = every path copies EN."""
    rng = np.random.default_rng(3)
    answers = {code: ["a" if rng.random() < 0.7 else str(rng.choice(["b", "c"])) for _ in range(n)] for code in PATHS}
    if agree:
        answers = {code: list(answers["EN"]) for code in PATHS}
    answers["ZH"][0] = ""
    return rq1k.PathBlock(model="m", dataset="d", item_ids=np.arange(n), gold=["a"] * n, answers=answers,
                          answered={c: np.array([a != "" for a in v]) for c, v in answers.items()},
                          correct={c: np.array([a == "a" for a in v]) for c, v in answers.items()},
                          tokens={c: np.full(n, 10.0) for c in answers}, compare=lambda x, y: x == y)


def checkMenuVote():
    """RQ1-K (Analysis/menuVote.py): vote and tie rule, subsets, the per-split identity, regrets and the four states."""
    eq = lambda x, y: x == y
    first = rq1k.vote(["a", "b", "b", "a"], [True] * 4, eq)
    rng_a = rq1k.vote(["a", "b", "c"], [True] * 3, eq, np.random.default_rng([0, 7]))
    rng_b = rq1k.vote(["a", "b", "c"], [True] * 3, eq, np.random.default_rng([0, 7]))
    check("17a. majority vote: a tie goes to the answer of the highest-priority path; missing answers do not vote; "
          "priority = EN, then codes alphabetically; random ties are reproducible",
          first == ("a", True) and rq1k.vote(["a", "b", "b", "c"], [True] * 4, eq) == ("b", False)
          and rq1k.vote(["", "b", "c"], [False, True, True], eq) == ("b", True) and rq1k.vote(["", ""], [False, False], eq) == ("", False)
          and rq1k.ordered(["ZH", "EN", "S1", "ES", "SR-EN", "R", "RU"]) == ["EN", "ES", "R", "RU", "S1", "SR-EN", "ZH"]
          and rng_a == rng_b and rng_a[1])

    block, splits = syntheticPathBlock(), makeSplits(80, 20, 0)
    rows = {menu: rq1k.evaluateMenu(block, menu, splits)["row"] for menu in rq1k.MENUS}
    h2 = ~splits[0]
    single = rq1k.evaluateMenu(block, "M3S", splits[:1])["row"]
    finals, _ = rq1k.menuFinals(block, "M3S")
    manual = np.mean([f == "a" for f, keep in zip(finals, h2) if keep]) - np.mean(block.correct[single["S_in_top"].split()[0]][h2])
    check(f"17b. per split, Excess_in = headroom × (recovery − recovery_blind) on every menu (max error "
          f"{max(r['identity_max_error'] for r in rows.values()):.1e}); subsets drop item 0 only for menus with ZH; "
          "M12's S_in is S_all; Excess_in = A − S_in on H2",
          all(r["identity_max_error"] < 1e-12 for r in rows.values())
          and rows["M3L"]["keep_in"] == 79 / 80 and rows["M3S"]["keep_in"] == 1.0 and rows["M3S"]["keep_all"] == 79 / 80
          and rows["M12"]["S_in_top"] == rows["M12"]["S_all_top"] and abs(single["excess_in"] - 100 * manual) < 1e-9)

    same = rq1k.evaluateMenu(syntheticPathBlock(agree=True), "M5L", splits)["row"]
    check("17c. when every path agrees: d = 0, Excess_in = 0 and recovery is undefined in every split",
          same["d"] == 0 and same["excess_in"] == 0 and same["undefined_splits"] == 20 and same["identity_max_error"] == 0)

    r = rq1k.regrets(np.array([1.0, -2.0, 0.5, -0.5]))
    states = [blockStats.fourState({"mean": m, "ci_low": lo, "ci_high": hi}, 0.5)
              for m, lo, hi in ((0.6, 0.1, 1.1), (-0.6, -1.1, -0.1), (0.1, -0.3, 0.4), (0.3, -0.2, 0.8), (0.45, 0.1, 0.8))]
    check("17d. regrets: always aggregate 0.625, always single 0.375, space 0.375; four states",
          (r["regret_aggregate"], r["regret_single"], r["space"]) == (0.625, 0.375, 0.375)
          and states == [blockStats.FORWARD, blockStats.REVERSE, blockStats.EQUIVALENT, blockStats.UNDETERMINED, blockStats.UNDETERMINED])

    finals = {menu: rq1k.evaluateMenu(block, menu, splits)["agg_correct"] for menu in rq1k.COMPARE_MENUS}
    keep = rq1k.allAnswered(block, rq1k.COMPARE_PATHS)
    expected = 100 * np.mean([finals["M3S"][~h & keep].mean() - finals["M3L"][~h & keep].mean() for h in splits])
    check("17e. M3S − M3L uses the items where all five paths answered, on H2, averaged over splits",
          abs(rq1k.compareMenus(block, splits, finals) - expected) < 1e-12 and keep.sum() == 79)


def checkMenuJudge(tmp: str):
    """RQ1-KJ MenuJudge on the arm / judge files of check 6–7 plus a third arm (generator = judge = gpt4omini; L:en item 5 has no answer)."""
    dataset = makeDataset(30)
    base = [ArmSpec.from_arm_id("L:en"), ArmSpec.from_arm_id("S:T0.7:seed1")]
    third = ArmSpec.from_arm_id("S:T0.7:seed2")
    Generate(StrategyConfig(strategyType="generate", languages=["english"]), FakeModel(), dataset, NoLog(), third,
             ResultStore(os.path.join(tmp, "arms", f"{third.file_stem}.json"))).getRes()
    arms3 = base + [third]
    files = {arm.arm_id: File(os.path.join(tmp, "arms", f"{arm.file_stem}.json")) for arm in arms3}

    def strategy(model, name, arms, menu="T3", ref=None, onlyItems=None, aggregator=None):
        return MenuJudge(StrategyConfig(strategyType="aggregate", languages=["english"] * len(arms)), model, dataset, NoLog(),
                         aggregator or MenuJudgeAggregator(len(arms), model, dataset), arms, [files[a.arm_id] for a in arms],
                         [dataset] * len(arms), ResultStore(os.path.join(tmp, "menu", f"{name}.json")),
                         menu=menu, referencePath=ref, onlyItems=onlyItems)

    ids = sorted(files["L:en"].records_map)
    recs = lambda i: [files[a.arm_id].getRecordById(i) for a in arms3]
    dis = [i for i in ids if len({r["parsed_answer"] for r in recs(i)}) > 1]
    judged = [i for i in dis if all(r["parse_ok"] for r in recs(i))]
    arm_ids = [a.arm_id for a in arms3]

    # 18a. §3.3 order on its own
    balanced, groups_ok, varied = True, True, True
    for n, k in ((37, 3), (100, 12), (5, 12), (24, 12), (1, 3)):
        names = [f"p{j}" for j in range(k)]
        orders = MenuJudge.groupedOrders(names, list(range(n)), 0)
        counts = {(name, pos): 0 for name in names for pos in range(k)}
        for order in orders.values():
            for pos, name in enumerate(order):
                counts[(name, pos)] += 1
        for name in names:
            per_position = [counts[(name, pos)] for pos in range(k)]
            balanced &= max(per_position) - min(per_position) <= 1
        shuffled = np.random.default_rng(0).permutation(n)   # the first draw of the same rng
        bases = [orders[int(shuffled[g * k])] for g in range((n + k - 1) // k)]   # rotation 0 of each group = B_g
        for i, index in enumerate(shuffled):
            group_base, r = bases[i // k], i % k
            groups_ok &= orders[int(index)] == group_base[r:] + group_base[:r]
        if n >= 2 * k:
            varied &= len({tuple(b) for b in bases}) > 1
    same_seed = MenuJudge.groupedOrders(names, list(range(50)), 0) == MenuJudge.groupedOrders(names, list(reversed(range(50))), 0)
    check("18a. §3.3 grouped orders: every arm in every position equally often (±1); the items of a group share one base order "
          "(rotated by i mod K); groups differ; reproducible and independent of the input order of the item_ids",
          balanced and groups_ok and varied and same_seed
          and MenuJudge.groupedOrders(names, list(range(50)), 0) != MenuJudge.groupedOrders(names, list(range(50)), 1))

    # 18b. K = 3 run
    model = RecordingModel("gpt4omini")
    failed = strategy(model, "k3", arms3).getRes()
    out = File(os.path.join(tmp, "menu", "k3.json"))
    meta = out.metadata
    sent = dict(zip(judged, model.sent))
    expected_orders = MenuJudge.groupedOrders(arm_ids, judged, 0)
    calls = len(model.sent)
    strategy(model, "k3", arms3).getRes()
    check(f"18b. MenuJudge (K = 3) judges only the {len(judged)}/{len(dis)} disagreements where every path answered, in the §3.3 "
          "orders; the prompt says 3 answers; each record keeps its call and the sha256 of the messages sent; a rerun makes no call",
          failed == [] and sorted(out.records_map) == judged and len(judged) < len(dis) and len(model.sent) == calls == len(judged)
          and all(out.records_map[i]["presentation_order"] == expected_orders[i] for i in judged)
          and all("There are 3 answers" in m[0]["content"] and "an integer from 1 to 3" in m[0]["content"] for m in model.sent)
          and all(out.records_map[i]["trace"]["prompt_sha256"] == PromptBuilder.promptSha256(sent[i]) for i in judged)
          and all(out.records_map[i]["call"]["model_version"] == "fake-model-v1" for i in judged)
          and (meta["prompt_version"], meta["menu"], meta["order_scheme"], meta["Aggregator"]["k"], meta["n_judged_items"],
               meta["model_versions"]) == ("choice-k-v1", "T3", ORDER_GROUPED, 3, len(judged), {"fake-model-v1": len(judged)}))

    # 18c. K = 2 with the recorded main-grid orders sends exactly the main-grid (choice-v1) prompt
    main = RecordingModel("gpt4omini")
    reference = os.path.join(tmp, "menu", "main_grid.json")
    Aggregate(StrategyConfig(strategyType="aggregate", languages=["english", "english"]), main, dataset, NoLog(),
              buildAggregator("judge", main, dataset, None), base, [files[a.arm_id] for a in base], [dataset, dataset],
              ResultStore(reference)).getRes()
    ref = File(reference)
    dis2 = [i for i in ids if files["L:en"].getRecordById(i)["parsed_answer"] != files["S:T0.7:seed1"].getRecordById(i)["parsed_answer"]]
    judged2 = [i for i in dis2 if files["L:en"].getRecordById(i)["parse_ok"] and files["S:T0.7:seed1"].getRecordById(i)["parse_ok"]]
    check2 = RecordingModel("gpt4omini")
    strategy(check2, "check_k2", base, menu="EN+S1", ref=reference).getRes()
    out2 = File(os.path.join(tmp, "menu", "check_k2.json"))
    main_sent = dict(zip(dis2, main.sent))
    check("18c. K = 2 with the recorded orders: the prompt sent equals the main-grid choice-v1 prompt item by item; "
          "the orders are the recorded ones; the file says order_scheme recorded_main_grid",
          sorted(out2.records_map) == judged2 and len(check2.sent) == len(judged2)
          and all(m == main_sent[i] for i, m in zip(judged2, check2.sent))
          and all(out2.records_map[i]["presentation_order"] == ref.records_map[i]["presentation_order"] for i in judged2)
          and out2.metadata["order_scheme"] == "recorded_main_grid" and out2.metadata["reference_file"] == reference)

    # 18d. refusals
    raised = []
    no_order = ResultStore(reference)
    no_order.records[judged2[0]]["presentation_order"] = None
    no_order.path = os.path.join(tmp, "menu", "reference_without_order.json")
    no_order.save()
    other_version = ResultStore(reference)
    other_version.metadata["prompt_version"] = None
    other_version.path = os.path.join(tmp, "menu", "reference_other_version.json")
    other_version.save()
    for name, kwargs, reason in (("missing_order", {"ref": no_order.path, "arms": base, "menu": "EN+S1"}, "no recorded presentation order"),
                                 ("other_version", {"ref": other_version.path, "arms": base, "menu": "EN+S1"}, "does not match the run"),
                                 ("k3", {"arms": arms3, "menu": "OTHER"}, "holds menu"),
                                 ("plain_judge", {"arms": base, "aggregator": buildAggregator("judge", main, dataset, None)}, "K-way Judge")):
        arms = kwargs.pop("arms")
        try:
            strategy(RecordingModel("gpt4omini"), name, arms, **kwargs).getRes()
            raised.append(False)
        except ValueError as e:
            raised.append(reason in str(e))
    check("18d. MenuJudge stops on a missing recorded order or a reference of another prompt version, refuses to resume a file "
          "of another menu, and only runs the K-way Judge", all(raised))

    # 18e. pilot subset, full resume, second run in its own file
    pilot = RecordingModel("gpt4omini")
    sample = set(judged[:3])
    strategy(pilot, "pilot", arms3, onlyItems=sample).getRes()
    after_pilot = len(File(os.path.join(tmp, "menu", "pilot.json")).records_map)
    strategy(pilot, "pilot", arms3).getRes()
    full_calls = len(pilot.sent)
    strategy(pilot, "rep2", arms3, onlyItems=sample).getRes()
    rep2 = File(os.path.join(tmp, "menu", "rep2.json"))
    check("18e. the pilot writes its sample first and the full run resumes the same file without repeating a call; "
          "the second pilot run goes to its own file with the same orders",
          after_pilot == 3 and full_calls == len(judged) and len(pilot.sent) == full_calls + 3 and sorted(rep2.records_map) == sorted(sample)
          and all(rep2.records_map[i]["presentation_order"] == expected_orders[i] for i in sample))

    # 18f. K = 12 parsing and scoring
    m12 = [ArmSpec.from_arm_id(PATHS[code]) for code in rq1kj.MENUS["M12"]]
    item = AggregationItem(item_id=0, candidates=[candidate(arm, f"x{j}") for j, arm in enumerate(m12)], question="Q",
                           presentation_order=[arm.arm_id for arm in reversed(m12)])

    class RefusingModel(FakeModel):
        def _complete(self, messages, temperature, seed):
            return SimpleNamespace(choices=[SimpleNamespace(message=None, finish_reason="content_filter")], model="fake-model-v1",
                                   usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0))

    results = {}
    for name, model in (("12", FixedReplyModel('reasoning\n{"choice":"12"}')), ("13", FixedReplyModel('reasoning\n{"choice":"13"}')),
                        ("none", FixedReplyModel("no choice at all")), ("refused", RefusingModel())):
        record = MenuJudgeAggregator(12, model, makeDataset(1), answerParser=PARSER).aggregate(item).to_dict()
        results[name] = (record["final_answer"], record["off_menu"], rq2.classifyRecord(record))
    check("18f. K = 12: choice 12 picks the 12th candidate shown; 13, no choice and a refused call give no answer (agg_no_answer) "
          "and are told apart",
          results["12"][0] == "x0" and not results["12"][1]
          and results["13"][:2] == ("", True) and results["13"][2]["out_of_range"]
          and results["none"][:2] == ("", True) and results["none"][2]["no_choice"]
          and results["refused"][:2] == ("", True) and results["refused"][2]["refused"])


def syntheticJudgeRecords(block, menu: str, pick) -> dict:
    """Judge records for the judged items of a synthetic block; pick(k, codes in display order) -> choice 1..K or None."""
    codes, arms = rq1kj.codesOf(menu), rq1kj.armIdsOf(menu)
    judged_mask = rq1k.allAnswered(block, codes) & ~rq1kjstats.agreeMask(block, codes)
    judged = [int(i) for i in block.item_ids[judged_mask]]
    orders = MenuJudge.groupedOrders(arms, judged, 0)
    records = {}
    for i in judged:
        order = orders[i]
        choice = pick(i, [rq1kj.ARM_TO_CODE[a] for a in order])
        final = block.answers[rq1kj.ARM_TO_CODE[order[choice - 1]]][i] if choice else ""
        records[i] = {"item_id": i, "final_answer": final, "off_menu": not choice, "presentation_order": order,
                      "tokens_in": 100, "tokens_out": 5, "call": {"usage_in": 101, "usage_out": 6},
                      "trace": {"judge_output": f'{{"choice":"{choice}"}}' if choice else "none", "choice": choice,
                                "chosen_arm": order[choice - 1] if choice else None, "prompt_sha256": "0" * 64}}
    return records


def checkMenuJudgeStats():
    """RQ1-KJ statistics (Analysis/menuJudgeStats.py) on the synthetic 14-path block of check 17."""
    block, splits = syntheticPathBlock(), makeSplits(80, 20, 0)
    rng = np.random.default_rng(9)
    random_pick = lambda i, codes: None if rng.random() < 0.05 else int(rng.integers(1, len(codes) + 1))
    perfect_pick = lambda i, codes: next((p for p, c in enumerate(codes, start=1) if block.correct[c][i]), 1)

    rows, ok_vote = {}, True
    judges = {}
    for menu in rq1kj.MENU_NAMES:
        judge = rq1kjstats.judgeArrays(block, rq1kj.codesOf(menu), syntheticJudgeRecords(block, menu, random_pick), menu)
        judges[menu] = judge
        rows[menu] = rq1kjstats.evaluateBlockMenu(block, menu, splits, judge)["row"]
        vote = rq1k.evaluateMenu(block, menu, splits)["row"]
        ok_vote &= all(abs(rows[menu][ours] - vote[theirs]) < 1e-12 for ours, theirs in (("A_V", "A_in"), ("S_in", "S_in"), ("excess_V", "excess_in")))
    h2 = ~splits[0]
    single = rq1kjstats.evaluateBlockMenu(block, "M3S", splits[:1], judges["M3S"])["row"]
    s_in = single["S_in_top"].split()[0]
    sub = rq1k.allAnswered(block, rq1kj.codesOf("M3S"))
    manual = np.mean(judges["M3S"].correct[h2 & sub]) - np.mean(block.correct[s_in][h2 & sub])
    check(f"18g. per split, Excess_J and Excess_V = headroom × (recovery − recovery_blind) on every menu (max error "
          f"{max(max(r['identity_max_error_J'], r['identity_max_error_V']) for r in rows.values()):.1e}); A_V, S_in, Excess_V equal "
          "RQ1-K's evaluateMenu; Excess_J = A_J − S_in on H2; agreement items keep the common answer",
          all(r["identity_max_error_J"] < 1e-12 and r["identity_max_error_V"] < 1e-12 for r in rows.values()) and ok_vote
          and abs(single["excess_J"] - 100 * manual) < 1e-9
          and all(judges["M12"].final[k] == block.answers["EN"][k] for k in range(80)
                  if rq1k.allAnswered(block, rq1kj.codesOf("M12"))[k] and not judges["M12"].judged[k]))

    perfect = rq1kjstats.judgeArrays(block, rq1kj.codesOf("M12"), syntheticJudgeRecords(block, "M12", perfect_pick), "M12")
    perfect_row = rq1kjstats.evaluateBlockMenu(block, "M12", splits, perfect)["row"]
    records = syntheticJudgeRecords(block, "M3L", random_pick)
    records.pop(next(iter(records)))
    try:
        rq1kjstats.judgeArrays(block, rq1kj.codesOf("M3L"), records, "M3L")
        raised = False
    except ValueError:
        raised = True
    result = rq1kjstats.evaluateBlockMenu(block, "M12", splits, judges["M12"])
    comparison = rq1kjstats.itemComparison(block, result)
    positions = rq1kjstats.positionStats(result)
    keep = rq1k.allAnswered(block, rq1kj.COMPARE_PATHS)
    expected = 100 * np.mean([judges["M3S"].correct[~h & keep].mean() - judges["M3L"].correct[~h & keep].mean() for h in splits])
    check("18h. a judge that always picks a right candidate has recovery_J = 1; a missing record fails loudly; M3S − M3L uses the "
          "five-path subset; the three gold-position classes and the four cells cover every judged item; positions are balanced",
          abs(perfect_row["recovery_J"] - 1) < 1e-12 and raised
          and abs(rq1kjstats.compareJudge(block, splits, judges["M3S"], judges["M3L"]) - expected) < 1e-12
          and comparison["n_plurality"] + comparison["n_minority"] + comparison["n_absent"] == comparison["n_judged"]
          == comparison["both_right"] + comparison["both_wrong"] + comparison["vote_right_judge_wrong"] + comparison["vote_wrong_judge_right"]
          and positions["position_spread"] <= 1 and rows["M12"]["n_agg_no_answer"] == rows["M12"]["n_no_choice"] > 0)


def checkTokenCounts():
    """TestTokenNums keeps the legacy strategies' output-token definitions (FakeModel counts words)."""
    model = FakeModel()
    cases = {
        "onelanguage": ({"Question": "q q", "Result": "a b c"}, 3),
        "challenge": ({"Record1": [{"role": "user", "content": "x x"}, {"role": "assistant", "content": "a b"}],
                       "Record2": [{"role": "assistant", "content": "c"}], "Result3": "d e f"}, 6),
        "selfreflection": ({"Response": "a b", "Reflection": "p p p p", "Result": "c d e"}, 5),
        "translate": ({"Translated": "a b"}, 2),
        "rewrite": ({"Rewritten": "a b c d"}, 4),
        "generate": ({"raw_text": "x y", "tokens_out": 7}, 7),
        "aggregate": ({"tokens_out": 0}, 0),
    }
    check("13. TestTokenNums counts only generated text, per legacy strategy definition",
          all(TOKEN_COUNTERS[StrategyType(name)](model, record) == expected for name, (record, expected) in cases.items()))


def main():
    checkPrompts()
    checkArmIds()
    checkRefinePrompt()
    checkRunners()
    checkAggregators()
    checkBalancing()
    checkDebateEquivalence()
    checkTokenCounts()
    checkModelRequests()
    checkProbe()
    with tempfile.TemporaryDirectory() as tmp:
        checkGenerateAndAggregate(tmp)
        checkRewrite(tmp)
        checkRefusals(tmp)
        checkAnalysis(tmp)
        checkCrossJudge(tmp)
        checkCrossAnalysis(tmp)
        checkMenuJudge(tmp)
    checkCrossJudgeStats()
    checkMenuVote()
    checkMenuJudgeStats()

    print(f"\n{'All checks passed' if not failures else f'{len(failures)} check(s) failed'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
