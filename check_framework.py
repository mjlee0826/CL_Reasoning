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

import numpy as np

from Model.Model import Model
from Model.ModelConfig import ModelConfig
from Dataset.Dataset import Dataset
from Dataset.DatasetConfig import DatasetConfig
from Strategy.Strategy import Strategy
from Strategy.StrategyConfig import StrategyConfig
from Strategy.OnlyOneLanguage import OnlyOneLanguage
from Strategy.Challenge import Challenge
from Strategy.Generate import Generate
from Strategy.Aggregate import Aggregate
from Strategy.Rewrite import Rewrite
from Dataset.path import rewriteFileName
from run_generate import defaultWorkers, interleaveByModel
from Arm.ArmSpec import ArmSpec
from Arm.PromptBuilder import PromptBuilder
from Arm.GenerationRecord import GenerationRecord
from Strategy.PromptAbstractFactory.PromptSelfReflectionCOTFactory import PromptSelfReflectionCOTFactory
from Strategy.PromptAbstractFactory.PromptFormatFactory import PromptFormatFactory
from Aggregator.Aggregator import AggregationItem, Candidate
from Aggregator.AggregatorConfig import AggregatorConfig
from Aggregator.AggregatorFactory import AggregatorFactory
from File.File import File
from File.ResultStore import ResultStore
from Log.NoLog import NoLog
from Test.Test import Test

# Hashes of the prompts produced by OnlyOneLanguage.getPrompt BEFORE it was refactored onto PromptBuilder
GOLDEN_QUESTION = 'There is a Problem: \nWhat is 2+2?.\nAnd there are 5 choices\na ) 1 , b ) 2 , c ) 3 , d ) 4 , e ) 5\n'
GOLDEN_PROMPT_HASHES = {
    "cot|english": "160f1ba2026c97bb", "cot|chinese": "a4ea73cd7c234d14", "cot|japanese": "2df8509d86f0c774",
    "cot|russian": "84b8ea137eb74ef1", "cot|spanish": "5851547b30a600a0",
    "short_cot|english": "49db993fa6501d02", "short_cot|chinese": "69bac09a68788b3d", "short_cot|japanese": "56fd32aa3e484053",
    "short_cot|russian": "c40bc84dd1ab4656", "short_cot|spanish": "e4a224f6b1b2049e",
    "direct|english": "8d3f8621bccd6f51", "direct|chinese": "8ad765b552510cda", "direct|japanese": "7b56d270722222bb",
    "direct|russian": "333cff9f9829cf8c", "direct|spanish": "10568bf1989bc3d6",
}
RECORD_FIELDS = {"item_id", "arm_id", "axis", "is_anchor", "lang", "raw_text", "parsed_answer", "parse_ok", "tokens_in",
                 "tokens_out", "model", "model_version_string", "temperature", "seed", "prompt_hash", "gold"}

failures = []


def check(name: str, condition: bool):
    print(("✅ " if condition else "❌ ") + name)
    if not condition:
        failures.append(name)


class FakeModel(Model):
    """Replies depend only on (messages, temperature, seed); markers in the last message inject failures."""
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
        old = OnlyOneLanguage(StrategyConfig(strategyType="onelanguage", languages=[lang], promptStyle=style),
                              None, None, None).getPrompt(GOLDEN_QUESTION)
        ok &= PromptBuilder.promptHash([{"role": "user", "content": built}]) == expected
        ok &= PromptBuilder.promptHash([{"role": "user", "content": old}]) == expected
    check("1. PromptBuilder and OnlyOneLanguage reproduce the pre-refactor prompts (3 styles × 5 languages)", ok)


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
    """A derived arm's prompt must match Strategy/SelfReflection.py so legacy SR prompts rebuild exactly."""
    arm = ArmSpec.from_arm_id("F:zh")
    question, previous = "問題：選一個字母", "先前輸出 {\"answer\":\"a\"}"
    expected = PromptSelfReflectionCOTFactory().getPrompt("chinese", question, previous) + PromptFormatFactory().getPrompt("chinese")
    built = PromptBuilder(arm).text(question, previous)
    raised = False
    try:
        PromptBuilder(arm).text(question)
    except ValueError:
        raised = True
    check("2e. derived prompt == SelfReflection prompt, and fails loudly without the base output",
          built == expected and raised)

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
        results[aggregator.config.aggregatorType] = Test.recoveryStats(cA, cB, np.array([f == "a" for f in final]), dis, np.ones(n, dtype=bool))
    check(f"3e. V2 recovery ≈ 0 ({results['v2']['recovery']:+.4f})", abs(results["v2"]["recovery"]) < 0.05)
    check("3f. Blind recovery == recovery_blind", abs(results["blind"]["recovery"] - results["blind"]["recovery_blind"]) < 1e-12)

    # Revise: mirror of Blind, always keeps the derived candidate
    revise = buildAggregator("revise", model, dataset)
    D = ArmSpec.from_arm_id("F:en")
    final = [revise.aggregate(AggregationItem(i, [candidate(A, x), candidate(D, y)], "Q")).final_answer
             for i, (x, y) in enumerate(zip(answersA, answersB))]
    stats = Test.recoveryStats(cA, cB, np.array([f == "a" for f in final]), dis, np.ones(n, dtype=bool))
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
    check("3h. off_menu when the judge picks neither candidate", record.off_menu and record.final_answer in "abc" and record.tokens_in > 0)


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
    challenge = Challenge.__new__(Challenge)
    challenge.config = StrategyConfig(strategyType="challenge")
    challenge.model, challenge.dataset, challenge.threshold = model, dataset, 3
    challenge.lang1, challenge.lang2 = "english", "japanese"
    arms = (ArmSpec.from_arm_id("L:en"), ArmSpec.from_arm_id("L:ja"))

    same, rounds, judged = 0, set(), 0
    for i in range(300):
        q1, q2 = f"Q{i} en", f"Q{i} ja"
        r1, r2 = f'init1 {i} {{"answer":"a"}}', f'init2 {i} {{"answer":"b"}}'
        rec1, rec2, res1, res2, ans1, ans2, ar1, ar2, turn = challenge.runChallenge(q1, q2, r1, r2, "a", "b")
        judge_output, final = "", ans1
        if not dataset.compareTwoAnswer(ans1, ans2):
            judge_output = model.getRes(challenge.getJudgePrompt("english", q1, res1, res2))
            final = challenge.parseAnswer(judge_output)

        record = debate.aggregate(AggregationItem(i, [candidate(arms[0], "a", r1, q1), candidate(arms[1], "b", r2, q2)], "Q"))
        trace = record.trace
        same += (trace["Record1"] == rec1[2:] and trace["Record2"] == rec2[2:] and trace["AnswerRecord1"] == ar1
                 and trace["AnswerRecord2"] == ar2 and trace["Result3"] == judge_output
                 and record.n_rounds == turn and record.final_answer == final)
        rounds.add(turn)
        judged += bool(judge_output)
    check(f"5. DebateAggregator == Challenge on 300 scripted items (rounds {sorted(rounds)}, judge called {judged})", same == 300)


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

    try:
        Generate(StrategyConfig(strategyType="generate", languages=[derived.language]), model, dataset, NoLog(),
                 derived, ResultStore(os.path.join(tmp, "arms", "F_en_generated.json")))
        check("8a. run_generate refuses derived arms", False)
    except ValueError:
        check("8a. run_generate refuses derived arms", True)

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


def main():
    checkPrompts()
    checkArmIds()
    checkRefinePrompt()
    checkRunners()
    checkAggregators()
    checkBalancing()
    checkDebateEquivalence()
    with tempfile.TemporaryDirectory() as tmp:
        checkGenerateAndAggregate(tmp)
        checkRewrite(tmp)

    print(f"\n{'All checks passed' if not failures else f'{len(failures)} check(s) failed'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
