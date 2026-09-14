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
from Arm.ArmSpec import ArmSpec
from Arm.PromptBuilder import PromptBuilder
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
            "R:short_cot", "R:direct", "P:expert", "W:rewrite"]
    check("2a. arm_id round trip", all(ArmSpec.from_arm_id(a).arm_id == a for a in good))
    check("2b. only L:en is the anchor", [a for a in good if ArmSpec.from_arm_id(a).is_anchor] == ["L:en"])
    rejected = 0
    for bad in ["R:cot", "S:T0.75:seed1", "S:T0.0:seed1", "L:de", "K:2", "W:original", "S:T0.7", "P:Expert", "L:EN"]:
        try:
            ArmSpec.from_arm_id(bad)
        except ValueError:
            rejected += 1
    try:
        ArmSpec("L", lang="ja", temperature=0.7, seed=1)
    except ValueError:
        rejected += 1
    check("2c. invalid / non-canonical / multi-factor arms are rejected (10)", rejected == 10)


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


def main():
    checkPrompts()
    checkArmIds()
    checkAggregators()
    checkBalancing()
    checkDebateEquivalence()
    with tempfile.TemporaryDirectory() as tmp:
        checkGenerateAndAggregate(tmp)

    print(f"\n{'All checks passed' if not failures else f'{len(failures)} check(s) failed'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
