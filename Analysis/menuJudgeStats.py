from collections import Counter
from dataclasses import dataclass

import numpy as np

from Analysis.crossJudge import classifyRecord
from Analysis.menuJudge import MODELS, DATASETS, MENUS, COMPARE_PATHS, ARM_TO_CODE, SEED, REPS
from Analysis.menuVote import PathBlock, allAnswered, loadPathBlock, menuFinals, ordered
from Analysis.probe import strongest
from Analysis.splitHalf import makeSplits

# ------------------------------------------------------------------
# RQ1-KJ（result/analysis/rq1kj/rq1kj_criteria.md）：逐切分的 A_J、A_V、S_in、EN、分解（§4、§7.5），
# 判定三（§5.5）、§7.3、§7.4、§7.6、§7.7、§7.10 的量。輸入是 Analysis.menuVote 的 PathBlock 與 Judge 紀錄。
# ------------------------------------------------------------------
FAILURE_KINDS = ("no_choice", "out_of_range", "refused")


def agreeMask(block: PathBlock, codes: list[str]) -> np.ndarray:
    """菜單內各 path 的答案完全相同（用該資料集的 compareTwoAnswer）。"""
    first = block.answers[codes[0]]
    return np.array([all(block.compare(first[k], block.answers[c][k]) for c in codes[1:]) for k in range(len(block.item_ids))],
                    dtype=bool)


@dataclass
class JudgeArrays:
    """一個區塊 × 菜單的 Judge 版本，逐題陣列依 block.item_ids 排列。"""
    judged: np.ndarray           # 有呼叫 Judge 的題目（子集內的不一致題）
    final: list[str]             # 一致題 = 共同答案；不一致題 = 被選中候選的答案（無效 = ""）；子集外 = ""
    correct: np.ndarray
    choice: list                 # 1..K；無效或沒呼叫 = None
    chosen: list                 # 被選中的 path 短代號
    order: list                  # presentation_order（短代號）
    failure: dict                # FAILURE_KINDS -> bool 陣列
    tokens_in: np.ndarray        # 重算（與主網格相同的定義）；沒呼叫 = 0
    tokens_out: np.ndarray
    api_in: np.ndarray           # API 計費；沒呼叫或沒有 usage = nan
    api_out: np.ndarray


def judgeArrays(block: PathBlock, codes: list[str], records: dict, label: str = "") -> JudgeArrays:
    """records 必須剛好涵蓋子集內的不一致題、prompt 是 choice-k-v1，否則報錯（rq1kj_criteria.md §2.4、§3.1）。"""
    sub, agree = allAnswered(block, codes), agreeMask(block, codes)
    judged = sub & ~agree
    expected = {int(i) for i in block.item_ids[judged]}
    if set(records) != expected:
        raise ValueError(f"{label}: the judge records cover {len(records)} items, expected the {len(expected)} judged items "
                         f"(missing {len(expected - set(records))}, extra {len(set(records) - expected)})")
    n = len(block.item_ids)
    final, choice, chosen, order = [""] * n, [None] * n, [None] * n, [None] * n
    failure = {kind: np.zeros(n, dtype=bool) for kind in FAILURE_KINDS}
    tokens_in, tokens_out = np.zeros(n), np.zeros(n)
    api_in, api_out = np.full(n, np.nan), np.full(n, np.nan)
    for k, item_id in enumerate(block.item_ids):
        if not sub[k]:
            continue
        if not judged[k]:
            final[k] = block.answers[codes[0]][k]
            continue
        record = records[int(item_id)]
        trace = record.get("trace") or {}
        if trace.get("prompt_sha256") is None:
            raise ValueError(f"{label}: item {item_id} has no prompt_sha256 (not written by MenuJudge)")
        final[k] = record["final_answer"]
        choice[k] = trace.get("choice")
        chosen[k] = ARM_TO_CODE.get(trace.get("chosen_arm"))
        order[k] = [ARM_TO_CODE[arm_id] for arm_id in record["presentation_order"]]
        for kind, flag in classifyRecord(record).items():
            if kind in failure:
                failure[kind][k] = flag
        tokens_in[k], tokens_out[k] = record["tokens_in"], record["tokens_out"]
        call = record.get("call") or {}
        if call.get("usage_in") is not None:
            api_in[k] = call["usage_in"]
        if call.get("usage_out") is not None:
            api_out[k] = call["usage_out"]
    correct = np.array([block.compare(g, f) if s else False for g, f, s in zip(block.gold, final, sub)], dtype=bool)
    return JudgeArrays(judged, final, correct, choice, chosen, order, failure, tokens_in, tokens_out, api_in, api_out)


# ------------------------------------------------------------------
# §4、§7.5、§7.8：逐切分的量
# ------------------------------------------------------------------
def evaluateBlockMenu(block: PathBlock, menu: str, splits: np.ndarray, judge: JudgeArrays | None = None) -> dict:
    """
    一個區塊 × 菜單：對 200 次切分平均的 A_J、A_V、S_in、EN（正確率）與 Excess_J、Excess_V、Diff_JV（百分點），
    以及分解（每次切分核對 Excess = headroom × (recovery − recovery_blind)，Judge 與多數決各一）與成本。
    judge = None：只算多數決這一側（§6.1 的離線重現）。
    """
    codes = ordered(MENUS[menu])            # 平手優先順序（§4.2）
    sub, agree = allAnswered(block, codes), agreeMask(block, codes)
    vote_finals, tied = menuFinals(block, menu)
    vote_correct = np.array([block.compare(g, f) for g, f in zip(block.gold, vote_finals)], dtype=bool)
    path_correct = np.array([block.correct[c] for c in codes])
    any_correct, mean_correct = path_correct.any(axis=0), path_correct.mean(axis=0)
    path_tokens = np.sum([block.tokens[c] for c in codes], axis=0)

    keys = ["A_V", "S_in", "EN", "excess_V", "d", "c", "m", "headroom", "recovery_V", "recovery_blind", "tokens_paths", "tokens_S_in"]
    if judge is not None:
        keys += ["A_J", "excess_J", "diff_JV", "recovery_J", "tokens_flow"]
    per_split = {key: [] for key in keys}
    chosen = Counter()
    undefined, max_error_V, max_error_J = 0, 0.0, 0.0
    for h1 in splits:
        in1, in2 = h1 & sub, ~h1 & sub
        s_in = strongest({c: int(block.correct[c][in1].sum()) for c in codes})
        chosen[s_in] += 1
        v = {"A_V": vote_correct[in2].mean(), "S_in": block.correct[s_in][in2].mean(), "EN": block.correct["EN"][in2].mean()}
        v["excess_V"] = v["A_V"] - v["S_in"]
        if judge is not None:
            v["A_J"] = judge.correct[in2].mean()
            v["excess_J"] = v["A_J"] - v["S_in"]
            v["diff_JV"] = v["A_J"] - v["A_V"]
        dis = in2 & ~agree
        d = dis.sum() / in2.sum()
        c = any_correct[dis].mean() if dis.any() else float("nan")
        m = mean_correct[dis].mean() if dis.any() else float("nan")
        if dis.any() and c > m:
            rb = (block.correct[s_in][dis].mean() - m) / (c - m)
            rv = (vote_correct[dis].mean() - m) / (c - m)
            max_error_V = max(max_error_V, abs(v["excess_V"] - d * (c - m) * (rv - rb)))
            if judge is not None:
                rj = (judge.correct[dis].mean() - m) / (c - m)
                max_error_J = max(max_error_J, abs(v["excess_J"] - d * (c - m) * (rj - rb)))
        else:   # 沒有不一致題或其中無人答對：recovery 無定義，Excess 必須是 0
            rb = rv = rj = float("nan")
            undefined += 1
            max_error_V = max(max_error_V, abs(v["excess_V"]))
            if judge is not None:
                max_error_J = max(max_error_J, abs(v["excess_J"]))
        v.update(d=d, c=c, m=m, headroom=d * (c - m) if dis.any() else 0.0, recovery_V=rv, recovery_blind=rb,
                 tokens_paths=path_tokens[in2].mean(), tokens_S_in=block.tokens[s_in][in2].mean())
        if judge is not None:
            v.update(recovery_J=rj, tokens_flow=(path_tokens + judge.tokens_out)[in2].mean())
        for key in keys:
            per_split[key].append(v[key])

    def mean(key: str, scale: float = 1.0) -> float:
        array = np.asarray(per_split[key], dtype=float)
        return float(np.nanmean(array) * scale) if np.isfinite(array).any() else float("nan")

    n_sub = int(sub.sum())
    row = {
        "model": block.model, "dataset": block.dataset, "menu": menu, "K": len(codes), "n": len(block.item_ids),
        "n_subset": n_sub, "keep": float(sub.mean()), "n_disagree": int((sub & ~agree).sum()),
        "disagree_rate": float((sub & ~agree).sum() / n_sub) if n_sub else float("nan"),
        "A_V": mean("A_V"), "S_in": mean("S_in"), "EN": mean("EN"), "excess_V": mean("excess_V", 100),
        "d": mean("d"), "c": mean("c"), "m": mean("m"), "headroom_pp": mean("headroom", 100),
        "recovery_V": mean("recovery_V"), "recovery_blind": mean("recovery_blind"),
        "undefined_splits": undefined, "identity_max_error_V": max_error_V,
        "tokens_paths": mean("tokens_paths"), "tokens_S_in": mean("tokens_S_in"),
        "S_in_top": " / ".join(f"{code} {count / len(splits):.0%}" for code, count in chosen.most_common(2)),
    }
    if judge is not None:
        on = judge.judged
        row.update({
            "A_J": mean("A_J"), "excess_J": mean("excess_J", 100), "diff_JV": mean("diff_JV", 100),
            "recovery_J": mean("recovery_J"), "identity_max_error_J": max_error_J,
            "calls_per_item": len(codes) + mean("d"),
            "judge_tokens_in_per_call": float(judge.tokens_in[on].mean()) if on.any() else float("nan"),
            "judge_tokens_out_per_call": float(judge.tokens_out[on].mean()) if on.any() else float("nan"),
            "judge_api_in_per_call": float(np.nanmean(judge.api_in[on])) if np.isfinite(judge.api_in[on]).any() else float("nan"),
            "judge_api_out_per_call": float(np.nanmean(judge.api_out[on])) if np.isfinite(judge.api_out[on]).any() else float("nan"),
            "tokens_flow": mean("tokens_flow"), "tokens_ratio": mean("tokens_flow") / mean("tokens_S_in"),
            "n_calls": int(on.sum()), **{f"n_{kind}": int(judge.failure[kind].sum()) for kind in FAILURE_KINDS},
            "n_agg_no_answer": int((on & np.any([judge.failure[kind] for kind in FAILURE_KINDS], axis=0)).sum()),
        })
    return {"row": row, "codes": codes, "sub": sub, "agree": agree, "vote_finals": vote_finals, "vote_correct": vote_correct,
            "tied": tied, "judge": judge}


def compareJudge(block: PathBlock, splits: np.ndarray, first: JudgeArrays, second: JudgeArrays) -> float:
    """判定三（§5.5）：五條都有答案的題目上，first 的 A_J − second 的 A_J 在 H2 上，對切分平均（百分點）。"""
    keep = allAnswered(block, COMPARE_PATHS)
    return float(np.mean([first.correct[~h1 & keep].mean() - second.correct[~h1 & keep].mean() for h1 in splits]) * 100)


def twoPathDiff(splits: np.ndarray, sub: np.ndarray, judge_correct: np.ndarray, two_correct: np.ndarray) -> float:
    """§7.7：菜單的子集上，A_J(M3) − A(兩條 path 的 Judge) 在 H2 上，對切分平均（百分點）。"""
    return float(np.mean([judge_correct[~h1 & sub].mean() - two_correct[~h1 & sub].mean() for h1 in splits]) * 100)


# ------------------------------------------------------------------
# §7.3、§7.4、§7.6：不切分，用全部子集題目
# ------------------------------------------------------------------
def voteGroups(answers: list[str], compare) -> list[list]:
    """[答案, 票數]，答案依第一次出現的順序。"""
    groups = []
    for answer in answers:
        for group in groups:
            if compare(group[0], answer):
                group[1] += 1
                break
        else:
            groups.append([answer, 1])
    return groups


def itemComparison(block: PathBlock, result: dict) -> dict:
    """§7.4：不一致題上 Judge 與多數決的逐題對照，以及依正確答案位置的三類。"""
    codes, judge = result["codes"], result["judge"]
    on = judge.judged
    vote, vote_correct = result["vote_finals"], result["vote_correct"]
    same = np.array([block.compare(judge.final[k], vote[k]) for k in range(len(vote))], dtype=bool)
    classes = {"plurality": [], "minority": [], "absent": []}
    ties = 0
    for k in np.flatnonzero(on):
        groups = voteGroups([block.answers[c][k] for c in codes], block.compare)
        top = max(count for _, count in groups)
        gold_votes = sum(int(block.correct[c][k]) for c in codes)
        if gold_votes == 0:
            classes["absent"].append(k)
        elif gold_votes == top:
            classes["plurality"].append(k)
            ties += sum(count == top for _, count in groups) > 1
        else:
            classes["minority"].append(k)
    out = {
        "n_judged": int(on.sum()), "same_answer_rate": float(same[on].mean()) if on.any() else float("nan"),
        "both_right": int((on & judge.correct & vote_correct).sum()), "both_wrong": int((on & ~judge.correct & ~vote_correct).sum()),
        "vote_right_judge_wrong": int((on & vote_correct & ~judge.correct).sum()),
        "vote_wrong_judge_right": int((on & ~vote_correct & judge.correct).sum()), "plurality_ties": int(ties),
    }
    for name, ks in classes.items():
        out[f"n_{name}"] = len(ks)
        out[f"judge_right_{name}"] = float(judge.correct[ks].mean()) if ks else float("nan")
    return out


def skepticComparison(block: PathBlock, result: dict) -> dict:
    """§7.3（M12）：子集內 P:skeptic 的答案和多數決不同的題目上，Judge 的歸屬比例與三者的正確率。"""
    judge, vote, sub = result["judge"], result["vote_finals"], result["sub"]
    skeptic = block.answers["P2"]
    ks = [k for k in np.flatnonzero(sub) if not block.compare(skeptic[k], vote[k])]
    if not ks:
        return {"n": 0}
    to_skeptic = [block.compare(judge.final[k], skeptic[k]) for k in ks]
    to_vote = [block.compare(judge.final[k], vote[k]) for k in ks]
    return {
        "n": len(ks), "judge_is_skeptic": float(np.mean(to_skeptic)), "judge_is_vote": float(np.mean(to_vote)),
        "judge_other": float(np.mean([not a and not b for a, b in zip(to_skeptic, to_vote)])),
        "acc_skeptic": float(block.correct["P2"][ks].mean()), "acc_vote": float(result["vote_correct"][ks].mean()),
        "acc_judge": float(judge.correct[ks].mean()),
    }


def positionStats(result: dict) -> dict:
    """§7.6：被選中的位置（1..K）與 path 的比例；每條 path 在各位置出現的次數（以及各 path 次數的最大差距）。"""
    codes, judge = result["codes"], result["judge"]
    k = len(codes)
    on = np.flatnonzero(judge.judged)
    valid = [i for i in on if judge.choice[i] is not None]
    positions = Counter(judge.choice[i] for i in valid)
    paths = Counter(judge.chosen[i] for i in valid)
    shown = {code: Counter() for code in codes}
    for i in on:
        for position, code in enumerate(judge.order[i], start=1):
            shown[code][position] += 1
    spread = max((max(shown[c][p] for p in range(1, k + 1)) - min(shown[c][p] for p in range(1, k + 1)) for c in codes),
                 default=0)
    return {
        "n_valid": len(valid),
        "position_share": {p: positions[p] / len(valid) if valid else float("nan") for p in range(1, k + 1)},
        "path_share": {c: paths[c] / len(valid) if valid else float("nan") for c in codes},
        "shown": {c: [shown[c][p] for p in range(1, k + 1)] for c in codes},
        "position_spread": int(spread),
    }



def blocksWithSplits(armdir: str, aggdir: str, models: list[str] = MODELS):
    """
    依序產生 (PathBlock, splits)：每個資料集用 makeSplits(n, REPS, SEED)（§4.1），同一資料集的所有模型共用，
    題目 id 不一致就報錯。
    """
    splits_by_dataset = {}
    for model in models:
        for dataset in DATASETS:
            block = loadPathBlock(armdir, model, dataset, aggdir)
            if dataset not in splits_by_dataset:
                splits_by_dataset[dataset] = (block.item_ids, makeSplits(len(block.item_ids), REPS, SEED))
            ids, splits = splits_by_dataset[dataset]
            if not np.array_equal(ids, block.item_ids):
                raise ValueError(f"{model} | {dataset}: item ids differ from the other models (the split would not be shared)")
            yield block, splits
