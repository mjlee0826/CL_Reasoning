import numpy as np

from Aggregator.Aggregator import Aggregator, AggregationItem, Resolution


class VoteAggregator(Aggregator):
    """
    Vote@K (v2 / vote3 / vote5): majority vote over the candidates' parsed answers.

    - Answers are grouped with dataset.compareTwoAnswer.
    - Ties among the top answers are broken with default_rng([seed, item_id]), so reruns are reproducible.
    - Empty answers count as ordinary votes (same as voting_5language.py); otherwise Vote@2 would
      "rescue" every item where one side failed to parse and its recovery would no longer be 0.
    """
    def resolve(self, item: AggregationItem) -> Resolution:
        groups: list[list[str]] = []  # equal answers grouped, in candidate order
        for candidate in item.candidates:
            for group in groups:
                if self.dataset.compareTwoAnswer(group[0], candidate.answer):
                    group.append(candidate.answer)
                    break
            else:
                groups.append([candidate.answer])

        top = max(len(group) for group in groups)
        tied = [group[0] for group in groups if len(group) == top]
        if len(tied) == 1:
            return Resolution(final_answer=tied[0])

        rng = np.random.default_rng([self.config.seed, int(item.item_id)])
        return Resolution(final_answer=tied[int(rng.integers(len(tied)))])
