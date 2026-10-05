from news_summarizer.evaluation import ner_scores, rouge_scores


def test_rouge_scores_reward_matching_summary_content():
    scores = rouge_scores("Maharashtra announced a farming program.", "Maharashtra announced a farming program.")

    assert scores["rouge1_f1"] == 1.0
    assert scores["rouge2_f1"] == 1.0
    assert scores["rougeL_f1"] == 1.0


def test_ner_scores_calculate_exact_match_metrics():
    scores = ner_scores(
        [("Mumbai", "LOCATION"), ("Ministry", "ORGANIZATION")],
        [("Mumbai", "LOCATION"), ("Delhi", "LOCATION")],
    )

    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["f1"] == 0.5
