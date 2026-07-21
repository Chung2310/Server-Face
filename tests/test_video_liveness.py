import numpy as np

from app.services.video_liveness import VideoLivenessService


def test_turn_motion_requires_neutral_turn_and_return():
    assert VideoLivenessService.score_turn([0, 1, 2, 18, 22, 4, 1], "turn_right") >= 0.8
    assert VideoLivenessService.score_turn([0, 1, 2, 18, 22], "turn_right") == 0.0
    assert VideoLivenessService.score_turn([0, -2, -18, -22, -3, 0], "turn_left") >= 0.8


def test_blink_motion_requires_open_closed_open():
    assert VideoLivenessService.score_blink([0.30, 0.31, 0.29, 0.10, 0.09, 0.28, 0.30]) >= 0.8
    assert VideoLivenessService.score_blink([0.30, 0.31, 0.29, 0.10, 0.09]) == 0.0


def test_composite_score_uses_conservative_passive_percentile():
    passive, final = VideoLivenessService.combine_scores(
        [0.99, 0.98, 0.97, 0.20, 0.96], 0.9
    )
    assert passive == np.percentile([0.99, 0.98, 0.97, 0.20, 0.96], 10)
    assert final == min(passive, 0.9)
