import random

from samg.training import default_schedule


def test_random_penalties_draw_once_per_epoch_and_are_reproducible():
    first = default_schedule("samg-random", reconstruction_epochs=1, switch_epoch=1, ramp_length=0)
    second = default_schedule("samg-random", reconstruction_epochs=1, switch_epoch=1, ramp_length=0)
    draws_a = first.resample(random.Random(2044))
    draws_b = second.resample(random.Random(2044))
    assert draws_a == draws_b
    assert set(draws_a) == {"kl", "cond_in", "cond_out", "poincare"}
    assert all(0.0 <= value < 1.0 for value in draws_a.values())
    assert len(set(draws_a.values())) == len(draws_a)
    assert "reconstruction" not in draws_a


def test_physical_penalty_reference_is_fitted_before_ramp():
    schedule = default_schedule("samg", reconstruction_epochs=1, switch_epoch=3, ramp_length=0)
    schedule.resample(random.Random(1))
    assert set(schedule.pending(1)) == {"cond_in", "cond_out", "poincare"}
    schedule.observe(1, {"cond_in": 100.0, "cond_out": 10.0, "poincare": 2.0})
    schedule.observe(2, {"cond_in": 400.0, "cond_out": 40.0, "poincare": 8.0})
    weights = schedule.weights(3)
    assert schedule.references == {"cond_in": 250.0, "cond_out": 25.0, "poincare": 5.0}
    assert weights["cond_in"] == 0.10 / 250.0
