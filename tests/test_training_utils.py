import pytest
import torch
from torch import nn
from torch.optim import SGD

from training_utils import ModelEMA, should_evaluate, warmup_cosine_scheduler


def test_ema_updates_toward_current_model() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEMA(model, decay=0.5)

    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)

    assert torch.allclose(ema.model.weight, torch.full_like(ema.model.weight, 2.0))
    assert not ema.model.weight.requires_grad


def test_ema_warmup_does_not_retain_the_untrained_initial_state() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEMA(model, decay=0.9998, warmup_tau=2000.0)

    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)

    assert ema.model.weight.item() > 2.99


def test_warmup_cosine_scheduler_warms_then_decays() -> None:
    model = nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=1.0)
    scheduler = warmup_cosine_scheduler(optimizer, total_steps=6, warmup_steps=2)
    rates = [optimizer.param_groups[0]["lr"]]

    for _ in range(6):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates[0] < rates[1]
    assert rates[1] == pytest.approx(1.0)
    assert rates[-1] == pytest.approx(0.0)


def test_evaluation_schedule_increases_frequency_near_end() -> None:
    evaluated_epochs = [
        epoch + 1 for epoch in range(9) if should_evaluate(epoch, epochs=9, interval=3)
    ]

    assert evaluated_epochs == [3, 6, 7, 8, 9]
