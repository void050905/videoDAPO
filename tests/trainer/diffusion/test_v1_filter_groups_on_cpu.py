# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU test for the DAPO group filter step of the v1 diffusion policy-gradient trainer."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from verl import DataProto

import verl_omni

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")


def compose_cfg(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides)


class _Stop(Exception):
    pass


def test_filter_groups_rejects_loss_without_rollout_is_weights():
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    with pytest.raises(ValueError, match="filter_groups"):
        PolicyGradientDiffusionTrainerV1Sync(
            compose_cfg(["algorithm.filter_groups.enable=true", "actor_rollout_ref.actor.diffusion_loss.loss_mode=awm"])
        )


@pytest.mark.parametrize("enable", [False, True])
def test_filter_groups_weights_reach_actor_update(monkeypatch, enable):
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    trainer = PolicyGradientDiffusionTrainerV1Sync(compose_cfg([f"algorithm.filter_groups.enable={enable}"]))
    steps = 2
    # Group a has tied rewards; the last row is a _balance_batch pad copy of a b row.
    data = DataProto.from_dict(
        tensors={
            "all_timesteps": torch.zeros(5, steps),
            "rm_scores": torch.tensor([[1.0], [1.0], [0.0], [2.0], [0.0]]),
        },
        non_tensors={"uid": np.array(["a", "a", "b", "b", "b"], dtype=object)},
    )
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=object())
    trainer.global_steps = 1
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.diffusion_tq_batch_to_dataproto",
        lambda meta, pad_token_id: data,
    )

    def mark_pad(d, metrics):
        d.non_tensor_batch["_balance_is_pad"] = np.array([False] * 4 + [True])
        return d

    monkeypatch.setattr(trainer, "_balance_batch", mark_pad)
    old_log_prob = DataProto.from_dict(tensors={"old_log_probs": torch.zeros(5, steps)})
    monkeypatch.setattr(trainer, "_compute_old_log_prob", lambda d: old_log_prob)
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.compute_rollout_corr_metrics_from_batch",
        lambda data, bypass_mode: {},
    )
    captured = {}

    def stop_at_update_actor(d):
        captured["data"] = d
        raise _Stop

    monkeypatch.setattr(trainer, "_update_actor", stop_at_update_actor)
    metrics = {}
    with pytest.raises(_Stop):
        trainer._train_sampled_batch(metrics, {}, object())

    batch = captured["data"].batch
    assert "_balance_is_pad" not in captured["data"].non_tensor_batch
    assert torch.all(batch["advantages"][4] == 0)
    if not enable:
        assert "rollout_is_weights" not in batch
        assert not any(key.startswith("training/filter_groups/") for key in metrics)
        return
    expected = torch.tensor([0.0, 0.0, 5 / 2, 5 / 2, 0.0])
    torch.testing.assert_close(batch["rollout_is_weights"], expected[:, None].expand(5, steps))
    assert metrics["training/filter_groups/kept_groups"] == 1
    assert metrics["training/filter_groups/num_groups"] == 2


def test_filter_groups_refill_redraws_zero_signal_groups(monkeypatch):
    """filter_groups.refill_rounds>0 evicts zero-signal groups and re-draws replacements."""
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    trainer = PolicyGradientDiffusionTrainerV1Sync(
        compose_cfg(
            [
                "algorithm.filter_groups.enable=true",
                "algorithm.filter_groups.min_std=0.0",
                "algorithm.filter_groups.refill_rounds=3",
                "trainer.v1.sampler.drop_incomplete_groups=true",
            ]
        )
    )
    assert trainer._filter_refill_rounds == 3
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.global_steps = 1

    # Round 1: group "a" is zero-signal; round 2: all clean.
    def fake_sample(global_steps, partition_id, batch_size):
        if not hasattr(fake_sample, "round"):
            fake_sample.round = 0
        fake_sample.round += 1
        if fake_sample.round == 1:
            keys = ["a_0_0", "a_0_1", "b_0_2", "b_0_3"]
            probe = DataProto.from_dict(
                tensors={"rm_scores": torch.tensor([[0.5], [0.5], [0.1], [0.9]])},
                non_tensors={"uid": np.array(["a", "a", "b", "b"], dtype=object)},
            )
        else:
            keys = ["c_1_0", "c_1_1", "d_1_2", "d_1_3"]
            probe = DataProto.from_dict(
                tensors={"rm_scores": torch.tensor([[0.2], [0.8], [0.3], [0.7]])},
                non_tensors={"uid": np.array(["c", "c", "d", "d"], dtype=object)},
            )
        fake_sample.probe = probe
        return SimpleNamespace(keys=keys, tags=[{}] * len(keys), partition_id="train"), {}

    monkeypatch.setattr(trainer.replay_buffer, "sample", fake_sample)
    monkeypatch.setattr(trainer.replay_buffer, "prompt_global_steps", {"train": {}})
    monkeypatch.setattr(trainer.replay_buffer, "finished_keys", {"train": set()})
    monkeypatch.setattr(trainer.replay_buffer, "failure_keys", {"train": set()})
    monkeypatch.setattr(trainer.replay_buffer, "partitions", {"train": {}})
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.diffusion_tq_batch_to_dataproto",
        lambda meta, pad_token_id: fake_sample.probe,
    )
    monkeypatch.setattr("verl_omni.trainer.diffusion.v1.trainer_base.tq.kv_clear", lambda **kwargs: None)
    refilled = []
    monkeypatch.setattr(trainer, "_add_prompts_to_generate", lambda n: refilled.append(n) or n)

    batch_meta, metrics = trainer._sample_training_batch(4)

    assert refilled == [1]  # group "a" (1 prompt x n samples) was re-drawn once
    assert [k.split("_")[0] for k in batch_meta.keys] == ["b", "b", "c", "c", "d", "d"]
    assert metrics["training/rollout_failure/evicted_groups"] == 1
    assert metrics["training/filter_groups/kept_groups"] == 2
    assert metrics["training/filter_groups/num_groups"] == 2


def test_filter_groups_refill_requires_sampler_flag():
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    with pytest.raises(ValueError, match="drop_incomplete_groups"):
        PolicyGradientDiffusionTrainerV1Sync(
            compose_cfg(
                [
                    "algorithm.filter_groups.enable=true",
                    "algorithm.filter_groups.refill_rounds=1",
                ]
            )
        )
