from abc import ABC, abstractmethod
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Module
import torch as ch

from trak.modelout_functions import AbstractModelOutput


class RLHFRewardModelingOutput(AbstractModelOutput):
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_output(
        model,
        weights: Iterable[Tensor],
        buffers: Iterable[Tensor],
        input_ids_chosen: Tensor,
        attention_mask_chosen: Tensor,
        token_type_ids_chosen: Tensor,
        input_ids_rejected: Tensor,
        attention_mask_rejected: Tensor,
        token_type_ids_rejected: Tensor,
    ) -> Tensor:
        kw_inputs = {
            "input_ids_chosen": input_ids_chosen.unsqueeze(0),
            "token_type_ids_chosen": token_type_ids_chosen.unsqueeze(0),
            "attention_mask_chosen": attention_mask_chosen.unsqueeze(0),
            "input_ids_rejected": input_ids_rejected.unsqueeze(0),
            "token_type_ids_rejected": token_type_ids_rejected.unsqueeze(0),
            "attention_mask_rejected": attention_mask_rejected.unsqueeze(0),
        }

        reward_difference = ch.func.functional_call(
            model, (weights, buffers), args=(), kwargs=kw_inputs
        )

        return reward_difference.sum().squeeze(-1)

    def get_out_to_loss_grad(self, model, weights, buffers, batch: Iterable[Tensor]) -> Tensor:
        (input_ids_chosen, token_type_ids_chosen, attention_mask_chosen, input_ids_rejected, token_type_ids_rejected,
         attention_mask_rejected) = batch
        kw_inputs = {
            "input_ids_chosen": input_ids_chosen,
            "token_type_ids_chosen": token_type_ids_chosen,
            "attention_mask_chosen": attention_mask_chosen,
            "input_ids_rejected": input_ids_rejected,
            "token_type_ids_rejected": token_type_ids_rejected,
            "attention_mask_rejected": attention_mask_rejected,
        }

        reward_difference = ch.func.functional_call(
            model, (weights, buffers), args=(), kwargs=kw_inputs
        )

        out_to_loss_grad = -torch.sigmoid(-reward_difference)

        return out_to_loss_grad.clone().detach()

