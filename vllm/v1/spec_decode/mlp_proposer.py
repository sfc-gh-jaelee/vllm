# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np
import torch
from typing import List
from vllm.model_executor.models.arctic_speculator import MLPSpeculator
from vllm.v1.spec_decode.tree_decoding import SequenceTreeV2

class MLPProposer:
    
    def link_model(
        self,
        model: MLPSpeculator,
    ):
        self.model = model
        self.device = next(model.parameters()).device

    def propose(
        self,
        context_token_ids: np.ndarray,
        previous_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = torch.tensor(context_token_ids, device=self.device)
        
        # next_tokens = self.model.generate_proposals(
        #     input_ids=input_ids,
        #     previous_hidden_states=previous_hidden_states,
        #     num_predict_tokens=3,
        # )

        more_tokens: List[torch.Tensor] = self.model.generate_proposals(
            input_ids=input_ids,
            previous_hidden_states=previous_hidden_states,
            num_predict_tokens=3,
        )

        return more_tokens[0]

        

        print("input_ids:", input_ids)
        print(f"more tokens: {more_tokens[0]}")
        batch_size, num_paths, path_length = more_tokens[0].shape
        assert input_ids.size(0) == batch_size

        print("more_tokens[0][b,:,:]", more_tokens[0][0,:,:])
        print("more_tokens[0][b,:,:].cpu().tolist()", more_tokens[0][0,:,:].cpu().tolist())


        sequence_trees = [SequenceTreeV2.from_paths(more_tokens[0][b,:,:].cpu().tolist()) for b in range(batch_size)]
        print("MLPProposer.propose returns: sequence_trees: ", sequence_trees)
        return sequence_trees

        # batch_size = input_ids.size(0)
        # all_seq_candidates : List[List[List[int]]] = []

        # for b in range(batch_size):
        #     seq_candidates : List[List[int]] = []
        #     for mt in more_tokens[0][b]:
        #         seq_candidates.append(mt.cpu().tolist())
        #     all_seq_candidates.append(seq_candidates)

        # return all_seq_candidates
        
