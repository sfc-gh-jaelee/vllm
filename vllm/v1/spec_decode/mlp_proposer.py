# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np
import torch

from vllm.model_executor.models.arctic_speculator import MLPSpeculator

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
    ) -> Optional[np.ndarray]:
        input_ids = torch.tensor(context_token_ids, device=self.device)
        
        # next_tokens = self.model.generate_proposals(
        #     input_ids=input_ids,
        #     previous_hidden_states=previous_hidden_states,
        #     num_predict_tokens=3,
        # )

        more_tokens = self.model.generate_proposals(
            input_ids=input_ids,
            previous_hidden_states=previous_hidden_states,
            num_predict_tokens=self.model.n_predict,
        )


        from typing import List

        batch_size = input_ids.size(0)
        all_seq_candidates : List[List[List[int]]] = []

        for b in range(batch_size):
            seq_candidates : List[List[int]] = []
            for mt in more_tokens[0][b]:
                seq_candidates.append(mt.cpu().tolist())
            all_seq_candidates.append(seq_candidates)

        return all_seq_candidates
        
