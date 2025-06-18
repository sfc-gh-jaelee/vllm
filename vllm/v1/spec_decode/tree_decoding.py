from typing import List, Dict, Any, Optional
from collections import deque

import torch
import numpy as np
from typing import List, Union

class SequenceTreeV2:
    def __init__(self):
        self.token_ids: List[int] = []
        self.parents: List[int] = []
        self.custom_mask: Optional[torch.Tensor] = None

    def __str__(self):
        return f"SequenceTreeV2(token_ids={self.token_ids}, parents={self.parents}), custom_mask={self.custom_mask}"

    def __repr__(self):
        return self.__str__()

    def construct_mask(self):
        num_tokens = len(self.token_ids)
        self.custom_mask = torch.eye(num_tokens, dtype=torch.long)
        for current_token_idx in range(1, num_tokens):
            parent_token_idx = self.parents[current_token_idx]
            self.custom_mask[current_token_idx, :] += self.custom_mask[parent_token_idx, :]

    def mask(self) -> torch.Tensor:
        if self.custom_mask is None:
            self.construct_mask()
        return self.custom_mask

    def construct_position_ids(self, position_ids: torch.Tensor, offset: int = 0) -> torch.Tensor:
        position_ids[0] = offset
        for i in range(1, len(self.token_ids)):
            parent_idx = self.parents[i]
            position_ids[i] = position_ids[parent_idx] + 1
    
    def verify(self, sequence: List[int]):
        assert len(sequence) == len(self.token_ids)
        assert len(self.token_ids) == len(self.parents)
        last_accepted_idx = 0
        accepted_tokens = []
        accepted_indices = []
        for i, tok in enumerate(self.token_ids):
            if i==0:
                continue
            if last_accepted_idx == self.parents[i] and sequence[self.parents[i]] == tok:
                last_accepted_idx = i
                accepted_tokens.append(tok)
                accepted_indices.append(self.parents[i])
        
        # Bonus token: can be found in sequence[last_accepted_idx]
        accepted_tokens.append(sequence[last_accepted_idx])
        accepted_indices.append(last_accepted_idx)

        return accepted_tokens, accepted_indices
    
    @classmethod
    def from_paths(cls, paths: List[List[int]]):
        
        instance = cls()
        
        if paths is None or len(paths) == 0:
            return []
        
        # Remove empty paths
        paths = [path for path in paths if len(path) > 0]
        if not paths:
            return []
        
        # Build a prefix tree structure implicitly using sets
        # This helps us understand the tree structure without building actual nodes
        all_prefixes = set()
        for path in paths:
            for i in range(1, len(path) + 1):
                all_prefixes.add(tuple(path[:i]))
        
        result = []
        parent_indices = []
        
        def dfs_recursive(prefix, parent_index):
            """
            Perform DFS traversal using prefix tuples
            
            Args:
                prefix: Tuple representing current path from root
                parent_index: Index of parent in the result list (-1 for root)
            """
            # Add current node to result
            current_index = len(result)
            result.append(prefix[-1])
            parent_indices.append(parent_index)
            
            # Find all children of current prefix
            children = set()
            for full_prefix in all_prefixes:
                if (len(full_prefix) == len(prefix) + 1 and 
                    full_prefix[:-1] == prefix):
                    children.add(full_prefix[-1])
            
            # Process children in sorted order
            for child_val in sorted(children):
                child_prefix = prefix + (child_val,)
                dfs_recursive(child_prefix, current_index)
        
        # Start from root
        root_val = paths[0][0]
        dfs_recursive((root_val,), -1)

        instance.token_ids = result
        instance.parents = parent_indices

        instance.construct_mask()

        return instance
        
    # return result, parent_indices
        
    #     instance.token_ids = dfs_order
    #     instance.parents = parents

    #     instance.construct_mask()
        
    #     return instance

class SequenceTree:

    def __init__(self):
        self.root: Dict[int, Dict[str, Any]] = {}
        self._flattened_sequence: Optional[List[int]] = None
        self._seqs: List[List[int]] = []
        self._tree_mask: Optional[torch.Tensor] = None

    def add_sequence(self, sequence: List[int]):
        self._seqs.append(sequence)

        current_level_nodes = self.root
        for number in sequence:
            if number not in current_level_nodes:
                current_level_nodes[number] = {'children': {}, 'index': None}
            current_level_nodes = current_level_nodes[number]['children']

    # def __get_sequences(self) -> List[List[int]]:
    #     return self._seqs

    def __get_indices(self, sequence: List[int]) -> List[int]:
        current_level_nodes = self.root
        indices = []
        for number in sequence:
            if number not in current_level_nodes:
                return indices
            indices.append(current_level_nodes[number]['index'])
            current_level_nodes = current_level_nodes[number]['children']
        return indices

    def verify(self, sequence: List[int]) -> tuple[List[int], List[int]]:
        assert self._flattened_sequence is not None, "Please call flat() before calling verify()"
        assert len(sequence) == len(
            self._flattened_sequence
        ), "flattened sequence and input sequence must have the same length"
        assert len(
            self.root.keys()) == 1, "There must be exactly one root node"

        current_level_nodes = self.root[list(self.root.keys())[0]]
        seq_idx = 0
        indices = [seq_idx]
        values = [sequence[seq_idx]]
        while sequence[seq_idx] in current_level_nodes['children']:
            current_level_nodes = current_level_nodes['children'][
                sequence[seq_idx]]
            seq_idx = current_level_nodes['index']
            indices.append(seq_idx)
            values.append(sequence[seq_idx])

        return values, indices

    def __create_tree_mask(self) -> torch.Tensor:
        assert self._flattened_sequence is not None, "Please call flat() before calling _create_tree_mask()"

        max_sequence_length = len(self._flattened_sequence)
        mask = torch.eye(max_sequence_length, dtype=torch.bool)
        mask[:, 0] = True

        paths = [self.__get_indices(seq) for seq in self._seqs]
        for path in paths:
            for i in range(1, len(path)):
                for j in range(i + 1, len(path)):
                    mask[path[j], path[i]] = True

        return mask
    
    def mask(self) -> torch.Tensor:
        return self._tree_mask

    def flat(self) -> tuple[List[int], torch.Tensor]:
        self._flattened_sequence = []
        queue = deque()
        current_index = 0

        for number in self.root.keys():
            node_data = self.root[number]
            queue.append((number, node_data))

        while queue:
            number, node_data = queue.popleft()
            node_data['index'] = current_index
            self._flattened_sequence.append(number)
            current_index += 1

            children_dict = node_data['children']
            for child_number in children_dict.keys():
                child_node_data = children_dict[child_number]
                queue.append((child_number, child_node_data))

        self._tree_mask = self.__create_tree_mask()
        return self._flattened_sequence

    def __str__(self) -> str:
        import json

        def default_serializer(obj):
            return str(obj)

        try:
            return json.dumps(self.root, indent=2, default=default_serializer)
        except TypeError as e:
            return f"Could not serialize tree structure: {e}\nRoot dictionary: {self.root}"