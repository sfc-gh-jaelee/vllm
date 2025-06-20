from typing import List, Dict, Any, Optional
from collections import deque

import torch


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
        self.depths = []

        for number in self.root.keys():
            node_data = self.root[number]
            queue.append((number, node_data))
            self.depths.append(0)
            node_data["depth"] =0

        while queue:
            number, node_data = queue.popleft()
            node_data['index'] = current_index
            self._flattened_sequence.append(number)
            current_index += 1

            children_dict = node_data['children']
            depth = node_data["depth"]
            for child_number in children_dict.keys():
                child_node_data = children_dict[child_number]
                queue.append((child_number, child_node_data))                
                self.depths.append(depth+1)
                child_node_data["depth"] = depth+1

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