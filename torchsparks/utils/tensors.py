#  Copyright 2024 Chuanwise and contributors.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import torch

import torchsparks.utils.shapes


class TensorBuilder:
    class _AutoIncreasingList:
        _EMPTY_TENSOR = torch.zeros(0)

        def __init__(self, data, parent=None):
            self.data = data
            self.parent = parent
            if parent is None:
                self.is_building = False

        def _root_is_building(self):
            node = self
            while node.parent is not None:
                node = node.parent
            return node.is_building

        def __getitem__(self, item):
            self._ensure_size(item)

            result = self.data[item]
            if result is self._EMPTY_TENSOR:
                if self._root_is_building():
                    result = self._EMPTY_TENSOR
                else:
                    result = TensorBuilder._AutoIncreasingList(self)
                    self.data[item] = result
            return result

        def _ensure_size(self, slice_or_index):
            if isinstance(slice_or_index, slice):
                size = slice_or_index.stop
            elif isinstance(slice_or_index, int):
                size = slice_or_index
            else:
                raise ValueError("slice_or_index must be a slice or an integer.")

            if size >= len(self.data):
                self.data.extend([self._EMPTY_TENSOR] * (size - len(self.data) + 1))

        def extend(self, sequence):
            self.data.extend(sequence)

        def append(self, value):
            self.data.append(value)

        def __setitem__(self, key, value):
            self._ensure_size(key)
            if isinstance(value, (int, float)):
                value = torch.tensor(value)
            if isinstance(key, slice):
                for i in range(key.start or 0, key.stop, key.step or 1):
                    self.data[i] = value
            else:
                self.data[key] = value

        def __setslice__(self, i, j, sequence):
            self._ensure_size(j - 1)
            self.data[i:j] = sequence

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

        def __repr__(self):
            return f"InfinityList({self.data})"

    def __init__(self):
        super().__init__()

        self.data = self._AutoIncreasingList([])

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, item):
        return self.data[item]

    def __setslice__(self, i, j, sequence):
        self.data[i:j] = sequence

    def build(self, *args, **kwargs):
        self.data.is_building = True
        try:
            return torchsparks.utils.shapes.pad(self.data, *args, **kwargs)
        finally:
            self.data.is_building = False
