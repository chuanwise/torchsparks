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

import torchsparks.utils.tensors

tensor = torchsparks.utils.tensors.TensorBuilder()

tensor[0] = torch.zeros(1, 3, 5)
tensor[2].append(torch.zeros(1, 3, 5))
tensor[2][5:6] = 1

tensor, _ = tensor.build([3, 6, 5])

print(tensor.shape)
print(tensor)
