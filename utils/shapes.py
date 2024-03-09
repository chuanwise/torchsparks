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

import utils.pythons


def shape(
        inputs,
        requirements=None,
        default_shape_requirement="max",
        truncation=True,
        need_len=True,
        return_dict=True,
):
    if requirements is None:
        num_requirements = 0
    elif isinstance(requirements, (int, str)):
        default_shape_requirement = requirements
        num_requirements = 0
    else:
        num_requirements = len(requirements)

    # add a dimension for convenience
    inputs = [inputs]
    if requirements is not None and isinstance(requirements, list):
        requirements.insert(0, 1)

    def shape_recursively(depth, depth_inputs):
        """
        Calculate the shape of the input tensors recursively.

        :param depth: index of dimension
        :param depth_inputs: tensor or list
        :return: shape, len or None if `need_len` is False
        """
        if isinstance(depth_inputs, torch.Tensor):
            return depth_inputs.size(), depth_inputs.size() if need_len else None
        elif utils.pythons.iterable(depth_inputs):
            element_shapes_and_lens = [shape_recursively(depth + 1, x) for x in depth_inputs]

            num_dimensions = max(len(x) for x, _ in element_shapes_and_lens)
            current_shape_results = [len(element_shapes_and_lens)]

            for i in range(num_dimensions):
                # get the shape value of corresponding requirement,
                # if the requirement is not specified, use the default value
                requirement_value = None

                # "+ 1" for the list itself
                requirement_index = i + depth + 1
                if requirement_index < num_requirements:
                    shape_requirement = requirements[requirement_index]
                else:
                    shape_requirement = default_shape_requirement
                if isinstance(shape_requirement, int):
                    if shape_requirement != -1:
                        requirement_value = shape_requirement
                elif shape_requirement not in ("min", "max"):
                    raise ValueError(f"Invalid shape requirement: {shape_requirement}")

                # get the shape value of the corresponding dimension
                for shape_value, _ in element_shapes_and_lens:
                    if len(shape_value) <= i:
                        continue

                    current_shape_value = shape_value[i]
                    if shape_requirement == "min":
                        if requirement_value is None or current_shape_value < requirement_value:
                            requirement_value = current_shape_value
                    elif shape_requirement == "max":
                        if requirement_value is None or current_shape_value > requirement_value:
                            requirement_value = current_shape_value
                    elif shape_requirement == -1:
                        # check if they are the same
                        if requirement_value is not None and current_shape_value != requirement_value:
                            raise ValueError(f"Shape requirement conflict: "
                                             f"{requirement_value} and {current_shape_value}")
                        requirement_value = current_shape_value
                    elif isinstance(shape_requirement, int):
                        if requirement_value < current_shape_value and not truncation:
                            requirement_value = current_shape_value
                    else:
                        raise ValueError(f"Invalid shape requirement: {shape_requirement}")

                current_shape_results.append(requirement_value)

            current_len_results = None
            if need_len:
                current_len_results = [
                    element_len for _, element_len in element_shapes_and_lens
                ]
            return current_shape_results, current_len_results

    # return results
    shape_results, len_results = shape_recursively(0, inputs)

    # remove the first dimension
    shape_results = shape_results[1:]
    len_results = len_results[0] if need_len else None

    if return_dict:
        return {
            "shape": shape_results,
            "len": len_results
        }
    else:
        return shape_results, len_results


def pad(
        data, requirements=None,
        default_shape_requirement="max",

        unsqueeze_location="after",

        padding_mode="constant",
        padding_location="after",
        padding_value=0,

        data_device=None,
        data_dtype=None,
        truncation=True,
        truncation_location="after",

        need_len=True,
        len_device=None,
        len_dtype=torch.int32,
        return_dict=True,
):
    """
    Pad tensors to given shape.

    The parameter `shape` can be a shape requirement or a list of it, where shape requirement can be:

    - a positive integer: the corresponding dimension of the tensor will be padded to the maximum of the
      integer and the original dimension if `truncation` is True, or the given shape.
    - "min" or "max": the corresponding dimension of the tensor will be padded to the minimum or maximum
      of the original dimensions.

    The parameter `empty_shape` can be:

    - "ignore": ignore empty shape.
    - "raise": raise an error if empty shape is found.

    """

    if data_device is not None and need_len and len_device is None:
        len_device = data_device

    # calculate shape
    shape_values, len_values = shape(
        data, requirements, default_shape_requirement, truncation, return_dict=False, need_len=need_len
    )

    # add a dimension for convenience
    data = [data]
    shape_values.insert(0, 1)
    len_values = [len_values] if need_len else None

    num_dimensions = len(shape_values)

    def pad_recursively(depth, depth_data, depth_lens):
        if isinstance(depth_data, torch.Tensor):
            # unsqueeze the tensor to the required dimensions
            if depth_data.ndim < (num_dimensions - depth):

                # unsqueeze the tensor to the required dimensions
                if unsqueeze_location == "before":
                    unsqueeze_location_value = 0
                elif unsqueeze_location == "after":
                    unsqueeze_location_value = depth_data.ndim
                else:
                    raise ValueError(f"Invalid value for `unsqueeze_location`: {unsqueeze_location}")

                for _ in range(num_dimensions - depth - depth_data.ndim):
                    depth_data = depth_data.unsqueeze(unsqueeze_location_value)

            # build the len of the tensor
            if need_len:
                if len(depth_lens) < (depth_data.ndim - 1):
                    # get the shape of len tensor, its dimension is always depth_data.ndim - 1
                    if unsqueeze_location == "before":
                        depth_len_shape = [1] * (depth_data.ndim - 1 - len(depth_lens)) + list(depth_lens)
                    elif unsqueeze_location == "after":
                        depth_len_shape = list(depth_lens) + [1] * (depth_data.ndim - 1 - len(depth_lens))
                    else:
                        raise ValueError(f"Invalid value for `unsqueeze_location`: {unsqueeze_location}")
                else:
                    depth_len_shape = list(depth_lens)[:depth_data.ndim - 1]

                # fill values and build tensor
                depth_lens = depth_data.size(-1)
                for i in reversed(range(len(depth_len_shape))):
                    depth_lens = [depth_lens] * depth_len_shape[i]
                depth_lens = torch.tensor(depth_lens, device=len_device, dtype=len_dtype)

            # pad
            argument_pad = []
            indexes_after_pad = [slice(None)] * depth_data.ndim
            for i in reversed(range(depth_data.ndim)):
                size = depth_data.size(i)
                shape_value = shape_values[i + depth]
                if size < shape_value:
                    size_to_pad = shape_value - size
                    if padding_location == "before":
                        argument_pad += [size_to_pad, 0]
                    elif padding_location == "after":
                        argument_pad += [0, size_to_pad]
                    else:
                        raise ValueError(f"Invalid value for `padding_location`: {padding_location}")
                    continue

                if size > shape_value:
                    if not truncation:
                        raise ValueError(f"Dimension {i} of tensor has size {size}, "
                                         f"which is greater than the shape requirement {shape_value}")
                    if truncation_location == "before":
                        indexes_after_pad[i] = slice(size - shape_value, None)
                    elif truncation_location == "after":
                        indexes_after_pad[i] = slice(None, shape_value)
                    else:
                        raise ValueError(f"Invalid value for `truncation_location`: {truncation_location}")
                argument_pad += [0, 0]

            depth_data = torch.nn.functional.pad(depth_data, argument_pad, value=padding_value, mode=padding_mode)

            # change dtype and device if required
            if data_dtype is not None:
                depth_data = depth_data.type(data_dtype)
            if data_device is not None:
                depth_data = depth_data.to(data_device)

            if need_len:
                depth_lens = torch.nn.functional.pad(depth_lens, argument_pad[2:], value=0, mode="constant")
                return depth_data[indexes_after_pad], depth_lens[indexes_after_pad[:-1]]
            else:
                return depth_data[indexes_after_pad], None
        elif utils.pythons.iterable(depth_data):
            # ignore truncated dimensions
            if len(depth_data) > shape_values[depth]:
                if not truncation:
                    raise ValueError(f"Length of list is greater than the shape requirement {shape_values[depth]}")

                if truncation_location == "before":
                    depth_data = depth_data[-shape_values[depth]:]
                    depth_lens = depth_lens[-shape_values[depth]:]
                elif truncation_location == "after":
                    depth_data = depth_data[:shape_values[depth]]
                    depth_lens = depth_lens[:shape_values[depth]]
                else:
                    raise ValueError(f"Invalid value for `truncation_location`: {truncation_location}")

            if need_len:
                pad_results = [pad_recursively(depth + 1, x, y) for x, y in zip(depth_data, depth_lens)]
            else:
                pad_results = [pad_recursively(depth + 1, x, None) for x in depth_data]

            # pad data
            current_data_results = torch.stack([x for x, _ in pad_results])
            current_len_results = torch.stack([y for _, y in pad_results]) if need_len else None
            if len(depth_data) < shape_values[depth]:
                argument_pad = [0, 0] * (num_dimensions - depth - 1)

                size_to_pad = shape_values[depth] - len(depth_data)
                if padding_location == "before":
                    argument_pad += [size_to_pad, 0]
                elif padding_location == "after":
                    argument_pad += [0, size_to_pad]
                else:
                    raise ValueError(f"Invalid value for `padding_location`: {padding_location}")

                current_data_results = torch.nn.functional.pad(
                    current_data_results, argument_pad, value=padding_value, mode=padding_mode
                )
                current_len_results = torch.nn.functional.pad(
                    current_len_results, argument_pad[2:], value=0, mode="constant"
                ) if need_len else None

            return current_data_results, current_len_results
        else:
            raise ValueError(f"Invalid value for `depth_data`: {depth_data}")

    # return results
    data_results, len_results = pad_recursively(0, data, len_values)

    # remove the first dimension
    data_results = data_results[0]
    len_results = len_results[0] if need_len else None

    if return_dict:
        return {
            "data": data_results,
            "len": len_results
        }
    else:
        return data_results, len_results
