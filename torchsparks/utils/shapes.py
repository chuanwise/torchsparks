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

import torchsparks.utils.pythons


def shape(
        inputs,
        requirement=None,
        default_requirement="max",
        truncation=True,
        need_len=True,
        returns="tuple",
):
    """
    Calculate corresponding shape of given requirement.

    The `requirement` can be:

    - a positive integer: All dimensions of the tensor will be padded to this integer, extra data will be
      truncated if `truncation` is True, or it'll be padded to the maximum of the integer and the original
      dimension.
    - "max" or "min": All dimensions of the tensor will be padded to the maximum or minimum of the original
      dimensions.
    - -1: All dimensions of the tensor will be left unchanged, but if their dimensions are not the same,
      an error will be raised.
    - a list of requirement, where the `i`-th element of the list will be the requirement of the `i`-th
      dimension of the tensor. The length of the list don't need to be the same as the depth of the tensor,
      if the depth of inputs is greater than it, the `default_requirement` will be used. If it's not allowed,
      make `default_requirement` to "raise", and an error will be raised if the requirement is not specified.

    Args:
        inputs: Item, where `item` can be a tensor, number or list of items. Their depth can be different.
        requirement: Requirements of the shape, where `requirement` was defined previously. Default: `None`
            (use `default_requirement`).
        default_requirement: Default requirement of the shape for unspecified dimensions, or "raise" to disable
            default requirement.
        truncation: Whether to truncate the data if the size is greater than the requirement. Default: `True`.
        need_len: Whether to calculate the length of the tensor, if true, tensor size, length of list or a
            nested list with above elements of the tensor will be returned. Default: `True`.
        returns: Whether to return a dictionary with keys "shape" and "len", or a tuple of the shape and length.
            Default: "tuple".

    Returns:
        If `returns` is "tuple", return a tuple of the shape and length of the tensor, where the shape is a list
        of integers, and the length is a list of integers or a nested list of integers. If `returns` is "dict",
        return a dictionary with keys "shape" and "len", where the value of "shape" is the shape of the tensor,
        and the value of "len" is the length of the tensor.

    Examples::

        >>> shape_values, shape_lengths = torchsparks.utils.shapes.shape([
        ...     torch.randn(3, 4, 5),
        ...     [
        ...         torch.randn(3, 4, 5),
        ...         torch.randn(9, 4)
        ...     ]
        ... ], [6])
        >>> print(shape_values)
        [6, 3, 9, 5, 5]
        >>> print(shape_lengths)
        [torch.Size([3, 4, 5]), [torch.Size([3, 4, 5]), torch.Size([9, 4])]]
    """

    if requirement is None:
        num_requirements = 0
    elif isinstance(requirement, (int, str)):
        default_requirement = requirement
        num_requirements = 0
    else:
        num_requirements = len(requirement)

    # add a dimension for convenience
    inputs = [inputs]
    if requirement is not None and isinstance(requirement, list):
        requirement.insert(0, 1)
        num_requirements += 1

    def shape_recursively(depth, depth_inputs):
        if isinstance(depth_inputs, torch.Tensor):
            return depth_inputs.size(), depth_inputs.size() if need_len else None
        elif torchsparks.utils.pythons.iterable(depth_inputs):
            element_shapes_and_lens = [shape_recursively(depth + 1, x) for x in depth_inputs]
            num_element_shapes_and_lens = len(element_shapes_and_lens)

            # empty shape
            if num_element_shapes_and_lens == 0:
                num_dimensions = 0
            else:
                num_dimensions = max(len(x) for x, _ in element_shapes_and_lens)

            current_shape_results = [len(element_shapes_and_lens)]

            for i in range(num_dimensions):
                # get the shape value of corresponding requirement,
                # if the requirement is not specified, use the default value
                requirement_value = None

                # "+ 1" for the list itself
                requirement_index = i + depth + 1
                if requirement_index < num_requirements:
                    shape_requirement = requirement[requirement_index]
                elif default_requirement == "raise":
                    raise ValueError(f"Shape requirement not specified for dimension {requirement_index}")
                else:
                    shape_requirement = default_requirement
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
                    elif isinstance(shape_requirement, int) and shape_requirement > 0:
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
        else:
            raise ValueError(f"Invalid value for `depth_inputs`: {depth_inputs}")

    # return results
    shape_results, len_results = shape_recursively(0, inputs)

    # remove the first dimension
    shape_results = shape_results[1:]
    len_results = len_results[0] if need_len and len_results is not None else None

    if returns == "dict":
        return {
            "shape": shape_results,
            "len": len_results
        }
    elif returns == "tuple":
        return shape_results, len_results
    else:
        raise ValueError(f"Invalid value for `returns`: {returns}")


def pad(
        inputs,
        requirement=None,
        default_requirement="max",
        unsqueeze_location="after",
        padding_mode="constant",
        padding_location="after",
        padding_value=0,
        data_device=None,
        data_dtype=None,
        truncation=True,
        truncation_location="after",
        need_len=True,
        len_dimensions="different",
        len_device=None,
        len_dtype=torch.int32,
        returns="tuple",
):
    """
    Pad tensors to given shape.

    Args:
        inputs: See docs of the parameter `inputs` in function `shape`.
        requirement: See docs of the parameter `requirement` in function `shape`.
        default_requirement: See docs of the parameter `default_requirement` in function `shape`.
        unsqueeze_location: Where to unsqueeze the tensor to the required dimensions if the number of tensor's
            dimensions is less than the corresponding dimensions. Default: "after".
        padding_mode: Padding mode of the tensor, see docs of the parameter `mode` in function
            `torch.nn.functional.pad() <https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html>`_.
            Default: "constant".
        padding_location: Where to pad the tensor, it will be used to build the parameter `pad` of function
            `torch.nn.functional.pad()`. See docs of the parameter `pad` in function
            `torch.nn.functional.pad() <https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html>`_.
            Default: "after".
        padding_value: Padding value of the tensor, used when the `padding_mode` is "constant". See docs of the
            parameter `value` in function `torch.nn.functional.pad()
            <https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html>`_. Default: 0.
        data_device: Device of the data tensor. Default: `None` (use the device of the input tensors, or default
            device if all input elements are numbers).
        data_dtype: Data type of the data tensor. Default: `None` (use the data type of the input tensor, or
            default dtype `torch.float32` if all input elements are numbers).
        truncation: Whether to truncate the data if the size is greater than the requirement. Default: `True`.
        truncation_location: Where to truncate the tensor. Default: "after".
        need_len: Whether to calculate the length of the tensor, if true, tensor size, length of list or a
            nested list with above elements of the tensor will be returned. Default: `True`.
        len_dimensions: The depth of the length tensor, or "different" to detect the depth that the deepest
            difference in dimensions appears automatically. If there are some dimensions that deeper than given
            deep are not the same, an error will be raised. Default: "different".
        len_device: Device of the length tensor. Default: `None` (use the device of the data tensor).
        len_dtype: Data type of the length tensor. Default: `torch.int32`.
        returns: Whether to return a dictionary with keys "data" and "len", or a tuple of the data and length.
            Default: "tuple".

    Examples::

        >>> padded_data, padded_len = torchsparks.utils.shapes.pad([
        ...     torch.ones(2, 2, 100) * 1,
        ...     [
        ...         torch.ones(2, 100) * 2,
        ...         torch.ones(2, 100) * 3
        ...     ],
        ...     [],
        ...     [
        ...         torch.ones(3, 100) * 4,
        ...         torch.ones(2, 100) * 5,
        ...         torch.ones(2, 100) * 6
        ...     ]
        ... ], [6])
        >>> print(padded_data.shape)
        torch.Size([6, 3, 3, 100])
        >>> # our function know that remaining dimensions are the same, so ignore deeper dimensions.
        >>> print(padded_len.shape)
        torch.Size([6, 3])
        >>> print(padded_len)
        tensor([[2, 2, 0],
                [2, 2, 0],
                [0, 0, 0],
                [3, 2, 2],
                [0, 0, 0],
                [0, 0, 0]], dtype=torch.int32)
        >>> # tensor `padded_len == 0` can be used to mask the padded data
        >>> print(padded_len == 0)
        tensor([[False, False,  True],
                [False, False,  True],
                [ True,  True,  True],
                [False, False, False],
                [ True,  True,  True],
                [ True,  True,  True]])
    """

    if data_device is not None and need_len and len_device is None:
        len_device = data_device

    # calculate shape
    shape_values, len_values = shape(
        inputs, requirement, default_requirement, truncation, returns="tuple", need_len=need_len
    )

    # add a dimension for convenience
    inputs = [inputs]
    shape_values.insert(0, 1)
    len_values = [len_values] if need_len else None

    num_dimensions = len(shape_values)

    len_dimension_depth = len_dimensions
    if need_len:
        if len_dimensions == "different":
            def len_dimension_recursively(depth, depth_data):
                if isinstance(depth_data, (int, float)):
                    return num_dimensions
                elif isinstance(depth_data, torch.Tensor):
                    for i in reversed(range(num_dimensions - depth)):
                        if depth_data.ndim <= i or depth_data.size(i) != shape_values[i + depth]:
                            return i + depth
                    return depth - 1
                elif torchsparks.utils.pythons.iterable(depth_data):
                    if len(depth_data) > 0:
                        len_dimension_result = max(len_dimension_recursively(depth + 1, x) for x in depth_data)
                        return depth - 1 if len_dimension_result == depth else len_dimension_result
                    else:
                        return depth - 1
                else:
                    raise ValueError(f"Invalid value for `depth_data`: {depth_data}")

            len_dimension_depth = len_dimension_recursively(0, inputs)
        elif not isinstance(len_dimensions, int) or len_dimensions < 0:
            raise ValueError(f"Invalid value for `len_dimensions`: {len_dimensions}")

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
            num_len_shape = len_dimension_depth - depth
            if need_len and depth <= len_dimension_depth:
                if len(depth_lens) < num_len_shape:
                    # get the shape of len tensor, its dimension is always depth_data.ndim - 1
                    if unsqueeze_location == "before":
                        depth_len_shape = [1] * (num_len_shape - len(depth_lens)) + list(depth_lens)
                    elif unsqueeze_location == "after":
                        depth_len_shape = list(depth_lens) + [1] * (num_len_shape - len(depth_lens))
                    else:
                        raise ValueError(f"Invalid value for `unsqueeze_location`: {unsqueeze_location}")
                else:
                    depth_len_shape = list(depth_lens)[:num_len_shape]

                # fill values and build tensor
                depth_lens = min(shape_values[len_dimension_depth], depth_data.size(num_len_shape))
                for i in reversed(depth_len_shape):
                    depth_lens = [depth_lens] * i
                depth_lens = torch.tensor(depth_lens, device=len_device, dtype=len_dtype)
            elif depth_data.size(0) != shape_values[depth]:
                raise ValueError(f"Dimension 0 of tensor has size {depth_data.size(0)}, "
                                 f"which is not equal to the shape requirement {shape_values[depth]}, "
                                 f"can not ignore the difference in their size! "
                                 f"Try to increase `len_dimensions`, "
                                 f"set it to `different` or set `need_len` to `False`! ")

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

            if need_len and depth <= len_dimension_depth:
                if depth < len_dimension_depth:
                    depth_lens = torch.nn.functional.pad(
                        depth_lens, argument_pad[-2 * num_len_shape:], value=0, mode="constant"
                    )
                    return depth_data[indexes_after_pad], depth_lens[indexes_after_pad[:num_len_shape]]
                else:
                    return depth_data[indexes_after_pad] if len(indexes_after_pad) else depth_data, depth_lens
            else:
                return depth_data[indexes_after_pad], None
        elif torchsparks.utils.pythons.iterable(depth_data):
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

            # create or pad len
            current_len_results = None
            if need_len and depth <= len_dimension_depth:
                if depth < len_dimension_depth:
                    pad_results = [pad_recursively(depth + 1, x, y) for x, y in zip(depth_data, depth_lens)]

                    if len(pad_results) > 0:
                        current_len_results = torch.stack([y for _, y in pad_results])
                    else:
                        # empty_shape
                        current_len_results = torch.zeros(
                            [0] + shape_values[depth + 1:len_dimension_depth], dtype=len_dtype, device=len_device
                        )
                else:
                    pad_results = [pad_recursively(depth + 1, x, None) for x in depth_data]
                    if depth == len_dimension_depth:
                        current_len_results = torch.tensor(
                            len(pad_results), dtype=len_dtype, device=len_device
                        ) if need_len else None
            else:
                pad_results = [pad_recursively(depth + 1, x, None) for x in depth_data]

            # pad data
            if len(pad_results) > 0:
                current_data_results = torch.stack([x for x, _ in pad_results])
            else:
                # empty_shape
                current_data_results = torch.zeros([0] + shape_values[depth + 1:], dtype=data_dtype, device=data_device)

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
                if need_len and depth < len_dimension_depth:
                    current_len_results = torch.nn.functional.pad(
                        current_len_results, argument_pad[-2 * len(current_len_results.size()):],
                        value=0, mode="constant"
                    )
                    # current_len_results = torch.nn.functional.pad(
                    #     current_len_results, argument_pad[2:], value=0, mode="constant"
                    # ) if need_len else None

            return current_data_results, current_len_results
        else:
            raise ValueError(f"Invalid value for `depth_data`: {depth_data}")

    # return results
    data_results, len_results = pad_recursively(0, inputs, len_values)

    # remove the first dimension
    data_results = data_results[0]
    len_results = len_results[0] if need_len else None

    if returns == "dict":
        return {
            "data": data_results,
            "len": len_results
        }
    elif returns == "tuple":
        return data_results, len_results
    else:
        raise ValueError(f"Invalid value for `returns`: {returns}")
