# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base transform implementations for composing environment-specific transforms."""

from .interfaces import Transform
from .types import Observation


class CompositeTransform(Transform):
    """Combines multiple transforms into a single transform."""

    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, observation: Observation) -> Observation:
        for transform in self.transforms:
            observation = transform(observation)
        return observation


class NullTransform(Transform):
    """Default transform that passes through unchanged."""

    def __call__(self, observation: Observation) -> Observation:
        return observation