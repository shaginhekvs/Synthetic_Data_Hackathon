# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FinRL Environment for OpenEnv.

This package provides a wrapper around FinRL's StockTradingEnv that conforms
to the OpenEnv specification, enabling stock trading RL tasks through a
simple HTTP API.

Example:
    >>> from envs.finrl_env import FinRLEnv, FinRLAction
    >>>
    >>> # Connect to server
    >>> client = FinRLEnv(base_url="http://localhost:8000")
    >>>
    >>> # Reset environment
    >>> result = client.reset()
    >>> print(result.observation.portfolio_value)
    >>>
    >>> # Execute trading action
    >>> action = FinRLAction(actions=[0.5])  # Buy
    >>> result = client.step(action)
    >>> print(result.reward)
"""

from .client import FinRLEnv
from .models import FinRLAction, FinRLObservation

__all__ = ["FinRLEnv", "FinRLAction", "FinRLObservation"]
