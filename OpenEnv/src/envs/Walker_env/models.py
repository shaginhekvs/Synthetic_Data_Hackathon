from typing import List, Dict, Any
from pydantic import BaseModel, Field

class Action(BaseModel):
    """
    Continuous action space: 4 motor speed values in the range [-1, 1].
    Controls the 2 hips and 2 knees of the walker.
    """
    action: List[float] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="Motor speeds for 4 joints (hips and knees). Range: [-1, 1]."
    )

class Observation(BaseModel):
    """
    24-dimensional observation vector representing the robot’s state:
    - hull angle and angular velocity
    - horizontal & vertical speed
    - joint angles and joint angular speeds
    - ground contact indicators
    - 10 lidar rangefinder measurements
    """
    values: List[float] = Field(
        ...,
        min_items=24,
        max_items=24,
        description="24-D continuous state vector as defined by the environment."
    )

class State(BaseModel):
    """
    Unified representation of a single timestep (from step() or reset()).
    """
    observation: Observation
    reward: float = Field(0.0, description="Reward for the current timestep.")
    done: bool = Field(False, description="Whether the episode has terminated.")
    info: Dict[str, Any] = Field(default_factory=dict, description="Extra environment information.")