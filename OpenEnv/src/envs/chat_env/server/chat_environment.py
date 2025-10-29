# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Chat Environment Implementation.

A chat-based environment for LLMs, designed as a blank canvas for conversation and RL.
"""

import torch

from core.env_server.interfaces import Environment, Message, ModelTokenizer, Transform

from ..models import ChatAction, ChatObservation, ChatState


class ChatEnvironment(Environment):
    """A chat-based environment for LLMs, designed as a blank canvas for conversation and RL.

    This environment is designed to work with language models. It provides the fundamental structure
    for managing conversation state but is intentionally minimal to allow maximum flexibility.

    The environment owns the tokenizer and is responsible for managing both message history and tokens.
    Actions contain only tokens that interface directly with models.

    Args:
        tokenizer: A tokenizer that will be used to tokenize the conversation
        system_prompt: An optional system prompt string to use during reset calls (optional)
        system_role: The role of the system (at reset time). Defaults to "system"
        transform: Optional transform to apply to observations
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        system_prompt: str | None = None,
        system_role: str = "system",
        transform: Transform | None = None,
    ):
        super().__init__(transform=transform)

        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer must have 'apply_chat_template' method")
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.system_role = system_role

        self._state = ChatState()

        if system_prompt:
            system_message: Message = {"role": system_role, "content": system_prompt}
            self._state.history_messages.append(system_message)
            # Tokenize the system message
            system_tokens = self.tokenizer.apply_chat_template(
                conversation=[system_message], tokenize=True, return_tensors="pt"  # type: ignore
            )
            self._state.history_tokens.append(system_tokens)

    def reset(self) -> ChatObservation:
        """Reset the environment to initial state.

        Returns:
            ChatObservation: Initial observation with system prompt (if any)
        """
        self._state.history_messages = []
        self._state.history_tokens = []
        if self.system_prompt:
            system_message: Message = {
                "role": self.system_role,
                "content": self.system_prompt,
            }
            self._state.history_messages = [system_message]
            # Tokenize the system message
            system_tokens = self.tokenizer.apply_chat_template(
                conversation=[system_message], tokenize=True, return_tensors="pt"  # type: ignore
            )
            self._state.history_tokens = [system_tokens]

        return self._create_observation()

    def step(self, action: ChatAction) -> ChatObservation:  # type: ignore[override]
        """Take a step in the environment by adding tokens to the chat history.

        Args:
            action: A ChatAction object containing tokens.

        Returns:
            ChatObservation: The updated observation with the new tokens added.
        """
        # Store the tokens directly from the action
        self._state.history_tokens.append(action.tokens)

        # Decode tokens to text and add as a message to history
        decoded_text = self.tokenizer.decode(
            action.tokens.squeeze(), skip_special_tokens=True
        )
        assistant_message: Message = {"role": "assistant", "content": decoded_text}
        self._state.history_messages.append(assistant_message)

        return self._create_observation()

    def _create_observation(self) -> ChatObservation:
        """Create a ChatObservation from the current state.

        Returns both the message history and the tokens flattened as a single tensor
        ready to be used by models.

        Returns:
            ChatObservation: Observation with messages and flattened tokens
        """
        if self._state.history_tokens:
            # Flatten all tokens into a single 1D tensor
            flattened_tokens = torch.cat(
                (t.flatten() for t in self._state.history_tokens), dim=0
            )
        else:
            flattened_tokens = torch.tensor([])

        observation = ChatObservation(
            messages=self._state.history_messages.copy(),  # Copy to prevent external mutation
            tokens=flattened_tokens,
        )

        transformed = self._apply_transform(observation)
        if isinstance(transformed, ChatObservation):
            return transformed
        else:
            # If transform returns base Observation, convert back to ChatObservation
            return ChatObservation(
                messages=getattr(transformed, "messages", []),
                tokens=getattr(transformed, "tokens", torch.tensor([])),
                done=transformed.done,
                reward=transformed.reward,
            )

    @property
    def state(self) -> ChatState:
        """Get the current state of the environment.

        Returns:
            ChatState: The current state.
        """
        return self._state

    def message_to_action(self, message: Message) -> ChatAction:
        """Convert a message dictionary to a ChatAction with tokens.

        Args:
            message: Dictionary with 'role' and 'content' keys

        Returns:
            ChatAction: A new ChatAction instance with tokenized content

        Raises:
            ValueError: If required keys are missing
        """
        if "role" not in message:
            raise ValueError("Message must contain a 'role' key")
        if "content" not in message:
            raise ValueError("Message must contain a 'content' key")
        if message["content"] is None:
            raise ValueError("Message content cannot be None")

        # Tokenize the single message
        tokens = self.tokenizer.apply_chat_template(
            conversation=[message], tokenize=True, return_tensors="pt"  # type: ignore
        )

        return ChatAction(tokens=tokens)
