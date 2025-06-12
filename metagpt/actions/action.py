#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : action.py
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from metagpt.actions.action_node import ActionNode
from metagpt.configs.models_config import ModelsConfig
from metagpt.context_mixin import ContextMixin
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.schema import (
    CodePlanAndChangeContext,
    CodeSummarizeContext,
    CodingContext,
    RunCodeContext,
    SerializationMixin,
    TestingContext,
)


class Action(SerializationMixin, ContextMixin, BaseModel):
    """Base class for all actions within the MetaGPT framework.

    An action represents a single step or operation performed by an agent.
    It can involve interacting with an LLM, processing data, or calling external tools.
    Actions are designed to be modular and reusable.

    Attributes:
        name: The name of the action. Defaults to the class name if not provided.
        i_context: The input context for the action. This can be a dictionary, a specific
                   context model (e.g., CodingContext), a string, or None. It's used
                   by ActionNode for templating.
        prefix: A system message prefix that is prepended to LLM prompts. This can be
                used to set the role or provide general instructions to the LLM.
        desc: A description of the action, primarily used by the skill manager.
        node: An optional ActionNode associated with this action. ActionNode allows
              for more structured, template-based interactions with LLMs.
        llm_name_or_type: Specifies the LLM model or API type to be used for this action,
                          referencing keys in the `models` section of `config2.yaml`.
                          If None, the default LLM from `config2.yaml` is used.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = ""
    i_context: Union[
        dict, CodingContext, CodeSummarizeContext, TestingContext, RunCodeContext, CodePlanAndChangeContext, str, None
    ] = ""
    prefix: str = ""  # System prompt prefix for LLM interaction.
    desc: str = ""  # Description for the skill manager.
    node: ActionNode = Field(default=None, exclude=True) # Optional ActionNode for structured LLM interaction.
    # The model name or API type of LLM of the `models` in the `config2.yaml`;
    #   Using `None` to use the `llm` configuration in the `config2.yaml`.
    llm_name_or_type: Optional[str] = None # Specifies a particular LLM configuration.

    @model_validator(mode="after")
    @classmethod
    def _update_private_llm(cls, data: Any) -> Any:
        """
        Pydantic validator that runs after model initialization.
        If `llm_name_or_type` is set, it creates a specific LLM instance for this action,
        overriding the default LLM. The cost manager from the default LLM is retained.
        """
        config = ModelsConfig.default().get(data.llm_name_or_type)
        if config:
            # Create a new LLM instance based on the specified configuration.
            llm = create_llm_instance(config)
            # Preserve the cost manager from the potentially globally/role-assigned LLM.
            if hasattr(data, "llm") and hasattr(data.llm, "cost_manager"):
                llm.cost_manager = data.llm.cost_manager
            data.llm = llm
        return data

    @property
    def prompt_schema(self) -> Optional[str]:
        """
        Returns the prompt schema defined in the action's configuration.
        The prompt schema dictates the format (e.g., json, markdown) for LLM interactions.
        """
        return self.config.prompt_schema

    @property
    def project_name(self) -> Optional[str]:
        """
        Returns the project name from the action's configuration.
        Useful for actions that operate within a specific project context.
        """
        return self.config.project_name

    @project_name.setter
    def project_name(self, value: str):
        """
        Sets the project name in the action's configuration.
        """
        self.config.project_name = value

    @property
    def project_path(self) -> Optional[str]:
        """
        Returns the project path from the action's configuration.
        This path typically points to the root directory of the current project.
        """
        return self.config.project_path

    @model_validator(mode="before")
    @classmethod
    def set_name_if_empty(cls, values: dict) -> dict:
        """
        Pydantic validator that runs before model initialization.
        If the 'name' field is not provided or is empty, it defaults to the class name.
        """
        if "name" not in values or not values["name"]:
            values["name"] = cls.__name__
        return values

    @model_validator(mode="before")
    @classmethod
    def _init_with_instruction(cls, values: dict) -> dict:
        """
        Pydantic validator that runs before model initialization.
        If an 'instruction' field is present in the input values, it initializes
        an ActionNode with this instruction. This allows for quick Action setup
        using a simple instruction string.
        """
        if "instruction" in values:
            name = values.get("name", cls.__name__) # Use class name if name not set yet
            instruction = values.pop("instruction")
            # Initialize an ActionNode with the provided instruction.
            # The schema is set to "raw" by default for simple instruction-based actions.
            values["node"] = ActionNode(key=name, expected_type=str, instruction=instruction, example="", schema="raw")
        return values

    def set_prefix(self, prefix: str) -> "Action":
        """
        Sets the system prompt prefix for the action and its associated LLM and ActionNode.

        This prefix is typically used to provide high-level instructions or context to the LLM
        before the main prompt.

        Args:
            prefix: The string to be used as the system prompt prefix.

        Returns:
            The Action instance with the updated prefix.
        """
        self.prefix = prefix
        if self.llm: # Ensure LLM is initialized
            self.llm.system_prompt = prefix
        if self.node:
            self.node.llm = self.llm # Propagate LLM (and its new prefix) to the node
        return self

    def __str__(self) -> str:
        """
        Returns the class name of the action.
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        """
        Returns the string representation of the action, same as __str__.
        """
        return self.__str__()

    async def _aask(self, prompt: str, system_msgs: Optional[list[str]] = None) -> str:
        """
        Performs an asynchronous call to the LLM with the given prompt.

        This is a convenience method that uses the action's configured LLM.
        The action's `prefix` (system_prompt) is automatically included by the LLM.

        Args:
            prompt: The main prompt to send to the LLM.
            system_msgs: An optional list of system messages to override the default system prompt.

        Returns:
            The text response from the LLM.
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized for this action. Call set_llm() or ensure context is set.")
        return await self.llm.aask(prompt, system_msgs)

    async def _run_action_node(self, *args, **kwargs) -> Any:
        """
        Executes the associated ActionNode with the provided arguments.

        It constructs a context from the history messages (passed as the first argument)
        and then calls the `fill` method of the ActionNode to get the result.

        Args:
            *args: Variable length argument list. The first argument is expected to be
                   a list of messages (history).
            **kwargs: Arbitrary keyword arguments (not directly used by this base method but available for overrides).

        Returns:
            The result from the ActionNode's `fill` method.

        Raises:
            RuntimeError: If the action does not have an associated ActionNode or if LLM is not set.
        """
        if not self.node:
            raise RuntimeError("ActionNode not initialized for this action.")
        if not self.llm:
            raise RuntimeError("LLM not initialized for this action. Cannot run ActionNode.")

        # The first argument is typically a list of history messages.
        msgs = args[0] if args else []
        context_str = "## History Messages\n"
        context_str += "\n".join([f"{idx}: {i}" for idx, i in enumerate(reversed(msgs))])
        return await self.node.fill(context=context_str, llm=self.llm, i_context=self.i_context)

    async def run(self, *args, **kwargs) -> Any:
        """
        Executes the action.

        If an ActionNode (`self.node`) is defined, this method will execute the node
        using `_run_action_node`. Subclasses are expected to override this method
        to implement their specific logic if they don't use an ActionNode or require
        more complex execution steps.

        Args:
            *args: Variable length argument list, typically passed to `_run_action_node`
                   or used by subclass implementations.
            **kwargs: Arbitrary keyword arguments, typically passed to `_run_action_node`
                      or used by subclass implementations.

        Returns:
            The result of the action's execution.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass and no
                                 ActionNode is defined.
        """
        if self.node:
            return await self._run_action_node(*args, **kwargs)
        raise NotImplementedError("The run method should be implemented in a subclass if no ActionNode is used.")

    def override_context(self):
        """
        Ensures that `private_context` and `context` refer to the same Context object.

        If `private_context` is not already set, it is assigned the value of `context`.
        This is useful for actions that need to operate on a shared context that might
        be updated by other components.
        """
        if not self.private_context:
            self.private_context = self.context
