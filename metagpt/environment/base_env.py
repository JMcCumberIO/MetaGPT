#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : base env of executing environment
"""
This module defines the base classes for environments in the MetaGPT framework.
It includes `ExtEnv` for integrating with external simulation/game environments
and a general-purpose `Environment` for multi-agent collaboration through message passing.
"""
import asyncio
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Set, Union

from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

from metagpt.base import BaseEnvironment, BaseRole
from metagpt.base.base_env_space import BaseEnvAction, BaseEnvObsParams
from metagpt.context import Context
from metagpt.environment.api.env_api import (
    EnvAPIAbstract,
    ReadAPIRegistry,
    WriteAPIRegistry,
)
from metagpt.logs import logger
from metagpt.memory import Memory
from metagpt.schema import Message
from metagpt.utils.common import get_function_schema, is_coroutine_func, is_send_to
from metagpt.utils.git_repository import GitRepository


class EnvType(Enum):
    """Enumeration of different environment types supported or conceptualized."""
    ANDROID = "Android"
    GYM = "Gym"  # For OpenAI Gym-like environments
    WEREWOLF = "Werewolf" # For Werewolf game simulation
    MINECRAFT = "Minecraft" # For Minecraft game integration
    STANFORDTOWN = "StanfordTown" # For Stanford Town simulation


# Global registry for writable APIs in external environments.
env_write_api_registry = WriteAPIRegistry()
# Global registry for readable APIs in external environments.
env_read_api_registry = ReadAPIRegistry()


def mark_as_readable(func):
    """Decorator to mark a function as a readable API for an ExtEnv.

    Readable APIs are functions that observe or get information from the external environment.
    The function's schema (signature) is registered in `env_read_api_registry`.

    Args:
        func: The function to be marked as readable.

    Returns:
        The original function, now registered.
    """
    env_read_api_registry[func.__name__] = get_function_schema(func)
    return func


def mark_as_writeable(func):
    """Decorator to mark a function as a writable API for an ExtEnv.

    Writable APIs are functions that perform an action or change the state of the
    external environment. The function's schema is registered in `env_write_api_registry`.

    Args:
        func: The function to be marked as writable.

    Returns:
        The original function, now registered.
    """
    env_write_api_registry[func.__name__] = get_function_schema(func)
    return func


class ExtEnv(BaseEnvironment, BaseModel):
    """Abstract base class for integrating with actual external (game, simulation) environments.

    This class provides a common interface for MetaGPT agents to interact with various
    external environments. It defines methods for observing the environment, taking actions,
    and resetting the state, similar to OpenAI Gym. It also introduces a mechanism for
    defining and discovering specific readable and writable APIs for more fine-grained
    interactions.

    Subclasses should implement the abstract methods: `reset`, `observe`, and `step`.

    Attributes:
        action_space: The space of possible actions, conforming to Gymnasium's Space interface.
                      Defaults to an empty `spaces.Space`.
        observation_space: The space of possible observations, conforming to Gymnasium's Space interface.
                           Defaults to an empty `spaces.Space`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action_space: spaces.Space[ActType] = Field(default_factory=spaces.Space, exclude=True)
    observation_space: spaces.Space[ObsType] = Field(default_factory=spaces.Space, exclude=True)

    def _check_api_exist(self, rw_api: Optional[Any] = None):
        """Internal helper to check if a retrieved API function actually exists (is not None).

        Args:
            rw_api: The API function retrieved from a registry.

        Raises:
            ValueError: If `rw_api` is None, indicating the API was not found or is not callable.
        """
        if not rw_api: # Check if it's None or otherwise falsy
            # The original error message was f"{rw_api} not exists", which would print "None not exists".
            # A more general message might be better if rw_api could be other non-callable types.
            raise ValueError("The specified API function does not exist or is not callable.")

    def get_all_available_apis(self, mode: str = "read") -> list[dict[str, Any]]:
        """Retrieves definitions of all available readable or writable APIs for this environment.

        Args:
            mode: Specifies whether to retrieve "read" or "write" APIs. Defaults to "read".

        Returns:
            A list of dictionaries, where each dictionary contains the schema/definition
            of an available API.

        Raises:
            AssertionError: If `mode` is not "read" or "write".
        """
        assert mode in ["read", "write"], "Mode must be 'read' or 'write'"
        if mode == "read":
            return env_read_api_registry.get_apis()
        else: # mode == "write"
            return env_write_api_registry.get_apis()

    async def read_from_api(self, env_action: Union[str, EnvAPIAbstract]) -> Any:
        """Reads data or makes an observation from the environment using a specified readable API.

        Args:
            env_action: Either the name of the readable API (string) or an `EnvAPIAbstract`
                        instance specifying the API name and any arguments.

        Returns:
            The result of calling the specified readable API. The type of the result
            depends on the API's implementation.

        Raises:
            ValueError: If the specified API does not exist or is not callable.
        """
        res: Any
        if isinstance(env_action, str):
            api_spec = env_read_api_registry.get(api_name=env_action)
            if not api_spec:
                raise ValueError(f"Readable API '{env_action}' not found in registry.")
            env_read_api = api_spec["func"]
            self._check_api_exist(env_read_api)
            if is_coroutine_func(env_read_api):
                res = await env_read_api(self)
            else:
                res = env_read_api(self)
        elif isinstance(env_action, EnvAPIAbstract):
            api_spec = env_read_api_registry.get(api_name=env_action.api_name)
            if not api_spec:
                raise ValueError(f"Readable API '{env_action.api_name}' not found in registry.")
            env_read_api = api_spec["func"]
            self._check_api_exist(env_read_api)
            if is_coroutine_func(env_read_api):
                res = await env_read_api(self, *env_action.args, **env_action.kwargs)
            else:
                res = env_read_api(self, *env_action.args, **env_action.kwargs)
        else:
            raise TypeError(f"Unsupported env_action type: {type(env_action)}")
        return res

    async def write_thru_api(self, env_action: Union[str, Message, EnvAPIAbstract, list[EnvAPIAbstract]]) -> Any:
        """Executes an action or sends data to the environment using a specified writable API.

        If `env_action` is a `Message`, it's published to the environment's roles.
        If `env_action` is an `EnvAPIAbstract` instance, the corresponding writable API
        is called with the specified arguments.

        Args:
            env_action: The action to perform. Can be a `Message` to publish,
                        an `EnvAPIAbstract` instance for a specific API call, or a list
                        of `EnvAPIAbstract` instances (though list handling is not explicitly shown).
                        A string representing an API name is not supported by this method directly
                        for write operations, unlike `read_from_api`.

        Returns:
            The result of the writable API call, if any. Returns None if a Message was published.

        Raises:
            ValueError: If a specified API in `EnvAPIAbstract` does not exist or is not callable.
            TypeError: If `env_action` is an unsupported type.
        """
        res: Any = None
        if isinstance(env_action, Message):
            self.publish_message(env_action) # Assuming ExtEnv might have roles or a message queue
        elif isinstance(env_action, EnvAPIAbstract):
            api_spec = env_write_api_registry.get(env_action.api_name)
            if not api_spec:
                raise ValueError(f"Writable API '{env_action.api_name}' not found in registry.")
            env_write_api = api_spec["func"]
            self._check_api_exist(env_write_api)
            if is_coroutine_func(env_write_api):
                res = await env_write_api(self, *env_action.args, **env_action.kwargs)
            else:
                res = env_write_api(self, *env_action.args, **env_action.kwargs)
        # Note: The type hint suggests `list[EnvAPIAbstract]` is possible, but it's not handled.
        # Also, `str` as an API name is not handled here for write operations.
        elif not isinstance(env_action, (Message, EnvAPIAbstract)): # Check if it's an unhandled type
             raise TypeError(f"Unsupported env_action type for write_thru_api: {type(env_action)}")
        return res

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment to its initial state and returns an initial observation.

        This method must be implemented by concrete environment subclasses.

        Args:
            seed: An optional seed for the environment's random number generator.
            options: Optional dictionary of environment-specific options.

        Returns:
            A tuple containing the initial observation and an info dictionary.
            The types should conform to `ObsType` and `dict[str, Any]`.
        """

    @abstractmethod
    def observe(self, obs_params: Optional[BaseEnvObsParams] = None) -> ObsType:
        """Returns an observation from the current state of the environment.

        This method must be implemented by concrete environment subclasses.
        It can be used to get full or partial observations depending on `obs_params`.

        Args:
            obs_params: Optional parameters to specify the type or scope of observation.

        Returns:
            The current observation from the environment, conforming to `ObsType`.
        """

    @abstractmethod
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Takes an action in the environment and returns the new state information.

        This method must be implemented by concrete environment subclasses. It follows
        the standard OpenAI Gym `step` signature.

        Args:
            action: The action to be performed in the environment, conforming to `ActType`.

        Returns:
            A tuple containing:
            - observation (ObsType): The observation of the new state.
            - reward (float): The reward obtained from the action.
            - terminated (bool): True if the episode has ended due to a terminal state.
            - truncated (bool): True if the episode has ended due to a time limit or other truncation.
            - info (dict[str, Any]): Additional information about the step.
        """


class Environment(ExtEnv):
    """A generic environment for multi-agent collaboration through message passing.

    The Environment class hosts a collection of roles (agents) and facilitates their
    interaction by managing a message queue and distributing messages to subscribed roles.
    It keeps a history of all messages exchanged. This environment is not tied to a
    specific external simulation but rather serves as a communication hub for agents
    performing tasks like software development.

    While it inherits from `ExtEnv`, its `reset`, `observe`, and `step` methods are
    currently placeholders (`pass`) as its primary mode of operation is through message
    exchange and role execution cycles, not traditional step-based simulation.

    Attributes:
        desc: A description of the environment's purpose or the project it represents.
        roles: A dictionary mapping role names to BaseRole instances active in the environment.
        member_addrs: A dictionary mapping BaseRole instances to a set of addresses (tags/topics)
                      they are subscribed to for receiving messages. Excluded from serialization.
        history: A Memory instance storing all messages published in the environment, for debugging
                 and logging purposes.
        context: A Context object providing shared configuration and resources (like cost manager,
                 project path) to all roles within the environment. Excluded from serialization
                 by default but handled manually in Team serialization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    desc: str = Field(default="", description="A description of the environment or project.")
    roles: dict[str, SerializeAsAny[BaseRole]] = Field(default_factory=dict, validate_default=True)
    member_addrs: Dict[BaseRole, Set[str]] = Field(default_factory=dict, exclude=True, description="Role address subscriptions.") # Changed Set to Set[str]
    history: Memory = Field(default_factory=Memory, description="Message history for debugging.")
    context: Context = Field(default_factory=Context, exclude=True, description="Shared context for roles.")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Placeholder for environment reset. Not actively used in this message-passing environment."""
        # In a typical Gym-like env, this would reset the state and return an initial observation.
        # For this Environment, state is managed by roles and messages.
        # Return a dummy observation and info if an interface is strictly needed.
        logger.debug("Environment.reset() called, but it's a placeholder for this message-passing environment.")
        return {}, {} # type: ignore # Conforms to ObsType, dict

    def observe(self, obs_params: Optional[BaseEnvObsParams] = None) -> ObsType:
        """Placeholder for environment observation. Not actively used."""
        # Observation in this context would be the current state of messages or roles.
        # Roles observe messages through their own _observe method.
        logger.debug("Environment.observe() called, but it's a placeholder.")
        return {} # type: ignore # Conforms to ObsType

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Placeholder for environment step. Not actively used in typical agent collaboration."""
        # Actions are taken by roles via their run() method, triggered by Environment.run().
        # This step function is more for Gym-like interaction.
        logger.debug(f"Environment.step({action}) called, but it's a placeholder.")
        return {}, 0.0, False, False, {} # type: ignore # Conforms to ObsType, float, bool, bool, dict

    @model_validator(mode="after")
    def init_roles(self) -> "Environment":
        """Pydantic validator to initialize roles after model creation.

        Ensures that all roles provided during initialization (or already present in `self.roles`)
        are properly configured with this environment instance and its context.

        Returns:
            The Environment instance itself.
        """
        self.add_roles(list(self.roles.values())) # Use list to avoid issues if roles dict changes during iteration
        return self

    def add_role(self, role: BaseRole):
        """Adds a single role to the environment.

        The role is stored in the `roles` dictionary and is configured to operate
        within this environment (its `env` attribute is set, and context is shared).

        Args:
            role: The BaseRole instance to add.
        """
        if role.name in self.roles:
            logger.warning(f"Role with name {role.name} already exists. Overwriting.")
        self.roles[role.name] = role
        role.set_env(self)
        role.context = self.context

    def add_roles(self, roles: Iterable[BaseRole]):
        """Adds multiple roles to the environment.

        Each role in the iterable is added and configured. This method is typically
        called during environment initialization.

        Args:
            roles: An iterable of BaseRole instances to add.
        """
        for role in roles:
            if role.name in self.roles and self.roles[role.name] is not role:
                logger.warning(f"Role with name {role.name} already exists. Overwriting with new instance.")
            self.roles[role.name] = role
            # Setting env and context should be done after all roles are in self.roles
            # if role setup depends on other roles being present (e.g., for system messages).
            # However, current BaseRole.set_env doesn't seem to have such dependencies.

        for role in roles:  # Second loop to ensure all roles are present before full setup if needed.
            role.context = self.context
            role.set_env(self) # This also sets addresses via role.set_addresses in set_env

    def publish_message(self, message: Message, peekable: bool = True) -> bool:
        """Distributes a message to all subscribed roles and records it in history.

        The method iterates through all roles registered in `member_addrs`. If a role's
        subscribed addresses (tags) match the message's `send_to` field or other
        routing criteria (evaluated by `is_send_to`), the message is put into that
        role's private message buffer using `role.put_message()`.

        The `peekable` argument is not currently used in this method's logic but is
        part of the signature, potentially for future use or compatibility.

        Args:
            message: The Message object to be published.
            peekable: If True, indicates the message might be observable by roles
                      not directly addressed (not currently implemented here). Defaults to True.

        Returns:
            True if the message was successfully processed (i.e., added to history,
            even if no roles received it).
        """
        logger.debug(f"Publishing message: {message.dump()}")
        found_recipient = False
        # Message routing based on role subscriptions (addresses).
        for role, addrs in self.member_addrs.items():
            if is_send_to(message, addrs):
                role.put_message(message)
                found_recipient = True

        if not found_recipient:
            logger.warning(f"Message '{message.content[:50]}...' from {message.sent_from} to {message.send_to} has no recipients among roles: {list(self.roles.keys())}")

        self.history.add(message)  # Add to environment's global message history.
        return True

    async def run(self, k: int = 1):
        """Executes role actions for a specified number of rounds.

        In each round, it iterates through all roles in the environment. If a role
        is not idle (i.e., has pending actions or messages to process), its `run()`
        method is called asynchronously. All such `run()` calls are gathered and
        awaited, allowing roles to perform their actions concurrently within a round.

        Args:
            k: The number of rounds to execute. Defaults to 1.
        """
        for i in range(k):
            logger.debug(f"Environment run round {i+1}/{k}")
            futures = []
            for role in self.roles.values():
                if not role.is_idle:
                    futures.append(role.run())

            if futures:
                await asyncio.gather(*futures)
            else:
                logger.debug("All roles are idle, no futures to gather.")
            logger.debug(f"Round {i+1}/{k} completed. Environment idle status: {self.is_idle}")

    def get_roles(self) -> dict[str, BaseRole]:
        """Returns a dictionary of all roles currently in the environment.

        Returns:
            A dictionary mapping role names (str) to BaseRole instances.
        """
        return self.roles

    def get_role(self, name: str) -> Optional[BaseRole]:
        """Retrieves a specific role by its name.

        Args:
            name: The name of the role to retrieve.

        Returns:
            The BaseRole instance if found, otherwise None.
        """
        return self.roles.get(name)

    def role_names(self) -> list[str]:
        """Returns a list of names of all roles in the environment.

        Returns:
            A list of strings, where each string is the name of a role.
        """
        return [role.name for role in self.roles.values()]

    @property
    def is_idle(self) -> bool:
        """Checks if all roles in the environment are currently idle.

        A role is considered idle if it has no pending actions or messages to process.
        The environment is idle if all its roles are idle.

        Returns:
            True if all roles are idle, False otherwise.
        """
        for role in self.roles.values():
            if not role.is_idle:
                return False
        return True

    def get_addresses(self, obj: BaseRole) -> Set[str]:
        """Retrieves the set of addresses (tags) a specific role object is subscribed to.

        Args:
            obj: The BaseRole instance whose addresses are requested.

        Returns:
            A set of strings representing the addresses. Returns an empty set if the
            role is not found or has no registered addresses.
        """
        return self.member_addrs.get(obj, set())

    def set_addresses(self, obj: BaseRole, addresses: Set[str]):
        """Sets or updates the addresses (tags) a specific role object is subscribed to.

        This is used by roles to declare what types of messages they are interested in.

        Args:
            obj: The BaseRole instance whose addresses are to be set.
            addresses: A set of strings representing the addresses.
        """
        self.member_addrs[obj] = addresses

    def archive(self, auto_archive: bool = True):
        """Archives the project associated with this environment, typically using Git.

        If `auto_archive` is True and a `project_path` is defined in the environment's
        context, this method initializes a `GitRepository` at that path and calls its
        `archive()` method. This usually involves committing any changes and creating
        a timestamped archive or tag.

        Args:
            auto_archive: A boolean flag to enable/disable automatic archiving.
                          Defaults to True.
        """
        if auto_archive and self.context and self.context.kwargs and self.context.kwargs.get("project_path"):
            project_path_str = self.context.kwargs.get("project_path")
            if project_path_str:
                try:
                    git_repo = GitRepository(Path(project_path_str)) # Ensure it's a Path
                    git_repo.archive()
                    logger.info(f"Project archived at {project_path_str}")
                except Exception as e:
                    logger.error(f"Failed to archive project at {project_path_str}: {e}")
            else:
                logger.warning("Auto-archive is enabled but project_path is not a valid string in context.")
        elif auto_archive:
            logger.warning("Auto-archive is enabled but project_path is not available in context.")
