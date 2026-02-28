#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 00:30
@Author  : alexanderwu
@File    : team.py
@Modified By: mashenquan, 2023/11/27. Add an archiving operation after completing the project, as specified in
        Section 2.2.3.3 of RFC 135.
"""

import warnings
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from metagpt.const import SERDESER_PATH
from metagpt.context import Context
from metagpt.environment import Environment
from metagpt.environment.mgx.mgx_env import MGXEnv
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.utils.common import (
    NoMoneyException,
    read_json_file,
    serialize_decorator,
    write_json_file,
)


class Team(BaseModel):
    """
    Represents a team of roles (agents) collaborating in an environment.

    The Team class orchestrates the interactions between different roles, manages the
    overall project investment, and tracks the project idea. It utilizes an
    Environment for message passing and role management.

    Attributes:
        env: The environment where roles interact. Can be a standard Environment or an MGXEnv.
        investment: The total budget allocated for the project.
        idea: The initial idea or requirement that drives the project.
        use_mgx: Flag to determine if MGXEnv (multi-agent graph execution environment) should be used.
                 Defaults to True. If False and no `env` is provided, a standard Environment is created.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Optional[Environment] = None
    investment: float = Field(default=10.0)
    idea: str = Field(default="")
    use_mgx: bool = Field(default=True)

    def __init__(self, context: Context = None, **data: Any):
        """
        Initializes a Team instance.

        Args:
            context: An optional Context object. If not provided, a new Context is created.
                     This context is used for the environment.
            **data: Arbitrary keyword arguments that are passed to the Pydantic BaseModel constructor.
                    This can include 'roles' to hire initially and 'env_desc' for the environment.
        """
        super(Team, self).__init__(**data)
        ctx = context or Context()
        if not self.env and not self.use_mgx:
            self.env = Environment(context=ctx)
        elif not self.env and self.use_mgx: # If use_mgx is True and no env is provided
            self.env = MGXEnv(context=ctx)
        else: # If env is provided (e.g., through deserialization)
            self.env.context = ctx  # The `env` object is allocated by deserialization
        if "roles" in data:
            self.hire(data["roles"])
        if "env_desc" in data and self.env:
            self.env.desc = data["env_desc"]

    def serialize(self, stg_path: Path = None):
        """
        Serializes the Team object to JSON files.

        This method saves the team's configuration and the context of its environment.
        The main team information is saved in 'team.json'.

        Args:
            stg_path: Optional path to the storage directory. If None, uses
                      `SERDESER_PATH / "team"`.
        """
        stg_path = SERDESER_PATH.joinpath("team") if stg_path is None else stg_path
        team_info_path = stg_path.joinpath("team.json")
        serialized_data = self.model_dump()
        if self.env and self.env.context:
            serialized_data["context"] = self.env.context.serialize()

        write_json_file(team_info_path, serialized_data)

    @classmethod
    def deserialize(cls, stg_path: Path, context: Context = None) -> "Team":
        """
        Deserializes a Team object from JSON files.

        This method reconstructs a Team instance from its serialized representation
        stored in the specified storage path.

        Args:
            stg_path: Path to the storage directory (e.g., "./storage/team").
            context: Optional Context object to use. If None, a new Context is created
                     and potentially updated from the serialized context.

        Returns:
            A deserialized Team instance.

        Raises:
            FileNotFoundError: If the 'team.json' file is not found in `stg_path`.
        """
        # recover team_info
        team_info_path = stg_path.joinpath("team.json")
        if not team_info_path.exists():
            raise FileNotFoundError(
                "Recovery storage meta file `team.json` not found. "
                "Cannot recover, please start a new project."
            )

        team_info: dict = read_json_file(team_info_path)
        ctx = context or Context()
        serialized_context = team_info.pop("context", None)
        if serialized_context:
            ctx.deserialize(serialized_context)
        team = Team(**team_info, context=ctx)
        return team

    def hire(self, roles: list[Role]):
        """Adds a list of roles to the team's environment.

        Args:
            roles: A list of Role instances to be added to the team.
        """
        if not self.env:
            logger.error("Environment is not initialized. Cannot hire roles.")
            return
        self.env.add_roles(roles)

    @property
    def cost_manager(self):
        """Provides access to the CostManager instance from the team's environment context.

        Returns:
            The CostManager instance if the environment and context are initialized,
            otherwise None.
        """
        if self.env and self.env.context:
            return self.env.context.cost_manager
        logger.warning("Environment or context not initialized, cannot get cost_manager.")
        return None

    def invest(self, investment: float):
        """Sets the project's investment budget.

        This budget is tracked by the CostManager in the team's environment.

        Args:
            investment: The amount of investment for the project.

        Raises:
            NoMoneyException: If the investment amount results in exceeding the max budget
                              (though this specific check seems to be in `_check_balance`).
        """
        self.investment = investment
        if self.cost_manager:
            self.cost_manager.max_budget = investment
        logger.info(f"Investment: ${investment}.")

    def _check_balance(self):
        """Checks if the total cost has exceeded the maximum budget.

        Raises:
            NoMoneyException: If the total cost is greater than or equal to the max budget.
        """
        if self.cost_manager and self.cost_manager.total_cost >= self.cost_manager.max_budget:
            raise NoMoneyException(self.cost_manager.total_cost, f"Insufficient funds: {self.cost_manager.max_budget}")

    def run_project(self, idea: str, send_to: str = ""):
        """Initiates a project by publishing the initial idea/requirement to the environment.

        Args:
            idea: The core idea or requirement for the project.
            send_to: Optional name of a specific role to send the initial message to.
                     If empty, the message is broadcast or handled by default environment routing.
        """
        self.idea = idea
        if not self.env:
            logger.error("Environment is not initialized. Cannot run project.")
            return

        # Create a message with the project idea.
        # If send_to is specified, it implies a targeted message, though the current
        # Message structure might not directly use it for recipient filtering here.
        # The environment's publish_message handles distribution.
        msg = Message(content=idea)
        if send_to: # This part might need review based on how `send_to` is actually used by `publish_message`
            msg.send_to = {send_to} # Assuming send_to is a set of recipients

        self.env.publish_message(msg)


    def start_project(self, idea: str, send_to: str = ""):
        """
        Deprecated: This method will be removed in a future version.
        Please use the `run_project` method instead.

        Args:
            idea: The core idea or requirement for the project.
            send_to: Optional name of a specific role to send the initial message to.
        """
        warnings.warn(
            "The 'start_project' method is deprecated and will be removed in the future. "
            "Please use the 'run_project' method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run_project(idea=idea, send_to=send_to)

    @serialize_decorator
    async def run(
        self,
        n_round: int = 3,
        idea: str = "",
        send_to: str = "",
        auto_archive: bool = True,
        on_round_end: Optional[Callable[["Team", int], None]] = None,
    ):
        """Run the team's project execution loop.

        Args:
            n_round: Maximum number of rounds to run.  Defaults to 3.
            idea: If provided, starts a new project with this idea.
            send_to: Optional recipient name for the initial project message.
            auto_archive: Archive environment state after the run.  Defaults to True.
            on_round_end: Optional callback invoked after every completed round.
                Signature: ``callback(team, rounds_remaining) -> None``.
                Useful for progress reporting, logging, or updating external state
                (e.g. a REST API or a WebSocket stream) without subclassing Team.
                The callback is called synchronously; use ``asyncio.ensure_future``
                inside it if you need non-blocking work.

        Returns:
            List of messages representing the environment's run history.

        Raises:
            NoMoneyException: If the budget is exhausted.

        Example::

            def report(team, remaining):
                cost = team.cost_manager.total_cost if team.cost_manager else 0
                print(f"Round done – {remaining} left, cost so far: ${cost:.4f}")

            await team.run(n_round=5, idea="Build a CLI todo app", on_round_end=report)
        """
        if idea:
            self.run_project(idea=idea, send_to=send_to)

        if not self.env:
            logger.error("Environment is not initialized. Cannot run.")
            return []

        max_round = n_round
        while n_round > 0:
            # All roles idle → project complete or permanently stuck.
            if self.env.is_idle:
                logger.debug("All roles are idle. Ending run.")
                break
            n_round -= 1
            self._check_balance()
            await self.env.run()

            rounds_done = max_round - n_round
            logger.debug(f"Round {rounds_done}/{max_round} completed. {n_round} rounds left.")

            if on_round_end is not None:
                try:
                    on_round_end(self, n_round)
                except Exception as cb_exc:
                    logger.warning(f"on_round_end callback raised an exception: {cb_exc}")

        self.env.archive(auto_archive=auto_archive)
        return self.env.history
