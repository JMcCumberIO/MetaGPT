#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : write_prd.py
@Modified By: mashenquan, 2023/11/27.
            1. According to Section 2.2.3.1 of RFC 135, replace file data in the message with the file name.
            2. According to the design in Section 2.2.3.5.2 of RFC 135, add incremental iteration functionality.
            3. Move the document storage operations related to WritePRD from the save operation of WriteDesign.
@Modified By: mashenquan, 2023/12/5. Move the generation logic of the project name to WritePRD.
@Modified By: mashenquan, 2024/5/31. Implement Chapter 3 of RFC 236.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from metagpt.actions import Action, ActionOutput
from metagpt.actions.action_node import ActionNode
from metagpt.actions.fix_bug import FixBug
from metagpt.actions.write_prd_an import (
    COMPETITIVE_QUADRANT_CHART,
    PROJECT_NAME,
    REFINED_PRD_NODE,
    WP_IS_RELATIVE_NODE,
    WP_ISSUE_TYPE_NODE,
    WRITE_PRD_NODE,
)
from metagpt.const import (
    BUGFIX_FILENAME,
    COMPETITIVE_ANALYSIS_FILE_REPO,
    REQUIREMENT_FILENAME,
)
from metagpt.logs import logger
from metagpt.schema import AIMessage, Document, Documents, Message
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import (
    CodeParser,
    aread,
    awrite,
    rectify_pathname,
    save_json_to_markdown,
    to_markdown_code_block,
)
from metagpt.utils.file_repository import FileRepository
from metagpt.utils.mermaid import mermaid_to_file
from metagpt.utils.project_repo import ProjectRepo
from metagpt.utils.report import DocsReporter, GalleryReporter

CONTEXT_TEMPLATE = """
### Project Name
{project_name}

### Original Requirements
{requirements}

### Search Information
-
"""

# Template for providing context when updating an existing PRD with new requirements.
NEW_REQ_TEMPLATE = """
### Legacy Content
{old_prd}

### Original Requirements
{requirements}

### Search Information
-
"""

# Template for providing context when updating an existing PRD with new requirements.
NEW_REQ_TEMPLATE = """
### Legacy Content
{old_prd}

### New Requirements
{requirements}
"""


@register_tool(include_functions=["run"]) # Registers the 'run' method of this action as a callable tool.
class WritePRD(Action):
    """
    Action to generate or update a Product Requirement Document (PRD).

    This action handles various scenarios:
    1.  **Bugfix**: If the input requirement is identified as a bugfix, it generates a bugfix document
        and triggers a `FixBug` action.
    2.  **New Requirement**: If the input is a new requirement, it generates a new PRD from scratch.
    3.  **Requirement Update**: If the input requirement is related to existing PRDs, it updates the
        relevant PRDs by merging the new information.

    The action interacts with a ProjectRepository for file management and uses various ActionNodes
    for specific LLM-driven tasks like determining issue types, checking relatedness, and refining PRDs.
    It can be invoked through an agentic workflow (via `with_messages`) or as a direct API call.

    Attributes:
        repo: An optional ProjectRepo instance for managing project files and documents.
              Initialized internally if not provided based on context.
        input_args: Optional Pydantic BaseModel storing parsed input arguments, typically
                    from an upstream action like `PrepareDocuments`.
    """

    repo: Optional[ProjectRepo] = Field(default=None, exclude=True) # Project repository for file operations.
    input_args: Optional[BaseModel] = Field(default=None, exclude=True) # Parsed arguments from input message.

    async def run(
        self,
        with_messages: Optional[List[Message]] = None,
        *,
        user_requirement: str = "",
        output_pathname: str = "",
        legacy_prd_filename: str = "",
        extra_info: str = "",
        **kwargs,
    ) -> Union[AIMessage, str]:
        """
        Main entry point for the WritePRD action. It orchestrates the PRD generation or update process.

        This method can be called in two ways:
        1.  Agentic Workflow (with `with_messages`):
            The action processes a list of messages, typically where the last message contains
            instructions or context (e.g., from `PrepareDocuments`). It uses this information
            to determine if the task is a bugfix, a new requirement, or an update to existing PRDs.
            It then calls the appropriate internal handler (`_handle_bugfix`, `_handle_new_requirement`,
            or `_handle_requirement_update`). The result is an AIMessage containing paths to
            changed/created PRD files and other relevant metadata.

        2.  Direct API-like Call (without `with_messages`):
            If `with_messages` is not provided, the method falls back to `_execute_api`.
            This mode is suitable for direct invocation with explicit parameters like `user_requirement`,
            `output_pathname`, etc. It returns a string indicating the path to the generated PRD.

        Args:
            with_messages: An optional list of Message objects. If provided, the action operates
                           in an agentic workflow, processing these messages.
            user_requirement: A string detailing the user's requirements. Used when `with_messages` is None.
            output_pathname: The output file path for the generated/updated document.
                             Used when `with_messages` is None. Defaults to a path within the project workspace.
            legacy_prd_filename: The file path of an existing PRD to use as a reference for updates.
                                 Used when `with_messages` is None.
            extra_info: Additional information to include in the document.
                        Used when `with_messages` is None.
            **kwargs: Additional keyword arguments (currently not used by the core logic but available for extensions).

        Returns:
            - If `with_messages` is provided: An AIMessage instance containing the paths of the
              generated/updated PRD and competitive analysis files, along with other project metadata.
            - If `with_messages` is None: A string message indicating the filename of the
              generated PRD, e.g., 'PRD filename: "/path/to/prd.json"'.

        Raises:
            FileNotFoundError: If `with_messages` is provided and the requirement document specified
                               in `self.input_args` cannot be found.

        Example (Direct API-like Call):
            >>> write_prd = WritePRD()
            >>> result = await write_prd.run(user_requirement="Create a snake game",
            ...                              output_pathname="snake_game/docs/prd.json")
            >>> print(result)
            PRD filename: ".../snake_game/docs/prd.json". The product requirement document (PRD) has been completed.
        """
        if not with_messages:
            # Direct API call mode
            return await self._execute_api(
                user_requirement=user_requirement,
                output_pathname=output_pathname,
                legacy_prd_filename=legacy_prd_filename,
                extra_info=extra_info,
            )

        # Agentic workflow mode
        self.input_args = with_messages[-1].instruct_content
        if not self.input_args:
            # If no structured input_args, assume raw requirement message and setup repo.
            # This might happen if WritePRD is the first action or called directly by a role.
            if not self.context.kwargs or not self.context.kwargs.project_path:
                raise ValueError("Project path not found in context for WritePRD.")
            self.repo = ProjectRepo(self.context.kwargs.project_path)
            await self.repo.docs.save(filename=REQUIREMENT_FILENAME, content=with_messages[-1].content)
            # Construct input_args as if PrepareDocuments had run.
            self.input_args = AIMessage.create_instruct_value(
                kvs={
                    "project_path": str(self.repo.workdir),
                    "requirements_filename": str(self.repo.docs.workdir / REQUIREMENT_FILENAME),
                    "prd_filenames": [str(self.repo.docs.prd.workdir / i) for i in self.repo.docs.prd.all_files],
                },
                class_name="PrepareDocumentsOutput",
            )
        else:
            if not hasattr(self.input_args, 'project_path') or not self.input_args.project_path: # type: ignore
                raise ValueError("project_path not found in input_args for WritePRD.")
            self.repo = ProjectRepo(self.input_args.project_path) # type: ignore

        req_doc_path = getattr(self.input_args, "requirements_filename", "")
        if not req_doc_path:
             raise FileNotFoundError("requirements_filename not found in input_args.")
        req = await Document.load(filename=req_doc_path)
        if not req:
            raise FileNotFoundError(f"Requirement document not found at {req_doc_path}.")

        prd_filenames = getattr(self.input_args, "prd_filenames", [])
        docs: list[Document] = []
        if self.repo: # Ensure repo is not None before accessing workdir
            for i in prd_filenames:
                doc = await Document.load(filename=i, project_path=self.repo.workdir)
                if doc:
                    docs.append(doc)


        if await self._is_bugfix(req.content):
            logger.info(f"Bugfix detected for requirement: {req.content[:100]}...")
            return await self._handle_bugfix(req)

        # Ensure bugfix file from a previous round is cleared if this is not a bugfix.
        if self.repo: # Ensure repo is not None
            await self.repo.docs.delete(filename=BUGFIX_FILENAME)

        related_docs = await self.get_related_docs(req, docs)
        if related_docs:
            logger.info(f"Requirement update detected for: {req.content[:100]}...")
            await self._handle_requirement_update(req=req, related_docs=related_docs)
        else:
            logger.info(f"New requirement detected: {req.content[:100]}...")
            await self._handle_new_requirement(req)

        # Prepare output message
        if not self.input_args: # Should be set by now, but as a safeguard
            raise ValueError("input_args not set before preparing output.")
        kvs = self.input_args.model_dump()
        changed_prd_files_keys: list[str] = []
        resources_prd_changed_files_keys: list[str] = []
        resources_ca_changed_files_keys: list[str] = []

        if self.repo: # Ensure repo is not None
            changed_prd_files_keys = list(self.repo.docs.prd.changed_files.keys())
            kvs["changed_prd_filenames"] = [str(self.repo.docs.prd.workdir / i) for i in changed_prd_files_keys]
            kvs["project_path"] = str(self.repo.workdir)
            kvs["requirements_filename"] = str(self.repo.docs.workdir / REQUIREMENT_FILENAME)
            self.context.kwargs.project_path = str(self.repo.workdir) # Update context
            resources_prd_changed_files_keys = list(self.repo.resources.prd.changed_files.keys())
            resources_ca_changed_files_keys = list(self.repo.resources.competitive_analysis.changed_files.keys())


        output_content = "PRD generation/update is completed. Changed files:\n" + "\n".join(
            changed_prd_files_keys
            + resources_prd_changed_files_keys
            + resources_ca_changed_files_keys
        )
        return AIMessage(
            content=output_content,
            instruct_content=AIMessage.create_instruct_value(kvs=kvs, class_name="WritePRDOutput"),
            cause_by=self,
        )

    async def _handle_bugfix(self, req: Document) -> AIMessage:
        """Handles a requirement identified as a bugfix.

        This method saves the bug description to a dedicated bugfix file (`BUGFIX_FILENAME`),
        clears the main requirement file (as the focus shifts to fixing the bug),
        and then constructs an AIMessage. This message is intended to be passed to
        an Engineer role (typically named 'Alex') to trigger the `FixBug` action.

        Args:
            req: The Document object containing the detailed bug description.

        Returns:
            An AIMessage configured to initiate the bug fixing process. This message
            includes paths to the project, the bug issue file, and the (now empty)
            requirements file. It specifies `FixBug` as the action to be caused
            and targets 'Alex' (Engineer) as the recipient.

        Raises:
            ValueError: If `self.repo` (ProjectRepo instance) is not initialized,
                        as file operations cannot be performed.
        """
        if not self.repo:
            raise ValueError("ProjectRepo not initialized for _handle_bugfix.")

        await self.repo.docs.save(filename=BUGFIX_FILENAME, content=req.content)
        # Clear the main requirement file as this is now a bugfix.
        await self.repo.docs.save(filename=REQUIREMENT_FILENAME, content="")

        bug_path = self.repo.docs.workdir / BUGFIX_FILENAME
        req_path = self.repo.docs.workdir / REQUIREMENT_FILENAME

        return AIMessage(
            content=f"A new bug fix task is received: {bug_path.name}",
            cause_by=FixBug, # Specifies that the next action should be FixBug
            instruct_content=AIMessage.create_instruct_value(
                {
                    "project_path": str(self.repo.workdir),
                    "issue_filename": str(bug_path),
                    "requirements_filename": str(req_path), # Though empty, pass for consistency
                },
                class_name="IssueDetail", # Expected input class for FixBug action
            ),
            send_to="Alex",  # Conventionally, Alex is the Engineer role.
        )

    async def _new_prd(self, requirement: str) -> ActionNode:
        """Generates content for a new Product Requirement Document (PRD) using an ActionNode.

        This method takes a user requirement, formats it into a context using `CONTEXT_TEMPLATE`
        (which includes the project name and original requirements), and then uses the
        `WRITE_PRD_NODE` ActionNode to generate the PRD content via an LLM.
        If the project name is already known (`self.project_name`), it's excluded from
        the LLM prompt to avoid redundancy.

        Args:
            requirement: A string detailing the user's requirements for the new PRD.

        Returns:
            An ActionNode instance that has been filled with the LLM-generated PRD content.
            The actual PRD data can be accessed via `node.instruct_content`.
        """
        project_name = self.project_name or self.context.kwargs.get("project_name", "Unnamed Project")
        context = CONTEXT_TEMPLATE.format(requirements=requirement, project_name=project_name)
        # Exclude PROJECT_NAME from LLM completion if already known, to avoid redundancy.
        exclude_keys = [PROJECT_NAME.key] if self.project_name else []
        node = await WRITE_PRD_NODE.fill(
            context=context, llm=self.llm, exclude=exclude_keys, schema=self.prompt_schema # type: ignore
        )
        return node

    async def _handle_new_requirement(self, req: Document) -> ActionOutput:
        """Handles a new requirement by generating a new Product Requirement Document (PRD).

        This method orchestrates the process of creating a PRD from a new user requirement.
        The key steps are:
        1. Generate initial PRD content using the `_new_prd` method, which involves an LLM call.
        2. If a project name is derived from the PRD content, rename the workspace directory accordingly
           using `_rename_workspace`.
        3. Save the generated PRD content as a JSON file in the project's document repository.
           A unique filename is generated for the new PRD.
        4. Extract and save any competitive analysis chart (e.g., a Mermaid diagram) found within
           the PRD content using `_save_competitive_analysis`.
        5. Convert the PRD from JSON to a PDF format (often via Markdown) and save it in the
           project's resource repository.
        6. Report the creation of these documents using `DocsReporter` and `GalleryReporter`.

        Args:
            req: A Document object containing the new user requirement.

        Returns:
            An ActionOutput instance containing the newly created PRD Document. This output
            can be used by subsequent actions in a workflow.

        Raises:
            ValueError: If `self.repo` (ProjectRepo instance) is not initialized.
        """
        if not self.repo:
            raise ValueError("ProjectRepo not initialized for _handle_new_requirement.")

        async with DocsReporter(enable_llm_stream=True) as reporter:
            await reporter.async_report({"type": "prd"}, "meta") # Report metadata for PRD generation
            prd_node = await self._new_prd(req.content)
            await self._rename_workspace(prd_node) # Rename workspace if project name is discovered

            prd_json_content = prd_node.instruct_content.model_dump_json() if prd_node.instruct_content else "{}" # type: ignore
            new_prd_doc = await self.repo.docs.prd.save(
                filename=FileRepository.new_filename() + ".json", # Generate a unique filename
                content=prd_json_content,
            )
            await self._save_competitive_analysis(new_prd_doc)

            # Save PRD as PDF (which internally converts JSON to Markdown first)
            md_doc = await self.repo.resources.prd.save_pdf(doc=new_prd_doc)
            if md_doc and self.repo.workdir: # Ensure md_doc and workdir are valid
                 await reporter.async_report(self.repo.workdir / md_doc.root_relative_path, "path") # type: ignore
            return Documents.from_iterable(documents=[new_prd_doc]).to_action_output()

    async def _handle_requirement_update(self, req: Document, related_docs: list[Document]) -> ActionOutput:
        """Handles a requirement update by merging it into related existing PRDs.

        This method iterates through each Document in `related_docs` (which are existing PRDs
        identified as relevant to the new requirement `req`). For each related PRD,
        it calls `_update_prd` to merge the new requirement information, thereby updating
        the PRD.

        Args:
            req: The Document object containing the new requirement or update details.
            related_docs: A list of existing PRD Document objects that have been
                          determined to be related to the new requirement.

        Returns:
            An ActionOutput instance containing all the PRD Documents that were updated.
            This allows subsequent actions to be aware of all modified PRDs.
        """
        updated_docs = []
        for doc in related_docs:
            updated_doc = await self._update_prd(req=req, prd_doc=doc)
            updated_docs.append(updated_doc)
        return Documents.from_iterable(documents=updated_docs).to_action_output()

    async def _is_bugfix(self, context: str) -> bool:
        """Determines if the given context (typically a user requirement) describes a bugfix.

        This method first checks if there are any code files in the project repository.
        If not, it assumes the requirement cannot be a bugfix for existing code.
        Otherwise, it uses the `WP_ISSUE_TYPE_NODE` ActionNode (which likely queries an LLM)
        to classify the given `context` string. If the classified issue type is "BUG",
        the method returns True.

        Args:
            context: A string containing the user requirement or issue description.

        Returns:
            True if the context is determined to be a bugfix, False otherwise.
        """
        if not self.repo or not self.repo.code_files_exists():
            # If no code files exist in the repo, it's unlikely to be a bugfix for existing code.
            return False
        node = await WP_ISSUE_TYPE_NODE.fill(context=context, llm=self.llm) # type: ignore
        return node.get("issue_type") == "BUG" # Compares the classified issue type

    async def get_related_docs(self, req: Document, docs: list[Document]) -> list[Document]:
        """Identifies which of the provided documents (typically existing PRDs) are related
        to the given new requirement.

        This method iterates through a list of existing documents (`docs`) and, for each one,
        calls `_is_related` to determine if it's relevant to the new requirement (`req`).
        It collects all such related documents into a list.

        Args:
            req: A Document object representing the new user requirement.
            docs: A list of Document objects (e.g., existing PRDs) to check against
                  the new requirement.

        Returns:
            A list of Document objects from `docs` that are considered related to `req`.
        """
        # TODO: Consider using asyncio.gather for concurrent _is_related calls if performance is an issue,
        # especially if 'docs' list can be very long.
        related_documents = []
        for doc in docs:
            if await self._is_related(req, doc):
                related_documents.append(doc)
        return related_documents

    async def _is_related(self, req: Document, old_prd: Document) -> bool:
        """Determines if a new requirement (`req`) is related to an existing PRD (`old_prd`).

        This is achieved by providing context to an LLM via `WP_IS_RELATIVE_NODE`.
        The context includes content from both the new requirement and the old PRD,
        formatted using `NEW_REQ_TEMPLATE`. The LLM is then expected to classify
        if the new requirement is relative to the old PRD.

        Args:
            req: A Document object representing the new user requirement.
            old_prd: A Document object representing an existing PRD.

        Returns:
            True if the LLM determines that the new requirement is related to the
            existing PRD (i.e., `is_relative` is "YES"). False otherwise.
        """
        context = NEW_REQ_TEMPLATE.format(old_prd=old_prd.content, requirements=req.content)
        node = await WP_IS_RELATIVE_NODE.fill(context=context, llm=self.llm) # type: ignore
        return node.get("is_relative") == "YES" # Checks the LLM's classification

    async def _merge(self, req: Document, related_doc: Document) -> Document:
        """Merges a new requirement into an existing, related Product Requirement Document (PRD).

        This method takes a new requirement (`req`) and an existing PRD (`related_doc`).
        It constructs a prompt using `NEW_REQ_TEMPLATE`, which presents both the legacy
        content of the PRD and the new requirements to an LLM via `REFINED_PRD_NODE`.
        The LLM's task is to generate a refined PRD content that incorporates the new
        requirements into the existing document.

        The `related_doc`'s content is then updated with this new, merged content.
        If a project name is derived during this process (e.g., from the LLM's output),
        the workspace may also be renamed.

        Args:
            req: A Document object containing the new user requirement or update.
            related_doc: The existing PRD Document object that needs to be updated.

        Returns:
            The `related_doc` Document object, now updated with the merged content.
        """
        if not self.project_name and self.project_path:
            self.project_name = Path(self.project_path).name # Default project name from path

        prompt = NEW_REQ_TEMPLATE.format(requirements=req.content, old_prd=related_doc.content)
        # Use REFINED_PRD_NODE to get the updated PRD content from LLM.
        prd_node = await REFINED_PRD_NODE.fill(context=prompt, llm=self.llm, schema=self.prompt_schema) # type: ignore

        related_doc.content = prd_node.instruct_content.model_dump_json() if prd_node.instruct_content else "{}" # type: ignore
        await self._rename_workspace(prd_node) # Rename workspace if project name is updated/discovered
        return related_doc

    async def _update_prd(self, req: Document, prd_doc: Document) -> Document:
        """Updates a single Product Requirement Document (PRD) with new requirements.

        This method orchestrates the update of an individual PRD. The process includes:
        1. Merging the new requirement (`req`) into the existing PRD (`prd_doc`) using the
           `_merge` method, which involves an LLM call to refine the content.
        2. Saving the updated PRD content back to its original file in the project's
           document repository (typically as a JSON file).
        3. Extracting and saving any competitive analysis chart (e.g., Mermaid diagram)
           found in the updated PRD content using `_save_competitive_analysis`.
        4. Converting the updated PRD to PDF format (often via Markdown) and saving it
           in the project's resource repository.
        5. Reporting the updated PRD and any generated artifacts (like the PDF or chart)
           using `DocsReporter` and `GalleryReporter`.

        Args:
            req: A Document object containing the new user requirement or update.
            prd_doc: The existing PRD Document object to be updated.

        Returns:
            The `prd_doc` Document object, now containing the updated content and reflecting
            any changes made during the process.

        Raises:
            ValueError: If `self.repo` (ProjectRepo instance) is not initialized.
        """
        if not self.repo:
            raise ValueError("ProjectRepo not initialized for _update_prd.")

        async with DocsReporter(enable_llm_stream=True) as reporter:
            await reporter.async_report({"type": "prd"}, "meta")
            updated_prd_doc: Document = await self._merge(req=req, related_doc=prd_doc)
            await self.repo.docs.prd.save_doc(doc=updated_prd_doc) # Save updated JSON
            await self._save_competitive_analysis(updated_prd_doc)

            md_doc = await self.repo.resources.prd.save_pdf(doc=updated_prd_doc) # Save PDF
            if md_doc and self.repo.workdir: # Ensure md_doc and workdir are valid
                await reporter.async_report(self.repo.workdir / md_doc.root_relative_path, "path") # type: ignore
        return updated_prd_doc

    async def _save_competitive_analysis(self, prd_doc: Document, output_filename: Optional[Path] = None):
        """Extracts a competitive quadrant chart from PRD content and saves it as an image.

        This method parses the PRD document's content (expected to be JSON) to find a
        competitive quadrant chart, which is typically represented as a Mermaid diagram string.
        If found, it uses `mermaid_to_file` to convert this diagram into an image file (e.g., SVG).
        The generated image is then reported using `GalleryReporter`.

        The output path for the saved chart can be specified directly. If not, it defaults
        to a structured path within the project's competitive analysis resources directory,
        derived from the PRD filename.

        Args:
            prd_doc: The Document object containing the PRD content (in JSON format).
            output_filename: An optional Path object specifying the base name and location
                             for the output chart image. If None, a default path is constructed.

        Side Effects:
            - Creates directories if they don't exist for the output path.
            - Writes a Mermaid diagram file (e.g., .mmd).
            - Writes an image file (e.g., .svg) generated from the Mermaid diagram.
            - Logs warnings or errors if parsing, chart extraction, or file generation fails.
        """
        if not self.repo:
            logger.warning("ProjectRepo not initialized. Cannot save competitive analysis.")
            return
        try:
            prd_data = json.loads(prd_doc.content)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse PRD content as JSON for doc: {prd_doc.filename}")
            return

        quadrant_chart = prd_data.get(COMPETITIVE_QUADRANT_CHART.key)
        if not quadrant_chart:
            logger.info(f"No competitive quadrant chart found in PRD: {prd_doc.filename}")
            return

        if output_filename:
            base_pathname = output_filename
        else:
            # Default path within the project's competitive analysis resources.
            if not self.repo.workdir: # Should be set if repo is initialized
                 logger.error("Repository workdir not set. Cannot save competitive analysis.")
                 return
            base_pathname = self.repo.workdir / COMPETITIVE_ANALYSIS_FILE_REPO / Path(prd_doc.filename).stem

        base_pathname.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Generate Mermaid file (.mmd) and then convert to SVG.
            await mermaid_to_file(self.config.mermaid.engine, quadrant_chart, base_pathname)
            # mermaid_to_file creates base_pathname.mmd and base_pathname.svg (or other format based on engine)
            # We assume SVG for reporting.
            image_path = base_pathname.with_suffix(".svg") # Or determine suffix from mermaid_to_file if it differs
            if image_path.exists():
                logger.info(f"Saved competitive analysis chart to {image_path}")
                await GalleryReporter().async_report(image_path, "path") # Report the generated image.
            else:
                logger.warning(f"Mermaid image not found at expected path: {image_path}")
        except Exception as e:
            logger.error(f"Failed to save competitive analysis chart: {e}")


    async def _rename_workspace(self, prd_data: Union[ActionNode, ActionOutput, str]):
        """Renames the project's workspace directory if a new project name is determined.

        This method attempts to extract a "Project Name" from the provided `prd_data`.
        `prd_data` can be an `ActionNode` or `ActionOutput` (where the project name is
        expected in `instruct_content`) or a raw string (parsed using `CodeParser`).

        If a project name is found and `self.project_name` is not already set,
        it updates `self.project_name` and attempts to rename the root directory
        of the project's Git repository via `self.repo.git_repo.rename_root()`.

        Args:
            prd_data: The data from which to extract the project name. This can be
                      an ActionNode, ActionOutput, or a string containing PRD content.

        Side Effects:
            - May update `self.project_name`.
            - May rename the project's root directory on the filesystem if a Git repository
              is being managed and a new project name is successfully extracted.
            - Logs information about the process, or warnings/errors if issues occur.
        """
        if self.project_name: # If project name is already set, no need to rename.
            return

        ws_name = None
        prd_instruct_content = getattr(prd_data, 'instruct_content', None)
        if prd_instruct_content:
            try:
                # Assumes instruct_content is a Pydantic model or dict with "Project Name"
                project_name_data = prd_instruct_content.model_dump()
                ws_name = project_name_data.get("Project Name")
            except Exception as e:
                logger.debug(f"Could not extract project name from instruct_content: {e}")
        elif isinstance(prd_data, str):
            # Fallback for raw string PRD, less reliable.
            ws_name = CodeParser.parse_str(block="Project Name", text=prd_data)

        if ws_name:
            self.project_name = ws_name
            logger.info(f"Project name set to: {ws_name}")
            if self.repo and self.repo.git_repo: # Ensure repo and git_repo are initialized
                try:
                    self.repo.git_repo.rename_root(self.project_name)
                    logger.info(f"Workspace renamed to: {self.project_name}")
                except Exception as e:
                    logger.error(f"Failed to rename workspace to {self.project_name}: {e}")
            elif not self.repo:
                logger.warning("ProjectRepo not available, cannot rename workspace.")

    async def _execute_api(
        self, user_requirement: str, output_pathname: str, legacy_prd_filename: str, extra_info: str
    ) -> str:
        """
        Handles direct API-like calls to generate or update a PRD.

        This method is used when `run` is called without `with_messages`. It bypasses
        the agentic workflow and directly generates/updates a PRD based on the
        provided string inputs.

        Args:
            user_requirement: The user's requirement string.
            output_pathname: The desired path for the output PRD (JSON).
            legacy_prd_filename: Path to an existing PRD if this is an update.
            extra_info: Additional information to be included.

        Returns:
            A string message indicating the success and path of the generated PRD.
        """
        content = "#### User Requirements\n{user_requirement}\n#### Extra Info\n{extra_info}\n".format(
            user_requirement=to_markdown_code_block(val=user_requirement),
            extra_info=to_markdown_code_block(val=extra_info),
        )
        async with DocsReporter(enable_llm_stream=True) as reporter:
            await reporter.async_report({"type": "prd"}, "meta")
            req = Document(content=content)
            if not legacy_prd_filename:
                node = await self._new_prd(requirement=req.content)
                new_prd = Document(content=node.instruct_content.model_dump_json())
            else:
                content = await aread(filename=legacy_prd_filename)
                old_prd = Document(content=content)
                new_prd = await self._merge(req=req, related_doc=old_prd)

            if not output_pathname:
                output_pathname = self.config.workspace.path / "docs" / "prd.json"
            elif not Path(output_pathname).is_absolute():
                output_pathname = self.config.workspace.path / output_pathname
            output_pathname = rectify_pathname(path=output_pathname, default_filename="prd.json")
            await awrite(filename=output_pathname, data=new_prd.content)
            competitive_analysis_filename = output_pathname.parent / f"{output_pathname.stem}-competitive-analysis"
            await self._save_competitive_analysis(prd_doc=new_prd, output_filename=Path(competitive_analysis_filename))
            md_output_filename = output_pathname.with_suffix(".md")
            await save_json_to_markdown(content=new_prd.content, output_filename=md_output_filename)
            await reporter.async_report(md_output_filename, "path")
        return f'PRD filename: "{str(output_pathname)}". The  product requirement document (PRD) has been completed.'
