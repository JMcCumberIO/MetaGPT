#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:26
@Author  : alexanderwu
@File    : design_api.py
@Modified By: mashenquan, 2023/11/27.
            1. According to Section 2.2.3.1 of RFC 135, replace file data in the message with the file name.
            2. According to the design in Section 2.2.3.5.3 of RFC 135, add incremental iteration functionality.
@Modified By: mashenquan, 2023/12/5. Move the generation logic of the project name to WritePRD.
@Modified By: mashenquan, 2024/5/31. Implement Chapter 3 of RFC 236.
"""
import json
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from metagpt.actions import Action
from metagpt.actions.design_api_an import (
    DATA_STRUCTURES_AND_INTERFACES,
    DESIGN_API_NODE,
    PROGRAM_CALL_FLOW,
    REFINED_DATA_STRUCTURES_AND_INTERFACES,
    REFINED_DESIGN_NODE,
    REFINED_PROGRAM_CALL_FLOW,
)
from metagpt.const import DATA_API_DESIGN_FILE_REPO, SEQ_FLOW_FILE_REPO
from metagpt.logs import logger
from metagpt.schema import AIMessage, Document, Documents, Message
from metagpt.tools.tool_registry import register_tool
from metagpt.utils.common import (
    aread,
    awrite,
    rectify_pathname,
    save_json_to_markdown,
    to_markdown_code_block,
)
from metagpt.utils.mermaid import mermaid_to_file
from metagpt.utils.project_repo import ProjectRepo
from metagpt.utils.report import DocsReporter, GalleryReporter

NEW_REQ_TEMPLATE = """
### Legacy Content
{old_design}

### New Requirements
{context}
"""


@register_tool(include_functions=["run"]) # Registers the 'run' method of this action as a callable tool.
class WriteDesign(Action):
    """
    Action to generate or update a System Design document.

    This action takes Product Requirement Documents (PRDs) and existing designs (if any)
    as input to produce a comprehensive system design. The design includes APIs,
    data structures, library tables, processes, and call flows.

    It handles several scenarios:
    1.  **New Design**: Generates a new system design based on PRD(s).
    2.  **Design Update**: Updates an existing system design document based on new requirements
        or changes in PRD(s).

    The process involves:
    - Interacting with a ProjectRepository for file management (PRDs, design documents, diagrams).
    - Utilizing ActionNodes (e.g., `DESIGN_API_NODE`, `REFINED_DESIGN_NODE`) for LLM-driven
      tasks like initial design generation and refinement.
    - Extracting and saving Mermaid diagrams for data structures/APIs and sequence flows.
    - Generating a PDF version of the system design document.

    Attributes:
        name: Name of the action.
        i_context: Optional input context (not heavily used in this specific action's run method,
                   as context is primarily derived from messages or direct params).
        desc: Description of the action, outlining its purpose.
        repo: An optional ProjectRepo instance for managing project files. Initialized internally
              if operating within a project context.
        input_args: Optional Pydantic BaseModel storing parsed input arguments, typically from
                    an upstream action like `WritePRD` when run in an agentic workflow.
    """
    name: str = ""
    i_context: Optional[str] = None # Input context, less critical here as context is built from messages/args.
    desc: str = (
        "Based on the PRD, think about the system design, and design the corresponding APIs, "
        "data structures, library tables, processes, and paths. Please provide your design, feedback "
        "clearly and in detail."
    ) # Description of the action's purpose.
    repo: Optional[ProjectRepo] = Field(default=None, exclude=True) # Project repository for file operations.
    input_args: Optional[BaseModel] = Field(default=None, exclude=True) # Parsed arguments from input message.

    async def run(
        self,
        with_messages: Optional[List[Message]] = None,
        *,
        user_requirement: str = "",
        prd_filename: str = "",
        legacy_design_filename: str = "",
        extra_info: str = "",
        output_pathname: str = "",
        **kwargs,
    ) -> Union[AIMessage, str]:
        """
        Main entry point for the WriteDesign action. Orchestrates system design generation or updates.

        This method operates in two primary modes:
        1.  **Agentic Workflow (with `with_messages`):**
            Processes a list of messages, where the last message typically contains `input_args`
            from a preceding action (e.g., `WritePRDOutput`). These arguments include paths to
            changed PRDs and existing system designs. The method iterates through these, calling
            `_update_system_design` for each relevant document. The result is an AIMessage
            summarizing the changes and providing paths to updated design files.

        2.  **Direct API-like Call (without `with_messages`):**
            Falls back to `_execute_api` for direct invocation. This mode uses explicit
            parameters like `user_requirement`, `prd_filename`, etc., to generate or update
            a single system design document. Returns a string message with the path to the
            generated design file.

        Args:
            with_messages: An optional list of Message objects. If provided, the action operates
                           in an agentic workflow.
            user_requirement: User's requirement string (used in API mode).
            prd_filename: Path to the PRD file (used in API mode).
            legacy_design_filename: Path to an existing design file to be updated (API mode).
            extra_info: Additional information for design generation (API mode).
            output_pathname: Desired output path for the design document (API mode).
            **kwargs: Additional keyword arguments.

        Returns:
            - If `with_messages` is provided: An AIMessage instance with paths to changed
              system design files and related artifacts (class diagrams, sequence diagrams).
            - If `with_messages` is None: A string message indicating the filename of the
              generated system design document.

        Raises:
            ValueError: If `input_args` or `project_path` is missing when expected in agentic mode.
        """
        if not with_messages:
            # Direct API call mode
            return await self._execute_api(
                user_requirement=user_requirement,
                prd_filename=prd_filename,
                legacy_design_filename=legacy_design_filename,
                extra_info=extra_info,
                output_pathname=output_pathname,
            )

        self.input_args = with_messages[-1].instruct_content
        self.repo = ProjectRepo(self.input_args.project_path)
        changed_prds = self.input_args.changed_prd_filenames
        changed_system_designs = [
            str(self.repo.docs.system_design.workdir / i)
            for i in list(self.repo.docs.system_design.changed_files.keys())
        ]

        # For those PRDs and design documents that have undergone changes, regenerate the design content.
        changed_files = Documents()
        for filename in changed_prds:
            doc = await self._update_system_design(filename=filename)
            changed_files.docs[filename] = doc

        for filename in changed_system_designs:
            if filename in changed_files.docs:
                continue
            doc = await self._update_system_design(filename=filename)
            changed_files.docs[filename] = doc
        if not changed_files.docs:
            logger.info("Nothing has changed.")
        # Wait until all files under `docs/system_designs/` are processed before sending the publish message,
        # leaving room for global optimization in subsequent steps.
        kvs = self.input_args.model_dump()
        kvs["changed_system_design_filenames"] = [
            str(self.repo.docs.system_design.workdir / i)
            for i in list(self.repo.docs.system_design.changed_files.keys())
        ]
        return AIMessage(
            content="Designing is complete. "
            + "\n".join(
                list(self.repo.docs.system_design.changed_files.keys())
                + list(self.repo.resources.data_api_design.changed_files.keys())
                + list(self.repo.resources.seq_flow.changed_files.keys())
            ),
            instruct_content=AIMessage.create_instruct_value(kvs=kvs, class_name="WriteDesignOutput"),
            cause_by=self,
        )

    async def _new_system_design(self, context: str) -> ActionNode:
        """Generates a new system design based on the provided context (typically PRD content).

        Uses the `DESIGN_API_NODE` ActionNode, which is expected to prompt an LLM
        to create the initial system design, including APIs, data structures, etc.

        Args:
            context: A string containing the context for design generation, usually the
                     content of a Product Requirement Document (PRD).

        Returns:
            An ActionNode instance filled with the LLM-generated system design content.
            The actual design data can be accessed via `node.instruct_content`.
        """
        node = await DESIGN_API_NODE.fill(context=context, llm=self.llm, schema=self.prompt_schema) # type: ignore
        return node

    async def _merge(self, prd_doc: Document, system_design_doc: Document) -> Document:
        """Merges new requirements from a PRD into an existing system design document.

        This method constructs a context using `NEW_REQ_TEMPLATE`, providing both the
        old system design content and the new PRD content to an LLM via `REFINED_DESIGN_NODE`.
        The LLM's role is to generate a refined system design that incorporates the
        new requirements from the PRD. The `system_design_doc` is updated in place.

        Args:
            prd_doc: A Document object containing the PRD content with new requirements.
            system_design_doc: The existing system design Document object to be updated.

        Returns:
            The updated `system_design_doc` Document object with its content modified to
            reflect the merged design.
        """
        context = NEW_REQ_TEMPLATE.format(old_design=system_design_doc.content, context=prd_doc.content)
        node = await REFINED_DESIGN_NODE.fill(context=context, llm=self.llm, schema=self.prompt_schema) # type: ignore
        system_design_doc.content = node.instruct_content.model_dump_json() if node.instruct_content else "{}" # type: ignore
        return system_design_doc

    async def _update_system_design(self, filename: str) -> Optional[Document]:
        """Updates or creates a system design document based on a PRD file.

        This is a core method in the agentic workflow. Given a `filename` (which should be
        the path to a PRD document), it performs the following:
        1. Loads the PRD document.
        2. Tries to find an existing system design document corresponding to this PRD
           (based on filename).
        3. If an old design exists, it merges the PRD content into it using `_merge`.
        4. If no old design exists, it creates a new one using `_new_system_design` based on the PRD.
        5. Saves the new or updated system design document (JSON).
        6. Extracts and saves data API design (class diagram) and sequence flow diagrams
           (Mermaid format) from the design content.
        7. Generates and saves a PDF version of the system design.
        8. Reports all generated artifacts.

        Args:
            filename: The path to the Product Requirement Document (PRD) that will
                      guide the system design update or creation. This path is typically
                      relative to the project's working directory.

        Returns:
            The updated or newly created Document object for the system design, or None
            if the PRD cannot be loaded or a new design cannot be generated.

        Raises:
            ValueError: If `self.repo` is not initialized.
        """
        if not self.repo: # Should be initialized by run method
            logger.error("ProjectRepo not initialized for _update_system_design.")
            return None

        # filename is expected to be a PRD filename, often relative to repo.workdir
        # Ensure we handle paths correctly, assuming filename might be absolute or relative.
        if Path(filename).is_absolute():
            try:
                # Ensure workdir is a Path object for relative_to, and it's an ancestor
                workdir_path = Path(self.repo.workdir) if self.repo.workdir else Path.cwd()
                if not Path(filename).is_relative_to(workdir_path): # Python 3.9+
                     # Fallback for older Python or if not directly relative (e.g. symlinks)
                     # This might need more robust handling depending on path structures.
                    logger.warning(f"File {filename} may not be relative to project workdir {workdir_path}.")
                    # Attempt to use just the name if full relative path fails. This is a simple heuristic.
                    root_relative_path = Path(Path(filename).name)
                else:
                    root_relative_path = Path(filename).relative_to(workdir_path)

            except ValueError: # Handles cases where filename is not under workdir_path
                logger.error(f"File {filename} is not relative to project workdir {self.repo.workdir}.")
                return None
        else: # Already a relative path
            root_relative_path = Path(filename)


        prd = await Document.load(filename=str(root_relative_path), project_path=self.repo.workdir)
        if not prd:
            logger.warning(f"PRD document not found at {filename} (relative: {root_relative_path}), cannot update system design.")
            return None

        # System design filename is assumed to match PRD filename (e.g., prd.json -> system_design.json)
        system_design_doc_name = root_relative_path.name # Use the same name part as PRD
        old_system_design_doc = await self.repo.docs.system_design.get(system_design_doc_name)

        doc: Optional[Document] = None # Initialize doc to ensure it's always defined
        async with DocsReporter(enable_llm_stream=True) as reporter:
            await reporter.async_report({"type": "design"}, "meta")
            if not old_system_design_doc:
                logger.info(f"No existing system design found for {prd.filename}. Creating new design.")
                system_design_node = await self._new_system_design(context=prd.content)
                if not system_design_node.instruct_content:
                    logger.error(f"Failed to generate new system design content for PRD: {prd.filename}")
                    return None
                doc = await self.repo.docs.system_design.save(
                    filename=prd.filename, # Save with the same base name as PRD
                    content=system_design_node.instruct_content.model_dump_json(),
                    dependencies={prd.root_relative_path},
                )
            else:
                logger.info(f"Updating existing system design {old_system_design_doc.filename} based on PRD {prd.filename}.")
                doc = await self._merge(prd_doc=prd, system_design_doc=old_system_design_doc)
                await self.repo.docs.system_design.save_doc(doc=doc, dependencies={prd.root_relative_path})

            if not doc: # Should not happen if logic above is correct, but as a safeguard
                logger.error(f"System design document creation/update failed for PRD: {prd.filename}")
                return None

            await self._save_data_api_design(doc)
            await self._save_seq_flow(doc)
            md_doc = await self.repo.resources.system_design.save_pdf(doc=doc)
            if md_doc and self.repo.workdir:
                await reporter.async_report(self.repo.workdir / md_doc.root_relative_path, "path") # type: ignore
        return doc

    async def _save_data_api_design(self, design_doc: Document, output_filename: Optional[Path] = None):
        """Extracts and saves the data API design (class diagram) from the system design document.

        The method parses the JSON content of the `design_doc` to find keys related to
        data structures and interfaces (e.g., `DATA_STRUCTURES_AND_INTERFACES.key`).
        If found, this data (assumed to be Mermaid code) is saved as a Mermaid file
        and then rendered as an image (e.g., SVG) using `_save_mermaid_file`.

        Args:
            design_doc: The Document object containing the system design JSON content.
            output_filename: Optional base path for the output diagram. If None, a default
                             path within the project's `DATA_API_DESIGN_FILE_REPO` is used,
                             derived from `design_doc.filename`.
        """
        if not self.repo: # self.repo should be set in run
            logger.warning("ProjectRepo not available. Skipping saving data API design.")
            return
        try:
            m = json.loads(design_doc.content)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse design_doc content as JSON for {design_doc.filename}")
            return

        data_api_design = m.get(DATA_STRUCTURES_AND_INTERFACES.key) or m.get(REFINED_DATA_STRUCTURES_AND_INTERFACES.key)
        if not data_api_design:
            logger.info(f"No data API design found in {design_doc.filename}")
            return

        pathname = output_filename or (self.repo.workdir / DATA_API_DESIGN_FILE_REPO / Path(design_doc.filename).with_suffix(""))
        await self._save_mermaid_file(data_api_design, pathname)
        logger.info(f"Saved data API design (class view) to {str(pathname)} and associated image.")

    async def _save_seq_flow(self, design_doc: Document, output_filename: Optional[Path] = None):
        """Extracts and saves the program call flow (sequence diagram) from the system design document.

        Similar to `_save_data_api_design`, this method parses the `design_doc` for keys
        related to program call flow (e.g., `PROGRAM_CALL_FLOW.key`). If Mermaid code
        for a sequence diagram is found, it's saved and rendered as an image using
        `_save_mermaid_file`.

        Args:
            design_doc: The Document object containing the system design JSON content.
            output_filename: Optional base path for the output diagram. If None, a default
                             path within the project's `SEQ_FLOW_FILE_REPO` is used,
                             derived from `design_doc.filename`.
        """
        if not self.repo: # self.repo should be set in run
            logger.warning("ProjectRepo not available. Skipping saving sequence flow.")
            return
        try:
            m = json.loads(design_doc.content)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse design_doc content as JSON for {design_doc.filename}")
            return

        seq_flow = m.get(PROGRAM_CALL_FLOW.key) or m.get(REFINED_PROGRAM_CALL_FLOW.key)
        if not seq_flow:
            logger.info(f"No sequence flow diagram found in {design_doc.filename}")
            return

        pathname = output_filename or (self.repo.workdir / Path(SEQ_FLOW_FILE_REPO) / Path(design_doc.filename).with_suffix(""))
        await self._save_mermaid_file(seq_flow, pathname)
        logger.info(f"Saved program call flow (sequence diagram) to {str(pathname)} and associated image.")

    async def _save_mermaid_file(self, data: str, pathname: Path):
        """Saves Mermaid diagram data to a file and renders it as an image.

        This helper method takes Mermaid diagram code (`data`) and a base `pathname`.
        It creates the necessary parent directories, saves the Mermaid code to a `.mmd`
        file (by convention, though `mermaid_to_file` handles the actual saving based
        on its logic, often just using `pathname` as a base for output), and then
        uses `mermaid_to_file` to convert this diagram into an image (e.g., SVG).
        The generated image is reported using `GalleryReporter`.

        Args:
            data: A string containing the Mermaid diagram definition.
            pathname: The base Path object for the output. `mermaid_to_file` will typically
                      create a `.mmd` file and an image file (e.g., `.svg`) based on this.
        """
        pathname.parent.mkdir(parents=True, exist_ok=True)
        # `mermaid_to_file` handles saving the .mmd and the image (e.g., .svg)
        await mermaid_to_file(self.config.mermaid.engine, data, pathname)
        # Determine the expected image suffix based on config or default to 'svg'
        image_suffix = self.config.mermaid.image_suffix or 'svg'
        image_path = pathname.with_suffix(f".{image_suffix}")

        if image_path.exists():
            await GalleryReporter().async_report(image_path, "path")
        else:
            logger.warning(f"Mermaid image not found at expected path: {image_path} (mermaid engine: {self.config.mermaid.engine})")

    async def _execute_api(
        self,
        user_requirement: str = "",
        prd_filename: str = "",
        legacy_design_filename: str = "",
        extra_info: str = "",
        output_pathname: str = "",
    ) -> str:
        prd_content = ""
        if prd_filename:
            prd_filename = rectify_pathname(path=prd_filename, default_filename="prd.json")
            prd_content = await aread(filename=prd_filename)
        context = "### User Requirements\n{user_requirement}\n### Extra_info\n{extra_info}\n### PRD\n{prd}\n".format(
            user_requirement=to_markdown_code_block(user_requirement),
            extra_info=to_markdown_code_block(extra_info),
            prd=to_markdown_code_block(prd_content),
        )
        async with DocsReporter(enable_llm_stream=True) as reporter:
            await reporter.async_report({"type": "design"}, "meta")
            if not legacy_design_filename:
                node = await self._new_system_design(context=context)
                design = Document(content=node.instruct_content.model_dump_json())
            else:
                old_design_content = await aread(filename=legacy_design_filename)
                design = await self._merge(
                    prd_doc=Document(content=context), system_design_doc=Document(content=old_design_content)
                )

            if not output_pathname:
                output_pathname = Path(output_pathname) / "docs" / "system_design.json"
            elif not Path(output_pathname).is_absolute():
                output_pathname = self.config.workspace.path / output_pathname
            output_pathname = rectify_pathname(path=output_pathname, default_filename="system_design.json")
            await awrite(filename=output_pathname, data=design.content)
            output_filename = output_pathname.parent / f"{output_pathname.stem}-class-diagram"
            await self._save_data_api_design(design_doc=design, output_filename=output_filename)
            output_filename = output_pathname.parent / f"{output_pathname.stem}-sequence-diagram"
            await self._save_seq_flow(design_doc=design, output_filename=output_filename)
            md_output_filename = output_pathname.with_suffix(".md")
            await save_json_to_markdown(content=design.content, output_filename=md_output_filename)
            await reporter.async_report(md_output_filename, "path")
        return f'System Design filename: "{str(output_pathname)}". \n The System Design has been completed.'
