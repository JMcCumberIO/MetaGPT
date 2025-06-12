#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : write_code.py
@Modified By: mashenquan, 2023-11-1. In accordance with Chapter 2.1.3 of RFC 116, modify the data type of the `cause_by`
            value of the `Message` object.
@Modified By: mashenquan, 2023-11-27.
        1. Mark the location of Design, Tasks, Legacy Code and Debug logs in the PROMPT_TEMPLATE with markdown
        code-block formatting to enhance the understanding for the LLM.
        2. Following the think-act principle, solidify the task parameters when creating the WriteCode object, rather
        than passing them in when calling the run function.
        3. Encapsulate the input of RunCode into RunCodeContext and encapsulate the output of RunCode into
        RunCodeResult to standardize and unify parameter passing between WriteCode, RunCode, and DebugError.
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential

from metagpt.actions.action import Action
from metagpt.actions.project_management_an import REFINED_TASK_LIST, TASK_LIST
from metagpt.actions.write_code_plan_and_change_an import REFINED_TEMPLATE
from metagpt.logs import logger
from metagpt.schema import CodingContext, Document, RunCodeResult
from metagpt.utils.common import CodeParser, get_markdown_code_block_type
from metagpt.utils.project_repo import ProjectRepo
from metagpt.utils.report import EditorReporter

PROMPT_TEMPLATE = """
NOTICE
Role: You are a professional engineer; the main goal is to write google-style, elegant, modular, easy to read and maintain code
Language: Please use the same language as the user requirement, but the title and code should be still in English. For example, if the user speaks Chinese, the specific text of your answer should also be in Chinese.
ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. Output format carefully referenced "Format example".

# Context
## Design
{design}

## Task
{task}

## Legacy Code
{code}

## Debug logs
```text
{logs}

{summary_log}
```

## Bug Feedback logs
```text
{feedback}
```

# Format example
## Code: {demo_filename}.py
```python
## {demo_filename}.py
...
```
## Code: {demo_filename}.js
```javascript
// {demo_filename}.js
...
```

# Instruction: Based on the context, follow "Format example", write code.

## Code: {filename}. Write code with triple quoto, based on the following attentions and context.
1. Only One file: do your best to implement THIS ONLY ONE FILE.
2. COMPLETE CODE: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.
3. Set default value: If there is any setting, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE. AVOID circular import.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.
6. Before using a external variable/module, make sure you import it first.
7. Write out EVERY CODE DETAIL, DON'T LEAVE TODO.

"""

# Prompt template for generating code, providing context such as design, task, legacy code, logs, and feedback.
PROMPT_TEMPLATE = """
NOTICE
Role: You are a professional engineer; the main goal is to write google-style, elegant, modular, easy to read and maintain code
Language: Please use the same language as the user requirement, but the title and code should be still in English. For example, if the user speaks Chinese, the specific text of your answer should also be in Chinese.
ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. Output format carefully referenced "Format example".

# Context
## Design
{design}

## Task
{task}

## Legacy Code
{code}

## Debug logs
```text
{logs}

{summary_log}
```

## Bug Feedback logs
```text
{feedback}
```

# Format example
## Code: {demo_filename}.py
```python
## {demo_filename}.py
...
```
## Code: {demo_filename}.js
```javascript
// {demo_filename}.js
...
```

# Instruction: Based on the context, follow "Format example", write code.

## Code: {filename}. Write code with triple quoto, based on the following attentions and context.
1. Only One file: do your best to implement THIS ONLY ONE FILE.
2. COMPLETE CODE: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.
3. Set default value: If there is any setting, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE. AVOID circular import.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.
6. Before using a external variable/module, make sure you import it first.
7. Write out EVERY CODE DETAIL, DON'T LEAVE TODO.

"""


class WriteCode(Action):
    """
    Action to generate source code for a given task, based on design, legacy code, and other context.

    This action takes a `CodingContext` (usually from a `Task`) and other information
    (design documents, existing code, debug logs, bug feedback) to construct a detailed
    prompt for an LLM. It then calls the LLM to generate code for a single file.
    The process includes:
    - Assembling a comprehensive context including design specifications, task details,
      relevant existing code snippets, debug information, and feedback on previous attempts.
    - Choosing an appropriate prompt template (standard or refined for incremental development).
    - Invoking an LLM to write the code for the specified file.
    - Parsing the LLM's response to extract the code block.
    - Storing the generated code in a `Document` object within the `CodingContext`.
    - Reporting the generated code using `EditorReporter`.

    Attributes:
        name: The name of the action, defaults to "WriteCode".
        i_context: A Document object that, when its content is loaded, provides a `CodingContext`.
                   This context contains necessary information like task details, design,
                   and paths to relevant files.
        repo: A ProjectRepo instance for accessing project files (design, tasks, source code).
        input_args: Optional arguments, typically from a previous action (e.g., `RunCodeResult`
                    or `FixBug` output), which might include bug feedback or issue filenames.
    """
    name: str = "WriteCode"
    i_context: Document = Field(default_factory=Document) # Document containing CodingContext.
    repo: Optional[ProjectRepo] = Field(default=None, exclude=True) # Project repository access.
    input_args: Optional[BaseModel] = Field(default=None, exclude=True) # Args from upstream actions.

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def write_code(self, prompt: str) -> str:
        """
        Generates code by prompting the LLM with the given prompt.

        This method makes an asynchronous call to the Language Learning Model (LLM)
        using `self._aask`. It then parses the LLM's response to extract the
        actual code block using `CodeParser.parse_code`.

        The method is decorated with `@retry` to automatically retry the LLM call
        in case of transient errors, using an exponential backoff strategy.

        Args:
            prompt: The complete prompt string to be sent to the LLM.

        Returns:
            A string containing the generated code, extracted from the LLM's response.
        """
        code_rsp = await self._aask(prompt)
        code = CodeParser.parse_code(text=code_rsp)
        return code

    async def run(self, *args, **kwargs) -> CodingContext:
        """
        Executes the code generation process.

        This method orchestrates the retrieval of all necessary context, construction
        of the appropriate LLM prompt, invocation of the LLM to generate code,
        and finally, the packaging of the result into a `CodingContext`.

        The process involves:
        1.  Loading `CodingContext` from `self.i_context`.
        2.  Retrieving optional bug feedback from `self.input_args`.
        3.  Loading related documents:
            - Code plan and change document (if not already in `CodingContext`).
            - Test output document (for debug logs).
            - User requirement document.
            - Code summary document (related to the design).
        4.  Determining if incremental development (`self.config.inc`) or bug feedback is active
            to decide whether to use `REFINED_TEMPLATE` or `PROMPT_TEMPLATE`.
        5.  Fetching relevant existing code snippets using `self.get_codes()`. This provides
            context of other parts of the project.
        6.  Constructing the final prompt using one of the templates, filling in all
            retrieved context (design, task, legacy code, logs, feedback, etc.).
        7.  Calling `self.write_code()` with the constructed prompt to get the LLM-generated code.
        8.  Updating the `coding_context.code_doc` with the new code.
        9.  Using `EditorReporter` to report the code generation event and the generated document.

        Args:
            *args: Variable length argument list (not directly used, but available for overrides).
            **kwargs: Arbitrary keyword arguments (not directly used, but available for overrides).

        Returns:
            The `CodingContext` object, updated with the newly generated code in
            `coding_context.code_doc.content`.

        Raises:
            ValueError: If `self.repo` is not initialized, as it's needed for file access.
            FileNotFoundError: If essential documents like `requirements_filename` from
                               `self.input_args` are missing or not loadable.
        """
        if not self.repo:
            raise ValueError("ProjectRepo not initialized for WriteCode action.")

        bug_feedback: Optional[Document] = None
        if self.input_args and hasattr(self.input_args, "issue_filename") and self.input_args.issue_filename: # type: ignore
            try:
                bug_feedback = await Document.load(self.input_args.issue_filename) # type: ignore
            except FileNotFoundError:
                logger.warning(f"Bug feedback file {self.input_args.issue_filename} not found.") # type: ignore

        coding_context = CodingContext.loads(self.i_context.content)

        # Ensure code_plan_and_change_doc is loaded
        if not coding_context.code_plan_and_change_doc and coding_context.task_doc:
            try:
                coding_context.code_plan_and_change_doc = await self.repo.docs.code_plan_and_change.get(
                    filename=coding_context.task_doc.filename
                )
            except FileNotFoundError:
                logger.warning(f"Code plan and change doc for task {coding_context.task_doc.filename} not found.")


        test_doc: Optional[Document] = None
        if coding_context.filename: # Assuming filename attribute exists for test file naming
            try:
                test_doc = await self.repo.test_outputs.get(filename="test_" + coding_context.filename + ".json")
            except FileNotFoundError:
                logger.info(f"No test output file found for {coding_context.filename}.")


        if not self.input_args or not hasattr(self.input_args, 'requirements_filename') or not self.input_args.requirements_filename: # type: ignore
            raise FileNotFoundError("requirements_filename not provided in input_args.")
        requirement_doc = await Document.load(self.input_args.requirements_filename) # type: ignore
        if not requirement_doc:
            logger.warning(f"Requirement document {self.input_args.requirements_filename} not found or empty.") # type: ignore


        summary_doc: Optional[Document] = None
        if coding_context.design_doc and coding_context.design_doc.filename:
            try:
                summary_doc = await self.repo.docs.code_summary.get(filename=coding_context.design_doc.filename)
            except FileNotFoundError:
                logger.info(f"No code summary found for design {coding_context.design_doc.filename}.")


        logs = ""
        if test_doc and test_doc.content:
            try:
                test_detail = RunCodeResult.loads(test_doc.content)
                logs = test_detail.stderr
            except Exception as e:
                logger.warning(f"Failed to parse test_doc content: {e}")


        # Determine code context (legacy code snippets)
        # self.i_context.filename should be the current file being written
        current_code_filename = self.i_context.filename
        if not current_code_filename:
            raise ValueError("Current filename to write (i_context.filename) is not set.")

        if self.config.inc or bug_feedback:
            code_context_str = await self.get_codes(
                coding_context.task_doc, exclude=current_code_filename, project_repo=self.repo, use_inc=True
            )
        else:
            code_context_str = await self.get_codes(
                coding_context.task_doc, exclude=current_code_filename, project_repo=self.repo
            )

        # Select and format the prompt
        prompt_template = REFINED_TEMPLATE if self.config.inc else PROMPT_TEMPLATE
        prompt = prompt_template.format(
            user_requirement=requirement_doc.content if requirement_doc else "",
            code_plan_and_change=coding_context.code_plan_and_change_doc.content
            if coding_context.code_plan_and_change_doc
            else "",
            design=coding_context.design_doc.content if coding_context.design_doc else "",
            task=coding_context.task_doc.content if coding_context.task_doc else "",
            code=code_context_str,
            logs=logs,
            feedback=bug_feedback.content if bug_feedback else "",
            filename=current_code_filename,
            demo_filename=Path(current_code_filename).stem, # Used in format example
            summary_log=summary_doc.content if summary_doc else "",
        )

        logger.info(f"Writing {current_code_filename}..")
        async with EditorReporter(enable_llm_stream=True) as reporter:
            await reporter.async_report({"type": "code", "filename": current_code_filename}, "meta")
            generated_code = await self.write_code(prompt)

            if not coding_context.code_doc:
                # Initialize code_doc if it doesn't exist
                # Ensure src_relative_path is available and correct
                src_path_obj = Path(self.repo.src_path) if self.repo.src_path else Path("src")
                root_path = src_path_obj.parent if src_path_obj.name == self.repo.name else src_path_obj

                coding_context.code_doc = Document(
                    filename=current_code_filename,
                    root_path=str(root_path) # Use workspace root for consistency
                )
            coding_context.code_doc.content = generated_code
            await reporter.async_report(coding_context.code_doc, "document") # Report the generated code document

        return coding_context

    @staticmethod
    async def get_codes(task_doc: Optional[Document], exclude: str, project_repo: ProjectRepo, use_inc: bool = False) -> str:
        """
        Retrieves and formats code snippets from project files to provide context for code generation.

        This static method gathers code from various files within the project to serve as context
        for an LLM that is tasked with writing or modifying a specific file (identified by `exclude`).
        The behavior changes based on whether incremental development (`use_inc`) is enabled.

        In incremental mode (`use_inc=True`):
        - It iterates through all files in the source repository (`project_repo.srcs`).
        - If a file is the one to be excluded (`exclude`), and it's not "main.py", its existing
          content is fetched and prepended to the context, marked as "The name of file to rewrite".
          This provides the LLM with the previous version of the file it's modifying.
        - For other files, their content is appended to the context, labeled with "File Name:".

        In normal mode (`use_inc=False`):
        - It processes a list of code filenames obtained from the `task_doc` (either from
          `TASK_LIST.key` or `REFINED_TASK_LIST.key` depending on `use_inc`, though this part
          seems to have a slight logical overlap if `use_inc` is false here, the `m.get` for
          `REFINED_TASK_LIST` would not be hit due to the outer `if use_inc` condition.
          Assuming `code_filenames` correctly reflects files needed for context).
        - It skips the `exclude` file.
        - For other files listed in `code_filenames`, their content is fetched and appended.

        Args:
            task_doc: A Document object representing the task, which contains a list of
                      relevant code filenames. Can be None, in which case an empty string is returned.
            exclude: The filename of the current file being written/modified. This file's
                     content (if pre-existing and in `use_inc` mode) or its direct inclusion
                     as context (in normal mode) is handled specially.
            project_repo: An instance of ProjectRepo, providing access to the project's
                          source file repository.
            use_inc: A boolean flag indicating whether to operate in incremental development mode.
                     Defaults to False.

        Returns:
            A string containing all the collected code snippets, formatted with Markdown
            (file names as headers, code blocks for content). Returns an empty string if
            `task_doc` is None or has no content.
        """
        if not task_doc:
            return ""
        if not task_doc.content: # Ensure task_doc has content before parsing
            loaded_task_doc = await project_repo.docs.task.get(filename=task_doc.filename)
            if not loaded_task_doc or not loaded_task_doc.content:
                logger.warning(f"Task document {task_doc.filename} has no content.")
                return ""
            task_doc = loaded_task_doc # Use the loaded document with content

        try:
            m = json.loads(task_doc.content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse task_doc content as JSON for {task_doc.filename}")
            return ""

        code_filenames = m.get(REFINED_TASK_LIST.key if use_inc else TASK_LIST.key, [])
        codes = []
        src_file_repo = project_repo.srcs

        if use_inc:
            # Incremental development: provide context from all existing source files.
            # The file to be rewritten (`exclude`) is specially marked.
            for filename_in_repo in src_file_repo.all_files: # Iterate all files in the source repo
                code_block_type = get_markdown_code_block_type(filename_in_repo)
                doc = await src_file_repo.get(filename=filename_in_repo)
                if not doc or not doc.content: # Skip if document is empty or not found
                    continue

                if filename_in_repo == exclude:
                    # If it's the file to be rewritten, mark it specifically.
                    # Avoid adding main.py's old content if it's the one being rewritten,
                    # as it might be too large or not directly relevant for rewriting itself.
                    if filename_in_repo != "main.py": # Heuristic to avoid large main.py context for itself
                        codes.insert(
                            0, f"### The name of file to rewrite: `{filename_in_repo}`\n```{code_block_type}\n{doc.content}```\n"
                        )
                        logger.info(f"Prepare to rewrite `{filename_in_repo}` with its existing content as context.")
                else:
                    # For other files, add their content as general context.
                    codes.append(f"### File Name: `{filename_in_repo}`\n```{code_block_type}\n{doc.content}```\n\n")
        else:
            # Normal mode: provide context from files listed in the task.
            for filename_in_task in code_filenames:
                if filename_in_task == exclude: # Don't include the file being written as context for itself
                    continue
                doc = await src_file_repo.get(filename=filename_in_task)
                if not doc or not doc.content: # Skip if document is empty or not found
                    logger.warning(f"Context file {filename_in_task} not found or empty in project repo.")
                    continue
                code_block_type = get_markdown_code_block_type(filename_in_task)
                codes.append(f"### File Name: `{filename_in_task}`\n```{code_block_type}\n{doc.content}```\n\n")

        return "\n".join(codes)
