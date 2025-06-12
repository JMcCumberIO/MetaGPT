#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 22:12
@Author  : alexanderwu
@File    : write_test.py
@Modified By: mashenquan, 2023-11-27. Following the think-act principle, solidify the task parameters when creating the
        WriteTest object, rather than passing them in when calling the run function.
"""

from typing import Optional

from metagpt.actions.action import Action
from metagpt.const import TEST_CODES_FILE_REPO
from metagpt.logs import logger
from metagpt.schema import Document, TestingContext
from metagpt.utils.common import CodeParser

PROMPT_TEMPLATE = """
NOTICE
1. Role: You are a QA engineer; the main goal is to design, develop, and execute PEP8 compliant, well-structured, maintainable test cases and scripts for Python 3.9. Your focus should be on ensuring the product quality of the entire project through systematic testing.
2. Requirement: Based on the context, develop a comprehensive test suite that adequately covers all relevant aspects of the code file under review. Your test suite will be part of the overall project QA, so please develop complete, robust, and reusable test cases.
3. Attention1: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the test case or script.
4. Attention2: If there are any settings in your tests, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE.
5. Attention3: YOU MUST FOLLOW "Data structures and interfaces". DO NOT CHANGE ANY DESIGN. Make sure your tests respect the existing design and ensure its validity.
6. Think before writing: What should be tested and validated in this document? What edge cases could exist? What might fail?
7. CAREFULLY CHECK THAT YOU DON'T MISS ANY NECESSARY TEST CASES/SCRIPTS IN THIS FILE.
Attention: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the test case or script and triple quotes.
-----
## Given the following code, please write appropriate test cases using Python's unittest framework to verify the correctness and robustness of this code:
```python
{code_to_test}
```
Note that the code to test is at {source_file_path}, we will put your test code at {workspace}/tests/{test_file_name}, and run your test code from {workspace},
you should correctly import the necessary classes based on these file locations!
## {test_file_name}: Write test code with triple quote. Do your best to implement THIS ONLY ONE FILE.
"""

# Prompt template for generating unit tests. It instructs the LLM to act as a QA engineer,
# write PEP8 compliant unittest cases for Python 3.9, and pay attention to sections,
# default values, design adherence, and comprehensive test coverage.
PROMPT_TEMPLATE = """
NOTICE
1. Role: You are a QA engineer; the main goal is to design, develop, and execute PEP8 compliant, well-structured, maintainable test cases and scripts for Python 3.9. Your focus should be on ensuring the product quality of the entire project through systematic testing.
2. Requirement: Based on the context, develop a comprehensive test suite that adequately covers all relevant aspects of the code file under review. Your test suite will be part of the overall project QA, so please develop complete, robust, and reusable test cases.
3. Attention1: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the test case or script.
4. Attention2: If there are any settings in your tests, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE.
5. Attention3: YOU MUST FOLLOW "Data structures and interfaces". DO NOT CHANGE ANY DESIGN. Make sure your tests respect the existing design and ensure its validity.
6. Think before writing: What should be tested and validated in this document? What edge cases could exist? What might fail?
7. CAREFULLY CHECK THAT YOU DON'T MISS ANY NECESSARY TEST CASES/SCRIPTS IN THIS FILE.
Attention: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the test case or script and triple quotes.
-----
## Given the following code, please write appropriate test cases using Python's unittest framework to verify the correctness and robustness of this code:
```python
{code_to_test}
```
Note that the code to test is at {source_file_path}, we will put your test code at {workspace}/tests/{test_file_name}, and run your test code from {workspace},
you should correctly import the necessary classes based on these file locations!
## {test_file_name}: Write test code with triple quote. Do your best to implement THIS ONLY ONE FILE.
"""


class WriteTest(Action):
    """
    Action class for generating unit test code for a given piece of source code.

    This action takes the source code to be tested as input, along with its file path,
    and uses an LLM to generate corresponding unit tests using Python's `unittest` framework.
    The generated tests are stored in a `TestingContext`.

    Attributes:
        name: The name of the action, defaults to "WriteTest".
        i_context: An optional `TestingContext` object that holds the input source code
                   document (`code_doc`) and will be updated with the generated test
                   document (`test_doc`).
    """
    name: str = "WriteTest"
    i_context: Optional[TestingContext] = None # Context containing code_doc and for storing test_doc.

    async def write_code(self, prompt: str) -> str:
        """
        Invokes the LLM to generate test code based on the provided prompt.

        This method sends the formatted prompt to the LLM and then parses the
        response to extract the generated code block. It includes basic error
        handling in case the code parsing fails, falling back to returning the
        raw LLM response.

        Args:
            prompt: The complete prompt string to be sent to the LLM, including
                    instructions and the code to be tested.

        Returns:
            A string containing the generated test code. If parsing fails, it may
            return the raw LLM response.
        """
        code_rsp = await self._aask(prompt)

        try:
            code = CodeParser.parse_code(text=code_rsp)
        except Exception:
            # Handle the exception if needed
            logger.error(f"Can't parse the code: {code_rsp}")

            # Return code_rsp in case of an exception, assuming llm just returns code as it is and doesn't wrap it inside ```
            code = code_rsp
        return code

    async def run(self, *args, **kwargs) -> TestingContext:
        """
        Executes the test code generation process.

        This method orchestrates the generation of unit tests for the source code
        provided in `self.i_context.code_doc`. It performs the following steps:
        1.  Ensures that `self.i_context.test_doc` is initialized. If not, it creates a
            new Document object for the test file, naming it conventionally by prepending
            "test_" to the source code's filename and setting its root path to
            `TEST_CODES_FILE_REPO`.
        2.  Constructs a detailed prompt using `PROMPT_TEMPLATE`. This prompt includes:
            - The actual source code to be tested (`self.i_context.code_doc.content`).
            - The intended filename for the test script (`self.i_context.test_doc.filename`).
            - The conceptual file path of the source code within a simulated workspace
              (using `fake_root` for LLM context).
            - The conceptual workspace path from where tests would be run.
            These path details help the LLM generate correct import statements in the tests.
        3.  Calls `self.write_code(prompt)` to get the LLM-generated test code.
        4.  Sets the content of `self.i_context.test_doc` to the generated test code.
        5.  Returns the updated `self.i_context` which now includes the generated tests.

        Args:
            *args: Variable length argument list (not directly used).
            **kwargs: Arbitrary keyword arguments (not directly used).

        Returns:
            The `TestingContext` (`self.i_context`) updated with the generated test code
            in `self.i_context.test_doc.content`.

        Raises:
            ValueError: If `self.i_context` or `self.i_context.code_doc` is not properly
                        initialized before calling this method.
        """
        if not self.i_context or not self.i_context.code_doc:
            raise ValueError("TestingContext and its code_doc must be initialized before running WriteTest.")

        if not self.i_context.test_doc:
            # Construct test document filename and path if not already defined.
            test_filename = "test_" + self.i_context.code_doc.filename
            self.i_context.test_doc = Document(filename=test_filename, root_path=TEST_CODES_FILE_REPO)

        # Use a conceptual root path for the LLM to understand the project structure for imports.
        # This helps in generating correct relative imports in the test file.
        fake_root = "/data" # Represents a generic workspace root for the LLM.
        source_file_relative_path = self.i_context.code_doc.root_relative_path
        if not source_file_relative_path: # Should ideally always be set if code_doc is from a repo
            source_file_relative_path = self.i_context.code_doc.filename # Fallback

        prompt = PROMPT_TEMPLATE.format(
            code_to_test=self.i_context.code_doc.content,
            test_file_name=self.i_context.test_doc.filename,
            source_file_path=fake_root + "/" + source_file_relative_path,
            workspace=fake_root,
        )
        self.i_context.test_doc.content = await self.write_code(prompt)
        return self.i_context
