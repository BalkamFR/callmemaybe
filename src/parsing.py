import json
from typing import Any
import sys

from pydantic import BaseModel


class Config(BaseModel):
    """Runtime configuration: file paths and model name.

    All fields have a default so the program works out of the box;
    each one can be overridden from the command line.
    """

    functions_definition: str = "data/input/functions_definition.json"
    input: str = "data/input/function_calling_tests.json"
    output: str = "data/output/function_calling_results.json"
    model: str = "Qwen/Qwen3-0.6B"

    @classmethod
    def parse_arguments(cls) -> "Config":
        """Build a Config from ``sys.argv``, falling back to defaults.

        Recognized flags: ``--functions_definition``, ``--input``,
        ``--output``, ``--model``. Any other argument is ignored.

        Returns:
            A Config with defaults overridden by matching flags.
        """
        config = cls()

        args = sys.argv[1:]

        for i in range(len(args) - 1):
            if args[i] == "--functions_definition":
                config.functions_definition = args[i + 1]
            elif args[i] == "--input":
                config.input = args[i + 1]
            elif args[i] == "--output":
                config.output = args[i + 1]
            elif args[i] == "--model":
                config.model = args[i + 1]

        return config

    def load_tools(self) -> dict[str, dict[str, Any]]:
        """Load and index the available functions by name.

        Returns:
            A mapping from function name to its parameter schema.

        Raises:
            ValueError: If the file is missing, contains invalid
                JSON, or an entry is missing the ``name`` or
                ``parameters`` key.
        """
        try:
            with open(self.functions_definition, "r", encoding="utf-8") as f:
                tools: list[dict[str, Any]] = json.load(f)
        except FileNotFoundError as e:
            raise ValueError(
                f"functions definition file not found: "
                f"{self.functions_definition}"
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"invalid JSON in {self.functions_definition}: {e}"
            ) from e

        try:
            return {fn["name"]: fn["parameters"] for fn in tools}
        except KeyError as e:
            raise ValueError(
                f"functions_definition.json entry is missing key {e}"
            ) from e

    def load_prompt(self) -> list[str]:
        """Load the natural-language prompts to process.

        Returns:
            The list of prompt strings from the input file.

        Raises:
            ValueError: If the file is missing, contains invalid
                JSON, or an entry is missing the ``prompt`` key.
        """
        try:
            with open(self.input, "r", encoding="utf-8") as f:
                tools: list[dict[str, Any]] = json.load(f)
        except FileNotFoundError as e:
            raise ValueError(f"input file not found: {self.input}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON in {self.input}: {e}") from e

        try:
            return [fn["prompt"] for fn in tools]
        except KeyError as e:
            raise ValueError(
                f"{self.input} entry is missing key {e}"
            ) from e
