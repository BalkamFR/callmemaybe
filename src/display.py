from enum import Enum
from typing import Any
import time


class Color(Enum):
    """ANSI escape codes used to style terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    GRAY = "\033[90m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


def print_animation(text: str, time_s: float) -> None:
    """Print text one character at a time with a delay between each."""
    for char in text:
        print(char, end="", flush=True)
        time.sleep(time_s)


def print_separator(number: int, color: Color = Color.GREEN) -> None:
    """Print a horizontal rule made of ``number`` colored dashes."""
    print()
    prefix = f"{Color.BOLD.value}{color.value}"
    for _ in range(number):
        print(f"{prefix}-{Color.RESET.value}", flush=True, end="")
        time.sleep(0.004)
    print()
    print()


def print_information(
        model: str,
        input: str,
        output: str,
        functions_definition: str) -> None:
    """Print the run configuration (model and file paths) at startup."""
    print_separator(200, color=Color.CYAN)
    label = f"{Color.BOLD.value}{Color.CYAN.value}"
    reset = Color.RESET.value
    print_animation(f"{label}Model:{reset} {model}", 0.01)
    print()
    print_animation(
        f"{label}Functions definition:{reset} {functions_definition}", 0.01
    )
    print()
    print_animation(f"{label}Input:{reset} {input}", 0.01)
    print()
    print_animation(f"{label}Output:{reset} {output}", 0.01)
    print()
    print_separator(200, color=Color.CYAN)


def print_color(answer: dict[str, Any]) -> None:
    """Print one prompt/answer result, colored green or red on error.

    Args:
        answer: A result entry with a ``prompt`` key and either
            ``name``/``parameters`` on success or an ``error`` key.
    """
    print_separator(200)
    prompt = answer.get("prompt", "")
    answer_clean = {
        key: value for key, value in answer.items() if key != "prompt"
    }

    is_error = "error" in answer
    color = Color.RED if is_error else Color.YELLOW
    bold_prompt = (
        f"{Color.BOLD.value}{Color.MAGENTA.value}Question:"
        f"{Color.MAGENTA.value} {prompt}{Color.RESET.value} "
    )
    bold_answer_label = (
        f"\n{Color.BOLD.value}{color.value}Answer:{Color.RESET.value}"
    )
    text = f"{color.value} {answer_clean}{Color.RESET.value}"
    print_animation(bold_prompt, 0.014)
    time.sleep(0.8)
    print_animation(bold_answer_label, 0.02)
    print_animation(text, 0.02)
    print()
