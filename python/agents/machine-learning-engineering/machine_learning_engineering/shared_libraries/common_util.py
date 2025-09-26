"""Common utility functions."""

import os
import random
import shutil
from typing import List, Union

import numpy as np
import torch

from google.adk.agents import callback_context as callback_context_module
from google.adk.models import llm_response


def get_text_from_response(
    response: llm_response.LlmResponse,
) -> str:
  """Extracts concatenated text from all textual parts of the response."""
  final_text = ""
  if response.content and response.content.parts:
    for part in response.content.parts:
        text = getattr(part, "text", None)
        if isinstance(text, str):
            final_text += text
  return final_text


def set_random_seed(seed: int) -> None:
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def copy_file(source_file_path: str, destination_dir: str) -> None:
    """Copies a file to the specified directory."""
    if not os.path.isdir(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
    shutil.copy2(source_file_path, destination_dir)


def get_run_guidance(
    context: Union[
        callback_context_module.CallbackContext,
        callback_context_module.ReadonlyContext,
    ],
    section: str,
) -> str:
    """Aggregates run guidance for a pipeline section."""

    guidance = {}
    if hasattr(context, "state") and isinstance(context.state, dict):
        guidance = context.state.get("run_guidance", {})
    if not isinstance(guidance, dict):
        return ""
    global_notes = guidance.get("global_notes", "")
    section_notes = guidance.get(section, "")
    notes = []
    if isinstance(global_notes, str) and global_notes.strip():
        notes.append(global_notes.strip())
    if isinstance(section_notes, str) and section_notes.strip():
        notes.append(section_notes.strip())
    return "\n".join(notes)


def extract_guidance_requirements(text: str) -> List[str]:
    """Pulls requirement-style bullet lines from guidance text."""

    if not isinstance(text, str):
        return []
    requirements: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line[0].isdigit() and line[1:2] in {".", ")"}:
            cleaned = line.split(None, 1)
            if len(cleaned) == 2:
                requirements.append(cleaned[1].strip())
                continue
        if line.startswith(('-', '*')):
            requirements.append(line[1:].strip())
            continue
    return requirements
