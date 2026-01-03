# llm_interpreter.py

import json
from typing import Dict, Any
import openai


import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")

# You can change this to your preferred model
MODEL_NAME = "gpt-4.1-mini"


SYSTEM_PROMPT = """You are a planning module for a robot that manipulates colored blocks on a table.

Your job:
- Read a natural language instruction from the user.
- Read a summary of the current scene (list of blocks with id, color, x_mm, y_mm).
- Output a single JSON object describing the task for the robot to execute.

You MUST follow one of these schemas:

1) Move a single block relative to another block:

{
  "task_type": "move_single",
  "source_selector": {
    "color": "<color or null>",
    "extremum": "<leftmost | rightmost | center | null>",
    "relation": "<furthest_from | closest_to | null>",
    "reference": {
      "color": "<color or null>",
      "extremum": "<leftmost | rightmost | center | null>"
    } or null
  },
  "target_selector": {
    "color": "<color or null>",
    "extremum": "<leftmost | rightmost | center | null>",
    "relation": null,
    "reference": null
  } or null,
  "placement": "<left_of | right_of | above | below | on_top | null>",
  "offset_mm": <number>
}

Examples:
- "grab the red block that is furthest away from the leftmost green block and put it to the right of the yellow block in the center"
{
  "task_type": "move_single",
  "source_selector": {
    "color": "red",
    "extremum": null,
    "relation": "furthest_from",
    "reference": {
      "color": "green",
      "extremum": "leftmost"
    }
  },
  "target_selector": {
    "color": "yellow",
    "extremum": "center",
    "relation": null,
    "reference": null
  },
  "placement": "right_of",
  "offset_mm": 30
}

- "put the leftmost blue block on top of the rightmost red block"
{
  "task_type": "move_single",
  "source_selector": {
    "color": "blue",
    "extremum": "leftmost",
    "relation": null,
    "reference": null
  },
  "target_selector": {
    "color": "red",
    "extremum": "rightmost",
    "relation": null,
    "reference": null
  },
  "placement": "on_top",
  "offset_mm": 0
}

2) Group all blocks of a certain color in a region:

{
  "task_type": "group_color",
  "group_selector": {
    "color": "<color>"
  },
  "group_region": "<left | right | center>",
  "layout": "line",
  "spacing_mm": <number>
}

Example:
- "put all the blue blocks together on the left"
{
  "task_type": "group_color",
  "group_selector": {
    "color": "blue"
  },
  "group_region": "left",
  "layout": "line",
  "spacing_mm": 30
}

Rules:
- ONLY output JSON, with no explanation or extra text.
- If the instruction is ambiguous, make a reasonable choice and still output JSON.
- Use colors exactly as: "red", "blue", "green", "yellow".
- For offsets, use a default of 30 mm if not specified.
"""


def interpret_prompt(prompt: str, scene_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the LLM to convert (prompt + scene) into a structured task JSON.
    `scene_summary` should be something like:
        {"blocks": [{"id":0,"color":"red","x_mm":300,"y_mm":10}, ...]}
    Returns a Python dict.
    """

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    user_message = {
        "role": "user",
        "content": json.dumps(
            {
                "instruction": prompt,
                "scene": scene_summary,
            }
        ),
    }

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            user_message,
        ],
        temperature=0.1,
    )

    raw_text = response.choices[0].message.content.strip()

    # The model should output pure JSON. Parse it.
    try:
        task = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to extract JSON if there's any extra text (fallback)
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            task = json.loads(raw_text[start : end + 1])
        else:
            raise ValueError(f"LLM output was not valid JSON: {raw_text}")

    return task
