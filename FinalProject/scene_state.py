# scene_state.py

from dataclasses import dataclass
from typing import List, Optional
import math


@dataclass
class Block:
    id: int
    color: str
    x_px: int
    y_px: int
    width_px: int
    height_px: int
    x_mm: Optional[float] = None
    y_mm: Optional[float] = None


class SceneState:
    def __init__(self, blocks: List[Block]):
        self.blocks = blocks

    # -----------------
    # Core selectors
    # -----------------

    def get_blocks_by_color(self, color: str):
        return [b for b in self.blocks if b.color == color]

    def leftmost(self, blocks=None):
        blocks = blocks or self.blocks
        return min(blocks, key=lambda b: b.x_mm)

    def rightmost(self, blocks=None):
        blocks = blocks or self.blocks
        return max(blocks, key=lambda b: b.x_mm)

    def closest_to(self, target_block: Block, blocks=None):
        blocks = blocks or self.blocks

        def dist(b):
            return math.hypot(b.x_mm - target_block.x_mm, b.y_mm - target_block.y_mm)

        return min(blocks, key=dist)

    def furthest_from(self, target_block: Block, blocks=None):
        blocks = blocks or self.blocks

        def dist(b):
            return math.hypot(b.x_mm - target_block.x_mm, b.y_mm - target_block.y_mm)

        return max(blocks, key=dist)

    def center_block(self, color=None):
        """
        Block closest to the centroid of the workspace or of the given color group.
        """
        if color:
            blocks = self.get_blocks_by_color(color)
        else:
            blocks = self.blocks

        if not blocks:
            return None

        avg_x = sum(b.x_mm for b in blocks) / len(blocks)
        avg_y = sum(b.y_mm for b in blocks) / len(blocks)

        def dist(b):
            return (b.x_mm - avg_x) ** 2 + (b.y_mm - avg_y) ** 2

        return min(blocks, key=dist)

    # -----------------
    # Convenience helpers
    # -----------------

    def summary(self):
        """
        Return a simple summary for feeding into the LLM:
        [{'id':0, 'color':'red','pos':[x_mm,y_mm]}...]
        """
        return [
            {
                "id": b.id,
                "color": b.color,
                "x_mm": b.x_mm,
                "y_mm": b.y_mm,
            }
            for b in self.blocks
        ]
