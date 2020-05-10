import numpy as np
import math
from controller.enums import PartEnum


def part_id_to_name(part_id):
    if part_id == PartEnum.BACKCOVER.value:
        return "Back cover"
    elif part_id == PartEnum.PCB.value:
        return "PCB"
    elif part_id == PartEnum.FUSE.value:
        return "Fuse"
    elif part_id == PartEnum.BLACKCOVER.value:
        return "Black front cover"
    elif part_id == PartEnum.WHITECOVER.value:
        return "White front cover"
    elif part_id == PartEnum.BLUECOVER.value:
        return "Blue front cover"
