from enum import Enum


class PartEnum(Enum):
    BACKCOVER = 0
    PCB = 1
    FUSE = 2
    BLACKCOVER = 3
    WHITECOVER = 4
    BLUECOVER = 5
    BACKCOVER_FLIPPED = 6
    PCB_FLIPPED = 7
    BLACKCOVER_FLIPPED = 8
    WHITECOVER_FLIPPED = 9
    BLUECOVER_FLIPPED = 10
    INVALID = 11



class OrientationEnum(Enum):
    HORIZONTAL = 0
    VERTICAL = 1
