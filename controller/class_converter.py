from enums import PartEnum


def convert_from_part_id(part_id):
    if part_id == PartEnum.FUSE.value:
        return 'Fuse', 'Fuse'
    elif part_id == PartEnum.BACKCOVER.value:
        return 'BottomCover', 'BottomCoverFlipped'
    elif part_id == PartEnum.WHITECOVER.value:
        return 'WhiteCover', 'WhiteCoverFlipped'
    elif part_id == PartEnum.BLUECOVER.value:
        return 'BlueCover', 'BlueCoverFlipped'
    elif part_id == PartEnum.BLACKCOVER.value:
        return 'BlackCover', 'BlackCoverFlipped'
    elif part_id == PartEnum.PCB.value:
        return 'PCB', 'PCBFlipped'
    else:
        print("[W] Could not convert class_id")
        return -1, -1


def convert_to_part_id(class_name):
    if class_name == 'Fuse':
        return PartEnum.FUSE.value
    elif class_name == 'BottomCover':
        return PartEnum.BACKCOVER.value
    elif class_name == 'BottomCoverFlipped':
        return PartEnum.BACKCOVER_FLIPPED.value
    elif class_name == 'WhiteCover':
        return PartEnum.WHITECOVER.value
    elif class_name == 'WhiteCoverFlipped':
        return PartEnum.WHITECOVER_FLIPPED.value
    elif class_name == 'BlueCover':
        return PartEnum.BLUECOVER.value
    elif class_name == 'BlueCoverFlipped':
        return PartEnum.BLUECOVER_FLIPPED.value
    elif class_name == 'BlackCover':
        return PartEnum.BLACKCOVER.value
    elif class_name == 'BlackCoverFlipped':
        return PartEnum.BLACKCOVER_FLIPPED.value
    elif class_name == 'PCB':
        return PartEnum.PCB.value
    elif class_name == 'PCBFlipped':
        return PartEnum.PCB_FLIPPED.value
    else:
        return PartEnum.INVALID.value

