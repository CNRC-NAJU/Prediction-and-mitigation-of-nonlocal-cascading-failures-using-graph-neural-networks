from enum import Enum, auto


class Stage(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


if __name__ == "__main__":
    print("This is module stage")
