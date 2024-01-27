from abc import ABC
from istream_player.core.module import ModuleInterface


class Renderer(ModuleInterface, ABC):
    """
    Video renderer
    """