from abc import ABC, abstractmethod
from typing import Dict, Tuple

from istream_player.core.module import ModuleInterface
from istream_player.models import AdaptationSet


class NESController(ModuleInterface, ABC):
    def __init__(self):
        self._min_bitrate_representations: Dict[int, int] = {}

    @abstractmethod
    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Update the representation selections
        """
        pass

    def update_selection_lowest(self, adaptation_sets: Dict[int, AdaptationSet]) -> Tuple[Dict[int, int], Dict[int, int]]:
        download_actions = {}
        enhance_actions = {}
        for adaptation_set in adaptation_sets.values():
            download_actions[adaptation_set.id] = self._find_representation_id_of_lowest_bitrate(adaptation_set)
            enhance_actions[adaptation_set.id] = 0  # todo: meaningful index
        return download_actions, enhance_actions

    def _find_representation_id_of_lowest_bitrate(self, adaptation_set: AdaptationSet) -> int:
        """
        Find the representation ID with the lowest bitrate in a given adaptation set
        Parameters
        ----------
        adaptation_set:
            The adaptation set to process

        Returns
        -------
            The representation ID with the lowest bitrate
        """
        if adaptation_set.id in self._min_bitrate_representations:
            return self._min_bitrate_representations[adaptation_set.id]

        representations = list(adaptation_set.representations.values())
        min_id = representations[0].id
        min_bandwidth = representations[0].bandwidth

        for representation in representations:
            if min_bandwidth is None:
                min_bandwidth = representation.bandwidth
                min_id = representation.id
            elif representation.bandwidth < min_bandwidth:
                min_bandwidth = representation.bandwidth
                min_id = representation.id
        self._min_bitrate_representations[adaptation_set.id] = min_id

        return min_id
