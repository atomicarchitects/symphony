from typing import Iterable, Dict, Sequence
import abc

from symphony import datatypes


class InMemoryDataset(abc.ABC):
    """Abstract base class for in-memory datasets."""

    def num_species(self) -> int:
        """Return the number of atom types."""
        return len(self.species_to_atomic_numbers())

    @abc.abstractmethod
    def get_atomic_numbers(self) -> Sequence[int]:
        """Returns a sorted list of the atomic numbers observed in the dataset."""

    @abc.abstractmethod
    def species_to_atomic_numbers(self) -> Dict[int, int]:
        """Returns a dictionary mapping species indices to atomic numbers."""

    @abc.abstractmethod
    def structures(self) -> Iterable[datatypes.Structures]:
        """Return a list of all completed structures."""

    @abc.abstractmethod
    def split_indices(self) -> Dict[str, Sequence[int]]:
        """Return a dictionary of split indices."""
        pass
