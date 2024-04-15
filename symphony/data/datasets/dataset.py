from typing import Iterable, Dict, Sequence
import abc

from symphony import datatypes


class InMemoryDataset(abc.ABC):
    """Abstract base class for in-memory datasets."""

    @abc.abstractmethod
    def species_to_atom_types() -> Dict[int, str]:
        """Return the mapping from (integer) species to atom types."""

    def num_species(self) -> int:
        return len(self.species_to_atom_types())

    @abc.abstractmethod
    def structures(self) -> Iterable[datatypes.Fragments]:
        """Return a list of all completed structures."""

    @abc.abstractmethod
    def split_indices(self) -> Dict[str, Sequence[int]]:
        """Return a dictionary of split indices."""
        pass
