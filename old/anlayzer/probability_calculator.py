from typing import Generic, TypeVar
from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

T = TypeVar('T')

class Accumulator:
    def __init__(self):
        self.positives = 0
        self.negatives = 0

    def add_positive(self):
        self.positives += 1

    def add_negative(self):
        self.negatives += 1
    
    def current_positive_odds(self):
        return self.positives / (self.positives + self.negatives) if (self.positives + self.negatives) > 0 else None
    
    def current_negative_odds(self):
        return self.negatives / (self.positives + self.negatives) if (self.positives + self.negatives) > 0 else None


class ProbabilityCalculator(ABC, Generic[T]):
    def __init__(self):
        self.accumulator = Accumulator()

    @abstractmethod
    def matches_scenario(self, item: T) -> Optional[bool]:
        pass

    def process(self, data: T):
        matches_scenario = self.matches_scenario(data)
        if matches_scenario is None:
            return
        elif matches_scenario:
            self.accumulator.add_positive()
        else:
            self.accumulator.add_negative()

