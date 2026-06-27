"""Task-rule synthesis backend for NeuroGolf optimizer."""

from .candidate_program import ProgramCandidate
from .program_search import search_programs

__all__ = ["ProgramCandidate", "search_programs"]
