"""Administrative container filtering for the NCBI Taxonomy.

This module provides the NoiseFilter class, which identifies and excludes
non-biological administrative nodes from the NCBI Taxonomy during
cascaded tree construction. The filter operates on two complementary
signals:

1. Name-based regex patterns: scientific names matching configured
   patterns are considered noise. Examples include 'unclassified X'
   containers, 'environmental samples', 'incertae sedis' clades, and
   clinical isolate groupings.

2. Rank-based blacklist: nodes carrying ranks below the species level
   (serotype, subtype, strain, etc.) are filtered regardless of their
   scientific name.

When a node is identified as noise, it is skipped during tree
construction and its associated sequences are reassigned to the next
valid ancestor in the lineage. The filter is configured via a JSON
file (default ``configs/noise_patterns.json``) whose structure is
documented in ``configs/noise_patterns.schema.json``.

Typical usage::

    from taxotreeset.io.noise_filter import NoiseFilter

    noise_filter = NoiseFilter()
    if noise_filter.is_noise("unclassified Caudoviricetes", rank="no_rank"):
        # Skip this node; reassign its sequences upward.
        pass

    print(noise_filter.stats())
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("TaxoTreeSet.IO.NoiseFilter")


class NoiseFilter:
    """Filter that identifies administrative containers in the NCBI Taxonomy.

    Loads regex patterns and rank filters from a JSON configuration file
    and evaluates taxonomic nodes against them. Maintains internal
    counters of evaluations and hits for diagnostic reporting.

    The filter is permissive when the configuration file is missing:
    instead of raising, it logs a warning and lets all nodes pass.
    This avoids breaking the pipeline during initial setup or in
    test environments where the configuration is not yet available.

    Attributes:
        config_path: Path to the noise patterns configuration JSON.
    """

    _DEFAULT_CONFIG_PATH = "configs/noise_patterns.json"
    _STATS_KEY_EVALUATED = "evaluated"
    _STATS_KEY_NAME_HITS = "name_hits"
    _STATS_KEY_RANK_HITS = "rank_hits"

    def __init__(self, config_path: str = _DEFAULT_CONFIG_PATH) -> None:
        """Initialize the filter and load configured patterns from disk.

        Args:
            config_path: Filesystem path to the noise patterns JSON file.
                Defaults to ``configs/noise_patterns.json``.
        """
        self.config_path: Path = Path(config_path)
        self._compiled_patterns: list[tuple[re.Pattern[str], str]] = []
        self._rank_blacklist: set[str] = set()
        self._stats: dict[str, int] = self._fresh_stats()
        self._load_configuration()

    @staticmethod
    def _fresh_stats() -> dict[str, int]:
        """Return a zero-initialized statistics dictionary.

        Returns:
            Dictionary with the three counters reset to zero.
        """
        return {
            NoiseFilter._STATS_KEY_EVALUATED: 0,
            NoiseFilter._STATS_KEY_NAME_HITS: 0,
            NoiseFilter._STATS_KEY_RANK_HITS: 0,
        }

    def _load_configuration(self) -> None:
        """Load and compile regex patterns and rank filters from disk.

        When the configuration file does not exist, the filter remains
        in a permissive state where ``is_noise`` always returns False.
        A warning is logged to make this situation visible.

        Invalid regex patterns are logged and skipped; the filter
        continues loading the remaining valid patterns.
        """
        if not self.config_path.exists():
            logger.warning(
                f"Noise patterns file not found at {self.config_path}. "
                "Filtering disabled - all nodes will pass."
            )
            return

        with self.config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)

        self._compile_name_patterns(config.get("name_patterns", []))
        self._load_rank_blacklist(config.get("rank_blacklist", {}))

        logger.info(
            f"NoiseFilter loaded: {len(self._compiled_patterns)} name "
            f"regexes, {len(self._rank_blacklist)} blocked ranks."
        )

    def _compile_name_patterns(
        self,
        pattern_entries: list[dict[str, str]],
    ) -> None:
        """Compile regex patterns from configuration entries.

        Each entry must contain a 'regex' field; the 'description'
        field is optional but recommended. Invalid regexes are logged
        and silently skipped to avoid failing the entire load.

        Args:
            pattern_entries: List of pattern dictionaries from the JSON
                configuration, each with 'regex' and 'description' keys.
        """
        for entry in pattern_entries:
            regex_source = entry.get("regex")
            description = entry.get("description", "")
            if not regex_source:
                continue
            try:
                compiled = re.compile(regex_source, re.IGNORECASE)
            except re.error as exc:
                logger.error(
                    f"Invalid regex '{regex_source}' in {self.config_path}: {exc}"
                )
                continue
            self._compiled_patterns.append((compiled, description))

    def _load_rank_blacklist(self, rank_config: dict) -> None:
        """Populate the rank blacklist from configuration.

        Ranks are lowercased for case-insensitive matching against
        node ranks during evaluation.

        Args:
            rank_config: The 'rank_blacklist' object from the JSON
                configuration, containing a 'ranks' list.
        """
        ranks = rank_config.get("ranks", [])
        self._rank_blacklist = {rank.lower() for rank in ranks}

    def is_noise(self, scientific_name: str, rank: str = "") -> bool:
        """Determine whether a taxonomic node should be filtered as noise.

        Rank check is performed first because it is cheaper than regex
        matching against a scientific name. If the rank is blacklisted,
        no name pattern matching is attempted.

        Increments the internal counters for diagnostic reporting via
        ``stats()``.

        Args:
            scientific_name: The scientific name of the taxonomic node.
                Empty strings are treated as non-noise.
            rank: The NCBI rank string of the node (e.g., 'species',
                'genus', 'strain'). Defaults to empty string.

        Returns:
            True if the node should be filtered out, False otherwise.

        Example:
            >>> filter = NoiseFilter()
            >>> filter.is_noise("unclassified Caudoviricetes")
            True
            >>> filter.is_noise("Escherichia coli K-12 substr. MG1655", rank="strain")
            True
            >>> filter.is_noise("Bacillus subtilis", rank="species")
            False
        """
        self._stats[self._STATS_KEY_EVALUATED] += 1

        if rank and rank.lower() in self._rank_blacklist:
            self._stats[self._STATS_KEY_RANK_HITS] += 1
            return True

        if not scientific_name:
            return False

        for pattern, _description in self._compiled_patterns:
            if pattern.search(scientific_name):
                self._stats[self._STATS_KEY_NAME_HITS] += 1
                return True

        return False

    def explain(
        self,
        scientific_name: str,
        rank: str = "",
    ) -> Optional[str]:
        """Return a human-readable explanation of why a node was filtered.

        Useful for debug logging. Returns the description of the first
        matching pattern (or rank blacklist hit), or None if the node
        does not match any filter.

        This method does not increment statistics counters; it is
        purely informational.

        Args:
            scientific_name: The scientific name of the taxonomic node.
            rank: The NCBI rank string of the node.

        Returns:
            A descriptive string identifying the matching pattern, or
            None if no pattern matches.

        Example:
            >>> filter = NoiseFilter()
            >>> filter.explain("unclassified Caudoviricetes")
            "matched /^unclassified\\\\b/: NCBI 'unclassified X' containers..."
        """
        if rank and rank.lower() in self._rank_blacklist:
            return f"rank '{rank}' is on the blacklist"

        if not scientific_name:
            return None

        for pattern, description in self._compiled_patterns:
            if pattern.search(scientific_name):
                return f"matched /{pattern.pattern}/: {description}"

        return None

    def stats(self) -> dict[str, int]:
        """Return a snapshot of accumulated filter statistics.

        Returns:
            Dictionary with three counters:
                - 'evaluated': total nodes examined
                - 'name_hits': nodes filtered by name pattern
                - 'rank_hits': nodes filtered by rank blacklist
        """
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset all internal counters to zero.

        Useful when running multiple census or generation passes in the
        same process and wanting separate statistics per run.
        """
        self._stats = self._fresh_stats()
