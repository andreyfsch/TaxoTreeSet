"""
Filtragem de contêineres administrativos do NCBI Taxonomy.

Identifica nós que são bolsões de curadoria (unclassified X, environmental samples,
incertae sedis, isolates clínicos) em vez de táxons biológicos legítimos. Esses
nós são pulados na construção da árvore taxonômica: suas sequências escalam para
o próximo ancestor válido na lineage.
"""
import json
import logging
import re
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("TaxoTreeSet.IO.NoiseFilter")


class NoiseFilter:
    """Avaliador de padrões de noise baseado em regex compilados."""

    def __init__(self, config_path: str = "configs/noise_patterns.json"):
        self.config_path = Path(config_path)
        self._compiled_patterns: list[tuple[re.Pattern, str]] = []
        self._rank_blacklist: set[str] = set()
        self._stats = {"name_hits": 0, "rank_hits": 0, "evaluated": 0}
        self._load()

    def _load(self) -> None:
        if not self.config_path.exists():
            logger.warning(
                f"Arquivo de padrões não encontrado em {self.config_path}. "
                f"Filtragem desabilitada — todos os nós passam."
            )
            return

        with self.config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        for entry in cfg.get("patterns", []):
            regex = entry.get("regex")
            desc = entry.get("description", "")
            if not regex:
                continue
            try:
                compiled = re.compile(regex, re.IGNORECASE)
            except re.error as e:
                logger.error(f"Regex inválido '{regex}' em {self.config_path}: {e}")
                continue
            self._compiled_patterns.append((compiled, desc))

        rank_cfg = cfg.get("rank_blacklist", {})
        self._rank_blacklist = {r.lower() for r in rank_cfg.get("ranks", [])}

        logger.info(
            f"NoiseFilter carregado: {len(self._compiled_patterns)} regex de nome, "
            f"{len(self._rank_blacklist)} ranks bloqueados"
        )

    def is_noise(self, scientific_name: str, rank: str = "") -> bool:
        """
        Retorna True se o nó for um contêiner administrativo que deve ser pulado.

        Verifica primeiro o rank (mais barato) e depois cada regex de nome.
        Mantém contadores internos para diagnóstico via `stats()`.
        """
        self._stats["evaluated"] += 1

        # Rank check (barato, faz primeiro)
        if rank and rank.lower() in self._rank_blacklist:
            self._stats["rank_hits"] += 1
            return True

        # Name check (regex)
        if not scientific_name:
            return False
        for pattern, _desc in self._compiled_patterns:
            if pattern.search(scientific_name):
                self._stats["name_hits"] += 1
                return True

        return False

    def explain(self, scientific_name: str, rank: str = "") -> str | None:
        """
        Retorna a descrição do primeiro padrão que casa, ou None.
        Útil para logs de debug.
        """
        if rank and rank.lower() in self._rank_blacklist:
            return f"rank '{rank}' está na blacklist"
        if not scientific_name:
            return None
        for pattern, desc in self._compiled_patterns:
            if pattern.search(scientific_name):
                return f"casou /{pattern.pattern}/: {desc}"
        return None

    def stats(self) -> dict:
        """Retorna estatísticas acumuladas para logging final."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        self._stats = {"name_hits": 0, "rank_hits": 0, "evaluated": 0}