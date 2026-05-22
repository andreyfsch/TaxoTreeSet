import os
import logging
import threading
import lmdb
import zlib

logger = logging.getLogger("TaxoTreeSet.Dataset.Utils")

# Cache de envs LMDB por processo. Detecta forks via PID e descarta
# o cache herdado, forçando reabertura local. Necessário porque
# lmdb.Environment não é fork-safe (mmap + FD nasceram no pai).
_LMDB_ENV_CACHE: dict[str, lmdb.Environment] = {}
_LMDB_CACHE_PID: int | None = None
_LMDB_CACHE_LOCK = threading.Lock()

def _pool_worker_initializer():
    """
    Roda uma vez no início de cada worker forked. Zera o cache de LMDB
    herdado do processo pai para evitar compartilhamento de handles mmap.
    """
    from src.taxotreeset.dataset import utils
    utils._LMDB_ENV_CACHE = {}
    utils._LMDB_CACHE_PID = None

def _get_lmdb_env(lmdb_path: str) -> lmdb.Environment:
    """
    Retorna um env LMDB cacheado por (processo, path). Em workers forked,
    detecta a mudança de PID e abre um novo handle em vez de reaproveitar
    o herdado do processo pai.
    """
    global _LMDB_ENV_CACHE, _LMDB_CACHE_PID
    current_pid = os.getpid()

    with _LMDB_CACHE_LOCK:
        if _LMDB_CACHE_PID != current_pid:
            if _LMDB_CACHE_PID is not None:
                logger.debug(f"[LMDB-FORK] PID mudou {_LMDB_CACHE_PID} → {current_pid}, descartando cache")
            _LMDB_ENV_CACHE = {}
            _LMDB_CACHE_PID = current_pid
            # Fork detectado. NÃO fechamos os envs herdados explicitamente:
            # o pai pode ainda estar usando o mesmo objeto Python no espaço dele,
            # e o LMDB do filho irá liberar o FD duplicado quando o GC rodar.
            # Apenas soltamos as referências para forçar reabertura neste PID.
            _LMDB_ENV_CACHE = {}
            _LMDB_CACHE_PID = current_pid

        env = _LMDB_ENV_CACHE.get(lmdb_path)
        if env is None:
            if not os.path.exists(lmdb_path):
                logger.error(f"LMDB execution database path does not exist: {lmdb_path}")
                raise FileNotFoundError(lmdb_path)
            env = lmdb.open(
                lmdb_path,
                readonly=True,
                lock=False,
                max_dbs=0,
                readahead=False,
            )
            logger.debug(f"[LMDB-OPEN] PID={current_pid} path={lmdb_path}")
            _LMDB_ENV_CACHE[lmdb_path] = env

    return env

def _read_single_sequence(lmdb_path: str, header_id: str) -> str:
    """
    Busca uma sequência comprimida no LMDB pelo header_id e descompacta.
    Retorna string vazia se a chave não existir ou der erro de I/O.
    """
    try:
        env = _get_lmdb_env(lmdb_path)
    except FileNotFoundError:
        return ""

    try:
        with env.begin(write=False) as txn:
            compressed_seq = txn.get(header_id.encode("utf-8"))
    except lmdb.Error as e:
        logger.error(f"LMDB I/O ao buscar '{header_id}' em {lmdb_path}: {e}")
        return ""

    if not compressed_seq:
        logger.warning(f"Header '{header_id}' não encontrado em {lmdb_path}")
        return ""

    try:
        return zlib.decompress(compressed_seq).decode("utf-8")
    except (zlib.error, UnicodeDecodeError) as e:
        logger.error(f"Falha descompactando '{header_id}': {e}")
        return ""

def _get_fasta_sequence_length(lmdb_path: str, header_id: str) -> int:
    """Computes sequence length boundaries directly by evaluating decompressed lookup length."""
    seq = _read_single_sequence(lmdb_path, header_id)
    return len(seq)