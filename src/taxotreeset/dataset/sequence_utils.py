import math
import random
import bisect

# Precomputação da tabela de tradução IUPAC estrita (Notação IUPAC - https://doi.org/10.1021/bi00822a023)
_IUPAC_MAP = {
    "A": "T", "T": "A", "C": "G", "G": "C",
    "Y": "R", "R": "Y", "W": "W", "S": "S",
    "K": "M", "M": "K", "D": "H", "H": "D",
    "V": "B", "B": "V", "N": "N"
}
_FROM_CHARS = "".join(chr(i) for i in range(256))
_TO_CHARS = ["N"] * 256
for _k, _v in _IUPAC_MAP.items():
    _TO_CHARS[ord(_k)] = _v
    _TO_CHARS[ord(_k.lower())] = _v
_IUPAC_TRANSLATE_TABLE = str.maketrans(_FROM_CHARS, "".join(_TO_CHARS))


def get_complement(seq: str) -> str:
    """Retorna o complemento reverso de uma sequência de DNA preservando a notação IUPAC completa em nível de C."""
    return seq[::-1].translate(_IUPAC_TRANSLATE_TABLE)


def extract_subseqs(
    seq: str,
    n: int,
    min_len: int,
    max_len: int,
    rng: random.Random | None = None
) -> list[str]:
    """
    Extrai n subsequências de uma string de DNA com tamanhos variando entre min_len e max_len.
    Preserva a lógica original de otimização de sobreposição baseada no comprimento do vetor.
    """
    if n <= 0:
        raise ValueError("n precisa ser positivo")
    if min_len > max_len:
        raise ValueError("min_len precisa ser <= max_len")
    
    if rng is None:
        rng = random
        
    subseqs = []
    if len(seq) < min_len:
        return []

    # Cenário 1: Sequência longa o suficiente para extrações 100% sem sobreposição
    if len(seq) >= 2 * n * max_len:
        blacklist = []  # Manterá uma lista ordenada de tuplas (start, end)
        while len(subseqs) < n:
            idx = rng.randrange(0, len(seq) - max_len + 1)
            start, end = idx, idx + max_len
            
            # Encontra o ponto de inserção ideal via busca binária O(log N)
            pos = bisect.bisect_left(blacklist, (start, end))
            
            # Valida sobreposição apenas com os vizinhos imediato esquerdo e direito
            if pos > 0 and start < blacklist[pos - 1][1]:
                continue
            if pos < len(blacklist) and end > blacklist[pos][0]:
                continue
                
            blacklist.insert(pos, (start, end))
            subseqs.append(seq[start:end])
            
    # Cenário 2: Sequência média, aplica blocos quase não-sobrepostos flanqueados
    elif len(seq) < 2 * n * max_len and len(seq) >= n * max_len:
        rest = (len(seq) // max_len) - n
        if rest <= 0:
            left_start = 0
            for _ in range(n):
                subseqs.append(seq[left_start:left_start + max_len])
                left_start += max_len
            return subseqs
            
        window_bases = max(0, (len(seq) - n * max_len) // (n + 1))
        left_start = 0
        right_start = len(seq) - max_len
        operations = int(n / 2) if n % 2 == 0 else int((n - 1) / 2)
        
        for _ in range(operations):
            subseqs.append(seq[left_start: left_start + max_len])
            left_start += max_len + window_bases
            subseqs.append(seq[right_start: right_start + max_len])
            right_start -= max_len + window_bases
            
        if n % 2 != 0:
            mid_seq = int(math.floor(len(seq) / 2))
            mid_max_len = int(math.floor(max_len / 2))
            subseqs.append(seq[mid_seq - mid_max_len: mid_seq + mid_max_len])
            
    # Cenário 3: Sequência curta, sorteio direto sem materializar produto cartesiano.
    # Inclui complemento reverso quando ainda há espaço no orçamento de n.
    else:
        max_L = min(max_len, len(seq))
        subseqs_set = set()
        attempts = 0
        # Teto de tentativas: protege contra loops infinitos quando a diversidade
        # disponível é menor do que n (caso real em viroides, ~300 bp).
        max_attempts = n * 50

        while len(subseqs) < n and attempts < max_attempts:
            attempts += 1
            L = rng.randint(min_len, max_L)
            start = rng.randint(0, len(seq) - L)
            s = seq[start:start + L]

            if s not in subseqs_set:
                subseqs.append(s)
                subseqs_set.add(s)

            if len(subseqs) < n:
                c = get_complement(s)
                if c not in subseqs_set:
                    subseqs.append(c)
                    subseqs_set.add(c)

    return subseqs