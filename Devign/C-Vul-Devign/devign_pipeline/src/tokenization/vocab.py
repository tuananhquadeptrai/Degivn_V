from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterable, Tuple
from collections import Counter
import json
import pickle
from pathlib import Path

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
SEP_TOKEN = '<SEP>'
CLS_TOKEN = '<CLS>'
MASK_TOKEN = '<MASK>'

DEFAULT_SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


@dataclass
class VocabConfig:
    min_freq: int = 2
    max_vocab_size: int = 50000
    special_tokens: List[str] = field(default_factory=lambda: DEFAULT_SPECIAL_TOKENS.copy())
    lowercase: bool = False


class Vocabulary:
    def __init__(self, config: VocabConfig = None):
        self.config = config or VocabConfig()
        self.token2id: Dict[str, int] = {}
        self.id2token: List[str] = []
        self.token_freqs: Dict[str, int] = {}
        self._built = False

    def build(self, token_iter: Iterable[List[str]], show_progress: bool = True) -> None:
        counter = Counter()
        iterator = token_iter
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(token_iter, desc="Building vocabulary")
            except ImportError:
                pass
        
        for tokens in iterator:
            if self.config.lowercase:
                tokens = [t.lower() for t in tokens]
            counter.update(tokens)
        
        self.build_from_counter(counter)

    def build_from_counter(self, counter: Counter) -> None:
        self.token2id.clear()
        self.id2token.clear()
        self.token_freqs.clear()
        
        self._add_special_tokens()
        
        filtered = [
            (token, freq) for token, freq in counter.items()
            if freq >= self.config.min_freq and token not in self.token2id
        ]
        filtered.sort(key=lambda x: (-x[1], x[0]))
        
        remaining_slots = self.config.max_vocab_size - len(self.token2id)
        for token, freq in filtered[:remaining_slots]:
            idx = len(self.id2token)
            self.token2id[token] = idx
            self.id2token.append(token)
            self.token_freqs[token] = freq
        
        self._built = True

    def _add_special_tokens(self) -> None:
        for token in self.config.special_tokens:
            if token not in self.token2id:
                idx = len(self.id2token)
                self.token2id[token] = idx
                self.id2token.append(token)
                self.token_freqs[token] = 0

    def token_to_id(self, token: str) -> int:
        if self.config.lowercase:
            token = token.lower()
        return self.token2id.get(token, self.unk_id)

    def id_to_token(self, idx: int) -> str:
        if 0 <= idx < len(self.id2token):
            return self.id2token[idx]
        return UNK_TOKEN

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(t) for t in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.id_to_token(i) for i in ids]

    def __len__(self) -> int:
        return len(self.token2id)

    def __contains__(self, token: str) -> bool:
        if self.config.lowercase:
            token = token.lower()
        return token in self.token2id

    @property
    def pad_id(self) -> int:
        return self.token2id.get(PAD_TOKEN, 0)

    @property
    def unk_id(self) -> int:
        return self.token2id.get(UNK_TOKEN, 1)

    @property
    def bos_id(self) -> int:
        return self.token2id.get(BOS_TOKEN, 2)

    @property
    def eos_id(self) -> int:
        return self.token2id.get(EOS_TOKEN, 3)

    def get_most_common(self, n: int = 100) -> List[Tuple[str, int]]:
        sorted_tokens = sorted(
            self.token_freqs.items(),
            key=lambda x: -x[1]
        )
        return sorted_tokens[:n]

    def get_oov_rate(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        oov_count = sum(1 for t in tokens if t not in self)
        return oov_count / len(tokens)

    def save(self, path: str, format: str = 'json') -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'config': {
                'min_freq': self.config.min_freq,
                'max_vocab_size': self.config.max_vocab_size,
                'special_tokens': self.config.special_tokens,
                'lowercase': self.config.lowercase,
            },
            'token2id': self.token2id,
            'id2token': self.id2token,
            'token_freqs': self.token_freqs,
        }
        
        if format == 'json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        
        config = VocabConfig(**data['config'])
        vocab = cls(config)
        vocab.token2id = data['token2id']
        vocab.id2token = data['id2token']
        vocab.token_freqs = data['token_freqs']
        vocab._built = True
        
        return vocab

    def merge(self, other: 'Vocabulary') -> 'Vocabulary':
        combined_counter = Counter()
        combined_counter.update(self.token_freqs)
        combined_counter.update(other.token_freqs)
        
        new_vocab = Vocabulary(self.config)
        new_vocab.build_from_counter(combined_counter)
        return new_vocab

    def get_stats(self) -> dict:
        freqs = list(self.token_freqs.values())
        non_special_freqs = [
            f for t, f in self.token_freqs.items()
            if t not in self.config.special_tokens
        ]
        
        return {
            'vocab_size': len(self),
            'num_special_tokens': len(self.config.special_tokens),
            'num_regular_tokens': len(self) - len(self.config.special_tokens),
            'total_occurrences': sum(freqs),
            'min_freq': min(non_special_freqs) if non_special_freqs else 0,
            'max_freq': max(non_special_freqs) if non_special_freqs else 0,
            'avg_freq': sum(non_special_freqs) / len(non_special_freqs) if non_special_freqs else 0,
            'config': {
                'min_freq_threshold': self.config.min_freq,
                'max_vocab_size': self.config.max_vocab_size,
                'lowercase': self.config.lowercase,
            }
        }


class VocabBuilder:
    def __init__(self, config: VocabConfig = None):
        self.config = config or VocabConfig()
        self.counter = Counter()

    def add_tokens(self, tokens: List[str]) -> None:
        if self.config.lowercase:
            tokens = [t.lower() for t in tokens]
        self.counter.update(tokens)

    def add_batch(self, token_lists: List[List[str]]) -> None:
        for tokens in token_lists:
            self.add_tokens(tokens)

    def build_vocab(self) -> Vocabulary:
        vocab = Vocabulary(self.config)
        vocab.build_from_counter(self.counter)
        return vocab

    def save_counter(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'config': {
                'min_freq': self.config.min_freq,
                'max_vocab_size': self.config.max_vocab_size,
                'special_tokens': self.config.special_tokens,
                'lowercase': self.config.lowercase,
            },
            'counter': dict(self.counter),
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load_counter(cls, path: str) -> 'VocabBuilder':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        config = VocabConfig(**data['config'])
        builder = cls(config)
        builder.counter = Counter(data['counter'])
        return builder

    def get_counter_stats(self) -> dict:
        if not self.counter:
            return {'total_unique': 0, 'total_occurrences': 0}
        
        freqs = list(self.counter.values())
        return {
            'total_unique': len(self.counter),
            'total_occurrences': sum(freqs),
            'tokens_above_min_freq': sum(1 for f in freqs if f >= self.config.min_freq),
            'top_10': self.counter.most_common(10),
        }
