import itertools
import random

from reasoning_core.tasks.generated.wave3.s36_regular_language_counting.s36_regular_language_counting import (
    _count_matches,
    _rand_pattern,
    _ALPHABETS,
)


def _brute(pattern, length, alphabet):
    import re

    regex = pattern
    count = 0
    for tup in itertools.product(alphabet, repeat=length):
        s = "".join(tup)
        if re.fullmatch(regex, s):
            count += 1
    return count


def test_count_matches_brute():
    random.seed(12345)
    for _ in range(200):
        for size in (2, 3):
            alphabet = _ALPHABETS[size]
            pattern = _rand_pattern(size)
            length = random.randint(1, 5)
            dp = _count_matches(pattern, length, alphabet)
            bf = _brute(pattern, length, alphabet)
            assert dp == bf, (pattern, length, dp, bf)
