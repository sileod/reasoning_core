
import difflib
import random
import re
import hashlib
from dataclasses import dataclass
from easydict import EasyDict as edict
from faker import Faker
from rapidfuzz.distance import Levenshtein
import whatthepatch
from reasoning_core.template import Task, Problem, Config, edict
from typing import List

fake = Faker()

def with_lineno(lines: List[str]) -> str:
    return "\n".join(f"{i+1:<4} | {line}" for i, line in enumerate(lines))

def get_short_hash():
    """Generates a git-style short hash (7 chars)."""
    r = str(random.random()).encode('utf-8')
    return hashlib.sha1(r).hexdigest()[:7]

@dataclass
class DiffConfig(Config):
    min_versions: int = 2
    max_versions: int = 5
    nb_lines: int = 5
    mutation_rate: float = 0.2

    def update(self, c):
        self.max_versions += c
        self.nb_lines += c

def mutate_words_in_line(line, vocab, rate):
    """Mutates words within a single string (line)."""
    words = line.split()
    if not words: return line
    
    if random.random() > rate:
        return line

    new_words = []
    for word in words:
        r = random.random()
        if r < 0.05:   # Delete word
            continue
        elif r < 0.15: # Substitute word
            new_words.append(random.choice(vocab))
        else:
            new_words.append(word)
    
    if not new_words and words:
        new_words = [random.choice(vocab)]
        
    return " ".join(new_words)

def mutate_lines(lines, vocab, rate):
    """Evolves a list of lines (sentences)."""
    new_lines = []
    
    for line in lines:
        r = random.random()
        if r < rate / 5:       # Delete entire line
            continue
        elif r < rate:         # Modify line
            new_lines.append(mutate_words_in_line(line, vocab, rate)) 
        elif r < rate * 1.2:   # Insert new line
            new_lines.append(" ".join(fake.words(nb=5)))
            new_lines.append(line)
        else:
            new_lines.append(line)
            
    if not new_lines:
        new_lines.append(" ".join(fake.words(nb=5)))
        
    return new_lines

def get_git_diff(src_lines, tgt_lines):
    """Generates a standard Git-style unified diff without file headers."""
    diff = difflib.unified_diff(src_lines, tgt_lines, lineterm='')
    # Strip the first two lines (--- and +++) to leave only chunks
    return "\n".join(list(diff)[2:])

class VersionedTask:
    def __init__(self, config=DiffConfig()):
        super().__init__(config=config)
        self.vocab = list(fake.words(nb=500, unique=True))
        self.balancing_key_ratio = 0.1

    def generate_version_chain(self):
        lines = [fake.sentence(nb_words=6).rstrip('.') for _ in range(self.config.nb_lines)]
        vid = get_short_hash()
        
        chain = [{'id': vid, 'lines': lines}]

        n_versions = random.randint(self.config.min_versions, self.config.max_versions)
        for _ in range(n_versions - 1):
            prev_lines = chain[-1]['lines']
            new_lines = mutate_lines(prev_lines, self.vocab, self.config.mutation_rate)
            new_vid = get_short_hash()
            chain.append({'id': new_vid, 'lines': new_lines})
            
        return chain

    def select_pair(self, chain):
        idxs = list(range(len(chain)))
        i = random.choice(idxs)
        j = random.choice([x for x in idxs if x != i])
        return chain[i], chain[j]

class DiffPrediction(VersionedTask, Task):
    def generate(self):
        chain = self.generate_version_chain()
        src, tgt = self.select_pair(chain)
        diff_str = get_git_diff(src['lines'], tgt['lines'])
        if not diff_str.strip() and  self.balancing_key_ratio<random.random():
            # No changes between versions; regenerate
            return self.generate()
        history_text = []
        for v in chain:
            content = with_lineno(v['lines'])
            history_text.append(f"Version {v['id']}:\n{content}\n")

        meta = edict(
            history="\n".join(history_text),
            src_id=src['id'],
            tgt_id=tgt['id'],
            src_text="\n".join(src['lines']),
            tgt_text="\n".join(tgt['lines'])
        )
        return Problem(meta, diff_str)

    def prompt(self, meta):
        return (f"Below is the version history of a file.\n\n"
                f"{meta.history}\n"
                f"Generate the Unified Diff to transform version {meta.src_id} into version {meta.tgt_id}.\n"
                f"The answer is the diff chunks only (no file headers), or empty if no changes.")

    def score_answer(self, answer, entry):
        meta = entry.get('metadata', {})
        src_text = meta.get('src_text')
        tgt_text = meta.get('tgt_text')
        
        if not src_text or not tgt_text:
            return Levenshtein.normalized_similarity(answer.strip(), entry['answer'].strip())

        try:
            patches = list(whatthepatch.parse_patch(answer))
            if not patches:
                patched_text = src_text
            else:
                patched_lines = whatthepatch.apply_diff(patches[0], src_text)
                patched_text = "\n".join(patched_lines)
            return Levenshtein.normalized_similarity(patched_text.strip(), tgt_text.strip())
        except Exception:
            return Levenshtein.normalized_similarity(answer.strip(), entry['answer'].strip())

class DiffPatching(VersionedTask, Task):
    def generate(self):
            chain = self.generate_version_chain()
            src, tgt = self.select_pair(chain)
            diff_str = get_git_diff(src['lines'], tgt['lines'])
            
            meta = edict(
                # Edit: Apply line numbering to source text
                src_text=with_lineno(src['lines']),
                src_id=src['id'],
                tgt_id=tgt['id'],
                diff=diff_str
            )
            return Problem(meta, "\n".join(tgt['lines']))

    def prompt(self, meta):
        return (f"Apply the following Unified Diff to the text.\n\n"
                f"Original Text (Version {meta.src_id}):\n"
                f"{meta.src_text}\n\n"
                f"Diff ({meta.src_id} -> {meta.tgt_id}):\n"
                f"{meta.diff}\n\n"
                f"The answer is the resulting text.")

    def score_answer(self, answer, entry):
        return Levenshtein.normalized_similarity(answer.strip(), entry['answer'].strip())