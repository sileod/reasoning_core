# Wave 7 standing rules

Every rule here was bought with a failed trial. They apply to every task in this wave.

## The prompt must determine the answer

Before anything else: could a careful reader who knows the algorithm get a *different*
answer from this prompt and still be right? If yes, the task is broken however cleanly
it validates.

- Every fact the answer depends on must be stated in the prompt. Not implied by the
  scenario, not conventional, not obvious -- stated.
- If two instances can render the same prompt text, they must have the same answer.
  Two different answers behind one prompt is the defect; a shared prompt on its own
  is fine.
- Where more than one answer could legitimately be written down for the same instance
  -- ties, orderings, equivalent representations -- name the tie-break **in the prompt**
  and enforce exactly it in `score_answer`.
- Say the answer format in the prompt, with a worked example of the format itself.

## The generator must check its own gold answer

Computing an answer is not the same as knowing it is right. Inside `generate_entry`:

- Apply the answer back to the instance and assert it reproduces the target, whenever
  the answer is a constructive object -- a script, a route, a schedule, a repaired
  string, a selection.
- Assert the defining property directly when it is not constructive: that the coset is
  closed, that the cut is consistent, that the offset respects the alignment.
- **A verifier that can give up must reject, never accept.** A brute-force check capped
  at N states that returns "don't know" and is read as "fine" is how a wave6 task
  shipped a non-minimal gold answer. If the check does not complete, throw the instance
  away.
- Claim optimality or uniqueness only where you verified it by exhaustive search at
  the sizes you actually generate. If you cannot verify it, do not claim it in the
  prompt.

## Answers

- A string, a list, a fraction, or an unbounded integer. Not a float, not JSON, not
  prose.
- Short. The answer is the thing being scored, not an explanation of it.
- **Balance the label when the answer set is small.** A yes/no task is welcome, but the
  generator must produce both answers at close to equal rates at every level, and the
  constant-guess audit measures how far your label prior sits above the 1/k floor. A
  balanced binary task passes; a 70/30 one does not.
- Never let the answer be readable off the surface of the prompt -- the last number,
  the first number, the largest number, the last word. The prompt-surface audit tests
  exactly those four.

## Prompt style

- Prose a person would write, in a setting the task actually fits. Not a symbolic dump
  with a question mark at the end.
- Name the standard polynomial algorithm in the prompt where one exists. The task is
  to execute known reasoning correctly, not to rediscover it.
- Difficulty comes from structural depth -- more interacting parts, longer dependency
  chains, more places to go wrong -- never from a longer input saying the same thing.

## Mechanics

- Do not override `validate`. Extend it by calling `super().validate(...)` first if you
  must; a replacement voids the whole contract and is rejected.
- Generation must be deterministic under a fixed seed, including across processes.
  Do not iterate an object-keyed dict, a set of non-strings, or anything else whose
  order is an id.
- Do not reseed the global RNG inside generation.
- Keep the whole trial inside your owned directory, tests included.
