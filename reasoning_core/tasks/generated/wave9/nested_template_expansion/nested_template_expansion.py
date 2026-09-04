"""Nested template expansion task.

Given a mapping of variables and a template containing references in the form
``{{ name }}``, expand each reference by substitution; references may nest
(one variable's value may itself contain references); a reference may carry an
optional default ``{{ name | default }}`` used only when the variable is
absent; a reference may be conditional ``{{ name ? yes | no }}`` evaluated by
truthiness of the (recursively expanded) name; and ``\\{{`` / ``\\}}`` escape
the literal braces. Substitution is applied outermost-inner recursively until
a fixed point, exactly one expansion pass over the original template is NOT
what we do: we recursively expand every produced value as well, stopping when
no reference remains.
"""

import random

from reasoning_core.template import Config, Entry, Task, edict, render_payload, stochastic_rounding as sround


class _ExpandError(Exception):
    pass


def _strip(s):
    return s.strip()


def _get_var(vars_, name, default_sentinel):
    # Returns (value, present) where present indicated the variable existed.
    for k in vars_:
        if k == name:
            return vars_[k], True
    return default_sentinel, False


def _defence(stripped):
    # If the answer is wrapped in a ``` fence (optionally with a language
    # tag and/or trailing whitespace/backticks), return the inner content,
    # else None.
    lines = stripped.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        if len(lines) == 1:
            # single-line fence: strip backticks
            core = _strip_ticks(stripped)
            return core if core != stripped else None
        inner = "\n".join(lines[1:])
        if lines[-1].strip() == "```":
            inner = "\n".join(lines[1:-1])
        inner = _strip_ticks(inner)
        return inner
    return None


def _strip_ticks(s):
    s = s.strip()
    s = s.strip("`")
    return s.strip()


def expand_template(template, vars_, max_depth=60):
    """Recursively expand nested references in template per explicit rules.

    A reference is a balanced {{ ... }} group; groups may nest (a default,
    condition or value may itself hold references). Substitution continues
    until the text is a fixed point, then escaped braces are unescaped.
    """
    expanded = template
    for _ in range(max_depth + 1):
        new = _expand_once(expanded, vars_)
        if new == expanded:
            return _unescape(expanded)
        expanded = new
    return _unescape(expanded)


def _unescape(text):
    # Escaped braces \{{ and \}} become single literal braces { and }.
    return text.replace("\\{{", "{").replace("\\}}", "}")


def _expand_once(text, vars_):
    # Single expansion pass: walk the text and replace balanced {{ ... }}
    # groups with the resolution of their inner content.
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\\" and i + 1 < n and text[i + 1] in "{}":
            # keep escaped brace pairs as literal text (unescaped at the end)
            out.append(text[i])
            out.append(text[i + 1])
            i += 2
            continue
        if text[i:i + 2] == "{{":
            j = _closing(text, i + 2)
            if j is None:
                out.append(text[i])
                i += 1
                continue
            inner = text[i + 2:j]
            out.append(_resolve(inner, vars_))
            i = j + 2
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _closing(text, start):
    # Find the index of the '}}' that closes the '{{' ending at start-2,
    # accounting for nested {{ ... }} groups. Returns None if unmatched.
    depth = 1
    j = start
    n = len(text)
    while j < n:
        if text[j:j + 2] == "{{":
            depth += 1
            j += 2
        elif text[j:j + 2] == "}}":
            depth -= 1
            if depth == 0:
                return j
            j += 2
        else:
            j += 1
    return None


def _resolve(inner, vars_):
    # Parse the reference grammar inside the braces:
    #   {{ name }}
    #   {{ name | default }}
    #   {{ cond ? yes | no }}
    inner = _strip(inner)
    if "?" not in inner and "|" not in inner:
        name = inner
        val, present = _get_var(vars_, name, None)
        if present:
            return str(val)
        raise _ExpandError("undefined variable: %s" % name)

    if "?" in inner:
        cond_part, _, rest = inner.partition("?")
        cond_name = _strip(cond_part)
        if "|" in rest:
            yes_part, _, no_part = rest.partition("|")
            yes = _strip(yes_part)
            no = _strip(no_part)
        else:
            yes = _strip(rest)
            no = ""
        cond_val, present = _get_var(vars_, cond_name, None)
        if present:
            cond_text = expand_template(str(cond_val), vars_)
            truthy = _truthy(cond_text)
        else:
            truthy = False
        return expand_template(yes if truthy else no, vars_)
    else:
        name, _, default = inner.partition("|")
        name = _strip(name)
        default = _strip(default)
        val, present = _get_var(vars_, name, None)
        if present:
            return str(val)
        return expand_template(default, vars_)


def _truthy(val):
    s = _strip(str(val))
    if s == "":
        return False
    try:
        f = float(s)
    except ValueError:
        # non numeric: truthy unless it is the word "false" (case-insensitive)
        return s.lower() != "false"
    return f != 0.0


TASK_META = {'parent_source_id': None,
 'idea': 'nested_template_expansion (draw 1 of 1)',
 'hypothesis': 'HV-072',
 'changes': 'new task in '
            'reasoning_core/tasks/generated/wave9/nested_template_expansion',
 'generation': {'provider_name': 'albert',
                'model_name': 'deepseek-v4-flash',
                'harness_name': 'opencode',
                'harness_version': None,
                'agent_name': 'task-search-worker',
                'settings': {'variant': None,
                             'requested_seed': 2045813452,
                             'seed_forwarded': True,
                             'temperature': None,
                             'top_p': None,
                             'pure': True,
                             'max_steps': 40,
                             'timeout_seconds': 1800,
                             'sandbox': {'name': 'bubblewrap',
                                         'version': 'bubblewrap 0.8.0'}}}}


class NestedTemplateExpansionConfig(Config):
    n_vars: int = 3
    depth: int = 2

    def apply_difficulty(self, level):
        self.n_vars = sround(self.n_vars + level)
        self.depth = sround(self.depth + level)


def _node_text(rng, words, names):
    choice = rng.random()
    if choice < 0.4 and names:
        name = rng.choice(names)
        pad = rng.choice(["", " ", ""])
        return "{{" + pad + name + pad + "}}"
    elif choice < 0.7:
        return str(rng.randint(0, 20))
    else:
        return rng.choice(words)


class NestedTemplateExpansion(Task):
    summary = "Expand templates with nested variables, defaults, conditional fragments, and escaping under explicit substitution rules, returning exact expanded text."
    config_cls = NestedTemplateExpansionConfig

    _WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "zeta",
              "kappa", "lambda", "rho", "tau", "phi", "psi", "nyx", "onyx"]

    def generate_entry(self):
        rng = random
        level_sz = self.config.n_vars
        depth = self.config.depth
        names = ["v%d" % i for i in range(level_sz)]
        vars_ = {}
        # build variables in order so later vars can reference earlier ones (acyclic forward refs)
        for i, name in enumerate(names):
            r = rng.random()
            if r < 0.45 and i > 0:
                # reference to a single earlier variable, maybe with extra nesting
                dep = rng.choice(names[:i])
                nested = "{{" + dep + "}}"
                # possibly double-nest
                if depth >= 2 and rng.random() < 0.4:
                    dep2 = rng.choice(names[:i])
                    nested = "{{" + dep2 + " | " + nested + "}}" if rng.random() < 0.5 else nested
                vars_[name] = _node_text(rng, self._WORDS, names[:i]) if rng.random() < 0.3 else nested
            elif r < 0.75:
                vars_[name] = _node_text(rng, self._WORDS, names[:i])
            else:
                vars_[name] = str(rng.randint(0, 20))

        # Build the template body: a sequence of fixed segments plus references,
        # with nested/conditional/default/escaped pieces.
        parts = []
        n_parts = 2 + level_sz
        for _ in range(n_parts):
            r = rng.random()
            used = names if names else ["v0"]
            if r < 0.35:
                name = rng.choice(names)
                inner = name
                # sometimes nest by wrapping in a default or plain
                mode = rng.random()
                if mode < 0.3 and name in vars_:
                    # nested default chain
                    default = rng.choice(self._WORDS)
                    inner = name + " | {{ " + rng.choice(used) + " | " + default + " }}"
                elif mode < 0.6 and rng.random() < 0.5:
                    # conditional
                    cond = rng.choice(names)
                    yes = rng.choice(self._WORDS)
                    no = rng.choice(self._WORDS)
                    inner = cond + "?" + yes + " | " + no
                elif mode < 0.75:
                    # reference with default
                    default = rng.choice(self._WORDS)
                    inner = name + " | " + default
                parts.append("{{" + inner + "}}")
            elif r < 0.5:
                # literal escaped braces
                parts.append("\\{{literal " + rng.choice(self._WORDS) + "\\}}")
            else:
                # plain literal
                parts.append(rng.choice(self._WORDS))

        # Join parts with varying separators
        seps = [" ", " / ", " - ", " ", "/"]
        body = ""
        for j, p in enumerate(parts):
            if j == 0:
                body = p
            else:
                body += rng.choice(seps) + p

        try:
            answer = expand_template(body, vars_)
        except _ExpandError:
            # avoid undefined-variable failures: ensure all referenced plain
            # names exist. Simplest: regenerate.
            return self.generate_entry()

        # --- verify the answer is a fixed point (defining property) ---
        again = expand_template(answer, vars_)
        if again != answer:
            # Not a fixed point; the recursion bound gave up. Discard.
            return self.generate_entry()
        # ensure the answer is not trivially readable
        if not answer:
            return self.generate_entry()

        payload = {
            "variables": {k: str(v) for k, v in vars_.items()},
            "template": body,
        }
        metadata = edict({
            "variables": payload["variables"],
            "template": body,
            "answer": answer,
        })
        metadata.payload = payload
        return Entry(metadata=metadata, answer=answer)

    def render_prompt(self, metadata):
        rules = (
            "Expand the template by repeatedly substituting every occurrence "
            "of {{ name }} with that variable's value, continuing to expand "
            "any references that the substituted value itself contains until "
            "no reference remains. "
            "A reference {{ name | D }} uses value of name, else literal D. "
            "A reference {{ C ? Y | N }} expands to Y if variable C is "
            "present and its recursively-expanded value is truthy (nonempty, "
            "nonzero number, or word other than 'false'), else to N. "
            "Escaped braces \\{{ ... }} are literal text with the backslash "
            "removed. Names are single words [a-z0-9]+. "
            "The final answer is the fully expanded template text with no "
            "remaining {{ }}, with backslashes removed from escaped braces, "
            "written verbatim."
        )
        body = (
            "Variables:\n" +
            "\n".join("  %s = %s" % (k, v) for k, v in sorted(metadata.variables.items())) +
            "\n\nTemplate:\n  " + metadata.template +
            "\n\n" + rules +
            "\n\nGive the expanded text as the answer."
        )
        return body

    def score_answer(self, answer, entry):
        if not isinstance(answer, str):
            return 0.0
        gold = entry.answer
        stripped = answer.strip()
        if stripped == gold:
            return 1.0
        # Accept a surrounding markdown code fence (with optional language tag).
        de_fenced = _defence(stripped)
        if de_fenced is not None and de_fenced == gold:
            return 1.0
        return 0.0
