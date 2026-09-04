# Level 0

## Example 1

### Prompt

Variables:
  v0 = 3
  v1 = {{v0}}
  v2 = {{v1 | {{v1}}}}

Template:
  {{v1 | {{ v2 | zeta }}}}/lambda sigma/omega / {{v0 | {{ v2 | kappa }}}}

Expand the template by repeatedly substituting every occurrence of {{ name }} with that variable's value, continuing to expand any references that the substituted value itself contains until no reference remains. A reference {{ name | D }} uses value of name, else literal D. A reference {{ C ? Y | N }} expands to Y if variable C is present and its recursively-expanded value is truthy (nonempty, nonzero number, or word other than 'false'), else to N. Escaped braces \{{ ... }} are literal text with the backslash removed. Names are single words [a-z0-9]+. The final answer is the fully expanded template text with no remaining {{ }}, with backslashes removed from escaped braces, written verbatim.

Give the expanded text as the answer.

### Answer

3/lambda sigma/omega / 3

## Example 2

### Prompt

Variables:
  v0 = 20
  v1 = {{v0}}
  v2 = {{v1}}

Template:
  zeta {{v0 | {{ v0 | rho }}}} - {{v2}} rho / \{{literal psi\}}

Expand the template by repeatedly substituting every occurrence of {{ name }} with that variable's value, continuing to expand any references that the substituted value itself contains until no reference remains. A reference {{ name | D }} uses value of name, else literal D. A reference {{ C ? Y | N }} expands to Y if variable C is present and its recursively-expanded value is truthy (nonempty, nonzero number, or word other than 'false'), else to N. Escaped braces \{{ ... }} are literal text with the backslash removed. Names are single words [a-z0-9]+. The final answer is the fully expanded template text with no remaining {{ }}, with backslashes removed from escaped braces, written verbatim.

Give the expanded text as the answer.

### Answer

zeta 20 - 20 rho / {literal psi}

# Level 2

## Example 1

### Prompt

Variables:
  v0 = 15
  v1 = 16
  v2 = {{ v1 }}
  v3 = {{v2 | {{v2}}}}
  v4 = 19

Template:
  alpha tau / {{v0 | nyx}} {{v0 | {{ v3 | omega }}}}/sigma/rho {{v4 | {{ v2 | psi }}}}

Expand the template by repeatedly substituting every occurrence of {{ name }} with that variable's value, continuing to expand any references that the substituted value itself contains until no reference remains. A reference {{ name | D }} uses value of name, else literal D. A reference {{ C ? Y | N }} expands to Y if variable C is present and its recursively-expanded value is truthy (nonempty, nonzero number, or word other than 'false'), else to N. Escaped braces \{{ ... }} are literal text with the backslash removed. Names are single words [a-z0-9]+. The final answer is the fully expanded template text with no remaining {{ }}, with backslashes removed from escaped braces, written verbatim.

Give the expanded text as the answer.

### Answer

alpha tau / 15 15/sigma/rho 19

## Example 2

### Prompt

Variables:
  v0 = 12
  v1 = beta
  v2 = 1
  v3 = 17
  v4 = {{v3}}

Template:
  \{{literal psi\}} / tau \{{literal tau\}} \{{literal psi\}}/lambda omega / beta

Expand the template by repeatedly substituting every occurrence of {{ name }} with that variable's value, continuing to expand any references that the substituted value itself contains until no reference remains. A reference {{ name | D }} uses value of name, else literal D. A reference {{ C ? Y | N }} expands to Y if variable C is present and its recursively-expanded value is truthy (nonempty, nonzero number, or word other than 'false'), else to N. Escaped braces \{{ ... }} are literal text with the backslash removed. Names are single words [a-z0-9]+. The final answer is the fully expanded template text with no remaining {{ }}, with backslashes removed from escaped braces, written verbatim.

Give the expanded text as the answer.

### Answer

{literal psi} / tau {literal tau} {literal psi}/lambda omega / beta

# Level 5

## Example 1

### Prompt

Variables:
  v0 = 7
  v1 = {{ v0 }}
  v2 = {{ v1 }}
  v3 = {{ v0 }}
  v4 = {{v3}}
  v5 = {{v2}}
  v6 = {{v3}}
  v7 = 8
  v8 = 19
  v9 = {{v3}}

Template:
  {{v4 | {{ v5 | psi }}}} - \{{literal zeta\}} beta {{v0?nyx | tau}} gamma / gamma/tau \{{literal beta\}}/{{v4?phi | psi}} {{v6 | {{ v5 | delta }}}} - phi - onyx

Expand the template by repeatedly substituting every occurrence of {{ name }} with that variable's value, continuing to expand any references that the substituted value itself contains until no reference remains. A reference {{ name | D }} uses value of name, else literal D. A reference {{ C ? Y | N }} expands to Y if variable C is present and its recursively-expanded value is truthy (nonempty, nonzero number, or word other than 'false'), else to N. Escaped braces \{{ ... }} are literal text with the backslash removed. Names are single words [a-z0-9]+. The final answer is the fully expanded template text with no remaining {{ }}, with backslashes removed from escaped braces, written verbatim.

Give the expanded text as the answer.

### Answer

7 - {literal zeta} beta nyx gamma / gamma/tau {literal beta}/phi 7 - phi - onyx

## Example 2

### Prompt

Variables:
  v0 = zeta
  v1 = 7
  v2 = {{ v1 }}
  v3 = {{v0 | {{v0}}}}
  v4 = {{v2}}
  v5 = {{v2}}
  v6 = {{v1}}
  v7 = 6
  v8 = 12
  v9 = tau

Template:
  {{v6 | alpha}} - {{v1 | onyx}} / {{v3 | zeta}}/nyx omega - delta {{v2 | {{ v9 | kappa }}}} - gamma / \{{literal alpha\}} / {{v5 | phi}} / {{v6}} tau

Expand the template by repeatedly substituting every occurrence of {{ name }} with that variable's value, continuing to expand any references that the substituted value itself contains until no reference remains. A reference {{ name | D }} uses value of name, else literal D. A reference {{ C ? Y | N }} expands to Y if variable C is present and its recursively-expanded value is truthy (nonempty, nonzero number, or word other than 'false'), else to N. Escaped braces \{{ ... }} are literal text with the backslash removed. Names are single words [a-z0-9]+. The final answer is the fully expanded template text with no remaining {{ }}, with backslashes removed from escaped braces, written verbatim.

Give the expanded text as the answer.

### Answer

7 - 7 / zeta/nyx omega - delta 7 - gamma / {literal alpha} / 7 / 7 tau

