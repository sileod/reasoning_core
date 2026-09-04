# Level 0
## Example
**Prompt:**
```
Nonterminals: A, B, C, D
Terminals: a, b, c
Start symbol: C
Parse table (rows "A b -> production", "eps" means empty production):
  A b -> eps
  A c -> B b
  B c -> A
  C b -> b
  D $ -> eps
  D b -> b
Input string to parse: b

Run the deterministic LL(1) predictive parser using a stack that starts as [$, start symbol] with $ the end-of-input marker and the top of stack on the right. The next input token is read left to right. Standard LL(1) algorithm: if the top of stack is a terminal (or $), match it against the current input token and pop; if it is a nonterminal, look up the parse table cell (top-nonterminal, current-input-token) where $ stands in for end of input, and expand: pop the nonterminal and push the production symbols in reverse order (an "eps" production pushes nothing).
Report every parse-table expansion as "expand X -> prod" and every successful terminal match as "match t", in execution order, ending with "accept". If at any point the current token does not match the top of stack, or the parse table has no entry (top-nonterminal, current-token), stop immediately and report the exact failure point as "fail no-entry X on t" or "fail mismatch X on t" (no "accept" is appended in the failure case).

The answer is that exact trace string.
```
**Answer:**
```
expand C -> b ; match b ; accept
```
## Example
**Prompt:**
```
Nonterminals: A, B, C, D
Terminals: a, b, c
Start symbol: C
Parse table (rows "A b -> production", "eps" means empty production):
  A $ -> b A
  B c -> C c
  C b -> b
  D $ -> c
  D a -> D B
Input string to parse: b

Run the deterministic LL(1) predictive parser using a stack that starts as [$, start symbol] with $ the end-of-input marker and the top of stack on the right. The next input token is read left to right. Standard LL(1) algorithm: if the top of stack is a terminal (or $), match it against the current input token and pop; if it is a nonterminal, look up the parse table cell (top-nonterminal, current-input-token) where $ stands in for end of input, and expand: pop the nonterminal and push the production symbols in reverse order (an "eps" production pushes nothing).
Report every parse-table expansion as "expand X -> prod" and every successful terminal match as "match t", in execution order, ending with "accept". If at any point the current token does not match the top of stack, or the parse table has no entry (top-nonterminal, current-token), stop immediately and report the exact failure point as "fail no-entry X on t" or "fail mismatch X on t" (no "accept" is appended in the failure case).

The answer is that exact trace string.
```
**Answer:**
```
expand C -> b ; match b ; accept
```
# Level 2
## Example
**Prompt:**
```
Nonterminals: A, B, C, D, E
Terminals: a, b, c, d, e
Start symbol: D
Parse table (rows "A b -> production", "eps" means empty production):
  A c -> D c
  B $ -> D B
  B b -> eps
  C c -> d c
  D a -> a
  E d -> E
  E e -> eps
Input string to parse: a

Run the deterministic LL(1) predictive parser using a stack that starts as [$, start symbol] with $ the end-of-input marker and the top of stack on the right. The next input token is read left to right. Standard LL(1) algorithm: if the top of stack is a terminal (or $), match it against the current input token and pop; if it is a nonterminal, look up the parse table cell (top-nonterminal, current-input-token) where $ stands in for end of input, and expand: pop the nonterminal and push the production symbols in reverse order (an "eps" production pushes nothing).
Report every parse-table expansion as "expand X -> prod" and every successful terminal match as "match t", in execution order, ending with "accept". If at any point the current token does not match the top of stack, or the parse table has no entry (top-nonterminal, current-token), stop immediately and report the exact failure point as "fail no-entry X on t" or "fail mismatch X on t" (no "accept" is appended in the failure case).

The answer is that exact trace string.
```
**Answer:**
```
expand D -> a ; match a ; accept
```
## Example
**Prompt:**
```
Nonterminals: A, B, C, D, E
Terminals: a, b, c, d, e
Start symbol: B
Parse table (rows "A b -> production", "eps" means empty production):
  A $ -> eps
  B b -> b
  B d -> B
  C d -> b
  D c -> e c
  D e -> a
  E c -> B
Input string to parse: b

Run the deterministic LL(1) predictive parser using a stack that starts as [$, start symbol] with $ the end-of-input marker and the top of stack on the right. The next input token is read left to right. Standard LL(1) algorithm: if the top of stack is a terminal (or $), match it against the current input token and pop; if it is a nonterminal, look up the parse table cell (top-nonterminal, current-input-token) where $ stands in for end of input, and expand: pop the nonterminal and push the production symbols in reverse order (an "eps" production pushes nothing).
Report every parse-table expansion as "expand X -> prod" and every successful terminal match as "match t", in execution order, ending with "accept". If at any point the current token does not match the top of stack, or the parse table has no entry (top-nonterminal, current-token), stop immediately and report the exact failure point as "fail no-entry X on t" or "fail mismatch X on t" (no "accept" is appended in the failure case).

The answer is that exact trace string.
```
**Answer:**
```
expand B -> b ; match b ; accept
```
# Level 5
## Example
**Prompt:**
```
Nonterminals: A, B, C, D, E
Terminals: a, b, c, d, e
Start symbol: B
Parse table (rows "A b -> production", "eps" means empty production):
  A b -> C
  B b -> b
  C e -> D
  D a -> D d
  E e -> eps
Input string to parse: b

Run the deterministic LL(1) predictive parser using a stack that starts as [$, start symbol] with $ the end-of-input marker and the top of stack on the right. The next input token is read left to right. Standard LL(1) algorithm: if the top of stack is a terminal (or $), match it against the current input token and pop; if it is a nonterminal, look up the parse table cell (top-nonterminal, current-input-token) where $ stands in for end of input, and expand: pop the nonterminal and push the production symbols in reverse order (an "eps" production pushes nothing).
Report every parse-table expansion as "expand X -> prod" and every successful terminal match as "match t", in execution order, ending with "accept". If at any point the current token does not match the top of stack, or the parse table has no entry (top-nonterminal, current-token), stop immediately and report the exact failure point as "fail no-entry X on t" or "fail mismatch X on t" (no "accept" is appended in the failure case).

The answer is that exact trace string.
```
**Answer:**
```
expand B -> b ; match b ; accept
```
## Example
**Prompt:**
```
Nonterminals: A, B, C, D, E
Terminals: a, b, c, d, e
Start symbol: C
Parse table (rows "A b -> production", "eps" means empty production):
  A $ -> D C
  A d -> eps
  B c -> C d
  B e -> eps
  C a -> a
  D e -> C
  E a -> eps
Input string to parse: a

Run the deterministic LL(1) predictive parser using a stack that starts as [$, start symbol] with $ the end-of-input marker and the top of stack on the right. The next input token is read left to right. Standard LL(1) algorithm: if the top of stack is a terminal (or $), match it against the current input token and pop; if it is a nonterminal, look up the parse table cell (top-nonterminal, current-input-token) where $ stands in for end of input, and expand: pop the nonterminal and push the production symbols in reverse order (an "eps" production pushes nothing).
Report every parse-table expansion as "expand X -> prod" and every successful terminal match as "match t", in execution order, ending with "accept". If at any point the current token does not match the top of stack, or the parse table has no entry (top-nonterminal, current-token), stop immediately and report the exact failure point as "fail no-entry X on t" or "fail mismatch X on t" (no "accept" is appended in the failure case).

The answer is that exact trace string.
```
**Answer:**
```
expand C -> a ; match a ; accept
```
