# Samples P027v1 (fd_attribute_closure)

## Level 0

### Prompt

Universe:
{ A, B, C, D, E, F, G }

Dependencies:
A -> DF
EFG -> A
ABC -> CDE
D -> E

Start:
{A, B}

Using the standard attribute-closure algorithm (repeatedly apply every functional dependency whose left side is already implied), compute the attribute closure of the starting set under these functional dependencies.

The answer is the closure written as a single string: its attribute letters concatenated, with no separators, sorted in alphabetical order. For example, a closure containing attributes A, C and B would be written ACB reordered as ABC.

### Answer

ABDEF

### Prompt

Universe:
{ A, B, C, D, E, F, G }

Dependencies:
F -> DE
BEG -> CDE
E -> AB
G -> ABD

Start:
{A, F}

Using the standard attribute-closure algorithm (repeatedly apply every functional dependency whose left side is already implied), compute the attribute closure of the starting set under these functional dependencies.

The answer is the closure written as a single string: its attribute letters concatenated, with no separators, sorted in alphabetical order. For example, a closure containing attributes A, C and B would be written ACB reordered as ABC.

### Answer

ABDEF

### Prompt

Universe:
{ A, B, C, D, E, F, G }

Dependencies:
AF -> BF
B -> CD
G -> A
EG -> BDE

Start:
{G}

Using the standard attribute-closure algorithm (repeatedly apply every functional dependency whose left side is already implied), compute the attribute closure of the starting set under these functional dependencies.

The answer is the closure written as a single string: its attribute letters concatenated, with no separators, sorted in alphabetical order. For example, a closure containing attributes A, C and B would be written ACB reordered as ABC.

### Answer

AG

## Level 2

### Prompt

Universe:
{ A, B, C, D, E, F, G, H, I }

Dependencies:
A -> AD
DFG -> BFI
CG -> BGI
BF -> D
CFG -> CDI
AEF -> AFG

Start:
{B, F}

Using the standard attribute-closure algorithm (repeatedly apply every functional dependency whose left side is already implied), compute the attribute closure of the starting set under these functional dependencies.

The answer is the closure written as a single string: its attribute letters concatenated, with no separators, sorted in alphabetical order. For example, a closure containing attributes A, C and B would be written ACB reordered as ABC.

### Answer

BDF

### Prompt

Universe:
{ A, B, C, D, E, F, G, H, I }

Dependencies:
E -> C
G -> DI
DEG -> DEF
BCF -> FI
G -> ACD
FI -> DE

Start:
{C, G}

Using the standard attribute-closure algorithm (repeatedly apply every functional dependency whose left side is already implied), compute the attribute closure of the starting set under these functional dependencies.

The answer is the closure written as a single string: its attribute letters concatenated, with no separators, sorted in alphabetical order. For example, a closure containing attributes A, C and B would be written ACB reordered as ABC.

### Answer

ACDGI

### Prompt

Universe:
{ A, B, C, D, E, F, G, H, I }

Dependencies:
AGH -> FH
DFG -> BGH
I -> A
H -> B
CDE -> AI
AE -> EG

Start:
{G, I}

Using the standard attribute-closure algorithm (repeatedly apply every functional dependency whose left side is already implied), compute the attribute closure of the starting set under these functional dependencies.

The answer is the closure written as a single string: its attribute letters concatenated, with no separators, sorted in alphabetical order. For example, a closure containing attributes A, C and B would be written ACB reordered as ABC.

### Answer

AGI

## Level 5

### Prompt

Universe:
{ A, B, C, D, E, F, G, H, I, J, K, L }

Dependencies:
EH -> F
EFG -> BK
AF -> FHJ
J -> ABC
D -> K
F -> CKL
BL -> GL
BCFK -> BCDG
EHJ -> GHL

Start:
{A, F}

Using the standard attribute-closure algorithm (repeatedly apply every functional dependency whose left side is already implied), compute the attribute closure of the starting set under these functional dependencies.

The answer is the closure written as a single string: its attribute letters concatenated, with no separators, sorted in alphabetical order. For example, a closure containing attributes A, C and B would be written ACB reordered as ABC.

### Answer

ABCDFGHJKL

### Prompt

Universe:
{ A, B, C, D, E, F, G, H, I, J, K, L }

Dependencies:
EL -> I
L -> L
BIK -> H
JL -> DFHI
J -> F
CDGJ -> G
IJ -> CH
FL -> CHK
ABK -> DF

Start:
{J}

Using the standard attribute-closure algorithm (repeatedly apply every functional dependency whose left side is already implied), compute the attribute closure of the starting set under these functional dependencies.

The answer is the closure written as a single string: its attribute letters concatenated, with no separators, sorted in alphabetical order. For example, a closure containing attributes A, C and B would be written ACB reordered as ABC.

### Answer

FJ

### Prompt

Universe:
{ A, B, C, D, E, F, G, H, I, J, K, L }

Dependencies:
IJL -> L
AEI -> ACEG
L -> I
KL -> CIJK
BI -> I
BCHI -> HJ
D -> F
E -> BCEJ
D -> AEJ

Start:
{E, F}

Using the standard attribute-closure algorithm (repeatedly apply every functional dependency whose left side is already implied), compute the attribute closure of the starting set under these functional dependencies.

The answer is the closure written as a single string: its attribute letters concatenated, with no separators, sorted in alphabetical order. For example, a closure containing attributes A, C and B would be written ACB reordered as ABC.

### Answer

BCEFJ
