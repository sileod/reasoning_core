# BCNF violation v2 samples

## Level 0

### Example 1

> Consider the relational schema R(A, B, C, D, E, F) with functional dependencies: {A, B, F} -> {D}; {A, B, C, F} -> {D}; {A, B, F} -> {C}; {A, B, D, F} -> {E}; {A, B, F} -> {E}; {A, B, C, E, F} -> {D}. Determine whether the schema is in Boyce-Codd Normal Form (BCNF). A schema is in BCNF exactly when every non-trivial functional dependency has a superkey as its determinant. If the schema is NOT in BCNF, state the FIRST BCNF violation as the functional dependency X -> A, where X is the non-superkey determinant, A is the determined attribute not in X, attributes within a set are listed in alphabetical order, and each set is written as {A, C}. If the schema IS in BCNF, the answer is the word None.
> 
> What is the first BCNF violation of this schema?

Answer: None

### Example 2

> Consider the relational schema R(A, B, C, D, E, F) with functional dependencies: {A, B, C, E, F} -> {D}; {A, B, E, F} -> {C}; {A, B, E, F} -> {D}; {A, B, D, E, F} -> {C}. Determine whether the schema is in Boyce-Codd Normal Form (BCNF). A schema is in BCNF exactly when every non-trivial functional dependency has a superkey as its determinant. If the schema is NOT in BCNF, state the FIRST BCNF violation as the functional dependency X -> A, where X is the non-superkey determinant, A is the determined attribute not in X, attributes within a set are listed in alphabetical order, and each set is written as {A, C}. If the schema IS in BCNF, the answer is the word None.
> 
> What is the first BCNF violation of this schema?

Answer: None

## Level 2

### Example 1

> Consider the relational schema R(A, B, C, D, E, F, G, H) with functional dependencies: {B, E, F} -> {G}; {B, E, F} -> {A}; {B, E, F} -> {D}; {A, B, E, F} -> {C}; {A, B, E, F, G} -> {D}; {B, E, F, G, H} -> {D}; {B, E, F} -> {H}; {B, E, F} -> {C}. Determine whether the schema is in Boyce-Codd Normal Form (BCNF). A schema is in BCNF exactly when every non-trivial functional dependency has a superkey as its determinant. If the schema is NOT in BCNF, state the FIRST BCNF violation as the functional dependency X -> A, where X is the non-superkey determinant, A is the determined attribute not in X, attributes within a set are listed in alphabetical order, and each set is written as {A, C}. If the schema IS in BCNF, the answer is the word None.
> 
> What is the first BCNF violation of this schema?

Answer: None

### Example 2

> Consider the relational schema R(A, B, C, D, E, F, G, H) with functional dependencies: {A, B, C, E, G, H} -> {D}; {D, F} -> {H}; {C} -> {H}; {C, D, F, G, H} -> {A}; {B, E} -> {H}; {A, C, E, F, H} -> {G}; {B, G, H} -> {C}; {C} -> {G}. Determine whether the schema is in Boyce-Codd Normal Form (BCNF). A schema is in BCNF exactly when every non-trivial functional dependency has a superkey as its determinant. If the schema is NOT in BCNF, state the FIRST BCNF violation as the functional dependency X -> A, where X is the non-superkey determinant, A is the determined attribute not in X, attributes within a set are listed in alphabetical order, and each set is written as {A, C}. If the schema IS in BCNF, the answer is the word None.
> 
> What is the first BCNF violation of this schema?

Answer: {A, B, C, E, G, H} -> D

## Level 5

### Example 1

> Consider the relational schema R(A, B, C, D, E, F, G, H, I, J, K) with functional dependencies: {B, D} -> {E}; {B, C, D, E, F, G, H, I, J, K} -> {A}; {A, B, D, E, F, H, I, J, K} -> {C}; {A, B, C, E, F, G, H, K} -> {I}; {A, B, E, F} -> {J}; {G} -> {B}; {B, E, G, H, I, J, K} -> {F}; {A, B, C, E, F, G, I, J} -> {H}; {A, B, C, D, E, H, I} -> {F}; {A, B, C, D, E, F, G, H, I, J} -> {K}; {E, H, J} -> {B}. Determine whether the schema is in Boyce-Codd Normal Form (BCNF). A schema is in BCNF exactly when every non-trivial functional dependency has a superkey as its determinant. If the schema is NOT in BCNF, state the FIRST BCNF violation as the functional dependency X -> A, where X is the non-superkey determinant, A is the determined attribute not in X, attributes within a set are listed in alphabetical order, and each set is written as {A, C}. If the schema IS in BCNF, the answer is the word None.
> 
> What is the first BCNF violation of this schema?

Answer: {B, D} -> E

### Example 2

> Consider the relational schema R(A, B, C, D, E, F, G, H, I, J, K) with functional dependencies: {A, D, F, I, K} -> {H}; {B, G} -> {I}; {A, B, D, E, F, G, H, I, J, K} -> {C}; {A, B, E, F, G, H, I, J, K} -> {D}; {A, D, H} -> {C}; {A, I} -> {H}; {A} -> {C}; {A, B, D, E, F, G, H, I, J} -> {C}; {C, D, E, F, G, H, J, K} -> {A}; {A, C, F, J} -> {G}; {A, B, F, G, H, J, K} -> {D}. Determine whether the schema is in Boyce-Codd Normal Form (BCNF). A schema is in BCNF exactly when every non-trivial functional dependency has a superkey as its determinant. If the schema is NOT in BCNF, state the FIRST BCNF violation as the functional dependency X -> A, where X is the non-superkey determinant, A is the determined attribute not in X, attributes within a set are listed in alphabetical order, and each set is written as {A, C}. If the schema IS in BCNF, the answer is the word None.
> 
> What is the first BCNF violation of this schema?

Answer: {A, D, F, I, K} -> H
