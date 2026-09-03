## Level 0
### Example 1
A relation has attributes A,B,C,D,E, with functional dependencies: {A,E} -> {B}; {B} -> {E}; {B,C} -> {E}; {E} -> {A}; {E} -> {B}. It is decomposed into the two projections {A,B,E} and {C,D,E}. A binary decomposition {XY, XZ} with shared attribute set X is lossless exactly when X -> Y or X -> Z holds in the closure of the dependencies. Decide losslessness and give the deciding witness. If lossless, state the shared key set, which of P1/P2 it determines, and the shortest chain of given dependencies (each as {L}->{R}, in the order they fire) that establishes it. If lossy, state the shared key set and the attributes it cannot determine. Begin the answer on one line with 'lossless:' or 'lossy:'.

Answer: lossless:{E}->P1 via {E}->{A};{A,E}->{B}

### Example 2
A relation has attributes A,B,C,D,E, with functional dependencies: {A,B} -> {D}; {A,E} -> {C}; {B,D} -> {A}; {C} -> {A}; {D,E} -> {A}. It is decomposed into the two projections {A,B,C} and {C,D,E}. A binary decomposition {XY, XZ} with shared attribute set X is lossless exactly when X -> Y or X -> Z holds in the closure of the dependencies. Decide losslessness and give the deciding witness. If lossless, state the shared key set, which of P1/P2 it determines, and the shortest chain of given dependencies (each as {L}->{R}, in the order they fire) that establishes it. If lossy, state the shared key set and the attributes it cannot determine. Begin the answer on one line with 'lossless:' or 'lossy:'.

Answer: lossy:{C} misses B,D,E

## Level 2
### Example 1
A relation has attributes A,B,C,D,E,F, with functional dependencies: {B,C} -> {E}; {C} -> {F}; {C,E} -> {B}; {D} -> {A}; {D} -> {E}; {D} -> {F}. It is decomposed into the two projections {A,B,C,D} and {D,E,F}. A binary decomposition {XY, XZ} with shared attribute set X is lossless exactly when X -> Y or X -> Z holds in the closure of the dependencies. Decide losslessness and give the deciding witness. If lossless, state the shared key set, which of P1/P2 it determines, and the shortest chain of given dependencies (each as {L}->{R}, in the order they fire) that establishes it. If lossy, state the shared key set and the attributes it cannot determine. Begin the answer on one line with 'lossless:' or 'lossy:'.

Answer: lossless:{D}->P2 via {D}->{A};{D}->{E};{D}->{F}

### Example 2
A relation has attributes A,B,C,D,E,F, with functional dependencies: {A,B} -> {D}; {A,B} -> {F}; {A,D} -> {F}; {A,E} -> {D}; {C} -> {A}; {C,F} -> {A}. It is decomposed into the two projections {A,B,C} and {A,D,E,F}. A binary decomposition {XY, XZ} with shared attribute set X is lossless exactly when X -> Y or X -> Z holds in the closure of the dependencies. Decide losslessness and give the deciding witness. If lossless, state the shared key set, which of P1/P2 it determines, and the shortest chain of given dependencies (each as {L}->{R}, in the order they fire) that establishes it. If lossy, state the shared key set and the attributes it cannot determine. Begin the answer on one line with 'lossless:' or 'lossy:'.

Answer: lossy:{A} misses B,C,D,E,F

## Level 5
### Example 1
A relation has attributes A,B,C,D,E,F,G, with functional dependencies: {B} -> {G}; {C} -> {F}; {C,D} -> {B}; {C,F} -> {D}; {D} -> {A}; {D,F} -> {B}; {D,F} -> {G}; {G} -> {B}. It is decomposed into the two projections {A,B,C,E} and {D,E,F,G}. A binary decomposition {XY, XZ} with shared attribute set X is lossless exactly when X -> Y or X -> Z holds in the closure of the dependencies. Decide losslessness and give the deciding witness. If lossless, state the shared key set, which of P1/P2 it determines, and the shortest chain of given dependencies (each as {L}->{R}, in the order they fire) that establishes it. If lossy, state the shared key set and the attributes it cannot determine. Begin the answer on one line with 'lossless:' or 'lossy:'.

Answer: lossy:{E} misses A,B,C,D,F,G

### Example 2
A relation has attributes A,B,C,D,E,F,G, with functional dependencies: {A} -> {D}; {A,B} -> {F}; {A,C} -> {E}; {B} -> {C}; {B,C} -> {D}; {B,E} -> {F}; {B,G} -> {D}; {C,G} -> {F}. It is decomposed into the two projections {A,B,F} and {C,D,E,F,G}. A binary decomposition {XY, XZ} with shared attribute set X is lossless exactly when X -> Y or X -> Z holds in the closure of the dependencies. Decide losslessness and give the deciding witness. If lossless, state the shared key set, which of P1/P2 it determines, and the shortest chain of given dependencies (each as {L}->{R}, in the order they fire) that establishes it. If lossy, state the shared key set and the attributes it cannot determine. Begin the answer on one line with 'lossless:' or 'lossy:'.

Answer: lossy:{F} misses A,B,C,D,E,G
