# Samples for critical_path_analysis (P068v1)

## Level 0

### Example 1

**Prompt:**

    Consider a project made of precedence-constrained tasks, each with a duration. A task can start only once all its predecessors have finished. Compute the earliest and latest start times for every task, then the slack of each task (latest minus earliest start). A task is critical when its slack is exactly zero. Using the critical-path method (CPM), find every critical activity.
    
    Tasks: A B C D
    Durations:
      A: 5  B: 4  C: 3  D: 1
    Precedence (a -> b means a must finish before b starts):
      A -> D B
      B -> C
      C -> (none)
      D -> (none)
    
    The answer is the space-separated, alphabetically sorted list of critical task labels (for example "A C E").

**Answer:**

    A B C

### Example 2

**Prompt:**

    Consider a project made of precedence-constrained tasks, each with a duration. A task can start only once all its predecessors have finished. Compute the earliest and latest start times for every task, then the slack of each task (latest minus earliest start). A task is critical when its slack is exactly zero. Using the critical-path method (CPM), find every critical activity.
    
    Tasks: A B C D
    Durations:
      A: 1  B: 3  C: 2  D: 4
    Precedence (a -> b means a must finish before b starts):
      A -> D
      B -> C
      C -> D
      D -> (none)
    
    The answer is the space-separated, alphabetically sorted list of critical task labels (for example "A C E").

**Answer:**

    B C D

## Level 2

### Example 1

**Prompt:**

    Consider a project made of precedence-constrained tasks, each with a duration. A task can start only once all its predecessors have finished. Compute the earliest and latest start times for every task, then the slack of each task (latest minus earliest start). A task is critical when its slack is exactly zero. Using the critical-path method (CPM), find every critical activity.
    
    Tasks: A B C D E F G H I J
    Durations:
      A: 6  B: 7  C: 2  D: 10  E: 5  F: 9  G: 6  H: 9  I: 2  J: 4
    Precedence (a -> b means a must finish before b starts):
      A -> I D
      B -> F
      C -> (none)
      D -> F H I
      E -> G J F
      F -> (none)
      G -> H
      H -> I J
      I -> J
      J -> (none)
    
    The answer is the space-separated, alphabetically sorted list of critical task labels (for example "A C E").

**Answer:**

    A D H I J

### Example 2

**Prompt:**

    Consider a project made of precedence-constrained tasks, each with a duration. A task can start only once all its predecessors have finished. Compute the earliest and latest start times for every task, then the slack of each task (latest minus earliest start). A task is critical when its slack is exactly zero. Using the critical-path method (CPM), find every critical activity.
    
    Tasks: A B C D E F G H I J
    Durations:
      A: 4  B: 4  C: 3  D: 6  E: 7  F: 3  G: 11  H: 3  I: 7  J: 4
    Precedence (a -> b means a must finish before b starts):
      A -> D F
      B -> I F
      C -> (none)
      D -> J I
      E -> (none)
      F -> (none)
      G -> (none)
      H -> J I
      I -> J
      J -> (none)
    
    The answer is the space-separated, alphabetically sorted list of critical task labels (for example "A C E").

**Answer:**

    A D I J

## Level 5

### Example 1

**Prompt:**

    Consider a project made of precedence-constrained tasks, each with a duration. A task can start only once all its predecessors have finished. Compute the earliest and latest start times for every task, then the slack of each task (latest minus earliest start). A task is critical when its slack is exactly zero. Using the critical-path method (CPM), find every critical activity.
    
    Tasks: A B C D E F G H I J K L M N O P Q R S
    Durations:
      A: 7  B: 5  C: 5  D: 10  E: 3  F: 2  G: 9  H: 11  I: 16  J: 15  K: 4  L: 6  M: 1  N: 17  O: 14  P: 5  Q: 17  R: 8  S: 11
    Precedence (a -> b means a must finish before b starts):
      A -> L D
      B -> (none)
      C -> I F G S
      D -> (none)
      E -> I
      F -> (none)
      G -> (none)
      H -> K O
      I -> Q P
      J -> M S O
      K -> S
      L -> Q N M
      M -> (none)
      N -> R
      O -> (none)
      P -> R Q S
      Q -> R
      R -> S
      S -> (none)
    
    The answer is the space-separated, alphabetically sorted list of critical task labels (for example "A C E").

**Answer:**

    C I P Q R S

### Example 2

**Prompt:**

    Consider a project made of precedence-constrained tasks, each with a duration. A task can start only once all its predecessors have finished. Compute the earliest and latest start times for every task, then the slack of each task (latest minus earliest start). A task is critical when its slack is exactly zero. Using the critical-path method (CPM), find every critical activity.
    
    Tasks: A B C D E F G H I J K L M N O P Q R S
    Durations:
      A: 14  B: 6  C: 3  D: 14  E: 12  F: 18  G: 12  H: 2  I: 14  J: 7  K: 15  L: 5  M: 1  N: 20  O: 15  P: 8  Q: 4  R: 3  S: 8
    Precedence (a -> b means a must finish before b starts):
      A -> D O I
      B -> K P
      C -> N J E
      D -> S M L F
      E -> G
      F -> (none)
      G -> J N
      H -> I R K
      I -> J K M
      J -> S R
      K -> L O P
      L -> M R
      M -> Q N P
      N -> (none)
      O -> Q
      P -> (none)
      Q -> S R
      R -> S
      S -> (none)
    
    The answer is the space-separated, alphabetically sorted list of critical task labels (for example "A C E").

**Answer:**

    A I K O Q R S

