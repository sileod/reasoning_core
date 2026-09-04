# Level 0

## Example 1

### Prompt

The control-flow graph has nodes 0..2 (node 0 is the entry, node 2 the exit).
Edges (from -> to):
0 -> 1, 1 -> 2

Each node k0 creates one definition with a unique id and may kill a set of older definition ids (those no longer valid past it). Node k0:
k0: gen {623}, kill {-}
k1: gen {407}, kill {623}
k2: gen {596}, kill {623}

Compute the set of definition ids that reach the entry of node 0.
A definition reaches a point if some execution path from its creating node to that point never passes through a node that kills it. At the entry of node 0 this set is empty.

Give the answer as the definition ids in the reaching set, space-separated in increasing order. If the set is empty, answer "-".

### Answer

-

## Example 2

### Prompt

The control-flow graph has nodes 0..2 (node 0 is the entry, node 2 the exit).
Edges (from -> to):
0 -> 1, 1 -> 2

Each node k0 creates one definition with a unique id and may kill a set of older definition ids (those no longer valid past it). Node k0:
k0: gen {904}, kill {-}
k1: gen {423}, kill {904}
k2: gen {242}, kill {423 904}

Compute the set of definition ids that reach the entry of node 0.
A definition reaches a point if some execution path from its creating node to that point never passes through a node that kills it. At the entry of node 0 this set is empty.

Give the answer as the definition ids in the reaching set, space-separated in increasing order. If the set is empty, answer "-".

### Answer

-



# Level 2

## Example 1

### Prompt

The control-flow graph has nodes 0..6 (node 0 is the entry, node 6 the exit).
Edges (from -> to):
0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5, 5 -> 6, 3 -> 4, 1 -> 2, 5 -> 4

Each node k0 creates one definition with a unique id and may kill a set of older definition ids (those no longer valid past it). Node k0:
k0: gen {813}, kill {-}
k1: gen {347}, kill {-}
k2: gen {385}, kill {-}
k3: gen {717}, kill {-}
k4: gen {506}, kill {-}
k5: gen {809}, kill {347 385 506 717 813}
k6: gen {105}, kill {347 385 506 717 809 813}

Compute the set of definition ids that reach the entry of node 6.
A definition reaches a point if some execution path from its creating node to that point never passes through a node that kills it. At the entry of node 0 this set is empty.

Give the answer as the definition ids in the reaching set, space-separated in increasing order. If the set is empty, answer "-".

### Answer

809

## Example 2

### Prompt

The control-flow graph has nodes 0..6 (node 0 is the entry, node 6 the exit).
Edges (from -> to):
0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5, 5 -> 6, 4 -> 5, 1 -> 2, 3 -> 2

Each node k0 creates one definition with a unique id and may kill a set of older definition ids (those no longer valid past it). Node k0:
k0: gen {139}, kill {-}
k1: gen {853}, kill {139}
k2: gen {508}, kill {-}
k3: gen {964}, kill {139 508 853}
k4: gen {819}, kill {508}
k5: gen {308}, kill {508 819}
k6: gen {234}, kill {139 308}

Compute the set of definition ids that reach the entry of node 6.
A definition reaches a point if some execution path from its creating node to that point never passes through a node that kills it. At the entry of node 0 this set is empty.

Give the answer as the definition ids in the reaching set, space-separated in increasing order. If the set is empty, answer "-".

### Answer

308 964



# Level 5

## Example 1

### Prompt

The control-flow graph has nodes 0..12 (node 0 is the entry, node 12 the exit).
Edges (from -> to):
0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5, 5 -> 6, 6 -> 7, 7 -> 8, 8 -> 9, 9 -> 10, 10 -> 11, 11 -> 12, 6 -> 7, 3 -> 4, 8 -> 9, 1 -> 2, 10 -> 11, 5 -> 4, 2 -> 1

Each node k0 creates one definition with a unique id and may kill a set of older definition ids (those no longer valid past it). Node k0:
k0: gen {275}, kill {-}
k1: gen {406}, kill {-}
k2: gen {781}, kill {-}
k3: gen {548}, kill {275 781}
k4: gen {967}, kill {275 406 548}
k5: gen {665}, kill {406 548}
k6: gen {672}, kill {275 406 548 665 781 967}
k7: gen {126}, kill {665 672 781}
k8: gen {570}, kill {-}
k9: gen {438}, kill {126 275 406 548 570 665 672 781}
k10: gen {416}, kill {570}
k11: gen {925}, kill {126 275 406 438 548 665 967}
k12: gen {769}, kill {416 438}

Compute the set of definition ids that reach the entry of node 8.
A definition reaches a point if some execution path from its creating node to that point never passes through a node that kills it. At the entry of node 0 this set is empty.

Give the answer as the definition ids in the reaching set, space-separated in increasing order. If the set is empty, answer "-".

### Answer

126

## Example 2

### Prompt

The control-flow graph has nodes 0..12 (node 0 is the entry, node 12 the exit).
Edges (from -> to):
0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5, 5 -> 6, 6 -> 7, 7 -> 8, 8 -> 9, 9 -> 10, 10 -> 11, 11 -> 12, 6 -> 7, 9 -> 10, 1 -> 2, 4 -> 5, 9 -> 8, 5 -> 4

Each node k0 creates one definition with a unique id and may kill a set of older definition ids (those no longer valid past it). Node k0:
k0: gen {645}, kill {-}
k1: gen {783}, kill {645}
k2: gen {919}, kill {645}
k3: gen {526}, kill {645}
k4: gen {289}, kill {-}
k5: gen {760}, kill {289 526 645 783}
k6: gen {535}, kill {289 526 645 760 919}
k7: gen {873}, kill {526 645 783}
k8: gen {515}, kill {289 645}
k9: gen {217}, kill {526 645 760 783 873}
k10: gen {576}, kill {645 783}
k11: gen {664}, kill {217 289 515 526 535 576 645 760 783 873}
k12: gen {494}, kill {289 873}

Compute the set of definition ids that reach the entry of node 10.
A definition reaches a point if some execution path from its creating node to that point never passes through a node that kills it. At the entry of node 0 this set is empty.

Give the answer as the definition ids in the reaching set, space-separated in increasing order. If the set is empty, answer "-".

### Answer

217 515 535


