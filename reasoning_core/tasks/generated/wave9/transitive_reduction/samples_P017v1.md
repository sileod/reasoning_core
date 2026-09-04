## Level 0

### Prompt

A directed acyclic graph has nodes n0..n6 and directed edges: n0 -> n1 n0 -> n2 n0 -> n3 n0 -> n5 n1 -> n4 n3 -> n2 n3 -> n4 n3 -> n5 n5 -> n4 n6 -> n3 n6 -> n4. The transitive reduction of a DAG is the unique minimal subgraph with the same reachability, obtained by removing every edge whose endpoints are already connected by an alternative directed path. List the edges of this DAG's transitive reduction, each in the form 'a -> b', separated by a semicolon and sorted lexicographically by source then target, for example 'n0 -> n1; n0 -> n3'.

### Answer

n0 -> n1; n0 -> n3; n1 -> n4; n3 -> n2; n3 -> n5; n5 -> n4; n6 -> n3

### Prompt

A directed acyclic graph has nodes n0..n6 and directed edges: n0 -> n1 n0 -> n5 n0 -> n6 n1 -> n4 n2 -> n0 n2 -> n1 n2 -> n3 n2 -> n6 n3 -> n1 n3 -> n4 n3 -> n5 n5 -> n1 n5 -> n4 n6 -> n1 n6 -> n4 n6 -> n5. The transitive reduction of a DAG is the unique minimal subgraph with the same reachability, obtained by removing every edge whose endpoints are already connected by an alternative directed path. List the edges of this DAG's transitive reduction, each in the form 'a -> b', separated by a semicolon and sorted lexicographically by source then target, for example 'n0 -> n1; n0 -> n3'.

### Answer

n0 -> n6; n1 -> n4; n2 -> n0; n2 -> n3; n3 -> n5; n5 -> n1; n6 -> n5

## Level 2

### Prompt

A directed acyclic graph has nodes n0..n9 and directed edges: n0 -> n4 n1 -> n8 n2 -> n5 n3 -> n0 n3 -> n5 n5 -> n4 n5 -> n7 n6 -> n4 n8 -> n4 n8 -> n7 n9 -> n8. The transitive reduction of a DAG is the unique minimal subgraph with the same reachability, obtained by removing every edge whose endpoints are already connected by an alternative directed path. List the edges of this DAG's transitive reduction, each in the form 'a -> b', separated by a semicolon and sorted lexicographically by source then target, for example 'n0 -> n1; n0 -> n3'.

### Answer

n0 -> n4; n1 -> n8; n2 -> n5; n3 -> n0; n3 -> n5; n5 -> n4; n5 -> n7; n6 -> n4; n8 -> n4; n8 -> n7; n9 -> n8

### Prompt

A directed acyclic graph has nodes n0..n9 and directed edges: n0 -> n5 n0 -> n6 n1 -> n4 n2 -> n4 n2 -> n7 n5 -> n3 n5 -> n9 n6 -> n1 n6 -> n4 n7 -> n3 n8 -> n0 n8 -> n1 n8 -> n9. The transitive reduction of a DAG is the unique minimal subgraph with the same reachability, obtained by removing every edge whose endpoints are already connected by an alternative directed path. List the edges of this DAG's transitive reduction, each in the form 'a -> b', separated by a semicolon and sorted lexicographically by source then target, for example 'n0 -> n1; n0 -> n3'.

### Answer

n0 -> n5; n0 -> n6; n1 -> n4; n2 -> n4; n2 -> n7; n5 -> n3; n5 -> n9; n6 -> n1; n7 -> n3; n8 -> n0

## Level 5

### Prompt

A directed acyclic graph has nodes n0..n14 and directed edges: n0 -> n12 n0 -> n2 n0 -> n3 n0 -> n6 n1 -> n0 n1 -> n13 n1 -> n7 n10 -> n13 n10 -> n4 n11 -> n10 n11 -> n13 n11 -> n5 n14 -> n0 n14 -> n10 n14 -> n12 n14 -> n2 n14 -> n3 n14 -> n8 n2 -> n10 n2 -> n3 n3 -> n10 n4 -> n13 n5 -> n2 n5 -> n6 n6 -> n3 n7 -> n13 n7 -> n14 n7 -> n3 n7 -> n5 n7 -> n6 n8 -> n0 n8 -> n3 n8 -> n4 n9 -> n12 n9 -> n3 n9 -> n8. The transitive reduction of a DAG is the unique minimal subgraph with the same reachability, obtained by removing every edge whose endpoints are already connected by an alternative directed path. List the edges of this DAG's transitive reduction, each in the form 'a -> b', separated by a semicolon and sorted lexicographically by source then target, for example 'n0 -> n1; n0 -> n3'.

### Answer

n0 -> n2; n0 -> n6; n0 -> n12; n1 -> n7; n2 -> n3; n3 -> n10; n4 -> n13; n5 -> n2; n5 -> n6; n6 -> n3; n7 -> n5; n7 -> n14; n8 -> n0; n9 -> n8; n10 -> n4; n11 -> n5; n14 -> n8

### Prompt

A directed acyclic graph has nodes n0..n14 and directed edges: n0 -> n11 n0 -> n14 n0 -> n3 n10 -> n14 n10 -> n8 n11 -> n1 n11 -> n5 n11 -> n7 n12 -> n1 n12 -> n14 n12 -> n6 n12 -> n8 n13 -> n8 n14 -> n7 n2 -> n1 n2 -> n11 n2 -> n13 n2 -> n14 n2 -> n5 n3 -> n12 n4 -> n10 n4 -> n12 n4 -> n14 n4 -> n3 n5 -> n3 n5 -> n7 n8 -> n14 n9 -> n10 n9 -> n13 n9 -> n14 n9 -> n3 n9 -> n7 n9 -> n8. The transitive reduction of a DAG is the unique minimal subgraph with the same reachability, obtained by removing every edge whose endpoints are already connected by an alternative directed path. List the edges of this DAG's transitive reduction, each in the form 'a -> b', separated by a semicolon and sorted lexicographically by source then target, for example 'n0 -> n1; n0 -> n3'.

### Answer

n0 -> n11; n2 -> n11; n2 -> n13; n3 -> n12; n4 -> n3; n4 -> n10; n5 -> n3; n8 -> n14; n9 -> n3; n9 -> n10; n9 -> n13; n10 -> n8; n11 -> n5; n12 -> n1; n12 -> n6; n12 -> n8; n13 -> n8; n14 -> n7
