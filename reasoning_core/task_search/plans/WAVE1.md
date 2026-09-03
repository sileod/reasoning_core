# WAVE1 — fully new procedural task candidates

```yaml
wave: 1
kind: generated_only
count: 80
novelty_screen:
  - GALLERY.md
  - reasoning_core/task_search/WAVE0.md
  - WAVE0 explicit exclusions

tasks:
  - id: W1-001
    name: strongly_connected_component
    family: graph
    description: Given a directed graph and node, output the sorted members of its strongly connected component.
    verifier: networkx

  - id: W1-002
    name: articulation_vertices
    family: graph
    description: Given an undirected graph, output all articulation vertices in sorted order.
    verifier: networkx

  - id: W1-003
    name: bridge_edges
    family: graph
    description: Given an undirected graph, output all bridge edges canonically.
    verifier: networkx

  - id: W1-004
    name: core_number
    family: graph
    description: Given an undirected graph and node, output that node's k-core number.
    verifier: networkx

  - id: W1-005
    name: eulerian_status
    family: graph
    description: Classify a graph as none, open Euler trail, or Euler circuit; give the canonical start if applicable.
    verifier: networkx

  - id: W1-006
    name: topological_generation
    family: graph
    description: Given a DAG and node, output its round under repeated zero-indegree removal.
    verifier: networkx

  - id: W1-007
    name: graph_planarity
    family: graph
    description: Given an undirected graph, answer whether it is planar.
    verifier: networkx

  - id: W1-008
    name: graph_chordality
    family: graph
    description: Given an undirected graph, answer whether every cycle of length at least four has a chord.
    verifier: networkx

  - id: W1-009
    name: heap_key_update
    family: data_structure
    description: Apply one key increase or decrease in a binary heap and output the key's final index.
    verifier: reference

  - id: W1-010
    name: open_addressing_slot
    family: data_structure
    description: Insert keys under a stated open-addressing probe rule and output the queried key's slot.
    verifier: reference

  - id: W1-011
    name: bloom_filter_membership
    family: data_structure
    description: Apply explicit hashes to a Bloom filter and classify a query as definitely absent or possibly present.
    verifier: reference

  - id: W1-012
    name: trie_unique_prefix
    family: data_structure
    description: Given strings, output the shortest prefix uniquely identifying the target.
    verifier: reference

  - id: W1-013
    name: fenwick_prefix_nodes
    family: data_structure
    description: Given an index, output the Fenwick-tree indices visited by its prefix query.
    verifier: reference

  - id: W1-014
    name: union_find_representative
    family: data_structure
    description: Execute canonical union-by-rank operations and output a queried element's representative.
    verifier: reference

  - id: W1-015
    name: btree_promoted_key
    family: data_structure
    description: Given a small B-tree and one insertion, output the first promoted key or None.
    verifier: reference

  - id: W1-016
    name: binary_search_probes
    family: algorithm
    description: Given a sorted array and target, output the indices probed by deterministic binary search.
    verifier: reference

  - id: W1-017
    name: skip_list_search_path
    family: data_structure
    description: Given explicit skip-list levels and a target, output visited keys in search order.
    verifier: reference

  - id: W1-018
    name: red_black_black_height
    family: data_structure
    description: Given a valid red-black tree and node, output its black-height.
    verifier: reference

  - id: W1-019
    name: finite_group_inverse
    family: algebra
    description: Given a Cayley table and element, output its inverse.
    verifier: reference

  - id: W1-020
    name: finite_group_element_order
    family: algebra
    description: Given a Cayley table and element, output the element's order.
    verifier: reference

  - id: W1-021
    name: generated_subgroup_membership
    family: algebra
    description: Given a finite group, generators, and element, answer whether it lies in the generated subgroup.
    verifier: reference

  - id: W1-022
    name: left_coset_identification
    family: algebra
    description: Given a subgroup and element, output the canonical sorted left coset.
    verifier: reference

  - id: W1-023
    name: group_homomorphism_check
    family: algebra
    description: Given two finite group tables and a mapping, answer whether the mapping is a homomorphism.
    verifier: reference

  - id: W1-024
    name: finite_relation_properties
    family: relations
    description: Given a finite relation, output which of reflexive, symmetric, antisymmetric, and transitive hold.
    verifier: reference

  - id: W1-025
    name: poset_cover_query
    family: relations
    description: Given a finite partial order, output the elements covering a queried element.
    verifier: reference

  - id: W1-026
    name: lattice_join_meet
    family: relations
    description: Given a finite lattice and two elements, output their join and meet.
    verifier: reference

  - id: W1-027
    name: fd_attribute_closure
    family: database
    description: Given functional dependencies and attributes, output their attribute closure.
    verifier: reference

  - id: W1-028
    name: candidate_key_minimality
    family: database
    description: Classify an attribute set as non-superkey, nonminimal superkey, or candidate key.
    verifier: reference

  - id: W1-029
    name: bcnf_violation
    family: database
    description: Given a schema and dependencies, output the first canonical BCNF violation or None.
    verifier: reference

  - id: W1-030
    name: lossless_binary_decomposition
    family: database
    description: Given dependencies and a binary schema decomposition, answer whether it is lossless.
    verifier: reference

  - id: W1-031
    name: conflict_serializability
    family: database
    description: Given a transaction schedule, determine conflict serializability and give the unique serial order if one exists.
    verifier: networkx

  - id: W1-032
    name: schedule_recoverability
    family: database
    description: Classify a schedule as strict, cascadeless, recoverable-only, or unrecoverable.
    verifier: reference

  - id: W1-033
    name: mvcc_visibility
    family: database
    description: Given transaction timestamps and row versions, output the version visible to a queried snapshot.
    verifier: reference

  - id: W1-034
    name: two_phase_lock_blocker
    family: database
    description: Given lock requests under strict 2PL, output the transaction blocking a queried request or None.
    verifier: reference

  - id: W1-035
    name: lamport_clock
    family: distributed
    description: Given local events and message edges, output the Lamport timestamp of a queried event.
    verifier: reference

  - id: W1-036
    name: vector_clock_order
    family: distributed
    description: Given two vector timestamps, classify them as before, after, equal, or concurrent.
    verifier: reference

  - id: W1-037
    name: consistent_distributed_cut
    family: distributed
    description: Given process prefixes and message edges, answer whether the selected global cut is consistent.
    verifier: reference

  - id: W1-038
    name: linearizability_check
    family: distributed
    description: Given a small concurrent history and sequential specification, answer whether the history is linearizable.
    verifier: exhaustive_small

  - id: W1-039
    name: wait_for_deadlock
    family: distributed
    description: Given a wait-for graph, output the transactions participating in deadlock cycles.
    verifier: networkx

  - id: W1-040
    name: quorum_intersection
    family: distributed
    description: Given replica and quorum sizes, answer whether every read quorum must intersect every write quorum.
    verifier: reference

  - id: W1-041
    name: paxos_chosen_value
    family: distributed
    description: Given acceptor votes for numbered proposals, output the value already chosen or None.
    verifier: reference

  - id: W1-042
    name: raft_vote_eligibility
    family: distributed
    description: Given candidate and voter terms plus log metadata, answer whether the voter may grant the vote.
    verifier: reference

  - id: W1-043
    name: virtual_address_translation
    family: systems
    description: Given page tables and a virtual address, output the physical address or page fault.
    verifier: reference

  - id: W1-044
    name: set_associative_cache
    family: systems
    description: Given cache state and an access, output hit or miss and the affected way.
    verifier: reference

  - id: W1-045
    name: clock_page_replacement
    family: systems
    description: Given Clock replacement state and a page fault, output the frame chosen for eviction.
    verifier: reference

  - id: W1-046
    name: longest_prefix_route
    family: network
    description: Given bit-prefix routes and a destination bitstring, output the selected next hop.
    verifier: reference

  - id: W1-047
    name: unix_mode_permission
    family: security
    description: Given Unix mode metadata and user identity, answer whether a requested access is permitted.
    verifier: reference

  - id: W1-048
    name: posix_acl_permission
    family: security
    description: Given POSIX ACL entries, mask, ownership, and groups, output the effective permission set.
    verifier: reference

  - id: W1-049
    name: buddy_allocator
    family: systems
    description: Apply one allocation or free to a buddy allocator and output the canonical free-block sizes.
    verifier: reference

  - id: W1-050
    name: round_robin_completion
    family: systems
    description: Given arrivals, burst lengths, and quantum, output process completion order under Round Robin.
    verifier: reference

  - id: W1-051
    name: bgp_best_path
    family: network
    description: Given BGP route attributes, output the route selected by a stated best-path rule sequence.
    verifier: reference

  - id: W1-052
    name: firewall_rule_shadowing
    family: security
    description: Given ordered finite-domain firewall rules, output the first fully shadowed rule or None.
    verifier: exhaustive_small

  - id: W1-053
    name: lexical_scope_resolution
    family: static_semantics
    description: Given nested scopes and declarations, output the declaration bound to a queried identifier occurrence.
    verifier: reference

  - id: W1-054
    name: c3_linearization
    family: static_semantics
    description: Given a multiple-inheritance DAG, output the C3 method-resolution order of a queried class.
    verifier: python_mro

  - id: W1-055
    name: nominal_subtyping
    family: static_semantics
    description: Given a nominal type hierarchy, answer whether one type is a subtype of another.
    verifier: reference

  - id: W1-056
    name: generic_variance_subtyping
    family: static_semantics
    description: Given variance declarations and ground types, answer a parameterized subtype query.
    verifier: reference

  - id: W1-057
    name: overload_resolution
    family: static_semantics
    description: Given overload signatures and argument types, output the unique most-specific applicable overload or Ambiguous.
    verifier: reference

  - id: W1-058
    name: struct_layout_alignment
    family: static_semantics
    description: Given field sizes, alignments, and ABI rules, output a field offset or total struct size.
    verifier: reference

  - id: W1-059
    name: virtual_method_dispatch
    family: static_semantics
    description: Given hierarchy, overrides, and runtime class, output the method definition that is invoked.
    verifier: reference

  - id: W1-060
    name: pattern_match_exhaustiveness
    family: static_semantics
    description: Given a finite algebraic datatype and pattern clauses, answer whether all constructors are covered.
    verifier: exhaustive_small

  - id: W1-061
    name: borda_winner
    family: social_choice
    description: Given ranked ballots, output the Borda winner with deterministic tie-breaking.
    verifier: reference

  - id: W1-062
    name: instant_runoff_winner
    family: social_choice
    description: Given ranked ballots, execute instant-runoff elimination and output the winner.
    verifier: reference

  - id: W1-063
    name: condorcet_winner
    family: social_choice
    description: Given ranked ballots, output the Condorcet winner or None.
    verifier: reference

  - id: W1-064
    name: dhondt_apportionment
    family: social_choice
    description: Given party vote totals and seat count, output the final D'Hondt seat vector.
    verifier: reference

  - id: W1-065
    name: approval_voting_winner
    family: social_choice
    description: Given approval ballots, output the winner with deterministic tie-breaking.
    verifier: reference

  - id: W1-066
    name: top_trading_cycles
    family: allocation
    description: Given agents, owned items, and strict preferences, output the item assigned to a queried agent.
    verifier: reference

  - id: W1-067
    name: canonical_huffman_code
    family: coding
    description: Given symbol frequencies, build a tie-broken Huffman tree and output one symbol's canonical codeword.
    verifier: heapq

  - id: W1-068
    name: crc_remainder
    family: coding
    description: Given a bitstring and generator polynomial, output the CRC remainder.
    verifier: reference

  - id: W1-069
    name: kraft_feasibility
    family: coding
    description: Given prefix-code word lengths, answer whether they satisfy Kraft's inequality.
    verifier: reference

  - id: W1-070
    name: prefix_code_decode
    family: coding
    description: Given a prefix-free codebook and bitstream, output the decoded symbol sequence.
    verifier: reference

  - id: W1-071
    name: suffix_array_rank
    family: string_algorithm
    description: Given a string and suffix start index, output that suffix's rank in the suffix array.
    verifier: reference

  - id: W1-072
    name: kmp_prefix_value
    family: string_algorithm
    description: Given a pattern and position, output the KMP prefix-function value at that position.
    verifier: reference

  - id: W1-073
    name: dimensional_consistency
    family: measurement
    description: Given dimensions for variables and an equation, answer whether the equation is dimensionally valid.
    verifier: sympy_or_vectors

  - id: W1-074
    name: missing_dimension
    family: measurement
    description: Given a valid formula with one unknown quantity dimension, output its base-dimension exponent vector.
    verifier: sympy_or_vectors

  - id: W1-075
    name: gregorian_weekday
    family: calendar
    description: Given a valid Gregorian date, output its weekday.
    verifier: datetime

  - id: W1-076
    name: cron_next_fire
    family: calendar
    description: Given a restricted numeric cron expression and timestamp, output the next matching timestamp.
    verifier: reference

  - id: W1-077
    name: petri_enabled_transitions
    family: resource_system
    description: Given a Petri-net marking and weighted arcs, output all currently enabled transitions.
    verifier: reference

  - id: W1-078
    name: petri_conflict_pair
    family: resource_system
    description: Output the canonical pair of individually enabled transitions that compete for insufficient shared tokens, or None.
    verifier: reference

  - id: W1-079
    name: incremental_build_rebuild_set
    family: dependency_system
    description: Given a build dependency DAG and changed sources, output all targets that must rebuild.
    verifier: networkx

  - id: W1-080
    name: semantic_version_precedence
    family: software_protocol
    description: Given two SemVer strings, output which has higher precedence or equal.
    verifier: packaging
```

The strongest underrepresented clusters are distributed/concurrent reasoning, static semantics, database transaction semantics, finite algebra, and systems semantics. These should be overweighted in an actual Wave1 screening run rather than allocating uniformly across all 80.
