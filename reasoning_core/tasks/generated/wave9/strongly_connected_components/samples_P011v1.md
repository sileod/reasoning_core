# Level 0

Consider the directed graph on nodes 0..6 with edges: 0->5;0->5;1->0;1->6;2->3;3->4;4->2;4->3;5->0;5->3;5->4;6->1;6->2;6->5. Partition its nodes into strongly connected components (SCCs), using Kosaraju's algorithm, where a component is a maximal set of nodes mutually reachable from each other. Give the component that contains node 0, with its nodes listed in increasing order separated by commas. The answer is that list, e.g. "1,2,5".

Answer: 0,5

Consider the directed graph on nodes 0..6 with edges: 0->2;0->2;1->3;1->5;1->6;2->4;3->0;3->2;3->5;4->0;5->2;5->3;6->1;6->2. Partition its nodes into strongly connected components (SCCs), where a component is a maximal set of nodes mutually reachable from each other. Give the complete partition: each component as its nodes in increasing order inside square brackets, components ordered by their smallest node, and components separated by semicolons. For example "[0,2];[1];[3,4]". The answer is that string.

Answer: [0,2,4];[1,6];[3,5]

# Level 2

Consider the directed graph on nodes 0..10 with edges: 0->1;0->9;1->4;2->3;2->5;2->9;3->5;3->8;3->10;3->10;4->6;5->8;6->4;6->5;6->7;7->0;7->8;8->9;9->5;10->2;10->5;10->8. Partition its nodes into strongly connected components (SCCs), using Kosaraju's algorithm, where a component is a maximal set of nodes mutually reachable from each other. Give the component that contains node 5, with its nodes listed in increasing order separated by commas. The answer is that list, e.g. "1,2,5".

Answer: 5,8,9

Consider the directed graph on nodes 0..10 with edges: 0->1;1->5;2->4;2->7;2->8;3->4;3->5;4->6;4->9;4->10;5->8;6->9;6->10;7->2;7->3;7->10;8->9;9->0;9->0;10->0;10->3. Partition its nodes into strongly connected components (SCCs), where a component is a maximal set of nodes mutually reachable from each other. Give the complete partition: each component as its nodes in increasing order inside square brackets, components ordered by their smallest node, and components separated by semicolons. For example "[0,2];[1];[3,4]". The answer is that string.

Answer: [0,1,5,8,9];[2,7];[3,4,6,10]

# Level 5

Consider the directed graph on nodes 0..16 with edges: 0->2;1->3;1->4;1->10;2->5;3->6;4->2;4->5;4->8;4->8;5->7;6->15;6->16;7->9;8->5;8->11;8->13;9->10;10->12;11->2;11->7;11->14;12->2;12->13;13->0;14->0;14->12;14->16;15->1;15->3;15->5;16->4;16->5. Partition its nodes into strongly connected components (SCCs), using Kosaraju's algorithm, where a component is a maximal set of nodes mutually reachable from each other. Give the component that contains node 0, with its nodes listed in increasing order separated by commas. The answer is that list, e.g. "1,2,5".

Answer: 0,2,5,7,9,10,12,13

Consider the directed graph on nodes 0..16 with edges: 0->4;0->7;1->2;1->3;1->4;2->7;3->0;3->8;3->9;3->11;4->2;4->6;5->10;5->10;6->4;6->7;6->12;7->8;7->8;8->11;9->6;9->13;9->15;10->2;10->14;11->15;12->0;12->8;12->11;13->0;13->1;13->8;14->1;14->16;15->2;16->5. Partition its nodes into strongly connected components (SCCs), where a component is a maximal set of nodes mutually reachable from each other. Give the complete partition: each component as its nodes in increasing order inside square brackets, components ordered by their smallest node, and components separated by semicolons. For example "[0,2];[1];[3,4]". The answer is that string.

Answer: [0,4,6,12];[1,3,9,13];[2,7,8,11,15];[5,10,14,16]
