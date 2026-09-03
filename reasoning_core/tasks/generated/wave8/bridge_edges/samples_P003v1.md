# bridge_edges samples

The answer format is the semicolon-separated list of bridge edges, each edge as the two node numbers with the smaller first (e.g. 1-4), listed in lexicographic order, or 'none' if the graph has no bridges.

## Level 0

Prompt:

Nodes:
[29]

Edges:
[[0, 1], [0, 2], [0, 4], [0, 5], [1, 2], [2, 3], [2, 4], [3, 4], [5, 6], [5, 8], [5, 9], [6, 7], [7, 8], [9, 10], [9, 11], [9, 12], [10, 11], [12, 13], [12, 14], [12, 15], [13, 14], [15, 16], [15, 17], [15, 18], [16, 17], [18, 19], [18, 21], [18, 22], [19, 20], [20, 21], [22, 23], [22, 24], [22, 25], [23, 24], [25, 26], [25, 28], [26, 27], [26, 28], [27, 28]]

The nodes are numbered 0 through 28 (the label lists the 29 nodes and 39 edges). Name every bridge edge -- an edge whose removal disconnects the graph -- each as the two node numbers with the smaller first, e.g. 1-4. List all bridges separated by semicolons, in lexicographic order of the pairs.

The answer is the semicolon-separated list of bridges, or 'none' if there are none.

Answer:

0-5; 5-9; 9-12; 12-15; 15-18; 18-22; 22-25

Prompt:

Nodes:
[17]

Edges:
[[0, 1], [0, 3], [0, 4], [0, 5], [1, 2], [1, 4], [2, 3], [3, 4], [5, 6], [5, 9], [5, 10], [6, 7], [6, 8], [7, 8], [8, 9], [10, 11], [10, 13], [10, 14], [11, 12], [12, 13], [14, 15], [14, 16], [15, 16]]

The nodes are numbered 0 through 16 (the label lists the 17 nodes and 23 edges). Name every bridge edge -- an edge whose removal disconnects the graph -- each as the two node numbers with the smaller first, e.g. 1-4. List all bridges separated by semicolons, in lexicographic order of the pairs.

The answer is the semicolon-separated list of bridges, or 'none' if there are none.

Answer:

0-5; 5-10; 10-14


## Level 2

Prompt:

Nodes:
[48]

Edges:
[[0, 1], [0, 4], [0, 5], [0, 6], [1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 5], [3, 4], [4, 5], [6, 7], [6, 12], [6, 13], [7, 8], [8, 9], [8, 12], [9, 10], [9, 11], [10, 11], [11, 12], [13, 14], [13, 18], [13, 19], [14, 15], [15, 16], [15, 17], [16, 17], [17, 18], [19, 20], [19, 22], [19, 23], [19, 24], [20, 21], [20, 22], [21, 22], [22, 23], [24, 25], [24, 29], [24, 30], [25, 26], [26, 27], [27, 28], [27, 29], [28, 29], [30, 31], [30, 36], [30, 37], [31, 32], [32, 33], [33, 34], [34, 35], [34, 36], [35, 36], [37, 38], [37, 41], [37, 42], [38, 39], [38, 41], [39, 40], [40, 41], [42, 43], [42, 47], [43, 44], [43, 45], [44, 45], [44, 46], [44, 47], [45, 46], [46, 47]]

The nodes are numbered 0 through 47 (the label lists the 48 nodes and 71 edges). Name every bridge edge -- an edge whose removal disconnects the graph -- each as the two node numbers with the smaller first, e.g. 1-4. List all bridges separated by semicolons, in lexicographic order of the pairs.

The answer is the semicolon-separated list of bridges, or 'none' if there are none.

Answer:

0-6; 6-13; 13-19; 19-24; 24-30; 30-37; 37-42

Prompt:

Nodes:
[64]

Edges:
[[0, 1], [0, 6], [0, 7], [1, 2], [1, 6], [2, 3], [3, 4], [4, 5], [5, 6], [7, 8], [7, 12], [7, 13], [8, 9], [8, 10], [9, 10], [9, 12], [10, 11], [11, 12], [13, 14], [13, 16], [13, 19], [13, 20], [14, 15], [14, 18], [15, 16], [16, 17], [17, 18], [18, 19], [20, 21], [20, 24], [20, 25], [21, 22], [22, 23], [23, 24], [25, 26], [25, 30], [25, 31], [25, 32], [26, 27], [27, 28], [27, 29], [27, 30], [27, 31], [28, 29], [29, 30], [30, 31], [32, 33], [32, 37], [32, 38], [33, 34], [33, 36], [33, 37], [34, 35], [35, 36], [35, 37], [36, 37], [38, 39], [38, 42], [38, 44], [38, 45], [39, 40], [40, 41], [41, 42], [42, 43], [43, 44], [45, 46], [45, 48], [45, 49], [45, 50], [46, 47], [47, 48], [48, 49], [50, 51], [50, 56], [50, 57], [51, 52], [51, 55], [52, 53], [52, 56], [53, 54], [54, 55], [55, 56], [57, 58], [57, 60], [57, 63], [58, 59], [59, 60], [60, 61], [61, 62], [61, 63], [62, 63]]

The nodes are numbered 0 through 63 (the label lists the 64 nodes and 91 edges). Name every bridge edge -- an edge whose removal disconnects the graph -- each as the two node numbers with the smaller first, e.g. 1-4. List all bridges separated by semicolons, in lexicographic order of the pairs.

The answer is the semicolon-separated list of bridges, or 'none' if there are none.

Answer:

0-7; 7-13; 13-20; 20-25; 25-32; 32-38; 38-45; 45-50; 50-57


## Level 5

Prompt:

Nodes:
[79]

Edges:
[[0, 1], [0, 2], [0, 5], [0, 7], [0, 8], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7], [8, 9], [8, 11], [8, 15], [8, 16], [9, 10], [10, 11], [10, 15], [11, 12], [12, 13], [12, 15], [13, 14], [14, 15], [16, 17], [16, 18], [16, 25], [16, 26], [17, 18], [17, 25], [18, 19], [19, 20], [19, 22], [19, 24], [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [26, 27], [26, 33], [26, 34], [27, 28], [27, 33], [28, 29], [28, 33], [29, 30], [30, 31], [31, 32], [31, 33], [32, 33], [34, 35], [34, 38], [34, 42], [34, 43], [35, 36], [36, 37], [37, 38], [37, 41], [37, 42], [38, 39], [38, 40], [39, 40], [40, 41], [41, 42], [43, 44], [43, 48], [43, 49], [43, 51], [43, 52], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49], [49, 50], [50, 51], [52, 53], [52, 60], [52, 61], [53, 54], [53, 57], [53, 60], [54, 55], [54, 59], [55, 56], [56, 57], [57, 58], [58, 59], [59, 60], [61, 62], [61, 68], [61, 69], [62, 63], [62, 67], [62, 68], [63, 64], [63, 66], [64, 65], [65, 66], [66, 67], [66, 68], [67, 68], [69, 70], [69, 72], [69, 78], [70, 71], [70, 75], [71, 72], [71, 78], [72, 73], [72, 76], [73, 74], [74, 75], [74, 77], [74, 78], [75, 76], [76, 77], [77, 78]]

The nodes are numbered 0 through 78 (the label lists the 79 nodes and 119 edges). Name every bridge edge -- an edge whose removal disconnects the graph -- each as the two node numbers with the smaller first, e.g. 1-4. List all bridges separated by semicolons, in lexicographic order of the pairs.

The answer is the semicolon-separated list of bridges, or 'none' if there are none.

Answer:

0-8; 8-16; 16-26; 26-34; 34-43; 43-52; 52-61; 61-69

Prompt:

Nodes:
[87]

Edges:
[[0, 1], [0, 6], [0, 7], [0, 8], [1, 2], [1, 5], [1, 6], [2, 3], [2, 7], [3, 4], [4, 5], [5, 6], [6, 7], [8, 9], [8, 11], [8, 16], [8, 17], [9, 10], [10, 11], [10, 16], [11, 12], [12, 13], [12, 14], [13, 14], [13, 15], [13, 16], [14, 15], [15, 16], [17, 18], [17, 21], [17, 26], [17, 27], [18, 19], [18, 21], [18, 23], [19, 20], [20, 21], [20, 22], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [27, 28], [27, 32], [27, 34], [27, 35], [28, 29], [28, 31], [29, 30], [30, 31], [31, 32], [32, 33], [33, 34], [35, 36], [35, 43], [35, 44], [36, 37], [36, 38], [37, 38], [38, 39], [38, 42], [38, 43], [39, 40], [39, 42], [39, 43], [40, 41], [41, 42], [42, 43], [44, 45], [44, 48], [44, 51], [44, 52], [45, 46], [45, 48], [46, 47], [46, 49], [47, 48], [48, 49], [49, 50], [50, 51], [52, 53], [52, 54], [52, 55], [52, 59], [52, 60], [53, 54], [53, 55], [53, 57], [54, 55], [54, 56], [55, 56], [56, 57], [57, 58], [58, 59], [60, 61], [60, 63], [60, 65], [60, 69], [60, 70], [61, 62], [61, 66], [62, 63], [63, 64], [64, 65], [65, 66], [65, 67], [66, 67], [66, 68], [67, 68], [68, 69], [70, 71], [70, 78], [70, 79], [71, 72], [72, 73], [72, 77], [73, 74], [73, 77], [74, 75], [74, 76], [74, 77], [75, 76], [76, 77], [77, 78], [79, 80], [79, 81], [79, 84], [79, 85], [79, 86], [80, 81], [81, 82], [81, 86], [82, 83], [82, 86], [83, 84], [84, 85], [85, 86]]

The nodes are numbered 0 through 86 (the label lists the 87 nodes and 138 edges). Name every bridge edge -- an edge whose removal disconnects the graph -- each as the two node numbers with the smaller first, e.g. 1-4. List all bridges separated by semicolons, in lexicographic order of the pairs.

The answer is the semicolon-separated list of bridges, or 'none' if there are none.

Answer:

0-8; 8-17; 17-27; 27-35; 35-44; 44-52; 52-60; 60-70; 70-79

