# DiagnosticTestSelection samples (P007v1)

## Level 0

### Prompt

    A diagnostic clinic must identify which of several hypotheses is true. Prior probabilities for the hypotheses are given as a list.
    Hypothesis priors: 0.0066, 0.3692, 0.2152, 0.1988, 0.2101
    
    Each candidate test partitions the hypotheses into outcome blocks. A test returns one outcome, and the true hypothesis lies in exactly one block. Blocks are written as comma-separated hypothesis numbers grouped by '|'.
    Candidate tests:
      Test A: {1,4 | 2,3,5}
      Test B: {1,3,4 | 5 | 2}
      Test C: {1,3 | 2 | 4,5}
      Test D: {2,5 | 1,3,4}
    
    The residual ambiguity of an outcome block is the Shannon entropy (bits) of the normalized posterior over the hypotheses in that block. The expected ambiguity of a test is the probability-weighted sum across its outcome blocks. Choose the test with the LOWEST expected ambiguity; on a tie, choose the test with the alphabetical-least label.
    
    The answer is a single letter, the label of the best test, e.g. "B".

**Answer:** C

---

### Prompt

    A diagnostic clinic must identify which of several hypotheses is true. Prior probabilities for the hypotheses are given as a list.
    Hypothesis priors: 0.0262, 0.4197, 0.2137, 0.2088, 0.1316
    
    Each candidate test partitions the hypotheses into outcome blocks. A test returns one outcome, and the true hypothesis lies in exactly one block. Blocks are written as comma-separated hypothesis numbers grouped by '|'.
    Candidate tests:
      Test A: {3 | 4 | 1,2 | 5}
      Test B: {3 | 1,2,5 | 4}
      Test C: {4 | 5 | 1,2,3}
      Test D: {1,2 | 4 | 3,5}
    
    The residual ambiguity of an outcome block is the Shannon entropy (bits) of the normalized posterior over the hypotheses in that block. The expected ambiguity of a test is the probability-weighted sum across its outcome blocks. Choose the test with the LOWEST expected ambiguity; on a tie, choose the test with the alphabetical-least label.
    
    The answer is a single letter, the label of the best test, e.g. "B".

**Answer:** A

---

## Level 2

### Prompt

    A diagnostic clinic must identify which of several hypotheses is true. Prior probabilities for the hypotheses are given as a list.
    Hypothesis priors: 0.2909, 0.0829, 0.045, 0.2452, 0.1667, 0.0642, 0.1051
    
    Each candidate test partitions the hypotheses into outcome blocks. A test returns one outcome, and the true hypothesis lies in exactly one block. Blocks are written as comma-separated hypothesis numbers grouped by '|'.
    Candidate tests:
      Test A: {1,3 | 4,5,6,7 | 2}
      Test B: {3 | 2,5 | 1,4,6,7}
      Test C: {2 | 4,7 | 1 | 3,5 | 6}
      Test D: {5,7 | 6 | 1,2,3 | 4}
      Test E: {1,3,6 | 4 | 2,5,7}
      Test F: {2 | 5 | 1 | 3,7 | 4,6}
    
    The residual ambiguity of an outcome block is the Shannon entropy (bits) of the normalized posterior over the hypotheses in that block. The expected ambiguity of a test is the probability-weighted sum across its outcome blocks. Choose the test with the LOWEST expected ambiguity; on a tie, choose the test with the alphabetical-least label.
    
    The answer is a single letter, the label of the best test, e.g. "B".

**Answer:** F

---

### Prompt

    A diagnostic clinic must identify which of several hypotheses is true. Prior probabilities for the hypotheses are given as a list.
    Hypothesis priors: 0.178, 0.1905, 0.108, 0.007, 0.0151, 0.2775, 0.2239
    
    Each candidate test partitions the hypotheses into outcome blocks. A test returns one outcome, and the true hypothesis lies in exactly one block. Blocks are written as comma-separated hypothesis numbers grouped by '|'.
    Candidate tests:
      Test A: {6 | 7 | 1 | 3 | 2,4 | 5}
      Test B: {2,5,7 | 1,3 | 4,6}
      Test C: {1,4 | 2,3,5 | 7 | 6}
      Test D: {3 | 7 | 1,4 | 2,5 | 6}
      Test E: {3,4,5,7 | 1,6 | 2}
      Test F: {4,5 | 3,7 | 1 | 2,6}
    
    The residual ambiguity of an outcome block is the Shannon entropy (bits) of the normalized posterior over the hypotheses in that block. The expected ambiguity of a test is the probability-weighted sum across its outcome blocks. Choose the test with the LOWEST expected ambiguity; on a tie, choose the test with the alphabetical-least label.
    
    The answer is a single letter, the label of the best test, e.g. "B".

**Answer:** A

---

## Level 5

### Prompt

    A diagnostic clinic must identify which of several hypotheses is true. Prior probabilities for the hypotheses are given as a list.
    Hypothesis priors: 0.044, 0.1195, 0.1431, 0.1499, 0.1264, 0.0386, 0.0473, 0.036, 0.1479, 0.1473
    
    Each candidate test partitions the hypotheses into outcome blocks. A test returns one outcome, and the true hypothesis lies in exactly one block. Blocks are written as comma-separated hypothesis numbers grouped by '|'.
    Candidate tests:
      Test A: {1,10 | 3,7 | 2,8,9 | 4,5,6}
      Test B: {1,4,5,6,7 | 2,3,8,9,10}
      Test C: {8,9 | 5,6,10 | 1 | 7 | 2,4 | 3}
      Test D: {4,5,6 | 1,2,3,7,8,9,10}
      Test E: {3,5 | 6 | 1,4,9 | 2,7,8,10}
      Test F: {8 | 4 | 6,9 | 1 | 5,7 | 2,3 | 10}
      Test G: {3,4,7,10 | 1,2,5,6,8,9}
      Test H: {3,7 | 6 | 5 | 1,4 | 2 | 8 | 9,10}
      Test I: {2,9 | 10 | 4 | 3 | 1 | 5,7 | 6,8}
    
    The residual ambiguity of an outcome block is the Shannon entropy (bits) of the normalized posterior over the hypotheses in that block. The expected ambiguity of a test is the probability-weighted sum across its outcome blocks. Choose the test with the LOWEST expected ambiguity; on a tie, choose the test with the alphabetical-least label.
    
    The answer is a single letter, the label of the best test, e.g. "B".

**Answer:** I

---

### Prompt

    A diagnostic clinic must identify which of several hypotheses is true. Prior probabilities for the hypotheses are given as a list.
    Hypothesis priors: 0.1149, 0.0676, 0.1321, 0.0865, 0.1502, 0.1422, 0.0098, 0.163, 0.058, 0.0757
    
    Each candidate test partitions the hypotheses into outcome blocks. A test returns one outcome, and the true hypothesis lies in exactly one block. Blocks are written as comma-separated hypothesis numbers grouped by '|'.
    Candidate tests:
      Test A: {1,8,10 | 7 | 3,4,6 | 5 | 2,9}
      Test B: {5 | 4,9 | 1,10 | 3 | 8 | 2 | 6,7}
      Test C: {2 | 6,10 | 7 | 4 | 1,8 | 9 | 5 | 3}
      Test D: {3,4,5,6,7,9 | 1,2,8,10}
      Test E: {5,7 | 6 | 9 | 4 | 8 | 10 | 2 | 1,3}
      Test F: {1,10 | 5 | 8 | 4,6,7 | 2,3,9}
      Test G: {2,4,5,6,8,9,10 | 1,3,7}
      Test H: {5 | 7 | 9 | 2,8 | 1,3,4 | 6,10}
      Test I: {5 | 8 | 7 | 9 | 2 | 1,4,6 | 3 | 10}
    
    The residual ambiguity of an outcome block is the Shannon entropy (bits) of the normalized posterior over the hypotheses in that block. The expected ambiguity of a test is the probability-weighted sum across its outcome blocks. Choose the test with the LOWEST expected ambiguity; on a tie, choose the test with the alphabetical-least label.
    
    The answer is a single letter, the label of the best test, e.g. "B".

**Answer:** E

---

