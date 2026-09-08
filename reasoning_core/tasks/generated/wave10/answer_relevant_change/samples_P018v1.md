## Level 0
### Example
**Prompt:**

```
- 1: 7
- 2: 6
- 3: 5
- 4: 6
- 5: 8
Ranking rule: larger value ranks higher; equal values break by smaller label.
You ranked every item above by this rule (larger value first, ties by smaller label). Then this fact is corrected: item 5 is changed from 8 to 4.

Keeping every other item unchanged, which of your previously ranked items now sit in a different relative rank than before? Give the answer as the labels of exactly the items whose rank order changed, comma-separated in ascending label order. Items whose relative order is unchanged are omitted. Write only the comma-separated list, nothing else.
```

**Answer:**

```
1,2,3,4,5
```

### Example
**Prompt:**

```
- 1: 5
- 2: 5
- 3: 9
- 4: 5
- 5: 0
Ranking rule: larger value ranks higher; equal values break by smaller label.
You ranked every item above by this rule (larger value first, ties by smaller label). Then this fact is corrected: item 2 is changed from 5 to 8.

Keeping every other item unchanged, which of your previously ranked items now sit in a different relative rank than before? Give the answer as the labels of exactly the items whose rank order changed, comma-separated in ascending label order. Items whose relative order is unchanged are omitted. Write only the comma-separated list, nothing else.
```

**Answer:**

```
1,2
```


## Level 2
### Example
**Prompt:**

```
- 1: 7
- 2: 6
- 3: 11
- 4: 11
- 5: 10
- 6: 6
- 7: 11
Ranking rule: larger value ranks higher; equal values break by smaller label.
You ranked every item above by this rule (larger value first, ties by smaller label). Then this fact is corrected: item 1 is changed from 7 to 1.

Keeping every other item unchanged, which of your previously ranked items now sit in a different relative rank than before? Give the answer as the labels of exactly the items whose rank order changed, comma-separated in ascending label order. Items whose relative order is unchanged are omitted. Write only the comma-separated list, nothing else.
```

**Answer:**

```
1,2,6
```

### Example
**Prompt:**

```
- 1: 9
- 2: 5
- 3: 10
- 4: 2
- 5: 5
- 6: 10
- 7: 10
Ranking rule: larger value ranks higher; equal values break by smaller label.
You ranked every item above by this rule (larger value first, ties by smaller label). Then this fact is corrected: item 3 is changed from 10 to 2.

Keeping every other item unchanged, which of your previously ranked items now sit in a different relative rank than before? Give the answer as the labels of exactly the items whose rank order changed, comma-separated in ascending label order. Items whose relative order is unchanged are omitted. Write only the comma-separated list, nothing else.
```

**Answer:**

```
1,2,3,5,6,7
```


## Level 5
### Example
**Prompt:**

```
- 1: 11
- 2: 7
- 3: 11
- 4: 8
- 5: 5
- 6: 5
- 7: 9
- 8: 7
- 9: 7
- 10: 8
Ranking rule: larger value ranks higher; equal values break by smaller label.
You ranked every item above by this rule (larger value first, ties by smaller label). Then this fact is corrected: item 3 is changed from 11 to 6.

Keeping every other item unchanged, which of your previously ranked items now sit in a different relative rank than before? Give the answer as the labels of exactly the items whose rank order changed, comma-separated in ascending label order. Items whose relative order is unchanged are omitted. Write only the comma-separated list, nothing else.
```

**Answer:**

```
2,3,4,7,8,9,10
```

### Example
**Prompt:**

```
- 1: 8
- 2: 1
- 3: 11
- 4: 2
- 5: 3
- 6: 4
- 7: 5
- 8: 6
- 9: 14
- 10: 1
Ranking rule: larger value ranks higher; equal values break by smaller label.
You ranked every item above by this rule (larger value first, ties by smaller label). Then this fact is corrected: item 9 is changed from 14 to 1.

Keeping every other item unchanged, which of your previously ranked items now sit in a different relative rank than before? Give the answer as the labels of exactly the items whose rank order changed, comma-separated in ascending label order. Items whose relative order is unchanged are omitted. Write only the comma-separated list, nothing else.
```

**Answer:**

```
1,2,3,4,5,6,7,8,9
```

