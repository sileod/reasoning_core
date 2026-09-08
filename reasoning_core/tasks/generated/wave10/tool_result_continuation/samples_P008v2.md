# Tool Result Continuation v2 samples

## Level 0

### Example 1

Prompt:

```
A collection script needs at least 2 items and makes 2 tool calls. Each call either accepts some items or errors.

Calls:
- call 1: errored with code 5
- call 2: accepted 1 items

Sum the items from calls that accepted items to get the running total. A call that errors makes the overall operation fail. Otherwise the operation succeeds if the running total is at least 2, is empty if the running total is 0, and is partial otherwise.

Classify the outcome as success, empty, partial, or failure, and report the running total.

The answer is exactly one line of the form STATE:TOTAL where STATE is one of success, empty, partial, failure and TOTAL is the running total (a non-negative integer).
```

Answer:

```
failure:1
```

### Example 2

Prompt:

```
A collection script needs at least 5 items and makes 2 tool calls. Each call either accepts some items or errors.

Calls:
- call 1: accepted 0 items
- call 2: accepted 2 items

Sum the items from calls that accepted items to get the running total. A call that errors makes the overall operation fail. Otherwise the operation succeeds if the running total is at least 5, is empty if the running total is 0, and is partial otherwise.

Classify the outcome as success, empty, partial, or failure, and report the running total.

The answer is exactly one line of the form STATE:TOTAL where STATE is one of success, empty, partial, failure and TOTAL is the running total (a non-negative integer).
```

Answer:

```
partial:2
```


## Level 2

### Example 1

Prompt:

```
A collection script needs at least 3 items and makes 6 tool calls. Each call either accepts some items or errors.

Calls:
- call 1: accepted 0 items
- call 2: accepted 0 items
- call 3: accepted 0 items
- call 4: accepted 0 items
- call 5: accepted 0 items
- call 6: accepted 0 items

Sum the items from calls that accepted items to get the running total. A call that errors makes the overall operation fail. Otherwise the operation succeeds if the running total is at least 3, is empty if the running total is 0, and is partial otherwise.

Classify the outcome as success, empty, partial, or failure, and report the running total.

The answer is exactly one line of the form STATE:TOTAL where STATE is one of success, empty, partial, failure and TOTAL is the running total (a non-negative integer).
```

Answer:

```
empty:0
```

### Example 2

Prompt:

```
A collection script needs at least 2 items and makes 5 tool calls. Each call either accepts some items or errors.

Calls:
- call 1: accepted 4 items
- call 2: accepted 6 items
- call 3: errored with code 3
- call 4: accepted 7 items
- call 5: errored with code 2

Sum the items from calls that accepted items to get the running total. A call that errors makes the overall operation fail. Otherwise the operation succeeds if the running total is at least 2, is empty if the running total is 0, and is partial otherwise.

Classify the outcome as success, empty, partial, or failure, and report the running total.

The answer is exactly one line of the form STATE:TOTAL where STATE is one of success, empty, partial, failure and TOTAL is the running total (a non-negative integer).
```

Answer:

```
failure:17
```


## Level 5

### Example 1

Prompt:

```
A collection script needs at least 29 items and makes 8 tool calls. Each call either accepts some items or errors.

Calls:
- call 1: accepted 9 items
- call 2: accepted 4 items
- call 3: accepted 3 items
- call 4: accepted 10 items
- call 5: accepted 1 items
- call 6: accepted 1 items
- call 7: accepted 0 items
- call 8: accepted 0 items

Sum the items from calls that accepted items to get the running total. A call that errors makes the overall operation fail. Otherwise the operation succeeds if the running total is at least 29, is empty if the running total is 0, and is partial otherwise.

Classify the outcome as success, empty, partial, or failure, and report the running total.

The answer is exactly one line of the form STATE:TOTAL where STATE is one of success, empty, partial, failure and TOTAL is the running total (a non-negative integer).
```

Answer:

```
partial:28
```

### Example 2

Prompt:

```
A collection script needs at least 9 items and makes 11 tool calls. Each call either accepts some items or errors.

Calls:
- call 1: accepted 0 items
- call 2: accepted 0 items
- call 3: accepted 0 items
- call 4: accepted 0 items
- call 5: accepted 0 items
- call 6: accepted 0 items
- call 7: accepted 0 items
- call 8: accepted 0 items
- call 9: accepted 0 items
- call 10: accepted 0 items
- call 11: accepted 0 items

Sum the items from calls that accepted items to get the running total. A call that errors makes the overall operation fail. Otherwise the operation succeeds if the running total is at least 9, is empty if the running total is 0, and is partial otherwise.

Classify the outcome as success, empty, partial, or failure, and report the running total.

The answer is exactly one line of the form STATE:TOTAL where STATE is one of success, empty, partial, failure and TOTAL is the running total (a non-negative integer).
```

Answer:

```
empty:0
```

