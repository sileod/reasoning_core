## Level 0
### Prompt
Activities:
[{'name': 'A', 'duration': 424, 'waits_for': []}, {'name': 'B', 'duration': 262, 'waits_for': ['A']}, {'name': 'C', 'duration': 579, 'waits_for': ['A', 'B']}, {'name': 'D', 'duration': 607, 'waits_for': []}, {'name': 'E', 'duration': 263, 'waits_for': []}, {'name': 'F', 'duration': 752, 'waits_for': ['A']}]

What is the earliest finish time of the whole project (the time at which all activities are complete)?

Return only the answer.
### Answer
1265

### Prompt
Activities:
[{'name': 'A', 'duration': 777, 'waits_for': []}, {'name': 'B', 'duration': 102, 'waits_for': []}, {'name': 'C', 'duration': 170, 'waits_for': []}, {'name': 'D', 'duration': 799, 'waits_for': ['A', 'C']}, {'name': 'E', 'duration': 314, 'waits_for': []}, {'name': 'F', 'duration': 124, 'waits_for': []}]

What is the earliest finish time of the whole project (the time at which all activities are complete)?

Return only the answer.
### Answer
1576

## Level 2
### Prompt
Activities:
[{'name': 'A', 'duration': 236, 'waits_for': []}, {'name': 'B', 'duration': 682, 'waits_for': []}, {'name': 'C', 'duration': 143, 'waits_for': ['A']}, {'name': 'D', 'duration': 281, 'waits_for': []}, {'name': 'E', 'duration': 223, 'waits_for': ['A', 'D']}, {'name': 'F', 'duration': 401, 'waits_for': ['A', 'B', 'C', 'D']}, {'name': 'G', 'duration': 369, 'waits_for': ['C', 'D', 'E']}, {'name': 'H', 'duration': 201, 'waits_for': ['B', 'C', 'D']}, {'name': 'I', 'duration': 107, 'waits_for': []}, {'name': 'J', 'duration': 687, 'waits_for': ['B', 'C', 'D', 'E']}]

Which activities are critical, that is, have zero slack (a delay to any of them delays the whole project)? Answer as a comma-separated list in alphabetical order.

Return only the answer.
### Answer
B,J

### Prompt
Activities:
[{'name': 'A', 'duration': 792, 'waits_for': []}, {'name': 'B', 'duration': 376, 'waits_for': ['A']}, {'name': 'C', 'duration': 798, 'waits_for': ['A', 'B']}, {'name': 'D', 'duration': 716, 'waits_for': ['C']}, {'name': 'E', 'duration': 156, 'waits_for': []}, {'name': 'F', 'duration': 743, 'waits_for': ['D']}, {'name': 'G', 'duration': 136, 'waits_for': ['A', 'C', 'D', 'E']}, {'name': 'H', 'duration': 794, 'waits_for': ['A', 'D', 'F', 'G']}, {'name': 'I', 'duration': 208, 'waits_for': ['D', 'E', 'H']}, {'name': 'J', 'duration': 263, 'waits_for': ['B', 'H']}]

Which activities are critical, that is, have zero slack (a delay to any of them delays the whole project)? Answer as a comma-separated list in alphabetical order.

Return only the answer.
### Answer
A,B,C,D,F,H,J

## Level 5
### Prompt
Activities:
[{'name': 'A', 'duration': 468, 'waits_for': []}, {'name': 'B', 'duration': 689, 'waits_for': []}, {'name': 'C', 'duration': 284, 'waits_for': []}, {'name': 'D', 'duration': 400, 'waits_for': []}, {'name': 'E', 'duration': 363, 'waits_for': ['A', 'B', 'C', 'D']}, {'name': 'F', 'duration': 751, 'waits_for': []}, {'name': 'G', 'duration': 422, 'waits_for': ['B', 'C', 'D']}, {'name': 'H', 'duration': 520, 'waits_for': ['A', 'B', 'C']}, {'name': 'I', 'duration': 658, 'waits_for': ['A', 'C', 'F']}, {'name': 'J', 'duration': 145, 'waits_for': ['B', 'C', 'D', 'G', 'H', 'I']}, {'name': 'K', 'duration': 703, 'waits_for': ['A', 'C', 'E', 'F', 'G', 'H', 'J']}, {'name': 'L', 'duration': 225, 'waits_for': ['A', 'E']}, {'name': 'M', 'duration': 342, 'waits_for': ['E', 'J', 'K']}, {'name': 'N', 'duration': 794, 'waits_for': ['E', 'H']}, {'name': 'O', 'duration': 123, 'waits_for': ['A', 'B', 'C']}, {'name': 'P', 'duration': 298, 'waits_for': []}]

Which activities are critical, that is, have zero slack (a delay to any of them delays the whole project)? Answer as a comma-separated list in alphabetical order.

Return only the answer.
### Answer
F,I,J,K,M

### Prompt
Activities:
[{'name': 'A', 'duration': 409, 'waits_for': []}, {'name': 'B', 'duration': 552, 'waits_for': []}, {'name': 'C', 'duration': 441, 'waits_for': ['A']}, {'name': 'D', 'duration': 534, 'waits_for': ['A', 'B', 'C']}, {'name': 'E', 'duration': 100, 'waits_for': ['B', 'D']}, {'name': 'F', 'duration': 667, 'waits_for': []}, {'name': 'G', 'duration': 706, 'waits_for': ['C']}, {'name': 'H', 'duration': 495, 'waits_for': ['G']}, {'name': 'I', 'duration': 344, 'waits_for': ['A', 'B', 'C', 'D', 'E', 'F', 'G']}, {'name': 'J', 'duration': 606, 'waits_for': ['A', 'B', 'D', 'E', 'F']}, {'name': 'K', 'duration': 390, 'waits_for': ['A', 'B', 'D', 'E']}, {'name': 'L', 'duration': 456, 'waits_for': ['A', 'B', 'D', 'H', 'I', 'J']}, {'name': 'M', 'duration': 253, 'waits_for': ['F', 'G', 'H']}, {'name': 'N', 'duration': 311, 'waits_for': ['B', 'E', 'F', 'I', 'J']}, {'name': 'O', 'duration': 725, 'waits_for': ['C', 'F', 'G', 'H']}, {'name': 'P', 'duration': 487, 'waits_for': ['F', 'I', 'J', 'K', 'L', 'N', 'O']}]

What is the earliest finish time of the whole project (the time at which all activities are complete)?

Return only the answer.
### Answer
3263

