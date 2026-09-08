## Level 0

At 8 hours into the day, classify each event below.

- A: starts 4 hours into the day, lasts 10 hours.
- B: starts 10 hours into the day, lasts 9 hours.
- C: starts 1 hours into the day, lasts 4 hours.

For each event, write a line `NAME: STATUS` where STATUS is `ongoing`, `completed`, or `not_yet_started`. An event is `completed` if the reference time is past start+duration, `ongoing` if the reference time is within [start, start+duration), and `not_yet_started` if it is before start.

Answer:
```
A: ongoing; 4 hours into the event
B: not_yet_started; 2 hours until start
C: completed; 3 hours since end
```

At 5 hours into the day, classify each event below.

- A: starts 5 hours into the day, lasts 7 hours.
- B: starts 10 hours into the day, lasts 2 hours.
- C: starts 1 hours into the day, lasts 1 hours.

For each event, write a line `NAME: STATUS` where STATUS is `ongoing`, `completed`, or `not_yet_started`. An event is `completed` if the reference time is past start+duration, `ongoing` if the reference time is within [start, start+duration), and `not_yet_started` if it is before start.

Answer:
```
A: ongoing; 0 hours into the event
B: not_yet_started; 5 hours until start
C: completed; 3 hours since end
```

## Level 2

At 21 hours into the day, classify each event below.

- A: starts 8 hours into the day, lasts 4 hours.
- B: starts 11 hours into the day, lasts 6 hours.
- C: starts 1 hours into the day, lasts 12 hours.
- D: starts 0 hours into the day, lasts 17 hours.
- E: starts 12 hours into the day, lasts 11 hours.

For each event, write a line `NAME: STATUS` where STATUS is `ongoing`, `completed`, or `not_yet_started`. An event is `completed` if the reference time is past start+duration, `ongoing` if the reference time is within [start, start+duration), and `not_yet_started` if it is before start.

Answer:
```
A: completed; 9 hours since end
B: completed; 4 hours since end
C: completed; 8 hours since end
D: completed; 4 hours since end
E: ongoing; 9 hours into the event
```

At 7 hours into the day, classify each event below.

- A: starts 12 hours into the day, lasts 3 hours.
- B: starts 15 hours into the day, lasts 8 hours.
- C: starts 4 hours into the day, lasts 2 hours.
- D: starts 14 hours into the day, lasts 15 hours.
- E: starts 1 hours into the day, lasts 14 hours.

For each event, write a line `NAME: STATUS` where STATUS is `ongoing`, `completed`, or `not_yet_started`. An event is `completed` if the reference time is past start+duration, `ongoing` if the reference time is within [start, start+duration), and `not_yet_started` if it is before start.

Answer:
```
A: not_yet_started; 5 hours until start
B: not_yet_started; 8 hours until start
C: completed; 1 hours since end
D: not_yet_started; 7 hours until start
E: ongoing; 6 hours into the event
```

## Level 5

At 18 hours into the day, classify each event below.

- A: starts 7 hours into the day, lasts 18 hours.
- B: starts 12 hours into the day, lasts 13 hours.
- C: starts 29 hours into the day, lasts 18 hours.
- D: starts 11 hours into the day, lasts 25 hours.
- E: starts 28 hours into the day, lasts 15 hours.
- F: starts 1 hours into the day, lasts 5 hours.
- G: starts 8 hours into the day, lasts 10 hours.
- H: starts 22 hours into the day, lasts 9 hours.

For each event, write a line `NAME: STATUS` where STATUS is `ongoing`, `completed`, or `not_yet_started`. An event is `completed` if the reference time is past start+duration, `ongoing` if the reference time is within [start, start+duration), and `not_yet_started` if it is before start.

Answer:
```
A: ongoing; 11 hours into the event
B: ongoing; 6 hours into the event
C: not_yet_started; 11 hours until start
D: ongoing; 7 hours into the event
E: not_yet_started; 10 hours until start
F: completed; 12 hours since end
G: completed; 0 hours since end
H: not_yet_started; 4 hours until start
```

At 9 hours into the day, classify each event below.

- A: starts 6 hours into the day, lasts 26 hours.
- B: starts 18 hours into the day, lasts 15 hours.
- C: starts 0 hours into the day, lasts 8 hours.
- D: starts 3 hours into the day, lasts 14 hours.
- E: starts 18 hours into the day, lasts 16 hours.
- F: starts 8 hours into the day, lasts 6 hours.
- G: starts 27 hours into the day, lasts 13 hours.
- H: starts 13 hours into the day, lasts 26 hours.

For each event, write a line `NAME: STATUS` where STATUS is `ongoing`, `completed`, or `not_yet_started`. An event is `completed` if the reference time is past start+duration, `ongoing` if the reference time is within [start, start+duration), and `not_yet_started` if it is before start.

Answer:
```
A: ongoing; 3 hours into the event
B: not_yet_started; 9 hours until start
C: completed; 1 hours since end
D: ongoing; 6 hours into the event
E: not_yet_started; 9 hours until start
F: ongoing; 1 hours into the event
G: not_yet_started; 18 hours until start
H: not_yet_started; 4 hours until start
```
