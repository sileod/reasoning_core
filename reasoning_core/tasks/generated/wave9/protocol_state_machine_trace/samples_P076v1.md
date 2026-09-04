# Level 0

## Example

Prompt:

```
state s0: on nack1 -> s2 (r=1 t=1) ; on timeout2 -> s0 (r=2 t=1) ; on ack3 -> s2 (r=1 t=1).
state s1: on nack1 -> s2 (r=2 t=3) ; on timeout2 -> s1 (r=1 t=2) ; on ack3 -> s0 (r=1 t=3).
state s2: on nack1 -> s1 (r=1 t=3) ; on timeout2 -> s1 (r=1 t=1) ; on ack3 -> s2 (r=2 t=1) [requires prev in {s2}].
start s1

event trace: ack3 nack1 timeout2 timeout2 timeout2

Run this protocol machine on the given event trace from the starting state. For each event, try the current state's transition for that event. The transition fires, changing state to its target, only if BOTH (a) the target's 'requires prev' set contains the previous state (an unguarded target always passes) AND (b) the number of events processed since the machine last changed state is less than that transition's timeout t. When a transition fires, reset that events-since-last-change counter to zero; otherwise increment it. The answer is the name of the state after the whole trace, e.g. 's0'.
```

Answer: s0

## Example

Prompt:

```
state s0: on nack1 -> s1 (r=2 t=1) ; on req2 -> s2 (r=1 t=2) ; on timeout3 -> s2 (r=1 t=3) [requires prev in {s2}].
state s1: on nack1 -> s2 (r=1 t=1) ; on req2 -> s1 (r=1 t=1) ; on timeout3 -> s0 (r=1 t=1) [requires prev in {s2}].
state s2: on nack1 -> s1 (r=1 t=1) ; on req2 -> s1 (r=1 t=3) ; on timeout3 -> s1 (r=1 t=1) [requires prev in {s2}].
start s2

event trace: timeout3 timeout3 nack1 timeout3 timeout3

Run this protocol machine on the given event trace from the starting state. For each event, try the current state's transition for that event. The transition fires, changing state to its target, only if BOTH (a) the target's 'requires prev' set contains the previous state (an unguarded target always passes) AND (b) the number of events processed since the machine last changed state is less than that transition's timeout t. When a transition fires, reset that events-since-last-change counter to zero; otherwise increment it. The answer is the name of the state after the whole trace, e.g. 's0'.
```

Answer: s0



# Level 2

## Example

Prompt:

```
state s0: on nack1 -> s2 (r=2 t=1) ; on msg2 -> s0 (r=1 t=1) ; on req3 -> s1 (r=2 t=1).
state s1: on nack1 -> s0 (r=2 t=2) ; on msg2 -> s1 (r=1 t=1) ; on req3 -> s0 (r=2 t=3) [requires prev in {s0, s1, s2}].
state s2: on nack1 -> s0 (r=1 t=1) ; on msg2 -> s0 (r=2 t=2) ; on req3 -> s1 (r=1 t=3) [requires prev in {s0, s1, s2}].
start s0

event trace: req3 msg2 msg2 req3 nack1

Run this protocol machine on the given event trace from the starting state. For each event, try the current state's transition for that event. The transition fires, changing state to its target, only if BOTH (a) the target's 'requires prev' set contains the previous state (an unguarded target always passes) AND (b) the number of events processed since the machine last changed state is less than that transition's timeout t. When a transition fires, reset that events-since-last-change counter to zero; otherwise increment it. The answer is the name of the state after the whole trace, e.g. 's0'.
```

Answer: s2

## Example

Prompt:

```
state s0: on timeout1 -> s0 (r=1 t=2) ; on req2 -> s2 (r=1 t=1) ; on recv3 -> s0 (r=2 t=2) [requires prev in {s0, s1}].
state s1: on timeout1 -> s1 (r=2 t=1) ; on req2 -> s2 (r=1 t=1) ; on recv3 -> s2 (r=1 t=3) [requires prev in {s0, s2}].
state s2: on timeout1 -> s2 (r=2 t=1) ; on req2 -> s0 (r=2 t=1) ; on recv3 -> s1 (r=1 t=1).
start s0

event trace: recv3 timeout1 recv3 timeout1 recv3

Run this protocol machine on the given event trace from the starting state. For each event, try the current state's transition for that event. The transition fires, changing state to its target, only if BOTH (a) the target's 'requires prev' set contains the previous state (an unguarded target always passes) AND (b) the number of events processed since the machine last changed state is less than that transition's timeout t. When a transition fires, reset that events-since-last-change counter to zero; otherwise increment it. The answer is the name of the state after the whole trace, e.g. 's0'.
```

Answer: s0



# Level 5

## Example

Prompt:

```
state s0: on req1 -> s0 (r=1 t=2) ; on req2 -> s0 (r=2 t=2) ; on msg3 -> s1 (r=2 t=1) [requires prev in {s0, s1, s2}].
state s1: on req1 -> s2 (r=1 t=3) ; on req2 -> s1 (r=2 t=1) ; on msg3 -> s2 (r=1 t=3) [requires prev in {s0}].
state s2: on req1 -> s0 (r=2 t=1) ; on req2 -> s2 (r=2 t=2) ; on msg3 -> s2 (r=2 t=2).
start s2

event trace: req2 msg3 req1 req1 msg3

Run this protocol machine on the given event trace from the starting state. For each event, try the current state's transition for that event. The transition fires, changing state to its target, only if BOTH (a) the target's 'requires prev' set contains the previous state (an unguarded target always passes) AND (b) the number of events processed since the machine last changed state is less than that transition's timeout t. When a transition fires, reset that events-since-last-change counter to zero; otherwise increment it. The answer is the name of the state after the whole trace, e.g. 's0'.
```

Answer: s1

## Example

Prompt:

```
state s0: on recv1 -> s2 (r=1 t=2) ; on msg2 -> s0 (r=1 t=1) ; on ack3 -> s0 (r=1 t=2).
state s1: on recv1 -> s1 (r=2 t=3) ; on msg2 -> s0 (r=1 t=2) ; on ack3 -> s0 (r=1 t=3) [requires prev in {s1, s2}].
state s2: on recv1 -> s2 (r=2 t=2) ; on msg2 -> s1 (r=1 t=1) ; on ack3 -> s2 (r=2 t=1) [requires prev in {s2}].
start s2

event trace: ack3 ack3 msg2 recv1 msg2

Run this protocol machine on the given event trace from the starting state. For each event, try the current state's transition for that event. The transition fires, changing state to its target, only if BOTH (a) the target's 'requires prev' set contains the previous state (an unguarded target always passes) AND (b) the number of events processed since the machine last changed state is less than that transition's timeout t. When a transition fires, reset that events-since-last-change counter to zero; otherwise increment it. The answer is the name of the state after the whole trace, e.g. 's0'.
```

Answer: s0


