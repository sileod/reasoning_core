## Level 0
### Prompt
An `unless` condition blocks its default rule.

Facts:
Bruno is exc1.

Rules:
Alpha-tagged people are bravo-tagged unless exc1.
Exc1 people are not bravo-tagged.
People who are not alpha-tagged are not bravo-tagged.

Hypothesis:
Clara is bravo-tagged.

Is the hypothesis true? Answer Yes, No, or Maybe.
### Answer
Maybe

### Prompt
An `unless` condition blocks its default rule.

Facts:
David is alpha-tagged and exc1.
Clara is exc1.
Bruno is exc1.

Rules:
Alpha-tagged people are bravo-tagged unless exc1.
Exc1 people are not bravo-tagged.
People who are not alpha-tagged are not bravo-tagged.

Hypothesis:
David is bravo-tagged.

Is the hypothesis true? Answer Yes, No, or Maybe.
### Answer
No

## Level 2
### Prompt
An `unless` condition blocks its default rule.

Facts:
Clara is alpha-tagged and exc1.
Alice is exc1.
Elena is exc1.

Rules:
Alpha-tagged people are bravo-tagged unless exc1.
Exc1 people are not bravo-tagged.
Bravo-tagged people are charlie-tagged unless exc2.
Exc2 people are not charlie-tagged.
Charlie-tagged people are delta-tagged unless exc3.
Exc3 people are not delta-tagged.
People who are not alpha-tagged are not bravo-tagged.
People who are not bravo-tagged are not charlie-tagged.
People who are not charlie-tagged are not delta-tagged.

Hypothesis:
Clara is delta-tagged.

Is the hypothesis true? Answer Yes, No, or Maybe.
### Answer
No

### Prompt
An `unless` condition blocks its default rule.

Facts:
Farah is exc3.
Alice is exc3.
Bruno is exc1.

Rules:
Alpha-tagged people are bravo-tagged unless exc1.
Exc1 people are not bravo-tagged.
Bravo-tagged people are charlie-tagged unless exc2.
Exc2 people are not charlie-tagged.
Charlie-tagged people are delta-tagged unless exc3.
Exc3 people are not delta-tagged.
People who are not alpha-tagged are not bravo-tagged.
People who are not bravo-tagged are not charlie-tagged.
People who are not charlie-tagged are not delta-tagged.

Hypothesis:
Elena is delta-tagged.

Is the hypothesis true? Answer Yes, No, or Maybe.
### Answer
Maybe

## Level 5
### Prompt
An `unless` condition blocks its default rule.

Facts:
David is alpha-tagged and exc4.
Clara is exc1.
Hannah is exc5.
Farah is exc6.
Elena is exc6.

Rules:
Alpha-tagged people are bravo-tagged unless exc1.
Exc1 people are not bravo-tagged.
Bravo-tagged people are charlie-tagged unless exc2.
Exc2 people are not charlie-tagged.
Charlie-tagged people are delta-tagged unless exc3.
Exc3 people are not delta-tagged.
Delta-tagged people are echo-tagged unless exc4.
Exc4 people are not echo-tagged.
Echo-tagged people are foxtrot-tagged unless exc5.
Exc5 people are not foxtrot-tagged.
Foxtrot-tagged people are gamma-tagged unless exc6.
Exc6 people are not gamma-tagged.
People who are not alpha-tagged are not bravo-tagged.
People who are not bravo-tagged are not charlie-tagged.
People who are not charlie-tagged are not delta-tagged.
People who are not delta-tagged are not echo-tagged.
People who are not echo-tagged are not foxtrot-tagged.
People who are not foxtrot-tagged are not gamma-tagged.

Hypothesis:
David is gamma-tagged.

Is the hypothesis true? Answer Yes, No, or Maybe.
### Answer
No

### Prompt
An `unless` condition blocks its default rule.

Facts:
Bruno is exc6.
George is exc3.
Alice is exc4.

Rules:
Alpha-tagged people are bravo-tagged unless exc1.
Exc1 people are not bravo-tagged.
Bravo-tagged people are charlie-tagged unless exc2.
Exc2 people are not charlie-tagged.
Charlie-tagged people are delta-tagged unless exc3.
Exc3 people are not delta-tagged.
Delta-tagged people are echo-tagged unless exc4.
Exc4 people are not echo-tagged.
Echo-tagged people are foxtrot-tagged unless exc5.
Exc5 people are not foxtrot-tagged.
Foxtrot-tagged people are gamma-tagged unless exc6.
Exc6 people are not gamma-tagged.
People who are not alpha-tagged are not bravo-tagged.
People who are not bravo-tagged are not charlie-tagged.
People who are not charlie-tagged are not delta-tagged.
People who are not delta-tagged are not echo-tagged.
People who are not echo-tagged are not foxtrot-tagged.
People who are not foxtrot-tagged are not gamma-tagged.

Hypothesis:
Elena is gamma-tagged.

Is the hypothesis true? Answer Yes, No, or Maybe.
### Answer
Maybe
