## Level 0

**Prompt:**

Consider the abstract argumentation framework with arguments: a b c d e.
Attacks (X attacks Y means the argument X attacks Y):
c attacks d
c attacks e
d attacks e
e attacks b

Using grounded semantics, compute the grounded extension by repeatedly marking an argument accepted whenever every argument attacking it is already rejected, and rejected whenever some accepted argument attacks it, until nothing changes. Name the arguments in the grounded extension, written as a single space-separated sequence in alphabetical order (write none if the grounded extension is empty).

**Answer:**

a b c

**Prompt:**

Consider the abstract argumentation framework with arguments: a b c d e.
Attacks (X attacks Y means the argument X attacks Y):
a attacks b
a attacks c
b attacks e
d attacks e
e attacks a
e attacks b

Using grounded semantics, compute the grounded extension by repeatedly marking an argument accepted whenever every argument attacking it is already rejected, and rejected whenever some accepted argument attacks it, until nothing changes. Name the arguments in the grounded extension, written as a single space-separated sequence in alphabetical order (write none if the grounded extension is empty).

**Answer:**

a d

## Level 2

**Prompt:**

Consider the abstract argumentation framework with arguments: a b c d e f g h i.
Attacks (X attacks Y means the argument X attacks Y):
a attacks d
a attacks e
b attacks g
c attacks b
d attacks c
d attacks e
d attacks g
e attacks c
e attacks g
f attacks b
f attacks e
g attacks c
g attacks i
h attacks b

Using grounded semantics, compute the grounded extension by repeatedly marking an argument accepted whenever every argument attacking it is already rejected, and rejected whenever some accepted argument attacks it, until nothing changes. Name the arguments in the grounded extension, written as a single space-separated sequence in alphabetical order (write none if the grounded extension is empty).

**Answer:**

a f g h

**Prompt:**

Consider the abstract argumentation framework with arguments: a b c d e f g h i.
Attacks (X attacks Y means the argument X attacks Y):
a attacks e
a attacks g
b attacks c
b attacks e
b attacks h
c attacks d
d attacks c
e attacks a
e attacks g
e attacks i
f attacks b
h attacks a
h attacks c
h attacks g
h attacks i
i attacks h

Using grounded semantics, compute the grounded extension by repeatedly marking an argument accepted whenever every argument attacking it is already rejected, and rejected whenever some accepted argument attacks it, until nothing changes. Name the arguments in the grounded extension, written as a single space-separated sequence in alphabetical order (write none if the grounded extension is empty).

**Answer:**

f

## Level 5

**Prompt:**

Consider the abstract argumentation framework with arguments: a b c d e f g h i j k l m n o.
Attacks (X attacks Y means the argument X attacks Y):
a attacks b
a attacks f
a attacks m
b attacks f
c attacks a
c attacks g
c attacks j
c attacks l
e attacks h
e attacks i
e attacks m
f attacks a
f attacks e
f attacks k
g attacks a
g attacks c
g attacks e
g attacks h
g attacks i
h attacks a
h attacks g
h attacks o
i attacks b
i attacks f
j attacks e
j attacks g
j attacks h
j attacks i
j attacks l
j attacks o
k attacks e
k attacks i
k attacks j
k attacks m
l attacks b
l attacks e
l attacks f
l attacks k
l attacks m
m attacks c
m attacks f
m attacks k
n attacks b
n attacks e
n attacks f
n attacks g
o attacks e
o attacks m

Using grounded semantics, compute the grounded extension by repeatedly marking an argument accepted whenever every argument attacking it is already rejected, and rejected whenever some accepted argument attacks it, until nothing changes. Name the arguments in the grounded extension, written as a single space-separated sequence in alphabetical order (write none if the grounded extension is empty).

**Answer:**

d n

**Prompt:**

Consider the abstract argumentation framework with arguments: a b c d e f g h i j k l m n o.
Attacks (X attacks Y means the argument X attacks Y):
a attacks b
a attacks d
a attacks f
a attacks h
a attacks k
a attacks o
b attacks a
b attacks f
b attacks m
b attacks o
c attacks d
c attacks i
d attacks l
e attacks m
f attacks d
f attacks j
f attacks k
g attacks h
h attacks a
i attacks c
i attacks e
i attacks f
i attacks n
j attacks b
j attacks i
k attacks b
k attacks m
k attacks o
l attacks b
l attacks h
l attacks i
l attacks n
n attacks f
n attacks o
o attacks b
o attacks f

Using grounded semantics, compute the grounded extension by repeatedly marking an argument accepted whenever every argument attacking it is already rejected, and rejected whenever some accepted argument attacks it, until nothing changes. Name the arguments in the grounded extension, written as a single space-separated sequence in alphabetical order (write none if the grounded extension is empty).

**Answer:**

g

