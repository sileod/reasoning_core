# Level 0

Prompt: Base types are 'int' and 'bool'. A type is written as alpha -> beta for a function from alpha to beta, right associative, and type variables are named t1, t2, ... in order of first appearance left to right.  Term: \x. \c. \s. c  The term is closed (no free variables), built from lambda abstraction \x. e, application (f x), integer constants, boolean constants true/false, and if-then-else (if c then a else b) where c has type bool and a and b have the same type. Give the principal type of the term in the notation above, or answer exactly 'untypable' if it has no type. The answer is the principal type, or 'untypable'.

Answer: t1 -> t2 -> t3 -> t2

Prompt: Base types are 'int' and 'bool'. A type is written as alpha -> beta for a function from alpha to beta, right associative, and type variables are named t1, t2, ... in order of first appearance left to right.  Term: \b. \c. \e. ((b e) c)  The term is closed (no free variables), built from lambda abstraction \x. e, application (f x), integer constants, boolean constants true/false, and if-then-else (if c then a else b) where c has type bool and a and b have the same type. Give the principal type of the term in the notation above, or answer exactly 'untypable' if it has no type. The answer is the principal type, or 'untypable'.

Answer: (t1 -> t2 -> t3) -> t2 -> t1 -> t3

# Level 2

Prompt: Base types are 'int' and 'bool'. A type is written as alpha -> beta for a function from alpha to beta, right associative, and type variables are named t1, t2, ... in order of first appearance left to right.  Term: \b. \c. b  The term is closed (no free variables), built from lambda abstraction \x. e, application (f x), integer constants, boolean constants true/false, and if-then-else (if c then a else b) where c has type bool and a and b have the same type. Give the principal type of the term in the notation above, or answer exactly 'untypable' if it has no type. The answer is the principal type, or 'untypable'.

Answer: t1 -> t2 -> t1

Prompt: Base types are 'int' and 'bool'. A type is written as alpha -> beta for a function from alpha to beta, right associative, and type variables are named t1, t2, ... in order of first appearance left to right.  Term: \a. \c. (c a)  The term is closed (no free variables), built from lambda abstraction \x. e, application (f x), integer constants, boolean constants true/false, and if-then-else (if c then a else b) where c has type bool and a and b have the same type. Give the principal type of the term in the notation above, or answer exactly 'untypable' if it has no type. The answer is the principal type, or 'untypable'.

Answer: t1 -> (t1 -> t2) -> t2

# Level 5

Prompt: Base types are 'int' and 'bool'. A type is written as alpha -> beta for a function from alpha to beta, right associative, and type variables are named t1, t2, ... in order of first appearance left to right.  Term: \a. \d. a  The term is closed (no free variables), built from lambda abstraction \x. e, application (f x), integer constants, boolean constants true/false, and if-then-else (if c then a else b) where c has type bool and a and b have the same type. Give the principal type of the term in the notation above, or answer exactly 'untypable' if it has no type. The answer is the principal type, or 'untypable'.

Answer: t1 -> t2 -> t1

Prompt: Base types are 'int' and 'bool'. A type is written as alpha -> beta for a function from alpha to beta, right associative, and type variables are named t1, t2, ... in order of first appearance left to right.  Term: \x. x  The term is closed (no free variables), built from lambda abstraction \x. e, application (f x), integer constants, boolean constants true/false, and if-then-else (if c then a else b) where c has type bool and a and b have the same type. Give the principal type of the term in the notation above, or answer exactly 'untypable' if it has no type. The answer is the principal type, or 'untypable'.

Answer: t1 -> t1
