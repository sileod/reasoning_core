# Virtual Method Dispatch - P059v1 samples

## Level 0

### Example 1

**Prompt:**

```
class A(object):
    def compute(self):
            return 'compute(A)'
    def describe(self):
            return 'describe(A)'
class C(A):
    def describe(self):
            return 'describe(C)'
class D(A):
    def compute(self):
            return 'compute(D)'
class E(A):
    def describe(self):
            return 'describe(E)'
class F(C):
    pass

A program holds an object whose runtime class is E and calls the method compute(). Virtual dispatch resolves this to the most-derived definition in the hierarchy. Which class provides the definition that is actually invoked? Answer with the class name only.
```

**Answer:**

```
A
```

### Example 2

**Prompt:**

```
class A(object):
    def compute(self):
            return 'compute(A)'
    def describe(self):
            return 'describe(A)'
class C(A):
    def compute(self):
            return 'compute(C)'
    def describe(self):
            return 'describe(C)'
class D(C):
    def compute(self):
            return 'compute(D)'
    def describe(self):
            return 'describe(D)'
class E(A):
    def compute(self):
            return 'compute(E)'
    def describe(self):
            return 'describe(E)'
class F(C):
    def compute(self):
            return 'compute(F)'
    def describe(self):
            return 'describe(F)'

A program holds an object whose runtime class is A and calls the method compute(). Virtual dispatch resolves this to the most-derived definition in the hierarchy. Which class provides the definition that is actually invoked? Answer with the class name only.
```

**Answer:**

```
A
```


## Level 2

### Example 1

**Prompt:**

```
class A(object):
    def compute(self):
            return 'compute(A)'
    def describe(self):
            return 'describe(A)'
    def render(self):
            return 'render(A)'
class C(A):
    def describe(self):
            return 'describe(C)'
    def render(self):
            return 'render(C)'
class D(A):
    def compute(self):
            return 'compute(D)'
    def describe(self):
            return 'describe(D)'
    def render(self):
            return 'render(D)'
class E(C):
    def describe(self):
            return 'describe(E)'
class F(D):
    def compute(self):
            return 'compute(F)'
    def describe(self):
            return 'describe(F)'
    def render(self):
            return 'render(F)'
class G(F):
    def compute(self):
            return 'compute(G)'
    def describe(self):
            return 'describe(G)'
class H(G):
    def compute(self):
            return 'compute(H)'

A program holds an object whose runtime class is D and calls the method describe(). Virtual dispatch resolves this to the most-derived definition in the hierarchy. Which class provides the definition that is actually invoked? Answer with the class name only.
```

**Answer:**

```
D
```

### Example 2

**Prompt:**

```
class A(object):
    def compute(self):
            return 'compute(A)'
    def describe(self):
            return 'describe(A)'
    def parse(self):
            return 'parse(A)'
    def render(self):
            return 'render(A)'
class C(A):
    def describe(self):
            return 'describe(C)'
    def parse(self):
            return 'parse(C)'
    def render(self):
            return 'render(C)'
class D(C):
    def parse(self):
            return 'parse(D)'
    def render(self):
            return 'render(D)'
class E(D):
    def compute(self):
            return 'compute(E)'
    def describe(self):
            return 'describe(E)'
    def parse(self):
            return 'parse(E)'
class F(D):
    def describe(self):
            return 'describe(F)'
    def parse(self):
            return 'parse(F)'
class G(C):
    def parse(self):
            return 'parse(G)'
    def render(self):
            return 'render(G)'
class H(E):
    def describe(self):
            return 'describe(H)'
    def parse(self):
            return 'parse(H)'
    def render(self):
            return 'render(H)'
class I(E):
    def compute(self):
            return 'compute(I)'
    def describe(self):
            return 'describe(I)'
    def parse(self):
            return 'parse(I)'
    def render(self):
            return 'render(I)'
class J(F):
    def compute(self):
            return 'compute(J)'
    def describe(self):
            return 'describe(J)'
    def render(self):
            return 'render(J)'

A program holds an object whose runtime class is I and calls the method parse(). Virtual dispatch resolves this to the most-derived definition in the hierarchy. Which class provides the definition that is actually invoked? Answer with the class name only.
```

**Answer:**

```
I
```


## Level 5

### Example 1

**Prompt:**

```
class A(object):
    def compute(self):
            return 'compute(A)'
    def describe(self):
            return 'describe(A)'
    def execute(self):
            return 'execute(A)'
    def fetch(self):
            return 'fetch(A)'
    def parse(self):
            return 'parse(A)'
    def render(self):
            return 'render(A)'
class C(A):
    def describe(self):
            return 'describe(C)'
    def execute(self):
            return 'execute(C)'
class D(C):
    def describe(self):
            return 'describe(D)'
    def execute(self):
            return 'execute(D)'
    def parse(self):
            return 'parse(D)'
class E(C):
    def compute(self):
            return 'compute(E)'
    def describe(self):
            return 'describe(E)'
    def execute(self):
            return 'execute(E)'
    def parse(self):
            return 'parse(E)'
    def render(self):
            return 'render(E)'
class F(C):
    def compute(self):
            return 'compute(F)'
    def describe(self):
            return 'describe(F)'
    def execute(self):
            return 'execute(F)'
    def fetch(self):
            return 'fetch(F)'
    def parse(self):
            return 'parse(F)'
    def render(self):
            return 'render(F)'
class G(A):
    def compute(self):
            return 'compute(G)'
    def describe(self):
            return 'describe(G)'
    def execute(self):
            return 'execute(G)'
    def parse(self):
            return 'parse(G)'
    def render(self):
            return 'render(G)'
class H(A):
    def compute(self):
            return 'compute(H)'
    def describe(self):
            return 'describe(H)'
    def execute(self):
            return 'execute(H)'
    def fetch(self):
            return 'fetch(H)'
    def parse(self):
            return 'parse(H)'
class I(E):
    def compute(self):
            return 'compute(I)'
    def describe(self):
            return 'describe(I)'
    def execute(self):
            return 'execute(I)'
    def fetch(self):
            return 'fetch(I)'
class J(E):
    def compute(self):
            return 'compute(J)'
    def describe(self):
            return 'describe(J)'
    def parse(self):
            return 'parse(J)'
class K(I):
    def compute(self):
            return 'compute(K)'
    def describe(self):
            return 'describe(K)'
    def parse(self):
            return 'parse(K)'
    def render(self):
            return 'render(K)'
class L(G):
    def execute(self):
            return 'execute(L)'
    def parse(self):
            return 'parse(L)'
    def render(self):
            return 'render(L)'
class M(D):
    def compute(self):
            return 'compute(M)'
    def execute(self):
            return 'execute(M)'
    def fetch(self):
            return 'fetch(M)'
    def parse(self):
            return 'parse(M)'
class N(E):
    def compute(self):
            return 'compute(N)'
    def fetch(self):
            return 'fetch(N)'
    def parse(self):
            return 'parse(N)'
class O(K):
    def fetch(self):
            return 'fetch(O)'
    def render(self):
            return 'render(O)'

A program holds an object whose runtime class is D and calls the method describe(). Virtual dispatch resolves this to the most-derived definition in the hierarchy. Which class provides the definition that is actually invoked? Answer with the class name only.
```

**Answer:**

```
D
```

### Example 2

**Prompt:**

```
class A(object):
    def compute(self):
            return 'compute(A)'
    def describe(self):
            return 'describe(A)'
    def execute(self):
            return 'execute(A)'
    def fetch(self):
            return 'fetch(A)'
    def parse(self):
            return 'parse(A)'
    def render(self):
            return 'render(A)'
    def serialize(self):
            return 'serialize(A)'
    def transform(self):
            return 'transform(A)'
class C(A):
    def describe(self):
            return 'describe(C)'
    def execute(self):
            return 'execute(C)'
    def fetch(self):
            return 'fetch(C)'
    def parse(self):
            return 'parse(C)'
    def serialize(self):
            return 'serialize(C)'
class D(C):
    def compute(self):
            return 'compute(D)'
    def execute(self):
            return 'execute(D)'
    def parse(self):
            return 'parse(D)'
    def render(self):
            return 'render(D)'
class E(C):
    def describe(self):
            return 'describe(E)'
    def execute(self):
            return 'execute(E)'
    def render(self):
            return 'render(E)'
class F(D):
    def compute(self):
            return 'compute(F)'
    def describe(self):
            return 'describe(F)'
    def render(self):
            return 'render(F)'
    def serialize(self):
            return 'serialize(F)'
class G(A):
    def describe(self):
            return 'describe(G)'
    def parse(self):
            return 'parse(G)'
    def render(self):
            return 'render(G)'
    def serialize(self):
            return 'serialize(G)'
    def transform(self):
            return 'transform(G)'
class H(F):
    def execute(self):
            return 'execute(H)'
    def fetch(self):
            return 'fetch(H)'
    def parse(self):
            return 'parse(H)'
    def render(self):
            return 'render(H)'
    def transform(self):
            return 'transform(H)'
class I(C):
    def describe(self):
            return 'describe(I)'
    def execute(self):
            return 'execute(I)'
    def fetch(self):
            return 'fetch(I)'
    def parse(self):
            return 'parse(I)'
    def render(self):
            return 'render(I)'
class J(I):
    def describe(self):
            return 'describe(J)'
    def execute(self):
            return 'execute(J)'
    def fetch(self):
            return 'fetch(J)'
    def render(self):
            return 'render(J)'
    def serialize(self):
            return 'serialize(J)'
    def transform(self):
            return 'transform(J)'
class K(D):
    def compute(self):
            return 'compute(K)'
    def fetch(self):
            return 'fetch(K)'
    def render(self):
            return 'render(K)'
    def transform(self):
            return 'transform(K)'
class L(J):
    def compute(self):
            return 'compute(L)'
    def describe(self):
            return 'describe(L)'
    def execute(self):
            return 'execute(L)'
    def render(self):
            return 'render(L)'
    def serialize(self):
            return 'serialize(L)'
    def transform(self):
            return 'transform(L)'
class M(H):
    def compute(self):
            return 'compute(M)'
    def describe(self):
            return 'describe(M)'
    def execute(self):
            return 'execute(M)'
    def fetch(self):
            return 'fetch(M)'
    def render(self):
            return 'render(M)'
class N(J):
    def compute(self):
            return 'compute(N)'
    def execute(self):
            return 'execute(N)'
    def fetch(self):
            return 'fetch(N)'
    def parse(self):
            return 'parse(N)'
    def render(self):
            return 'render(N)'
    def serialize(self):
            return 'serialize(N)'
    def transform(self):
            return 'transform(N)'
class O(I):
    def render(self):
            return 'render(O)'
    def serialize(self):
            return 'serialize(O)'
    def transform(self):
            return 'transform(O)'

A program holds an object whose runtime class is O and calls the method execute(). Virtual dispatch resolves this to the most-derived definition in the hierarchy. Which class provides the definition that is actually invoked? Answer with the class name only.
```

**Answer:**

```
I
```
