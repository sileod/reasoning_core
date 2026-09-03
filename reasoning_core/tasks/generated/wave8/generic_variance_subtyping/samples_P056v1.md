# Samples for P056v1 (generic_variance_subtyping)

## Level 0

### Example 1
**Prompt:**

The generic type Box takes one type parameter marked * (bivariant).
Ground types (each marker-set is the type's position): Apple={Fruit}; Berry={Fruit, Color}; Even={Number}; Food={Fruit, Shape, Color}; Hard={Matter, Shape}; Left={Direction}; Mint={Color, Number}; Num={Number, Direction}; Red={Color}; Round={Shape}; Stone={Matter}; Sweet={Fruit, Shape}.
Under bivariant variance, Box<A> <: Box<B> requires:
  covariant (+): A's markers are a subset of B's markers;
  contravariant (-): B's markers are a subset of A's markers;
  invariant (o): A and B have exactly the same markers;
  bivariant (*): the relation always holds.
Compare A = Left with B = Even under bivariant variance.
Give the minimal set of markers responsible for the failure of that subtype relation, sorted alphabetically and separated by commas; if it holds, write none.
Example: A = Mint (Color,Number), B = Round (Shape), covariant -> Mint's markers not in Round are Color, Number.

**Answer:**

none

### Example 2
**Prompt:**

The generic type Box takes one type parameter marked o (invariant).
Ground types (each marker-set is the type's position): Apple={Fruit}; Berry={Fruit, Color}; Even={Number}; Food={Fruit, Shape, Color}; Hard={Matter, Shape}; Left={Direction}; Mint={Color, Number}; Num={Number, Direction}; Red={Color}; Round={Shape}; Stone={Matter}; Sweet={Fruit, Shape}.
Under invariant variance, Box<A> <: Box<B> requires:
  covariant (+): A's markers are a subset of B's markers;
  contravariant (-): B's markers are a subset of A's markers;
  invariant (o): A and B have exactly the same markers;
  bivariant (*): the relation always holds.
Compare A = Mint with B = Sweet under invariant variance.
Give the minimal set of markers responsible for the failure of that subtype relation, sorted alphabetically and separated by commas; if it holds, write none.
Example: A = Mint (Color,Number), B = Round (Shape), covariant -> Mint's markers not in Round are Color, Number.

**Answer:**

Color, Fruit, Number, Shape

## Level 2

### Example 1
**Prompt:**

The generic type Box takes one type parameter marked o (invariant).
Ground types (each marker-set is the type's position): Apple={Fruit}; Berry={Fruit, Color}; Even={Number}; Food={Fruit, Shape, Color}; Hard={Matter, Shape}; Left={Direction}; Mint={Color, Number}; Num={Number, Direction}; Red={Color}; Round={Shape}; Stone={Matter}; Sweet={Fruit, Shape}.
Under invariant variance, Box<A> <: Box<B> requires:
  covariant (+): A's markers are a subset of B's markers;
  contravariant (-): B's markers are a subset of A's markers;
  invariant (o): A and B have exactly the same markers;
  bivariant (*): the relation always holds.
Compare A = Apple with B = Num under invariant variance.
Give the minimal set of markers responsible for the failure of that subtype relation, sorted alphabetically and separated by commas; if it holds, write none.
Example: A = Mint (Color,Number), B = Round (Shape), covariant -> Mint's markers not in Round are Color, Number.

**Answer:**

Direction, Fruit, Number

### Example 2
**Prompt:**

The generic type Box takes one type parameter marked * (bivariant).
Ground types (each marker-set is the type's position): Apple={Fruit}; Berry={Fruit, Color}; Even={Number}; Food={Fruit, Shape, Color}; Hard={Matter, Shape}; Left={Direction}; Mint={Color, Number}; Num={Number, Direction}; Red={Color}; Round={Shape}; Stone={Matter}; Sweet={Fruit, Shape}.
Under bivariant variance, Box<A> <: Box<B> requires:
  covariant (+): A's markers are a subset of B's markers;
  contravariant (-): B's markers are a subset of A's markers;
  invariant (o): A and B have exactly the same markers;
  bivariant (*): the relation always holds.
Compare A = Hard ⊔ Food with B = Mint under bivariant variance.
Give the minimal set of markers responsible for the failure of that subtype relation, sorted alphabetically and separated by commas; if it holds, write none.
Example: A = Mint (Color,Number), B = Round (Shape), covariant -> Mint's markers not in Round are Color, Number.

**Answer:**

none

## Level 5

### Example 1
**Prompt:**

The generic type Box takes one type parameter marked * (bivariant).
Ground types (each marker-set is the type's position): Apple={Fruit}; Berry={Fruit, Color}; Even={Number}; Food={Fruit, Shape, Color}; Hard={Matter, Shape}; Left={Direction}; Mint={Color, Number}; Num={Number, Direction}; Red={Color}; Round={Shape}; Stone={Matter}; Sweet={Fruit, Shape}.
Under bivariant variance, Box<A> <: Box<B> requires:
  covariant (+): A's markers are a subset of B's markers;
  contravariant (-): B's markers are a subset of A's markers;
  invariant (o): A and B have exactly the same markers;
  bivariant (*): the relation always holds.
Compare A = Sweet with B = Left under bivariant variance.
Give the minimal set of markers responsible for the failure of that subtype relation, sorted alphabetically and separated by commas; if it holds, write none.
Example: A = Mint (Color,Number), B = Round (Shape), covariant -> Mint's markers not in Round are Color, Number.

**Answer:**

none

### Example 2
**Prompt:**

The generic type Box takes one type parameter marked + (covariant).
Ground types (each marker-set is the type's position): Apple={Fruit}; Berry={Fruit, Color}; Even={Number}; Food={Fruit, Shape, Color}; Hard={Matter, Shape}; Left={Direction}; Mint={Color, Number}; Num={Number, Direction}; Red={Color}; Round={Shape}; Stone={Matter}; Sweet={Fruit, Shape}.
Under covariant variance, Box<A> <: Box<B> requires:
  covariant (+): A's markers are a subset of B's markers;
  contravariant (-): B's markers are a subset of A's markers;
  invariant (o): A and B have exactly the same markers;
  bivariant (*): the relation always holds.
Compare A = Red with B = Round ⊔ Apple ⊔ Food under covariant variance.
Give the minimal set of markers responsible for the failure of that subtype relation, sorted alphabetically and separated by commas; if it holds, write none.
Example: A = Mint (Color,Number), B = Round (Shape), covariant -> Mint's markers not in Round are Color, Number.

**Answer:**

none
