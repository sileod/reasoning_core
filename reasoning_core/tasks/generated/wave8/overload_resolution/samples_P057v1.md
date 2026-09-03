# Samples P057v1: overload_resolution

## Level 0

Prompt:

```
Consider these overloads of function f:
  f0(list)
  f1(number)
  f2(str)
f is called with arguments of types (str).
Using standard overload resolution (numeric conversions cost more than exact matches, and a most-specific applicable overload is chosen when unique), which overload is chosen? Output the parameter-type list of the unique most-specific applicable overload as comma-separated types with no spaces (for example int, float), otherwise output the word Ambiguous.
Format: comma-separated types or Ambiguous.
```

Answer:

```
str
```

Prompt:

```
Consider these overloads of function f:
  f0(bytes)
  f1(str)
f is called with arguments of types (str).
Using standard overload resolution (numeric conversions cost more than exact matches, and a most-specific applicable overload is chosen when unique), which overload is chosen? Output the parameter-type list of the unique most-specific applicable overload as comma-separated types with no spaces (for example int, float), otherwise output the word Ambiguous.
Format: comma-separated types or Ambiguous.
```

Answer:

```
str
```

## Level 2

Prompt:

```
Consider these overloads of function f:
  f0(number, bytes)
  f1(number, number)
  f2(list, bool)
  f3(int, tuple)
  f4(str, list)
f is called with arguments of types (str, list).
Using standard overload resolution (numeric conversions cost more than exact matches, and a most-specific applicable overload is chosen when unique), which overload is chosen? Output the parameter-type list of the unique most-specific applicable overload as comma-separated types with no spaces (for example int, float), otherwise output the word Ambiguous.
Format: comma-separated types or Ambiguous.
```

Answer:

```
str, list
```

Prompt:

```
Consider these overloads of function f:
  f0(float, int)
  f1(float, number)
  f2(tuple, char)
  f3(tuple, bool)
  f4(str, list)
f is called with arguments of types (float, int).
Using standard overload resolution (numeric conversions cost more than exact matches, and a most-specific applicable overload is chosen when unique), which overload is chosen? Output the parameter-type list of the unique most-specific applicable overload as comma-separated types with no spaces (for example int, float), otherwise output the word Ambiguous.
Format: comma-separated types or Ambiguous.
```

Answer:

```
float, int
```

## Level 5

Prompt:

```
Consider these overloads of function f:
  f0(char, tuple)
  f1(list, str)
  f2(tuple, str)
  f3(char, bytes)
  f4(list, float)
  f5(str, list)
  f6(float, char)
  f7(float, int)
f is called with arguments of types (tuple, str).
Using standard overload resolution (numeric conversions cost more than exact matches, and a most-specific applicable overload is chosen when unique), which overload is chosen? Output the parameter-type list of the unique most-specific applicable overload as comma-separated types with no spaces (for example int, float), otherwise output the word Ambiguous.
Format: comma-separated types or Ambiguous.
```

Answer:

```
tuple, str
```

Prompt:

```
Consider these overloads of function f:
  f0(int, float)
  f1(str, tuple)
  f2(char, int)
  f3(bytes, str)
  f4(number, float)
  f5(list, bool)
  f6(bytes, int)
  f7(char, float)
f is called with arguments of types (list, bool).
Using standard overload resolution (numeric conversions cost more than exact matches, and a most-specific applicable overload is chosen when unique), which overload is chosen? Output the parameter-type list of the unique most-specific applicable overload as comma-separated types with no spaces (for example int, float), otherwise output the word Ambiguous.
Format: comma-separated types or Ambiguous.
```

Answer:

```
list, bool
```

