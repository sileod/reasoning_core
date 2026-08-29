# Quarantined: generator is stateful

`m02_regex_boolean_languages` writes different bytes on its first call than on
every later call in a fresh process: sha256 over three runs at PYTHONHASHSEED
0/0/1 gave ebf90188 cd27c9e8 cd27c9e8. Two runs at the same salt disagreeing means
state carried across calls (a memo or a lazily-built table consumed from the RNG),
not unsorted-set iteration order. Same-salt runs must agree before this ships.
