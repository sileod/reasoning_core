# Quarantined: fails the gameability gate at the promotion width

`n02_modular_congruence_system` passed the in-trial gate at n=30 and failed the
promotion battery at n=40: constant guess 0.45 > 0.40, the winning constant being
the literal `NONE` returned when no solution exists. The generator makes the
unsatisfiable case too common; widening the modulus/residue draw so a witness is
usually returned would fix it. The gate is noisy near the threshold -- n=40 is the
one that decides.
