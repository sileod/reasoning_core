# Data Lineage Trace samples (P037v1)

## Level 0

### Example 1

A data warehouse task: a query reads from a base fact table T0 and several dimension tables.
The base table T0 is a list of source rows; each row shows its source id and a key value:
  T0 row 0: source_id 1000, key 21
  T0 row 1: source_id 1001, key 3
  T0 row 2: source_id 1002, key 15
  T0 row 3: source_id 1003, key 6
  T0 row 4: source_id 1004, key 19
  T0 row 5: source_id 1005, key 23
  T0 row 6: source_id 1006, key 1
Table T1 (dimension) is a list of source rows; each row shows its source id and a key value:
  T1 row 0: source_id 1007, key 17
  T1 row 1: source_id 1008, key 8
  T1 row 2: source_id 1009, key 22
  T1 row 3: source_id 1010, key 7
  T1 row 4: source_id 1011, key 3
  T1 row 5: source_id 1012, key 20
  T1 row 6: source_id 1013, key 5
Table T2 (dimension) is a list of source rows; each row shows its source id and a key value:
  T2 row 0: source_id 1014, key 21
  T2 row 1: source_id 1015, key 7
  T2 row 2: source_id 1016, key 2
  T2 row 3: source_id 1017, key 10
  T2 row 4: source_id 1018, key 11
  T2 row 5: source_id 1019, key 3
A SQL-like pipeline runs over T0 in this order:
  step 1: keep only rows whose key is less than or equal to 6 (drop the rest)
  step 2: inner-join with table T2 on equal key values
  step 3: aggregate by summing the keys of all remaining rows into one row; the source-id set of that row is the union of the source-id sets of all remaining rows
Question: which source rows (by source id) end up contributing to the final output key value 3? Contributing rows are the T0 rows kept after filtering plus every dimension row added by a join that survives to the aggregation.
Give the answer as a single line: the contributing source ids, sorted in ascending order, separated by commas and spaces. Every pipeline in this exercise leaves at least one source row reaching the output.

**Answer**: 1001, 1019

### Example 2

A data warehouse task: a query reads from a base fact table T0 and several dimension tables.
The base table T0 is a list of source rows; each row shows its source id and a key value:
  T0 row 0: source_id 1000, key 8
  T0 row 1: source_id 1001, key 5
  T0 row 2: source_id 1002, key 2
  T0 row 3: source_id 1003, key 15
  T0 row 4: source_id 1004, key 6
  T0 row 5: source_id 1005, key 19
Table T1 (dimension) is a list of source rows; each row shows its source id and a key value:
  T1 row 0: source_id 1006, key 15
  T1 row 1: source_id 1007, key 13
  T1 row 2: source_id 1008, key 14
  T1 row 3: source_id 1009, key 22
  T1 row 4: source_id 1010, key 16
  T1 row 5: source_id 1018, key 8
Table T2 (dimension) is a list of source rows; each row shows its source id and a key value:
  T2 row 0: source_id 1011, key 12
  T2 row 1: source_id 1012, key 1
  T2 row 2: source_id 1013, key 6
  T2 row 3: source_id 1014, key 17
  T2 row 4: source_id 1015, key 12
  T2 row 5: source_id 1016, key 5
  T2 row 6: source_id 1017, key 15
A SQL-like pipeline runs over T0 in this order:
  step 1: keep only rows whose key is less than or equal to 14 (drop the rest)
  step 2: inner-join with table T1 on equal key values
  step 3: aggregate by summing the keys of all remaining rows into one row; the source-id set of that row is the union of the source-id sets of all remaining rows
Question: which source rows (by source id) end up contributing to the final output key value 8? Contributing rows are the T0 rows kept after filtering plus every dimension row added by a join that survives to the aggregation.
Give the answer as a single line: the contributing source ids, sorted in ascending order, separated by commas and spaces. Every pipeline in this exercise leaves at least one source row reaching the output.

**Answer**: 1000, 1018

## Level 2

### Example 1

A data warehouse task: a query reads from a base fact table T0 and several dimension tables.
The base table T0 is a list of source rows; each row shows its source id and a key value:
  T0 row 0: source_id 1000, key 20
  T0 row 1: source_id 1001, key 8
  T0 row 2: source_id 1002, key 6
  T0 row 3: source_id 1003, key 22
  T0 row 4: source_id 1004, key 17
  T0 row 5: source_id 1005, key 3
  T0 row 6: source_id 1006, key 9
Table T1 (dimension) is a list of source rows; each row shows its source id and a key value:
  T1 row 0: source_id 1007, key 5
  T1 row 1: source_id 1008, key 7
  T1 row 2: source_id 1009, key 7
  T1 row 3: source_id 1010, key 1
  T1 row 4: source_id 1011, key 15
  T1 row 5: source_id 1012, key 10
  T1 row 6: source_id 1013, key 17
  T1 row 7: source_id 1038, key 3
Table T2 (dimension) is a list of source rows; each row shows its source id and a key value:
  T2 row 0: source_id 1014, key 13
  T2 row 1: source_id 1015, key 2
  T2 row 2: source_id 1016, key 10
  T2 row 3: source_id 1017, key 19
  T2 row 4: source_id 1018, key 3
  T2 row 5: source_id 1019, key 25
  T2 row 6: source_id 1020, key 21
Table T3 (dimension) is a list of source rows; each row shows its source id and a key value:
  T3 row 0: source_id 1021, key 4
  T3 row 1: source_id 1022, key 12
  T3 row 2: source_id 1023, key 8
  T3 row 3: source_id 1024, key 4
  T3 row 4: source_id 1025, key 22
  T3 row 5: source_id 1026, key 8
  T3 row 6: source_id 1027, key 8
  T3 row 7: source_id 1028, key 12
Table T4 (dimension) is a list of source rows; each row shows its source id and a key value:
  T4 row 0: source_id 1029, key 23
  T4 row 1: source_id 1030, key 6
  T4 row 2: source_id 1031, key 11
  T4 row 3: source_id 1032, key 14
  T4 row 4: source_id 1033, key 18
  T4 row 5: source_id 1034, key 4
  T4 row 6: source_id 1035, key 21
  T4 row 7: source_id 1036, key 17
  T4 row 8: source_id 1037, key 3
A SQL-like pipeline runs over T0 in this order:
  step 1: keep only rows whose key is less than or equal to 22 (drop the rest)
  step 2: keep only rows whose key is less than or equal to 5 (drop the rest)
  step 3: keep only rows whose key is greater than 2 (drop the rest)
  step 4: inner-join with table T4 on equal key values
  step 5: inner-join with table T1 on equal key values
  step 6: aggregate by summing the keys of all remaining rows into one row; the source-id set of that row is the union of the source-id sets of all remaining rows
  step 7: subtract 3 the aggregated key value
  step 8: subtract 1 the aggregated key value
Question: which source rows (by source id) end up contributing to the final output key value -1? Contributing rows are the T0 rows kept after filtering plus every dimension row added by a join that survives to the aggregation.
Give the answer as a single line: the contributing source ids, sorted in ascending order, separated by commas and spaces. Every pipeline in this exercise leaves at least one source row reaching the output.

**Answer**: 1005, 1037, 1038

### Example 2

A data warehouse task: a query reads from a base fact table T0 and several dimension tables.
The base table T0 is a list of source rows; each row shows its source id and a key value:
  T0 row 0: source_id 1000, key 24
  T0 row 1: source_id 1001, key 3
  T0 row 2: source_id 1002, key 4
  T0 row 3: source_id 1003, key 20
  T0 row 4: source_id 1004, key 4
  T0 row 5: source_id 1005, key 21
  T0 row 6: source_id 1006, key 21
  T0 row 7: source_id 1007, key 8
Table T1 (dimension) is a list of source rows; each row shows its source id and a key value:
  T1 row 0: source_id 1008, key 6
  T1 row 1: source_id 1009, key 22
  T1 row 2: source_id 1010, key 18
  T1 row 3: source_id 1011, key 7
  T1 row 4: source_id 1012, key 12
  T1 row 5: source_id 1013, key 21
  T1 row 6: source_id 1014, key 11
  T1 row 7: source_id 1015, key 23
Table T2 (dimension) is a list of source rows; each row shows its source id and a key value:
  T2 row 0: source_id 1016, key 17
  T2 row 1: source_id 1017, key 21
  T2 row 2: source_id 1018, key 7
  T2 row 3: source_id 1019, key 18
  T2 row 4: source_id 1020, key 19
  T2 row 5: source_id 1021, key 23
  T2 row 6: source_id 1022, key 24
  T2 row 7: source_id 1023, key 22
Table T3 (dimension) is a list of source rows; each row shows its source id and a key value:
  T3 row 0: source_id 1024, key 9
  T3 row 1: source_id 1025, key 17
  T3 row 2: source_id 1026, key 10
  T3 row 3: source_id 1027, key 3
  T3 row 4: source_id 1028, key 4
  T3 row 5: source_id 1029, key 14
  T3 row 6: source_id 1030, key 19
Table T4 (dimension) is a list of source rows; each row shows its source id and a key value:
  T4 row 0: source_id 1031, key 22
  T4 row 1: source_id 1032, key 4
  T4 row 2: source_id 1033, key 12
  T4 row 3: source_id 1034, key 1
  T4 row 4: source_id 1035, key 20
  T4 row 5: source_id 1036, key 2
  T4 row 6: source_id 1037, key 4
  T4 row 7: source_id 1038, key 8
  T4 row 8: source_id 1039, key 21
A SQL-like pipeline runs over T0 in this order:
  step 1: keep only rows whose key is greater than 7 (drop the rest)
  step 2: keep only rows whose key is less than or equal to 21 (drop the rest)
  step 3: keep only rows whose key is greater than 19 (drop the rest)
  step 4: inner-join with table T2 on equal key values
  step 5: inner-join with table T1 on equal key values
  step 6: aggregate by summing the keys of all remaining rows into one row; the source-id set of that row is the union of the source-id sets of all remaining rows
  step 7: multiply by 6 the aggregated key value
  step 8: multiply by 6 the aggregated key value
Question: which source rows (by source id) end up contributing to the final output key value 1512? Contributing rows are the T0 rows kept after filtering plus every dimension row added by a join that survives to the aggregation.
Give the answer as a single line: the contributing source ids, sorted in ascending order, separated by commas and spaces. Every pipeline in this exercise leaves at least one source row reaching the output.

**Answer**: 1005, 1006, 1013, 1017

## Level 5

### Example 1

A data warehouse task: a query reads from a base fact table T0 and several dimension tables.
The base table T0 is a list of source rows; each row shows its source id and a key value:
  T0 row 0: source_id 1000, key 13
  T0 row 1: source_id 1001, key 18
  T0 row 2: source_id 1002, key 25
  T0 row 3: source_id 1003, key 2
  T0 row 4: source_id 1004, key 11
  T0 row 5: source_id 1005, key 13
  T0 row 6: source_id 1006, key 22
  T0 row 7: source_id 1007, key 10
  T0 row 8: source_id 1008, key 1
  T0 row 9: source_id 1009, key 24
Table T1 (dimension) is a list of source rows; each row shows its source id and a key value:
  T1 row 0: source_id 1010, key 19
  T1 row 1: source_id 1011, key 11
  T1 row 2: source_id 1012, key 15
  T1 row 3: source_id 1013, key 8
  T1 row 4: source_id 1014, key 8
  T1 row 5: source_id 1015, key 20
  T1 row 6: source_id 1016, key 4
  T1 row 7: source_id 1017, key 12
  T1 row 8: source_id 1018, key 23
  T1 row 9: source_id 1019, key 3
  T1 row 10: source_id 1020, key 3
  T1 row 11: source_id 1021, key 17
  T1 row 12: source_id 1075, key 10
Table T2 (dimension) is a list of source rows; each row shows its source id and a key value:
  T2 row 0: source_id 1022, key 4
  T2 row 1: source_id 1023, key 1
  T2 row 2: source_id 1024, key 3
  T2 row 3: source_id 1025, key 21
  T2 row 4: source_id 1026, key 6
  T2 row 5: source_id 1027, key 7
  T2 row 6: source_id 1028, key 2
  T2 row 7: source_id 1029, key 20
  T2 row 8: source_id 1030, key 4
  T2 row 9: source_id 1031, key 12
  T2 row 10: source_id 1032, key 10
Table T3 (dimension) is a list of source rows; each row shows its source id and a key value:
  T3 row 0: source_id 1033, key 21
  T3 row 1: source_id 1034, key 15
  T3 row 2: source_id 1035, key 3
  T3 row 3: source_id 1036, key 18
  T3 row 4: source_id 1037, key 5
  T3 row 5: source_id 1038, key 18
  T3 row 6: source_id 1039, key 12
  T3 row 7: source_id 1040, key 2
  T3 row 8: source_id 1041, key 5
  T3 row 9: source_id 1042, key 21
Table T4 (dimension) is a list of source rows; each row shows its source id and a key value:
  T4 row 0: source_id 1043, key 17
  T4 row 1: source_id 1044, key 18
  T4 row 2: source_id 1045, key 5
  T4 row 3: source_id 1046, key 25
  T4 row 4: source_id 1047, key 8
  T4 row 5: source_id 1048, key 3
  T4 row 6: source_id 1049, key 3
  T4 row 7: source_id 1050, key 4
  T4 row 8: source_id 1051, key 4
  T4 row 9: source_id 1052, key 9
  T4 row 10: source_id 1053, key 7
Table T5 (dimension) is a list of source rows; each row shows its source id and a key value:
  T5 row 0: source_id 1054, key 11
  T5 row 1: source_id 1055, key 23
  T5 row 2: source_id 1056, key 25
  T5 row 3: source_id 1057, key 8
  T5 row 4: source_id 1058, key 7
  T5 row 5: source_id 1059, key 1
  T5 row 6: source_id 1060, key 16
  T5 row 7: source_id 1061, key 2
  T5 row 8: source_id 1062, key 19
  T5 row 9: source_id 1063, key 25
Table T6 (dimension) is a list of source rows; each row shows its source id and a key value:
  T6 row 0: source_id 1064, key 6
  T6 row 1: source_id 1065, key 3
  T6 row 2: source_id 1066, key 23
  T6 row 3: source_id 1067, key 18
  T6 row 4: source_id 1068, key 5
  T6 row 5: source_id 1069, key 3
  T6 row 6: source_id 1070, key 7
  T6 row 7: source_id 1071, key 20
  T6 row 8: source_id 1072, key 21
  T6 row 9: source_id 1073, key 2
  T6 row 10: source_id 1074, key 7
A SQL-like pipeline runs over T0 in this order:
  step 1: keep only rows whose key is less than or equal to 21 (drop the rest)
  step 2: keep only rows whose key is greater than 2 (drop the rest)
  step 3: keep only rows whose key is less than or equal to 14 (drop the rest)
  step 4: keep only rows whose key is less than or equal to 13 (drop the rest)
  step 5: inner-join with table T2 on equal key values
  step 6: inner-join with table T1 on equal key values
  step 7: inner-join with table T2 on equal key values
  step 8: aggregate by summing the keys of all remaining rows into one row; the source-id set of that row is the union of the source-id sets of all remaining rows
  step 9: integer-divide by 5 the aggregated key value
  step 10: subtract 5 the aggregated key value
  step 11: integer-divide by 5 the aggregated key value
  step 12: integer-divide by 5 the aggregated key value
Question: which source rows (by source id) end up contributing to the final output key value -1? Contributing rows are the T0 rows kept after filtering plus every dimension row added by a join that survives to the aggregation.
Give the answer as a single line: the contributing source ids, sorted in ascending order, separated by commas and spaces. Every pipeline in this exercise leaves at least one source row reaching the output.

**Answer**: 1007, 1032, 1075

### Example 2

A data warehouse task: a query reads from a base fact table T0 and several dimension tables.
The base table T0 is a list of source rows; each row shows its source id and a key value:
  T0 row 0: source_id 1000, key 17
  T0 row 1: source_id 1001, key 18
  T0 row 2: source_id 1002, key 11
  T0 row 3: source_id 1003, key 24
  T0 row 4: source_id 1004, key 22
  T0 row 5: source_id 1005, key 3
  T0 row 6: source_id 1006, key 11
  T0 row 7: source_id 1007, key 12
  T0 row 8: source_id 1008, key 25
  T0 row 9: source_id 1009, key 13
  T0 row 10: source_id 1010, key 23
Table T1 (dimension) is a list of source rows; each row shows its source id and a key value:
  T1 row 0: source_id 1011, key 10
  T1 row 1: source_id 1012, key 24
  T1 row 2: source_id 1013, key 13
  T1 row 3: source_id 1014, key 11
  T1 row 4: source_id 1015, key 22
  T1 row 5: source_id 1016, key 14
  T1 row 6: source_id 1017, key 18
  T1 row 7: source_id 1018, key 9
  T1 row 8: source_id 1019, key 8
  T1 row 9: source_id 1020, key 4
  T1 row 10: source_id 1021, key 8
Table T2 (dimension) is a list of source rows; each row shows its source id and a key value:
  T2 row 0: source_id 1022, key 7
  T2 row 1: source_id 1023, key 4
  T2 row 2: source_id 1024, key 13
  T2 row 3: source_id 1025, key 13
  T2 row 4: source_id 1026, key 14
  T2 row 5: source_id 1027, key 13
  T2 row 6: source_id 1028, key 11
  T2 row 7: source_id 1029, key 10
  T2 row 8: source_id 1030, key 12
  T2 row 9: source_id 1031, key 21
  T2 row 10: source_id 1032, key 6
Table T3 (dimension) is a list of source rows; each row shows its source id and a key value:
  T3 row 0: source_id 1033, key 24
  T3 row 1: source_id 1034, key 10
  T3 row 2: source_id 1035, key 6
  T3 row 3: source_id 1036, key 10
  T3 row 4: source_id 1037, key 17
  T3 row 5: source_id 1038, key 12
  T3 row 6: source_id 1039, key 5
  T3 row 7: source_id 1040, key 9
  T3 row 8: source_id 1041, key 10
  T3 row 9: source_id 1042, key 9
Table T4 (dimension) is a list of source rows; each row shows its source id and a key value:
  T4 row 0: source_id 1043, key 21
  T4 row 1: source_id 1044, key 17
  T4 row 2: source_id 1045, key 15
  T4 row 3: source_id 1046, key 19
  T4 row 4: source_id 1047, key 2
  T4 row 5: source_id 1048, key 3
  T4 row 6: source_id 1049, key 16
  T4 row 7: source_id 1050, key 12
  T4 row 8: source_id 1051, key 7
  T4 row 9: source_id 1052, key 18
Table T5 (dimension) is a list of source rows; each row shows its source id and a key value:
  T5 row 0: source_id 1053, key 12
  T5 row 1: source_id 1054, key 2
  T5 row 2: source_id 1055, key 23
  T5 row 3: source_id 1056, key 23
  T5 row 4: source_id 1057, key 21
  T5 row 5: source_id 1058, key 23
  T5 row 6: source_id 1059, key 21
  T5 row 7: source_id 1060, key 22
  T5 row 8: source_id 1061, key 2
  T5 row 9: source_id 1062, key 4
  T5 row 10: source_id 1063, key 21
  T5 row 11: source_id 1076, key 3
Table T6 (dimension) is a list of source rows; each row shows its source id and a key value:
  T6 row 0: source_id 1064, key 19
  T6 row 1: source_id 1065, key 5
  T6 row 2: source_id 1066, key 15
  T6 row 3: source_id 1067, key 16
  T6 row 4: source_id 1068, key 5
  T6 row 5: source_id 1069, key 19
  T6 row 6: source_id 1070, key 13
  T6 row 7: source_id 1071, key 1
  T6 row 8: source_id 1072, key 2
  T6 row 9: source_id 1073, key 17
  T6 row 10: source_id 1074, key 13
  T6 row 11: source_id 1075, key 1
A SQL-like pipeline runs over T0 in this order:
  step 1: keep only rows whose key is less than or equal to 4 (drop the rest)
  step 2: keep only rows whose key is less than or equal to 3 (drop the rest)
  step 3: keep only rows whose key is greater than 2 (drop the rest)
  step 4: keep only rows whose key is less than or equal to 3 (drop the rest)
  step 5: inner-join with table T4 on equal key values
  step 6: inner-join with table T4 on equal key values
  step 7: inner-join with table T5 on equal key values
  step 8: aggregate by summing the keys of all remaining rows into one row; the source-id set of that row is the union of the source-id sets of all remaining rows
  step 9: subtract 4 the aggregated key value
  step 10: integer-divide by 6 the aggregated key value
  step 11: multiply by 2 the aggregated key value
  step 12: multiply by 6 the aggregated key value
Question: which source rows (by source id) end up contributing to the final output key value -12? Contributing rows are the T0 rows kept after filtering plus every dimension row added by a join that survives to the aggregation.
Give the answer as a single line: the contributing source ids, sorted in ascending order, separated by commas and spaces. Every pipeline in this exercise leaves at least one source row reaching the output.

**Answer**: 1005, 1048, 1076
