# Counterfactual twin model samples (P009v1)

## Level 0

Consider a deterministic structural causal model on variables V0, V1, ..., V2.
The structural equations are (U_i are the unobserved exogenous terms):
  V0 = U0
  V1 = -2 * V0 + U1
  V2 = 2 * V1 + U2
You observe the evidence: V0 = 1, V1 = 1, V2 = 3.
An intervention sets V1 := 3, replacing V1's structural equation.
Using Pearl's three-step procedure (abduction, action, prediction), what would V2 equal after this intervention?
The answer is the single integer value of V2.

Answer: 7

Consider a deterministic structural causal model on variables V0, V1, ..., V2.
The structural equations are (U_i are the unobserved exogenous terms):
  V0 = U0
  V1 = -2 * V0 + U1
  V2 = 1 * V1 + U2
You observe the evidence: V0 = -3, V1 = 4, V2 = 1.
An intervention sets V1 := 2, replacing V1's structural equation.
Using Pearl's three-step procedure (abduction, action, prediction), what would V2 equal after this intervention?
The answer is the single integer value of V2.

Answer: -1

## Level 2

Consider a deterministic structural causal model on variables V0, V1, ..., V4.
The structural equations are (U_i are the unobserved exogenous terms):
  V0 = U0
  V1 = 4 * V0 + U1
  V2 = -4 * V1 + U2
  V3 = -2 * V2 + U3
  V4 = -3 * V3 + U4
You observe the evidence: V0 = -6, V1 = 6, V2 = 2, V3 = 2, V4 = 5.
An intervention sets V3 := 1, replacing V3's structural equation.
Using Pearl's three-step procedure (abduction, action, prediction), what would V4 equal after this intervention?
The answer is the single integer value of V4.

Answer: 8

Consider a deterministic structural causal model on variables V0, V1, ..., V4.
The structural equations are (U_i are the unobserved exogenous terms):
  V0 = U0
  V1 = -4 * V0 + U1
  V2 = 1 * V1 + U2
  V3 = -3 * V2 + U3
  V4 = -2 * V3 + U4
You observe the evidence: V0 = 5, V1 = 5, V2 = -6, V3 = -5, V4 = -5.
An intervention sets V0 := 3, replacing V0's structural equation.
Using Pearl's three-step procedure (abduction, action, prediction), what would V3 equal after this intervention?
The answer is the single integer value of V3.

Answer: -29

## Level 5

Consider a deterministic structural causal model on variables V0, V1, ..., V5.
The structural equations are (U_i are the unobserved exogenous terms):
  V0 = U0
  V1 = -6 * V0 + U1
  V2 = -3 * V1 + U2
  V3 = -5 * V2 + U3
  V4 = 2 * V3 + U4
  V5 = 7 * V4 + U5
You observe the evidence: V0 = 2, V1 = 1, V2 = -3, V3 = 0, V4 = -7, V5 = -4.
An intervention sets V0 := 3, replacing V0's structural equation.
Using Pearl's three-step procedure (abduction, action, prediction), what would V3 equal after this intervention?
The answer is the single integer value of V3.

Answer: -90

Consider a deterministic structural causal model on variables V0, V1, ..., V6.
The structural equations are (U_i are the unobserved exogenous terms):
  V0 = U0
  V1 = 3 * V0 + U1
  V2 = 1 * V1 + U2
  V3 = -1 * V2 + U3
  V4 = 3 * V3 + U4
  V5 = -4 * V4 + U5
  V6 = 3 * V5 + U6
You observe the evidence: V0 = 1, V1 = -2, V2 = -1, V3 = -8, V4 = -3, V5 = -5, V6 = 0.
An intervention sets V0 := 2, replacing V0's structural equation.
Using Pearl's three-step procedure (abduction, action, prediction), what would V3 equal after this intervention?
The answer is the single integer value of V3.

Answer: -11
