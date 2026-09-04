# Samples P045v1

## Level 0
### Example 1
**Prompt:**

A token bucket rate limiter has a maximum fill capacity of 5 tokens. Initially the bucket holds 1 tokens. Every 3 time units the bucket is refilled by 1 tokens, stopping when it reaches capacity.
at time 5 a request for 1 tokens arrives, at time 6 a request for 1 tokens arrives, at time 9 a request for 2 tokens arrives, at time 15 a request for 2 tokens arrives.
Process the requests in increasing time order. A request is accepted only if the bucket currently holds enough tokens for it. Accepted requests consume their tokens from the bucket; rejected requests consume none.
What is the token balance of the bucket right after the last request is handled?
The answer is an integer.

**Answer:**

3

## Level 0
### Example 2
**Prompt:**

A token bucket rate limiter has a maximum fill capacity of 2 tokens. Initially the bucket holds 2 tokens. Every 2 time units the bucket is refilled by 4 tokens, stopping when it reaches capacity.
at time 3 a request for 2 tokens arrives, at time 8 a request for 2 tokens arrives.
Additionally, a sliding window of 3 time units limits the total demand of requests within any window of that length to at most 3 tokens.
Process the requests in increasing time order. A request is accepted only if the bucket currently holds enough tokens for it, AND the combined demand of all requests whose times lie within 3 time units of it (including itself) does not exceed 3 tokens. Accepted requests consume their tokens from the bucket; rejected requests consume none.
List the times of the accepted requests, in increasing order, separated by commas.
The answer is a comma-separated list of integers.

**Answer:**

3,8

## Level 2
### Example 1
**Prompt:**

A token bucket rate limiter has a maximum fill capacity of 6 tokens. Initially the bucket holds 4 tokens. Every 2 time units the bucket is refilled by 1 tokens, stopping when it reaches capacity.
at time 2 a request for 2 tokens arrives, at time 3 a request for 3 tokens arrives, at time 8 a request for 2 tokens arrives, at time 13 a request for 2 tokens arrives.
Process the requests in increasing time order. A request is accepted only if the bucket currently holds enough tokens for it. Accepted requests consume their tokens from the bucket; rejected requests consume none.
What is the token balance of the bucket right after the last request is handled?
The answer is an integer.

**Answer:**

4

## Level 2
### Example 2
**Prompt:**

A token bucket rate limiter has a maximum fill capacity of 1 tokens. Initially the bucket holds 1 tokens. Every 2 time units the bucket is refilled by 2 tokens, stopping when it reaches capacity.
at time 5 a request for 1 tokens arrives, at time 7 a request for 3 tokens arrives, at time 11 a request for 1 tokens arrives.
Process the requests in increasing time order. A request is accepted only if the bucket currently holds enough tokens for it. Accepted requests consume their tokens from the bucket; rejected requests consume none.
What is the token balance of the bucket right after the last request is handled?
The answer is an integer.

**Answer:**

0

## Level 5
### Example 1
**Prompt:**

A token bucket rate limiter has a maximum fill capacity of 1 tokens. Initially the bucket holds 1 tokens. Every 2 time units the bucket is refilled by 2 tokens, stopping when it reaches capacity.
at time 4 a request for 3 tokens arrives, at time 6 a request for 1 tokens arrives, at time 8 a request for 1 tokens arrives.
Additionally, a sliding window of 1 time units limits the total demand of requests within any window of that length to at most 1 tokens.
Process the requests in increasing time order. A request is accepted only if the bucket currently holds enough tokens for it, AND the combined demand of all requests whose times lie within 1 time units of it (including itself) does not exceed 1 tokens. Accepted requests consume their tokens from the bucket; rejected requests consume none.
List the times of the accepted requests, in increasing order, separated by commas.
The answer is a comma-separated list of integers.

**Answer:**

6,8

## Level 5
### Example 2
**Prompt:**

A token bucket rate limiter has a maximum fill capacity of 1 tokens. Initially the bucket holds 1 tokens. Every 1 time units the bucket is refilled by 1 tokens, stopping when it reaches capacity.
at time 2 a request for 1 tokens arrives, at time 3 a request for 1 tokens arrives, at time 6 a request for 3 tokens arrives.
Process the requests in increasing time order. A request is accepted only if the bucket currently holds enough tokens for it. Accepted requests consume their tokens from the bucket; rejected requests consume none.
What is the token balance of the bucket right after the last request is handled?
The answer is an integer.

**Answer:**

1
