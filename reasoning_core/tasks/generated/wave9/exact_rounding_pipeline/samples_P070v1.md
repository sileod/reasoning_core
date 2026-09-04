
## Level 0

### Example 1

A fixed-point pipeline starts with x0 = -8.6 and applies stages in order.
Each stage k computes xk = round(x(k-1) * m + b, p, rule), where rounding to p decimals keeps only p digits after the point:
  'truncate' discards extra decimals towards zero;
  'floor' rounds down towards minus infinity;
  'ceiling' rounds up towards plus infinity;
  'half-even' rounds the half to the nearest even final digit.
Stage list (m, b, p, rule):
  stage 1: m = 1/3, b = 7/6, p = 1, rule = truncate
  stage 2: m = 7/6, b = 1/3, p = 3, rule = floor
  stage 3: m = 1.6, b = 1.2, p = 1, rule = floor
What is the exact integer xN after the final stage? The answer is a single integer.

Answer:
-2

### Example 2

A fixed-point pipeline starts with x0 = -62/3 and applies stages in order.
Each stage k computes xk = round(x(k-1) * m + b, p, rule), where rounding to p decimals keeps only p digits after the point:
  'truncate' discards extra decimals towards zero;
  'floor' rounds down towards minus infinity;
  'ceiling' rounds up towards plus infinity;
  'half-even' rounds the half to the nearest even final digit.
Stage list (m, b, p, rule):
  stage 1: m = 1.5, b = 7/6, p = 1, rule = floor
  stage 2: m = 1.5, b = 1.25, p = 1, rule = floor
  stage 3: m = 7/6, b = 2/3, p = 3, rule = floor
What is the exact integer xN after the final stage? The answer is a single integer.

Answer:
-50


## Level 2

### Example 1

A fixed-point pipeline starts with x0 = -23.6 and applies stages in order.
Each stage k computes xk = round(x(k-1) * m + b, p, rule), where rounding to p decimals keeps only p digits after the point:
  'truncate' discards extra decimals towards zero;
  'floor' rounds down towards minus infinity;
  'ceiling' rounds up towards plus infinity;
  'half-even' rounds the half to the nearest even final digit.
Stage list (m, b, p, rule):
  stage 1: m = 1.8, b = 0.5, p = 3, rule = ceiling
  stage 2: m = 11/6, b = 0.5, p = 2, rule = truncate
  stage 3: m = 5/3, b = 1.25, p = 3, rule = floor
  stage 4: m = 5/3, b = 2/3, p = 4, rule = truncate
  stage 5: m = 5/3, b = 2/3, p = 4, rule = ceiling
What is the exact integer xN after the final stage? The answer is a single integer.

Answer:
-349

### Example 2

A fixed-point pipeline starts with x0 = -97/3 and applies stages in order.
Each stage k computes xk = round(x(k-1) * m + b, p, rule), where rounding to p decimals keeps only p digits after the point:
  'truncate' discards extra decimals towards zero;
  'floor' rounds down towards minus infinity;
  'ceiling' rounds up towards plus infinity;
  'half-even' rounds the half to the nearest even final digit.
Stage list (m, b, p, rule):
  stage 1: m = 5/3, b = 1/3, p = 4, rule = half-even
  stage 2: m = 2/3, b = 1.25, p = 1, rule = ceiling
  stage 3: m = 7/6, b = 1.2, p = 1, rule = ceiling
  stage 4: m = 2/3, b = 4/3, p = 4, rule = truncate
  stage 5: m = 1.6, b = 1.25, p = 3, rule = half-even
What is the exact integer xN after the final stage? The answer is a single integer.

Answer:
-38


## Level 5

### Example 1

A fixed-point pipeline starts with x0 = 31.75 and applies stages in order.
Each stage k computes xk = round(x(k-1) * m + b, p, rule), where rounding to p decimals keeps only p digits after the point:
  'truncate' discards extra decimals towards zero;
  'floor' rounds down towards minus infinity;
  'ceiling' rounds up towards plus infinity;
  'half-even' rounds the half to the nearest even final digit.
Stage list (m, b, p, rule):
  stage 1: m = 2.5, b = 1.2, p = 5, rule = half-even
  stage 2: m = 1.5, b = 2/3, p = 4, rule = half-even
  stage 3: m = 2/3, b = 1/3, p = 4, rule = truncate
  stage 4: m = 0.75, b = 7/6, p = 3, rule = truncate
  stage 5: m = 7/6, b = 1.25, p = 3, rule = half-even
  stage 6: m = 11/6, b = 1/3, p = 5, rule = half-even
  stage 7: m = 1/3, b = 7/6, p = 5, rule = floor
What is the exact integer xN after the final stage? The answer is a single integer.

Answer:
46

### Example 2

A fixed-point pipeline starts with x0 = -45 and applies stages in order.
Each stage k computes xk = round(x(k-1) * m + b, p, rule), where rounding to p decimals keeps only p digits after the point:
  'truncate' discards extra decimals towards zero;
  'floor' rounds down towards minus infinity;
  'ceiling' rounds up towards plus infinity;
  'half-even' rounds the half to the nearest even final digit.
Stage list (m, b, p, rule):
  stage 1: m = 2.5, b = 0.75, p = 3, rule = half-even
  stage 2: m = 1.2, b = 1.25, p = 4, rule = truncate
  stage 3: m = 11/6, b = 2/3, p = 2, rule = floor
  stage 4: m = 11/6, b = 1.25, p = 4, rule = half-even
  stage 5: m = 5/3, b = 4/3, p = 1, rule = truncate
  stage 6: m = 0.75, b = 1.2, p = 1, rule = ceiling
  stage 7: m = 1/3, b = 0.5, p = 3, rule = half-even
What is the exact integer xN after the final stage? The answer is a single integer.

Answer:
-184
