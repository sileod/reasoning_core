## Level 0

Transactions follow strict two-phase locking: a transaction may request any number of locks but releases them only after all its requests, and a lock request is granted only if no other transaction currently holds it.

T2 requests lock on D1
T2 releases lock on D1
T1 requests lock on D4
T1 requests lock on D2
T1 releases lock on D4
T1 requests lock on D2
T2 requests lock on D1

The final request is the queried request. Name the transaction whose held lock blocks that request; if no transaction blocks it, the answer is the token None.
The answer is a single transaction id such as T1, or the token None when no transaction blocks the queried request.

Answer: None

Transactions follow strict two-phase locking: a transaction may request any number of locks but releases them only after all its requests, and a lock request is granted only if no other transaction currently holds it.

T1 requests lock on D3
T3 requests lock on D5
T1 requests lock on D2
T3 releases lock on D5
T2 requests lock on D4
T2 requests lock on D2
T2 requests lock on D4

The final request is the queried request. Name the transaction whose held lock blocks that request; if no transaction blocks it, the answer is the token None.
The answer is a single transaction id such as T1, or the token None when no transaction blocks the queried request.

Answer: None

## Level 2

Transactions follow strict two-phase locking: a transaction may request any number of locks but releases them only after all its requests, and a lock request is granted only if no other transaction currently holds it.

T2 requests lock on D4
T4 requests lock on D2
T2 requests lock on D2
T5 requests lock on D4
T4 requests lock on D8
T5 requests lock on D5
T3 requests lock on D3
T1 requests lock on D1
T4 releases lock on D2
T1 requests lock on D6
T4 requests lock on D8
T1 requests lock on D6
T4 releases lock on D8
T5 requests lock on D4
T5 requests lock on D2

The final request is the queried request. Name the transaction whose held lock blocks that request; if no transaction blocks it, the answer is the token None.
The answer is a single transaction id such as T1, or the token None when no transaction blocks the queried request.

Answer: None

Transactions follow strict two-phase locking: a transaction may request any number of locks but releases them only after all its requests, and a lock request is granted only if no other transaction currently holds it.

T4 requests lock on D8
T1 requests lock on D1
T2 requests lock on D7
T5 requests lock on D1
T4 requests lock on D4
T1 releases lock on D1
T4 releases lock on D4
T4 requests lock on D5
T2 requests lock on D4
T4 releases lock on D5
T4 releases lock on D8
T1 requests lock on D3
T2 releases lock on D7
T2 releases lock on D4
T1 requests lock on D3

The final request is the queried request. Name the transaction whose held lock blocks that request; if no transaction blocks it, the answer is the token None.
The answer is a single transaction id such as T1, or the token None when no transaction blocks the queried request.

Answer: None

## Level 5

Transactions follow strict two-phase locking: a transaction may request any number of locks but releases them only after all its requests, and a lock request is granted only if no other transaction currently holds it.

T2 requests lock on D8
T2 releases lock on D8
T7 requests lock on D12
T7 releases lock on D12
T2 requests lock on D11
T3 requests lock on D8
T6 requests lock on D14
T7 requests lock on D2
T1 requests lock on D1
T1 requests lock on D11
T1 releases lock on D1
T7 requests lock on D2
T3 requests lock on D1
T1 requests lock on D13
T6 releases lock on D14
T1 requests lock on D14
T2 requests lock on D4
T2 releases lock on D4
T1 requests lock on D4
T2 requests lock on D11
T2 requests lock on D6
T2 releases lock on D11
T5 requests lock on D9
T3 requests lock on D15
T8 requests lock on D10
T4 requests lock on D8
T3 requests lock on D14

The final request is the queried request. Name the transaction whose held lock blocks that request; if no transaction blocks it, the answer is the token None.
The answer is a single transaction id such as T1, or the token None when no transaction blocks the queried request.

Answer: T1

Transactions follow strict two-phase locking: a transaction may request any number of locks but releases them only after all its requests, and a lock request is granted only if no other transaction currently holds it.

T1 requests lock on D3
T5 requests lock on D14
T6 requests lock on D5
T2 requests lock on D14
T5 requests lock on D15
T7 requests lock on D2
T4 requests lock on D7
T6 requests lock on D2
T4 requests lock on D14
T6 releases lock on D5
T3 requests lock on D5
T6 requests lock on D12
T7 requests lock on D4
T2 requests lock on D1
T5 requests lock on D1
T3 requests lock on D8
T2 requests lock on D3
T3 requests lock on D2
T1 requests lock on D7
T4 requests lock on D3
T5 releases lock on D15
T8 requests lock on D12
T4 releases lock on D7
T3 releases lock on D5
T5 requests lock on D15
T7 requests lock on D9
T6 requests lock on D15

The final request is the queried request. Name the transaction whose held lock blocks that request; if no transaction blocks it, the answer is the token None.
The answer is a single transaction id such as T1, or the token None when no transaction blocks the queried request.

Answer: T5

