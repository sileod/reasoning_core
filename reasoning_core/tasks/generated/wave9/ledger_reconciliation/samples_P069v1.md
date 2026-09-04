## Level 0

### Example 1

**Prompt:**

Starting balances: acct0=4, acct1=7, acct2=3, acct3=9, acct4=1
Ordered ledger operations (dest is the account, debit adds to it, credit deducts if the account has enough, hold reserves the amount against available funds, release first frees held funds then deducts, reversal subtracts the amount from the account):
  debit acct2 7
  debit acct4 3
  hold acct0 3
  release acct1 9
  hold acct3 6
  reversal acct3 2
  reversal acct3 1
  debit acct2 9
Processing stops at the first operation that fails (a credit or release whose amount cannot be covered). If every operation succeeds, give the final balances as account=value pairs joined by semicolons, sorted by account name. If processing stops early, give the unmatched operation as one line like 'unmatched credit acct3 5'. State the answer alone.

**Answer:**

unmatched release acct1 9

### Example 2

**Prompt:**

Starting balances: acct0=5, acct1=0, acct2=9, acct3=9, acct4=6
Ordered ledger operations (dest is the account, debit adds to it, credit deducts if the account has enough, hold reserves the amount against available funds, release first frees held funds then deducts, reversal subtracts the amount from the account):
  credit acct1 9
  reversal acct4 1
  hold acct4 7
  debit acct3 8
  debit acct1 9
  reversal acct1 2
  credit acct4 8
  debit acct1 3
Processing stops at the first operation that fails (a credit or release whose amount cannot be covered). If every operation succeeds, give the final balances as account=value pairs joined by semicolons, sorted by account name. If processing stops early, give the unmatched operation as one line like 'unmatched credit acct3 5'. State the answer alone.

**Answer:**

unmatched credit acct1 9

## Level 2

### Example 1

**Prompt:**

Starting balances: acct0=5, acct1=1, acct2=2, acct3=9, acct4=2, acct5=1, acct6=4
Ordered ledger operations (dest is the account, debit adds to it, credit deducts if the account has enough, hold reserves the amount against available funds, release first frees held funds then deducts, reversal subtracts the amount from the account):
  release acct0 8
  release acct2 5
  hold acct5 11
  release acct2 1
  debit acct1 10
  debit acct0 7
  credit acct5 6
  release acct0 7
  reversal acct6 4
  credit acct1 3
  release acct6 3
  credit acct2 1
Processing stops at the first operation that fails (a credit or release whose amount cannot be covered). If every operation succeeds, give the final balances as account=value pairs joined by semicolons, sorted by account name. If processing stops early, give the unmatched operation as one line like 'unmatched credit acct3 5'. State the answer alone.

**Answer:**

unmatched release acct0 8

### Example 2

**Prompt:**

Starting balances: acct0=1, acct1=6, acct2=4, acct3=5, acct4=11, acct5=1, acct6=0
Ordered ledger operations (dest is the account, debit adds to it, credit deducts if the account has enough, hold reserves the amount against available funds, release first frees held funds then deducts, reversal subtracts the amount from the account):
  credit acct3 5
  release acct6 3
  release acct6 4
  reversal acct5 7
  credit acct3 1
  reversal acct3 7
  hold acct2 9
  reversal acct5 1
  release acct1 7
  hold acct2 3
  credit acct2 4
  credit acct3 9
Processing stops at the first operation that fails (a credit or release whose amount cannot be covered). If every operation succeeds, give the final balances as account=value pairs joined by semicolons, sorted by account name. If processing stops early, give the unmatched operation as one line like 'unmatched credit acct3 5'. State the answer alone.

**Answer:**

unmatched release acct6 3

## Level 5

### Example 1

**Prompt:**

Starting balances: acct0=2, acct1=0, acct2=12, acct3=12, acct4=1, acct5=8, acct6=5, acct7=9, acct8=9, acct9=4
Ordered ledger operations (dest is the account, debit adds to it, credit deducts if the account has enough, hold reserves the amount against available funds, release first frees held funds then deducts, reversal subtracts the amount from the account):
  debit acct8 5
  debit acct7 6
  debit acct4 2
  release acct3 12
  debit acct0 7
  credit acct9 1
  hold acct3 14
  reversal acct6 1
  hold acct1 8
  hold acct9 2
  hold acct4 14
  credit acct6 13
  hold acct2 9
  reversal acct7 3
  credit acct7 9
  hold acct3 5
  reversal acct7 14
  credit acct6 14
Processing stops at the first operation that fails (a credit or release whose amount cannot be covered). If every operation succeeds, give the final balances as account=value pairs joined by semicolons, sorted by account name. If processing stops early, give the unmatched operation as one line like 'unmatched credit acct3 5'. State the answer alone.

**Answer:**

unmatched credit acct6 13

### Example 2

**Prompt:**

Starting balances: acct0=2, acct1=2, acct2=12, acct3=13, acct4=5, acct5=7, acct6=14, acct7=2, acct8=4, acct9=9
Ordered ledger operations (dest is the account, debit adds to it, credit deducts if the account has enough, hold reserves the amount against available funds, release first frees held funds then deducts, reversal subtracts the amount from the account):
  release acct2 4
  debit acct5 12
  reversal acct9 8
  release acct0 3
  credit acct9 13
  release acct7 4
  release acct3 6
  credit acct8 13
  debit acct8 3
  debit acct7 11
  release acct6 8
  debit acct8 4
  credit acct9 9
  reversal acct3 12
  hold acct0 4
  debit acct3 8
  debit acct4 7
  hold acct6 1
Processing stops at the first operation that fails (a credit or release whose amount cannot be covered). If every operation succeeds, give the final balances as account=value pairs joined by semicolons, sorted by account name. If processing stops early, give the unmatched operation as one line like 'unmatched credit acct3 5'. State the answer alone.

**Answer:**

unmatched credit acct9 13

