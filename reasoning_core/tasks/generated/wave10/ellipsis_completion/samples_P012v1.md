# Ellipsis Completion — P012v1 samples

## Level 0

**Prompt:**

```
Dialogue:
Customer 1 handed over the bill for 26 euros.
Customer 2 settled the tab for 79 pounds.
Customer 3 settled the tab for 46 euros.
Later the customer asked what the 1st transaction, 2nd transaction, and 3rd transaction in total came to.
The cashier replied, elliptically: "that ran to <blank>."

The cashier's elliptical reply "that ran to <blank>" is missing its amount. Recover the amount by resolving the ellipsis: the missing figure is the sum of the amounts of the transactions the customer's question referenced (1st, 2nd, 3rd). What integer amount does the ellipsis resolve to?

The answer is a single integer.
```

**Answer:**

`151`

**Prompt:**

```
Dialogue:
Customer 1 handed over the order for 12 dollars.
Customer 2 handed over the order for 75 dollars.
Customer 3 covered the bill for 33 pounds.
Later the customer asked what the 2nd transaction and 3rd transaction in total came to.
The cashier replied, elliptically: "that ran to <blank>."

The cashier's elliptical reply "that ran to <blank>" is missing its amount. Recover the amount by resolving the ellipsis: the missing figure is the sum of the amounts of the transactions the customer's question referenced (2nd, 3rd). What integer amount does the ellipsis resolve to?

The answer is a single integer.
```

**Answer:**

`108`

## Level 2

**Prompt:**

```
Dialogue:
Customer 1 paid the total for 62 dollars.
Customer 2 settled the order for 57 euros.
Customer 3 paid the total for 51 euros.
Customer 4 rang up the order for 43 euros.
Customer 5 settled the order for 66 pounds.
Later the customer asked what the 1st transaction, 3rd transaction, and 4th transaction altogether came to.
The cashier replied, elliptically: "it came to <blank>."

The cashier's elliptical reply "it came to <blank>" is missing its amount. Recover the amount by resolving the ellipsis: the missing figure is the sum of the amounts of the transactions the customer's question referenced (1st, 3rd, 4th). What integer amount does the ellipsis resolve to?

The answer is a single integer.
```

**Answer:**

`156`

**Prompt:**

```
Dialogue:
Customer 1 handed over the tab for 71 pounds.
Customer 2 settled the bill for 22 pounds.
Customer 3 rang up the bill for 68 pounds.
Customer 4 handed over the tab for 44 euros.
Customer 5 covered the bill for 58 pounds.
Later the customer asked what the 1st transaction, 2nd transaction, and 4th transaction put together came to.
The cashier replied, elliptically: "that totalled <blank>."

The cashier's elliptical reply "that totalled <blank>" is missing its amount. Recover the amount by resolving the ellipsis: the missing figure is the sum of the amounts of the transactions the customer's question referenced (1st, 2nd, 4th). What integer amount does the ellipsis resolve to?

The answer is a single integer.
```

**Answer:**

`137`

## Level 5

**Prompt:**

```
Dialogue:
Customer 1 paid the bill for 784 pounds.
Customer 2 rang up the tab for 218 pounds.
Customer 3 covered the order for 512 pounds.
Customer 4 handed over the total for 699 dollars.
Customer 5 settled the bill for 614 pounds.
Customer 6 settled the total for 898 pounds.
Customer 7 covered the tab for 676 dollars.
Later the customer asked what the 4th transaction and 5th transaction in total came to.
The cashier replied, elliptically: "the amount was <blank>."

The cashier's elliptical reply "the amount was <blank>" is missing its amount. Recover the amount by resolving the ellipsis: the missing figure is the sum of the amounts of the transactions the customer's question referenced (4th, 5th). What integer amount does the ellipsis resolve to?

The answer is a single integer.
```

**Answer:**

`1313`

**Prompt:**

```
Dialogue:
Customer 1 rang up the order for 511 pounds.
Customer 2 covered the tab for 979 dollars.
Customer 3 paid the total for 511 dollars.
Customer 4 handed over the order for 485 dollars.
Customer 5 paid the bill for 487 euros.
Customer 6 paid the order for 574 dollars.
Customer 7 paid the order for 540 dollars.
Later the customer asked what the 4th transaction and 5th transaction added up came to.
The cashier replied, elliptically: "it came to <blank>."

The cashier's elliptical reply "it came to <blank>" is missing its amount. Recover the amount by resolving the ellipsis: the missing figure is the sum of the amounts of the transactions the customer's question referenced (4th, 5th). What integer amount does the ellipsis resolve to?

The answer is a single integer.
```

**Answer:**

`972`

