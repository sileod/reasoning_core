# Samples for P043v1: virtual_address_translation

## Level 0

### Example 1

**Prompt:**

A process uses a 2-level page table with 5 bits per level and a 10-bit offset field.
The page table (as nested dicts mapping level index to subtree, with leaf values being physical frame numbers) is:
0:
{'18': 16068}

1:
{'26': 12665}
The virtual address is the 20-bit binary value 00000100101100011110.
The answer is the physical address as a decimal integer, or the exact two-word phrase 'page fault' if the translation faults.

**Answer:**

16454430

### Example 2

**Prompt:**

A process uses a 2-level page table with 5 bits per level and a 10-bit offset field.
The page table (as nested dicts mapping level index to subtree, with leaf values being physical frame numbers) is:
22:
{'1': 9679}

26:
{'11': 416}
The virtual address is the 20-bit binary value 10110000010001010101.
The answer is the physical address as a decimal integer, or the exact two-word phrase 'page fault' if the translation faults.

**Answer:**

9911381

## Level 2

### Example 1

**Prompt:**

A process uses a 2-level page table with 7 bits per level and a 10-bit offset field.
The page table (as nested dicts mapping level index to subtree, with leaf values being physical frame numbers) is:
23:
{'71': 56821}

55:
{'77': 37795}

75:
{'56': 25217}

113:
{'66': 82924}
The virtual address is the 24-bit binary value 100101101110000101100110.
The answer is the physical address as a decimal integer, or the exact two-word phrase 'page fault' if the translation faults.

**Answer:**

25822566

### Example 2

**Prompt:**

A process uses a 2-level page table with 7 bits per level and a 10-bit offset field.
The page table (as nested dicts mapping level index to subtree, with leaf values being physical frame numbers) is:
67:
{'98': 21638}

70:
{'25': 23521}

72:
{'29': 108798}

82:
{'75': 77288}

85:
{'42': 78920}

86:
{'5': 85659}

90:
{'60': 108565, '77': 34971}

92:
{'28': 122310}
The virtual address is the 24-bit binary value 101101010011010101110111.
The answer is the physical address as a decimal integer, or the exact two-word phrase 'page fault' if the translation faults.

**Answer:**

35810679

## Level 5

### Example 1

**Prompt:**

A process uses a 3-level page table with 7 bits per level and a 9-bit offset field.
The page table (as nested dicts mapping level index to subtree, with leaf values being physical frame numbers) is:
16:
{'84': {'41': 60287}}

46:
{'28': {'85': 12568}}

56:
{'126': {'99': 19929}}

80:
{'126': {'53': 24734}}

82:
{'17': {'127': 3449}}

127:
{'75': {'49': 41873}}
The virtual address is the 30-bit binary value 010111000111001010101010000100.
The answer is the physical address as a decimal integer, or the exact two-word phrase 'page fault' if the translation faults.

**Answer:**

6434948

### Example 2

**Prompt:**

A process uses a 3-level page table with 7 bits per level and a 9-bit offset field.
The page table (as nested dicts mapping level index to subtree, with leaf values being physical frame numbers) is:
113:
{'120': {'37': 27174}}

127:
{'12': {'107': 21007}}
The virtual address is the 30-bit binary value 111111100011001101011101101001.
The answer is the physical address as a decimal integer, or the exact two-word phrase 'page fault' if the translation faults.

**Answer:**

10755945
