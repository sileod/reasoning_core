# ConfigurationPrecedenceResolution samples (P073v1)

## Level 0

### Example 1

**Prompt:**

A program reads configuration from four sources, applied in order from weakest to strongest: defaults, inherited, environment, local. Later sources override earlier ones for the same key. A setting marked `deleted` removes that key from the effective configuration entirely; a later source may re-add it. Given:
defaults settings: vdh = deleted, vw = 1T
inherited settings: vw = WFS, zll = deleted
environment settings: 
local settings: vdh = DW7C5, zll = ZI8FNF

Write the effective configuration as comma-separated entries `key = value`, sorted alphabetically by key, omitting any key that ends up deleted. For a working config with keys `host` and `port` the answer format is: host = x, port = 123

**Answer:**

vdh = DW7C5, vw = WFS, zll = ZI8FNF

### Example 2

**Prompt:**

A program reads configuration from four sources, applied in order from weakest to strongest: defaults, inherited, environment, local. Later sources override earlier ones for the same key. A setting marked `deleted` removes that key from the effective configuration entirely; a later source may re-add it. Given:
defaults settings: fbn = deleted, is = VX, jcqkw = QU
inherited settings: fbn = AKSED
environment settings: fbn = O4, is = EWXPW9
local settings: is = deleted, jcqkw = 5YOX

Write the effective configuration as comma-separated entries `key = value`, sorted alphabetically by key, omitting any key that ends up deleted. For a working config with keys `host` and `port` the answer format is: host = x, port = 123

**Answer:**

fbn = O4, jcqkw = 5YOX

## Level 2

### Example 1

**Prompt:**

A program reads configuration from four sources, applied in order from weakest to strongest: defaults, inherited, environment, local. Later sources override earlier ones for the same key. A setting marked `deleted` removes that key from the effective configuration entirely; a later source may re-add it. Given:
defaults settings: fttdp = 5PQ8, lf = deleted, wsl = 5YWM, xtqg = deleted
inherited settings: lf = Z7L, wsl = QSZ8MF, xla = EF
environment settings: fttdp = deleted, lf = C7MB, wsl = 7S, xla = SLS
local settings: fttdp = 50D4X0, wsl = XP, xla = LH7N, xtqg = WOAU4F

Write the effective configuration as comma-separated entries `key = value`, sorted alphabetically by key, omitting any key that ends up deleted. For a working config with keys `host` and `port` the answer format is: host = x, port = 123

**Answer:**

fttdp = 50D4X0, lf = C7MB, wsl = XP, xla = LH7N, xtqg = WOAU4F

### Example 2

**Prompt:**

A program reads configuration from four sources, applied in order from weakest to strongest: defaults, inherited, environment, local. Later sources override earlier ones for the same key. A setting marked `deleted` removes that key from the effective configuration entirely; a later source may re-add it. Given:
defaults settings: awja = B1P7N, mtgw = deleted, pvmkc = FAR0
inherited settings: awja = EZZ, kgcf = 3UB, pvmkc = deleted, zgavn = GDPF
environment settings: awja = P48S, mtgw = GSHF, pvmkc = P76, zgavn = I240C
local settings: awja = ZOKYK, mtgw = CBJ156, pvmkc = K93Z, zgavn = BHB

Write the effective configuration as comma-separated entries `key = value`, sorted alphabetically by key, omitting any key that ends up deleted. For a working config with keys `host` and `port` the answer format is: host = x, port = 123

**Answer:**

awja = ZOKYK, kgcf = 3UB, mtgw = CBJ156, pvmkc = K93Z, zgavn = BHB

## Level 5

### Example 1

**Prompt:**

A program reads configuration from four sources, applied in order from weakest to strongest: defaults, inherited, environment, local. Later sources override earlier ones for the same key. A setting marked `deleted` removes that key from the effective configuration entirely; a later source may re-add it. Given:
defaults settings: bgt = 311Y, cxsu = deleted, hwyba = deleted, oz = VHDAS, pa = BNH0, raw = 5H, ss = JJF
inherited settings: bgt = deleted, hwyba = deleted, oz = deleted, pa = QVNW48, raw = QUVVY, ss = deleted
environment settings: bgt = 6Q3N25, cxsu = deleted, hwyba = CM, raw = AC1F, ss = 80A, ssri = ZZXIA
local settings: bgt = JOA, hwyba = 3I, pa = CXTL7S, raw = G86IO, ssri = 78YTYW

Write the effective configuration as comma-separated entries `key = value`, sorted alphabetically by key, omitting any key that ends up deleted. For a working config with keys `host` and `port` the answer format is: host = x, port = 123

**Answer:**

bgt = JOA, hwyba = 3I, pa = CXTL7S, raw = G86IO, ss = 80A, ssri = 78YTYW

### Example 2

**Prompt:**

A program reads configuration from four sources, applied in order from weakest to strongest: defaults, inherited, environment, local. Later sources override earlier ones for the same key. A setting marked `deleted` removes that key from the effective configuration entirely; a later source may re-add it. Given:
defaults settings: cm = TYLP7T, dham = AED963, hhmtx = H1NV9U, qhr = TDWZM, tngoo = VOZ, tqhqn = RUG, tw = M4
inherited settings: cm = QDAR4, dham = KRM99, hhmtx = 77FOJ9, ogm = WXFW, qhr = 39QYP, tngoo = 23K8C, tqhqn = deleted, tw = 16PJD
environment settings: cm = WUXMZM, dham = FJNQ7I, ogm = MXU, qhr = TLZ, tqhqn = deleted, tw = deleted
local settings: cm = deleted, dham = 0PIY, hhmtx = deleted, ogm = Q6X4, qhr = CJRUJ4, tqhqn = deleted, tw = RG3VR0

Write the effective configuration as comma-separated entries `key = value`, sorted alphabetically by key, omitting any key that ends up deleted. For a working config with keys `host` and `port` the answer format is: host = x, port = 123

**Answer:**

dham = 0PIY, ogm = Q6X4, qhr = CJRUJ4, tngoo = 23K8C, tw = RG3VR0
