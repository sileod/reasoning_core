## Level 0

### Example

**Prompt**:

Coordinate frames are related by discrete rigid transforms applied to coordinates in the source frame to obtain coordinates in the target frame, in this fixed order: first rotate counterclockwise by the given amount, then reflect across the named axis, then translate. A reflection across the vertical axis maps (x, y) to (-x, y); a reflection across the horizontal axis maps (x, y) to (x, -y). A counterclockwise rotation by theta maps a vector to (x*cos(theta) - y*sin(theta), x*sin(theta) + y*cos(theta)).

Point Location:
A point P has coordinates (-2, 3) in frame B.

Relations:
- frame B from frame A: rotate 270 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (-1, -4)
- frame C from frame B: rotate 0 degree(s) counterclockwise, then translate by (5, 5)
- frame D from frame C: rotate 90 degree(s) counterclockwise, then reflect across the horizontal axis, then translate by (5, 3)
- frame E from frame D: rotate 180 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (-3, -2)

Question:
What are the coordinates of P in frame C?

Give the answer as (x, y).

**Answer**:

(3, 8)

### Example

**Prompt**:

Coordinate frames are related by discrete rigid transforms applied to coordinates in the source frame to obtain coordinates in the target frame, in this fixed order: first rotate counterclockwise by the given amount, then reflect across the named axis, then translate. A reflection across the vertical axis maps (x, y) to (-x, y); a reflection across the horizontal axis maps (x, y) to (x, -y). A counterclockwise rotation by theta maps a vector to (x*cos(theta) - y*sin(theta), x*sin(theta) + y*cos(theta)).

Point Location:
A point P has coordinates (2, 3) in frame B.

Relations:
- frame B from frame A: rotate 0 degree(s) counterclockwise, then reflect across the horizontal axis, then translate by (-5, -2)
- frame C from frame B: rotate 270 degree(s) counterclockwise, then translate by (5, -5)
- frame D from frame C: rotate 0 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (-3, 0)
- frame E from frame D: rotate 180 degree(s) counterclockwise, then reflect across the horizontal axis, then translate by (6, 4)

Question:
What are the coordinates of P in frame A?

Give the answer as (x, y).

**Answer**:

(7, -5)

## Level 2

### Example

**Prompt**:

Coordinate frames are related by discrete rigid transforms applied to coordinates in the source frame to obtain coordinates in the target frame, in this fixed order: first rotate counterclockwise by the given amount, then reflect across the named axis, then translate. A reflection across the vertical axis maps (x, y) to (-x, y); a reflection across the horizontal axis maps (x, y) to (x, -y). A counterclockwise rotation by theta maps a vector to (x*cos(theta) - y*sin(theta), x*sin(theta) + y*cos(theta)).

Point Location:
A point P has coordinates (-10, 14) in frame A.

Relations:
- frame B from frame A: rotate 180 degree(s) counterclockwise, then reflect across the horizontal axis, then translate by (-3, -9)
- frame C from frame B: rotate 90 degree(s) counterclockwise, then reflect across the horizontal axis, then translate by (-11, 4)
- frame D from frame C: rotate 270 degree(s) counterclockwise, then translate by (-10, -9)
- frame E from frame D: rotate 90 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (13, -6)
- frame F from frame E: rotate 90 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (-13, -14)
- frame G from frame F: rotate 270 degree(s) counterclockwise, then translate by (0, 9)

Question:
What are the coordinates of P in frame G?

Give the answer as (x, y).

**Answer**:

(6, 41)

### Example

**Prompt**:

Coordinate frames are related by discrete rigid transforms applied to coordinates in the source frame to obtain coordinates in the target frame, in this fixed order: first rotate counterclockwise by the given amount, then reflect across the named axis, then translate. A reflection across the vertical axis maps (x, y) to (-x, y); a reflection across the horizontal axis maps (x, y) to (x, -y). A counterclockwise rotation by theta maps a vector to (x*cos(theta) - y*sin(theta), x*sin(theta) + y*cos(theta)).

Point Location:
A point P has coordinates (-3, 6) in frame B.

Relations:
- frame B from frame A: rotate 270 degree(s) counterclockwise, then translate by (-9, -4)
- frame C from frame B: rotate 0 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (6, -5)
- frame D from frame C: rotate 0 degree(s) counterclockwise, then translate by (-2, -6)
- frame E from frame D: rotate 270 degree(s) counterclockwise, then reflect across the horizontal axis, then translate by (-13, 3)
- frame F from frame E: rotate 180 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (-9, -9)
- frame G from frame F: rotate 270 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (-9, 5)

Question:
What are the coordinates of P in frame D?

Give the answer as (x, y).

**Answer**:

(7, -5)

## Level 5

### Example

**Prompt**:

Coordinate frames are related by discrete rigid transforms applied to coordinates in the source frame to obtain coordinates in the target frame, in this fixed order: first rotate counterclockwise by the given amount, then reflect across the named axis, then translate. A reflection across the vertical axis maps (x, y) to (-x, y); a reflection across the horizontal axis maps (x, y) to (x, -y). A counterclockwise rotation by theta maps a vector to (x*cos(theta) - y*sin(theta), x*sin(theta) + y*cos(theta)).

Point Location:
A point P has coordinates (-17, 4) in frame A.

Relations:
- frame B from frame A: rotate 180 degree(s) counterclockwise, then translate by (-7, -9)
- frame C from frame B: rotate 0 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (-21, 13)
- frame D from frame C: rotate 90 degree(s) counterclockwise, then translate by (-5, 26)
- frame E from frame D: rotate 270 degree(s) counterclockwise, then reflect across the horizontal axis, then translate by (11, -12)
- frame F from frame E: rotate 180 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (25, -22)
- frame G from frame F: rotate 180 degree(s) counterclockwise, then reflect across the horizontal axis, then translate by (-14, -22)
- frame H from frame G: rotate 180 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (18, -10)
- frame I from frame H: rotate 180 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (15, -10)
- frame J from frame I: rotate 180 degree(s) counterclockwise, then reflect across the horizontal axis, then translate by (-4, 7)

Question:
What are the coordinates of P in frame I?

Give the answer as (x, y).

**Answer**:

(-12, -27)

### Example

**Prompt**:

Coordinate frames are related by discrete rigid transforms applied to coordinates in the source frame to obtain coordinates in the target frame, in this fixed order: first rotate counterclockwise by the given amount, then reflect across the named axis, then translate. A reflection across the vertical axis maps (x, y) to (-x, y); a reflection across the horizontal axis maps (x, y) to (x, -y). A counterclockwise rotation by theta maps a vector to (x*cos(theta) - y*sin(theta), x*sin(theta) + y*cos(theta)).

Point Location:
A point P has coordinates (-7, 9) in frame H.

Relations:
- frame B from frame A: rotate 90 degree(s) counterclockwise, then reflect across the horizontal axis, then translate by (-17, -19)
- frame C from frame B: rotate 270 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (-6, 6)
- frame D from frame C: rotate 180 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (-17, 16)
- frame E from frame D: rotate 90 degree(s) counterclockwise, then reflect across the horizontal axis, then translate by (1, 10)
- frame F from frame E: rotate 180 degree(s) counterclockwise, then translate by (3, 24)
- frame G from frame F: rotate 180 degree(s) counterclockwise, then translate by (16, 15)
- frame H from frame G: rotate 270 degree(s) counterclockwise, then reflect across the vertical axis, then translate by (16, 1)
- frame I from frame H: rotate 180 degree(s) counterclockwise, then reflect across the horizontal axis, then translate by (-11, -19)
- frame J from frame I: rotate 270 degree(s) counterclockwise, then translate by (17, 4)

Question:
What are the coordinates of P in frame C?

Give the answer as (x, y).

**Answer**:

(-5, -6)
