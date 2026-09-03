## Level 0

**Example 1**

Prompt:
```
Text:
There are 4 intervals on a single number line running from 5:00 to 19:00.
1. return pickup from 17:00 to 20:00.
2. courier pickup from 7:00 to 11:00.
3. loading-dock slot from 10:00 to 11:00.
4. courier pickup from 9:00 to 11:00.
Pick the smallest set of integer times (in hours) so that every interval contains at least one chosen time. Ties break toward the smallest point. Give the chosen times as a comma-separated increasing list, e.g. 9,12.
```

Answer:
```
11,20
```

**Example 2**

Prompt:
```
Text:
There are 4 intervals on a single number line running from 7:00 to 21:00.
1. parcel drop from 14:00 to 16:00.
2. parcel drop from 15:00 to 16:00.
3. package handoff from 14:00 to 15:00.
4. loading-dock slot from 7:00 to 9:00.
Pick the smallest set of integer times (in hours) so that every interval contains at least one chosen time. Ties break toward the smallest point. Give the chosen times as a comma-separated increasing list, e.g. 9,12.
```

Answer:
```
9,15
```

## Level 2

**Example 1**

Prompt:
```
Text:
There are 7 intervals on a single number line running from 6:00 to 28:00.
1. loading-dock slot from 18:00 to 24:00.
2. loading-dock slot from 11:00 to 15:00.
3. loading-dock slot from 6:00 to 7:00.
4. driver swap from 19:00 to 21:00.
5. courier pickup from 16:00 to 19:00.
6. depot refuel stop from 20:00 to 25:00.
7. driver swap from 8:00 to 14:00.
Pick the smallest set of integer times (in hours) so that every interval contains at least one chosen time. Ties break toward the smallest point. Give the chosen times as a comma-separated increasing list, e.g. 9,12.
```

Answer:
```
7,14,19,25
```

**Example 2**

Prompt:
```
Text:
There are 7 intervals on a single number line running from 10:00 to 32:00.
1. workshop slot from 10:00 to 11:00.
2. onboarding slot from 16:00 to 20:00.
3. workshop slot from 21:00 to 27:00.
4. review slot from 11:00 to 14:00.
5. video-call slot from 14:00 to 18:00.
6. interview slot from 13:00 to 19:00.
7. feedback slot from 25:00 to 26:00.
Pick the smallest set of integer times (in hours) so that every interval contains at least one chosen time. Ties break toward the smallest point. Give the chosen times as a comma-separated increasing list, e.g. 9,12.
```

Answer:
```
11,18,26
```

## Level 5

**Example 1**

Prompt:
```
Text:
There are 11 intervals on a single number line running from 9:00 to 43:00.
1. onboarding slot from 14:00 to 16:00.
2. feedback slot from 38:00 to 40:00.
3. video-call slot from 9:00 to 14:00.
4. feedback slot from 21:00 to 27:00.
5. video-call slot from 19:00 to 24:00.
6. workshop slot from 39:00 to 48:00.
7. meeting slot from 28:00 to 30:00.
8. feedback slot from 21:00 to 31:00.
9. review slot from 26:00 to 35:00.
10. video-call slot from 13:00 to 15:00.
11. video-call slot from 40:00 to 49:00.
Pick the smallest set of integer times (in hours) so that every interval contains at least one chosen time. Ties break toward the smallest point. Give the chosen times as a comma-separated increasing list, e.g. 9,12.
```

Answer:
```
14,24,30,40
```

**Example 2**

Prompt:
```
Text:
There are 12 intervals on a single number line running from 8:00 to 42:00.
1. parcel drop from 11:00 to 12:00.
2. sorting-bay slot from 40:00 to 43:00.
3. package handoff from 13:00 to 19:00.
4. driver swap from 37:00 to 40:00.
5. sorting-bay slot from 27:00 to 29:00.
6. courier pickup from 33:00 to 39:00.
7. depot refuel stop from 18:00 to 21:00.
8. loading-dock slot from 12:00 to 18:00.
9. courier pickup from 24:00 to 28:00.
10. courier pickup from 35:00 to 39:00.
11. parcel drop from 20:00 to 22:00.
12. parcel drop from 39:00 to 48:00.
Pick the smallest set of integer times (in hours) so that every interval contains at least one chosen time. Ties break toward the smallest point. Give the chosen times as a comma-separated increasing list, e.g. 9,12.
```

Answer:
```
12,19,22,28,39,43
```
