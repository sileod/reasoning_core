# Level 0

## Example

**Prompt:**

```
Schema (keys are validated in this order):
- mean: string
- node: number in [3, 7]
- max: array of:
    - price: integer in [-2, 1]
- mode: array of:
    - length: number in [-3, 0]

JSON document:
{
 "mean": "s0",
 "node": 10,
 "max": [
  0,
  0
 ],
 "mode": [
  -3.0,
  0.0,
  -3.0
 ]
}

Validate the document against the schema. Keys are checked in the order the schema lists them, and array elements by index. Report the dotted path of the FIRST violation (e.g. 'k0.items.1'). If the document is fully valid, answer exactly 'valid'.
```

**Answer:**

```
node
```


## Example

**Prompt:**

```
Schema (keys are validated in this order):
- count: string
- col: string
- list: number in [-3, 3]
- limit9: integer in [-5, -1]

JSON document:
{
 "count": 23,
 "col": "s5",
 "list": -1.0,
 "limit9": -1
}

Validate the document against the schema. Keys are checked in the order the schema lists them, and array elements by index. Report the dotted path of the FIRST violation (e.g. 'k0.items.1'). If the document is fully valid, answer exactly 'valid'.
```

**Answer:**

```
count
```


# Level 2

## Example

**Prompt:**

```
Schema (keys are validated in this order):
- rank3: string
- id: integer in [2, 8]
- edge: array of:
    - score: integer in [-5, -1]
- rank: number in [9, 14]
- depth: number in [1, 4]
- flag0: object {
    - flag: number in [-3, -2]
    - mode: array of:
        - size: string
    - status: array of:
        - min: number in [6, 12]
    - mean5: integer in [9, 15]
    - height3: string
  }

JSON document:
{
 "rank3": "s5",
 "id": 7,
 "edge": [
  -4,
  -5
 ],
 "rank": 10.0,
 "depth": 3.0,
 "flag0": {
  "flag": -2.0,
  "mode": [
   "s9",
   "s4",
   "s0"
  ],
  "status": 61,
  "mean5": 14,
  "height3": "s7"
 }
}

Validate the document against the schema. Keys are checked in the order the schema lists them, and array elements by index. Report the dotted path of the FIRST violation (e.g. 'k0.items.1'). If the document is fully valid, answer exactly 'valid'.
```

**Answer:**

```
flag0.status
```


## Example

**Prompt:**

```
Schema (keys are validated in this order):
- name: object {
    - rate: number in [-4, -3]
    - width: integer in [5, 8]
    - offset: integer in [7, 8]
    - field2: object {
        - mean0: object {
            - depth: string
            - total: number in [1, 2]
            - list: integer in [0, 1]
          }
        - speed2: integer in [1, 5]
        - status: string
        - type: object {
            - unit: integer in [8, 14]
            - label: number in [-2, -1]
            - row: integer in [-4, 2]
          }
      }
    - col: string
  }
- id: integer in [-5, 0]
- sum4: object {
    - order: string
    - min6: number in [1, 6]
    - depth6: array of:
        - value: integer in [-3, -1]
    - meta: object {
        - row2: integer in [1, 7]
        - pair0: integer in [-1, 0]
        - node2: number in [-1, 2]
        - node0: integer in [7, 9]
      }
    - key: string
  }
- height: array of:
    - limit: number in [10, 11]
- length6: string
- width4: object {
    - sum: array of:
        - offset0: string
    - items0: integer in [-3, 1]
    - meta1: object {
        - id0: integer in [-3, -1]
        - max: string
        - value4: number in [7, 10]
        - limit8: number in [-3, 0]
      }
    - edge: string
    - items: number in [0, 5]
  }

JSON document:
{
 "name": {
  "rate": -3.0,
  "width": 7,
  "offset": 7,
  "field2": {
   "mean0": {
    "depth": "s4",
    "total": 1.0,
    "list": 0
   },
   "speed2": 3,
   "status": "s1",
   "type": {
    "unit": 14,
    "label": -1.0,
    "row": 0
   }
  },
  "col": "s4"
 },
 "id": 0,
 "sum4": {
  "order": "s5",
  "min6": 3.0,
  "depth6": [
   -3,
   -2
  ],
  "meta": {
   "row2": 7,
   "pair0": -1,
   "node2": 2.0,
   "node0": 9
  },
  "key": "s2"
 },
 "height": [
  11.0
 ],
 "length6": "s7",
 "width4": {
  "sum": [
   "s8",
   "s1",
   "s2"
  ],
  "items0": 1,
  "meta1": {
   "id0": -2,
   "max": "s9",
   "value4": 8.0,
   "limit8": -2.0
  },
  "edge": "s4",
  "items": 0.0
 }
}

Validate the document against the schema. Keys are checked in the order the schema lists them, and array elements by index. Report the dotted path of the FIRST violation (e.g. 'k0.items.1'). If the document is fully valid, answer exactly 'valid'.
```

**Answer:**

```
valid
```


# Level 5

## Example

**Prompt:**

```
Schema (keys are validated in this order):
- sum: integer in [-4, -2]
- speed: object {
    - height: object {
        - weight7: array of:
            - unit: number in [-2, 2]
        - name: string
        - sum2: string
        - meta: string
        - max7: integer in [2, 5]
        - max: object {
            - row: integer in [2, 5]
            - size: integer in [4, 5]
            - size3: number in [-2, 3]
            - depth: object {
                - mode: string
                - price: number in [-2, 0]
                - id: array of:
                    - field: string
                - index7: number in [0, 5]
                - limit: string
              }
            - mean: object {
                - col5: string
                - rate: integer in [8, 9]
                - meta0: object {
                    - index: number in [-2, 1]
                    - items: string
                    - total: integer in [9, 10]
                    - col4: object {
                        - list: string
                        - rank: integer in [7, 13]
                        - type: string
                      }
                  }
                - status3: number in [-5, -4]
                - status: number in [-1, 2]
              }
            - flag: integer in [1, 5]
          }
        - label: integer in [9, 10]
      }
    - length8: string
    - col: string
    - weight: number in [7, 13]
    - unit8: string
    - pair: array of:
        - max4: integer in [-3, 3]
    - row7: object {
        - min8: string
        - temp: object {
            - node4: array of:
                - edge: string
            - count: integer in [-1, 4]
            - meta6: object {
                - weight2: number in [-1, 0]
                - limit5: number in [-4, -2]
                - row9: array of:
                    - min: string
                - items3: number in [2, 7]
                - name2: string
              }
            - order: object {
                - sum9: array of:
                    - meta2: string
                - value: integer in [6, 12]
                - field5: array of:
                    - edge9: number in [-5, -3]
                - width: array of:
                    - rank3: string
                - depth5: number in [6, 11]
              }
            - key: string
            - size8: string
          }
        - node2: integer in [-5, -1]
        - offset1: integer in [-1, 1]
        - offset: object {
            - rate1: number in [10, 11]
            - id6: string
            - min2: integer in [3, 9]
            - key5: integer in [-4, -3]
            - depth3: array of:
                - temp6: integer in [7, 12]
            - mode9: array of:
                - id9: integer in [1, 4]
          }
        - price0: string
        - node: number in [-5, 1]
      }
    - id0: object {
        - weight4: string
        - max5: array of:
            - mean9: string
        - unit7: string
        - count0: number in [-1, 2]
        - length: array of:
            - key6: number in [8, 9]
        - score8: object {
            - field4: integer in [7, 9]
            - flag1: object {
                - temp1: number in [-3, -1]
                - status2: string
                - value5: array of:
                    - size1: string
                - size4: number in [-5, -4]
                - weight8: string
              }
            - count4: object {
                - row4: object {
                    - edge2: integer in [0, 1]
                    - value7: integer in [8, 14]
                    - row0: integer in [-1, 0]
                    - length4: number in [0, 2]
                  }
                - col3: integer in [1, 2]
                - score: object {
                    - max9: integer in [8, 9]
                    - count5: number in [-2, 0]
                    - pair3: number in [2, 6]
                    - weight5: number in [1, 4]
                  }
                - rank2: number in [7, 8]
                - mode5: string
              }
            - edge7: integer in [-2, 1]
            - max6: array of:
                - sum0: string
            - limit1: array of:
                - field8: integer in [8, 11]
          }
        - list9: object {
            - total3: integer in [-1, 3]
            - label6: string
            - price8: integer in [9, 12]
            - height8: string
            - rank4: integer in [2, 6]
            - node7: object {
                - key9: number in [2, 6]
                - speed3: number in [6, 7]
                - row8: object {
                    - name0: string
                    - flag3: string
                    - offset8: number in [5, 7]
                    - flag7: object {
                        - type3: number in [7, 9]
                        - size5: string
                        - length5: integer in [0, 4]
                      }
                  }
                - temp8: object {
                    - score3: integer in [5, 10]
                    - row6: string
                    - count8: array of:
                        - width9: integer in [4, 9]
                    - max1: number in [3, 4]
                  }
                - limit0: object {
                    - mean3: number in [4, 7]
                    - list5: string
                    - edge6: object {
                        - flag2: integer in [4, 9]
                        - index9: string
                        - max0: integer in [7, 8]
                      }
                    - offset4: integer in [-3, -2]
                  }
              }
          }
      }
  }
- mean2: integer in [4, 6]
- height7: object {
    - total6: integer in [10, 11]
    - edge1: number in [7, 13]
    - field7: string
    - type6: integer in [1, 7]
    - meta1: object {
        - sum5: object {
            - id4: integer in [-2, -1]
            - price1: array of:
                - label4: number in [10, 11]
            - col6: integer in [9, 11]
            - speed5: integer in [-4, 1]
            - total2: number in [2, 8]
            - key1: string
          }
        - node0: integer in [7, 11]
        - temp2: integer in [8, 11]
        - mean4: number in [-2, -1]
        - order1: array of:
            - offset5: string
        - key7: object {
            - rate7: number in [-2, 2]
            - key2: integer in [0, 2]
            - flag9: number in [0, 1]
            - status5: string
            - length9: array of:
                - field9: integer in [1, 6]
            - depth8: array of:
                - name4: string
          }
        - length7: number in [1, 6]
      }
    - min4: string
    - value9: string
    - name6: array of:
        - node1: number in [8, 11]
  }
- items0: object {
    - height5: number in [10, 16]
    - type5: integer in [1, 3]
    - mode3: integer in [-5, 1]
    - status1: integer in [2, 6]
    - temp3: integer in [0, 4]
    - list2: integer in [0, 4]
    - type2: integer in [8, 10]
    - width0: integer in [4, 6]
  }
- rank5: array of:
    - rate2: integer in [8, 13]
- order6: object {
    - type0: string
    - id3: integer in [2, 4]
    - label8: number in [6, 8]
    - width8: object {
        - rank7: object {
            - limit6: string
            - size0: number in [2, 6]
            - id7: integer in [-5, 1]
            - order4: object {
                - key8: number in [9, 14]
                - mode8: object {
                    - offset0: number in [8, 13]
                    - length6: integer in [1, 5]
                    - limit7: number in [0, 5]
                    - width4: number in [-1, 0]
                  }
                - height0: object {
                    - mode2: object {
                        - rank0: integer in [-3, 1]
                        - status4: number in [5, 7]
                        - list7: string
                      }
                    - value4: string
                    - total8: string
                    - unit5: object {
                        - index2: integer in [6, 9]
                        - node8: number in [-2, 0]
                        - type1: number in [3, 4]
                      }
                  }
                - width7: integer in [10, 15]
                - mode0: string
              }
            - rate6: object {
                - height4: array of:
                    - unit4: number in [0, 3]
                - col0: array of:
                    - order7: integer in [0, 2]
                - status8: array of:
                    - depth7: integer in [-2, 3]
                - pair8: number in [-1, 2]
                - height6: integer in [10, 12]
              }
            - mean8: number in [8, 12]
          }
        - edge0: string
        - temp5: number in [-5, -4]
        - label2: integer in [6, 9]
        - meta4: number in [0, 6]
        - rate8: array of:
            - total7: string
        - depth4: array of:
            - type8: number in [7, 8]
      }
    - mean6: object {
        - score2: number in [5, 9]
        - temp0: array of:
            - type7: number in [-5, -3]
        - id2: string
        - offset9: integer in [4, 7]
        - price6: string
        - rank8: number in [-1, 3]
        - count2: object {
            - value1: string
            - sum6: object {
                - col1: array of:
                    - speed7: integer in [-5, -1]
                - field1: number in [10, 14]
                - node9: integer in [7, 12]
                - order2: string
                - type9: integer in [1, 4]
              }
            - pair9: integer in [0, 3]
            - mode7: array of:
                - edge4: string
            - row2: object {
                - speed6: array of:
                    - row1: string
                - pair1: number in [-5, -4]
                - price3: integer in [4, 8]
                - rate0: object {
                    - order5: object {
                        - list4: integer in [9, 15]
                        - label9: string
                        - unit3: integer in [5, 8]
                      }
                    - f267: object {
                        - order3: number in [0, 3]
                        - unit9: number in [2, 4]
                        - field0: integer in [6, 8]
                      }
                    - mode6: string
                    - list1: number in [6, 7]
                  }
                - items1: string
              }
            - list3: string
          }
      }
    - weight1: integer in [8, 9]
    - min6: string
    - list0: number in [7, 10]
  }
- sum8: integer in [-5, 0]
- flag6: object {
    - width5: number in [-1, 3]
    - speed2: string
    - width3: number in [1, 7]
    - value2: string
    - f284: array of:
        - index8: number in [2, 8]
    - col9: array of:
        - weight3: string
    - mean7: number in [2, 3]
    - limit4: object {
        - length0: number in [7, 8]
        - depth9: object {
            - order0: array of:
                - offset6: number in [-4, 1]
            - status7: number in [8, 11]
            - index1: integer in [6, 11]
            - status6: integer in [-4, -2]
            - height9: array of:
                - id5: number in [9, 15]
            - index0: array of:
                - mean1: string
          }
        - order9: number in [2, 7]
        - mode1: string
        - f303: object {
            - id8: integer in [-2, 3]
            - order8: number in [2, 8]
            - flag8: integer in [-1, 3]
            - f307: array of:
                - depth2: string
            - limit2: integer in [2, 4]
            - unit0: object {
                - height3: object {
                    - speed8: number in [8, 13]
                    - row3: object {
                        - length2: number in [-5, -4]
                        - rate5: number in [-5, 0]
                        - offset3: number in [3, 7]
                      }
                    - temp9: integer in [0, 3]
                    - value0: number in [2, 8]
                  }
                - price7: integer in [8, 13]
                - f320: object {
                    - f321: string
                    - score6: number in [-3, -2]
                    - pair2: number in [6, 7]
                    - offset2: array of:
                        - label0: string
                  }
                - flag5: number in [3, 6]
                - weight9: integer in [5, 6]
              }
          }
        - score1: object {
            - key3: array of:
                - f330: string
            - items6: number in [6, 11]
            - f332: object {
                - label5: string
                - f334: string
                - f335: array of:
                    - pair5: string
                - value3: number in [0, 5]
                - unit2: string
              }
            - total0: integer in [1, 7]
            - speed1: number in [7, 10]
            - size7: number in [6, 7]
          }
        - index5: string
      }
  }

JSON document:
{
 "sum": -4,
 "speed": {
  "height": {
   "weight7": [
    -2.0
   ],
   "name": "s8",
   "sum2": "s8",
   "meta": "s6",
   "max7": 3,
   "max": {
    "row": 5,
    "size": 4,
    "size3": 2.0,
    "depth": {
     "mode": "s9",
     "price": -1.0,
     "id": [
      "s1",
      "s9"
     ],
     "index7": 2.0,
     "limit": "s2"
    },
    "mean": {
     "col5": "s9",
     "rate": 9,
     "meta0": {
      "index": 0.0,
      "items": "s0",
      "total": 9,
      "col4": {
       "list": "s5",
       "rank": 10,
       "type": "s6"
      }
     },
     "status3": -5.0,
     "status": 0.0
    },
    "flag": 3
   },
   "label": 9
  },
  "length8": "s9",
  "col": "s1",
  "weight": 7.0,
  "unit8": "s4",
  "pair": [
   -3,
   -1
  ],
  "row7": {
   "min8": "s3",
   "temp": {
    "node4": [
     "s4",
     "s4"
    ],
    "count": 0,
    "meta6": {
     "weight2": -1.0,
     "limit5": -3.0,
     "row9": [
      "s8",
      "s2"
     ],
     "items3": 7.0,
     "name2": "s2"
    },
    "order": {
     "sum9": [
      "s1"
     ],
     "value": 6,
     "field5": [
      -3.0
     ],
     "width": [
      "s8",
      "s1"
     ],
     "depth5": 7.0
    },
    "key": "s6",
    "size8": "s2"
   },
   "node2": -5,
   "offset1": 1,
   "offset": {
    "rate1": 10.0,
    "id6": "s6",
    "min2": 6,
    "key5": -3,
    "depth3": [
     10
    ],
    "mode9": [
     3
    ]
   },
   "price0": "s6",
   "node": -3.0
  },
  "id0": {
   "weight4": "s6",
   "max5": [
    "s9"
   ],
   "unit7": "s8",
   "count0": 2.0,
   "length": [
    9.0,
    9.0
   ],
   "score8": {
    "field4": 8,
    "flag1": {
     "temp1": -3.0,
     "status2": "s1",
     "value5": [
      "s0",
      "s1",
      "s0"
     ],
     "size4": -5.0,
     "weight8": "s1"
    },
    "count4": {
     "row4": {
      "edge2": 0,
      "value7": 14,
      "row0": -1,
      "length4": 2.0
     },
     "col3": 1,
     "score": {
      "max9": 9,
      "count5": -1.0,
      "pair3": 3.0,
      "weight5": 4.0
     },
     "rank2": 7.0,
     "mode5": "s2"
    },
    "edge7": 1,
    "max6": [
     "s5",
     "s2",
     "s4"
    ],
    "limit1": [
     11
    ]
   },
   "list9": {
    "total3": 2,
    "label6": "s7",
    "price8": 9,
    "height8": "s6",
    "rank4": 3,
    "node7": {
     "key9": 4.0,
     "speed3": 7.0,
     "row8": {
      "name0": "s0",
      "flag3": "s1",
      "offset8": 7.0,
      "flag7": {
       "type3": 7.0,
       "size5": "s3",
       "length5": 1
      }
     },
     "temp8": {
      "score3": 6,
      "row6": "s7",
      "count8": [
       8,
       7,
       5
      ],
      "max1": 3.0
     },
     "limit0": {
      "mean3": 4.0,
      "list5": "s2",
      "edge6": {
       "flag2": 4,
       "index9": "s5",
       "max0": 8
      },
      "offset4": -2
     }
    }
   }
  }
 },
 "mean2": 5,
 "height7": {
  "total6": 10,
  "edge1": 9.0,
  "field7": "s6",
  "type6": 5,
  "meta1": {
   "sum5": {
    "id4": -2,
    "price1": [
     11.0,
     11.0
    ],
    "col6": 11,
    "speed5": -2,
    "total2": 6.0,
    "key1": "s2"
   },
   "node0": 11,
   "temp2": 11,
   "mean4": -2.0,
   "order1": [
    "s1",
    "s7"
   ],
   "key7": {
    "rate7": 1.0,
    "key2": 0,
    "flag9": 1.0,
    "status5": "s4",
    "length9": [
     6
    ],
    "depth8": [
     "s4",
     "s3",
     "s4"
    ]
   },
   "length7": 6.0
  },
  "min4": 68,
  "value9": "s3",
  "name6": [
   10.0
  ]
 },
 "items0": {
  "height5": 13.0,
  "type5": 3,
  "mode3": -1,
  "status1": 4,
  "temp3": 3,
  "list2": 3,
  "type2": 10,
  "width0": 6
 },
 "rank5": [
  12,
  9
 ],
 "order6": {
  "type0": "s0",
  "id3": 2,
  "label8": 8.0,
  "width8": {
   "rank7": {
    "limit6": "s7",
    "size0": 2.0,
    "id7": -5,
    "order4": {
     "key8": 13.0,
     "mode8": {
      "offset0": 9.0,
      "length6": 3,
      "limit7": 1.0,
      "width4": -1.0
     },
     "height0": {
      "mode2": {
       "rank0": -1,
       "status4": 7.0,
       "list7": "s1"
      },
      "value4": "s3",
      "total8": "s2",
      "unit5": {
       "index2": 6,
       "node8": 0.0,
       "type1": 4.0
      }
     },
     "width7": 10,
     "mode0": "s7"
    },
    "rate6": {
     "height4": [
      3.0
     ],
     "col0": [
      2
     ],
     "status8": [
      2
     ],
     "pair8": -1.0,
     "height6": 12
    },
    "mean8": 8.0
   },
   "edge0": "s8",
   "temp5": -4.0,
   "label2": 7,
   "meta4": 5.0,
   "rate8": [
    "s7",
    "s9"
   ],
   "depth4": [
    7.0,
    8.0
   ]
  },
  "mean6": {
   "score2": 5.0,
   "temp0": [
    -4.0,
    -3.0,
    -3.0
   ],
   "id2": "s8",
   "offset9": 5,
   "price6": "s0",
   "rank8": 0.0,
   "count2": {
    "value1": "s6",
    "sum6": {
     "col1": [
      -5,
      -4,
      -3
     ],
     "field1": 12.0,
     "node9": 11,
     "order2": "s3",
     "type9": 2
    },
    "pair9": 2,
    "mode7": [
     "s3"
    ],
    "row2": {
     "speed6": [
      "s5"
     ],
     "pair1": -4.0,
     "price3": 6,
     "rate0": {
      "order5": {
       "list4": 15,
       "label9": "s9",
       "unit3": 6
      },
      "f267": {
       "order3": 1.0,
       "unit9": 4.0,
       "field0": 7
      },
      "mode6": "s4",
      "list1": 6.0
     },
     "items1": "s8"
    },
    "list3": "s8"
   }
  },
  "weight1": 8,
  "min6": "s5",
  "list0": 10.0
 },
 "sum8": -2,
 "flag6": {
  "width5": 3.0,
  "speed2": "s4",
  "width3": 2.0,
  "value2": "s5",
  "f284": [
   6.0,
   8.0
  ],
  "col9": [
   "s4",
   "s1"
  ],
  "mean7": 3.0,
  "limit4": {
   "length0": 8.0,
   "depth9": {
    "order0": [
     -2.0,
     -4.0
    ],
    "status7": 11.0,
    "index1": 7,
    "status6": -3,
    "height9": [
     14.0
    ],
    "index0": [
     "s1",
     "s2"
    ]
   },
   "order9": 4.0,
   "mode1": "s3",
   "f303": {
    "id8": 1,
    "order8": 7.0,
    "flag8": 3,
    "f307": [
     "s3"
    ],
    "limit2": 4,
    "unit0": {
     "height3": {
      "speed8": 13.0,
      "row3": {
       "length2": -5.0,
       "rate5": -5.0,
       "offset3": 7.0
      },
      "temp9": 3,
      "value0": 6.0
     },
     "price7": 13,
     "f320": {
      "f321": "s5",
      "score6": -2.0,
      "pair2": 7.0,
      "offset2": [
       "s9",
       "s1"
      ]
     },
     "flag5": 3.0,
     "weight9": 6
    }
   },
   "score1": {
    "key3": [
     "s8"
    ],
    "items6": 6.0,
    "f332": {
     "label5": "s8",
     "f334": "s9",
     "f335": [
      "s7"
     ],
     "value3": 2.0,
     "unit2": "s9"
    },
    "total0": 7,
    "speed1": 7.0,
    "size7": 7.0
   },
   "index5": "s2"
  }
 }
}

Validate the document against the schema. Keys are checked in the order the schema lists them, and array elements by index. Report the dotted path of the FIRST violation (e.g. 'k0.items.1'). If the document is fully valid, answer exactly 'valid'.
```

**Answer:**

```
height7.min4
```


## Example

**Prompt:**

```
Schema (keys are validated in this order):
- total: array of:
    - sum: number in [5, 9]
- min: object {
    - size0: object {
        - mode: integer in [3, 8]
        - flag: string
        - min1: string
        - col: string
        - list: integer in [-1, 5]
        - pair: string
        - order: object {
            - field: object {
                - row: integer in [-1, 4]
                - depth: number in [2, 8]
                - status: string
                - width: array of:
                    - unit1: integer in [7, 12]
                - node9: number in [0, 5]
              }
            - price: string
            - limit0: string
            - order2: object {
                - speed: object {
                    - key6: string
                    - rate: integer in [2, 4]
                    - value3: integer in [2, 8]
                    - row2: integer in [4, 8]
                  }
                - meta3: number in [9, 14]
                - mean2: object {
                    - node: integer in [7, 8]
                    - height: object {
                        - status6: string
                        - rank: string
                        - name: string
                      }
                    - unit: integer in [6, 12]
                    - list4: string
                  }
                - mean5: array of:
                    - limit: number in [-3, -2]
                - label: number in [0, 4]
              }
            - meta: number in [-5, -4]
            - items: array of:
                - score3: string
          }
      }
    - type: object {
        - col8: number in [7, 9]
        - name4: array of:
            - size: number in [4, 10]
        - max4: array of:
            - key: number in [-2, -1]
        - temp1: array of:
            - temp4: string
        - index8: object {
            - flag7: number in [2, 4]
            - id5: array of:
                - width9: string
            - mode1: integer in [0, 1]
            - depth3: number in [-3, 1]
            - order1: integer in [-2, 4]
            - type1: string
          }
        - items1: array of:
            - label7: number in [4, 6]
        - edge3: number in [8, 13]
      }
    - temp: object {
        - row8: number in [8, 9]
        - field3: number in [0, 3]
        - type9: string
        - rank4: string
        - value: string
        - offset: integer in [2, 3]
        - type0: object {
            - index: integer in [3, 6]
            - id7: integer in [-2, 1]
            - status1: integer in [5, 11]
            - max: object {
                - mean9: number in [8, 14]
                - score: array of:
                    - flag4: number in [8, 14]
                - weight4: array of:
                    - temp6: string
                - count: number in [9, 12]
                - node1: string
              }
            - value9: integer in [-1, 4]
            - length: number in [7, 9]
          }
      }
    - id: array of:
        - node5: number in [7, 10]
    - count1: object {
        - mean: object {
            - width3: string
            - rate5: object {
                - min7: integer in [9, 11]
                - label0: string
                - weight1: object {
                    - limit2: array of:
                        - depth7: integer in [4, 8]
                    - height6: integer in [3, 4]
                    - weight: integer in [10, 13]
                    - height8: integer in [6, 9]
                  }
                - edge: integer in [-4, -2]
                - value2: string
              }
            - depth5: integer in [-1, 3]
            - value4: integer in [5, 9]
            - length7: string
            - total0: integer in [5, 10]
          }
        - type4: object {
            - depth8: integer in [9, 14]
            - row7: array of:
                - limit9: number in [9, 11]
            - flag0: object {
                - flag1: string
                - meta8: object {
                    - id1: array of:
                        - speed6: integer in [-3, 2]
                    - total3: string
                    - row1: integer in [-4, 1]
                    - field0: object {
                        - min5: integer in [6, 12]
                        - name1: number in [3, 9]
                        - id3: integer in [-1, 5]
                      }
                  }
                - status7: integer in [5, 6]
                - limit1: string
                - score1: object {
                    - field5: string
                    - rate1: string
                    - rate8: array of:
                        - key7: integer in [4, 5]
                    - index2: number in [-3, 3]
                  }
              }
            - rank6: object {
                - mean4: integer in [-2, 4]
                - col7: array of:
                    - height5: integer in [0, 6]
                - temp9: object {
                    - field2: number in [3, 8]
                    - index0: array of:
                        - speed8: string
                    - meta7: integer in [2, 6]
                    - meta4: number in [-5, 1]
                  }
                - unit5: array of:
                    - pair0: number in [7, 11]
                - mode0: number in [3, 4]
              }
            - row6: string
            - price6: object {
                - id9: array of:
                    - order0: number in [1, 7]
                - rate0: integer in [6, 12]
                - field1: number in [2, 8]
                - sum7: string
                - count7: integer in [8, 12]
              }
          }
        - unit0: integer in [-2, 0]
        - flag3: string
        - unit8: array of:
            - count0: integer in [-4, 2]
        - node2: number in [9, 13]
        - node3: object {
            - length1: number in [4, 5]
            - speed4: array of:
                - flag6: number in [7, 11]
            - rank0: object {
                - offset8: object {
                    - height3: number in [4, 9]
                    - length9: object {
                        - edge2: string
                        - total9: string
                        - items4: string
                      }
                    - max2: integer in [5, 11]
                    - max6: integer in [3, 5]
                  }
                - rate9: integer in [0, 4]
                - size5: object {
                    - sum3: integer in [-2, 4]
                    - list0: string
                    - type6: number in [1, 4]
                    - f168: string
                  }
                - flag2: string
                - min9: string
              }
            - depth6: object {
                - min8: integer in [4, 7]
                - offset0: array of:
                    - edge5: number in [10, 16]
                - status9: string
                - edge6: integer in [4, 7]
                - index6: string
              }
            - price2: object {
                - offset2: integer in [2, 7]
                - height1: string
                - count8: integer in [-5, -4]
                - node6: integer in [-3, -1]
                - col2: number in [0, 3]
              }
            - index1: object {
                - items3: integer in [0, 1]
                - speed0: object {
                    - pair5: string
                    - key3: string
                    - length8: array of:
                        - type7: number in [4, 10]
                    - key8: object {
                        - total6: string
                        - id4: integer in [5, 8]
                        - speed2: integer in [-2, 2]
                      }
                  }
                - max3: integer in [6, 8]
                - label6: number in [9, 12]
                - depth2: number in [2, 4]
              }
          }
      }
    - items8: number in [-2, 2]
    - speed5: integer in [-4, -3]
    - pair9: object {
        - order9: string
        - count2: number in [2, 3]
        - value1: string
        - row3: integer in [5, 6]
        - index5: object {
            - offset3: string
            - list1: object {
                - price3: object {
                    - price0: string
                    - value7: number in [-4, 0]
                    - order3: integer in [6, 12]
                    - depth1: string
                  }
                - edge0: string
                - pair3: number in [-2, 4]
                - flag5: object {
                    - weight5: number in [3, 8]
                    - rate4: integer in [6, 10]
                    - field6: string
                    - id8: string
                  }
                - field9: integer in [3, 8]
              }
            - col0: object {
                - width6: integer in [5, 6]
                - col3: number in [5, 6]
                - max0: integer in [-5, -1]
                - total7: integer in [8, 11]
                - field4: string
              }
            - flag9: number in [6, 10]
            - row4: string
            - offset6: array of:
                - max7: string
          }
        - min3: number in [8, 12]
        - total5: object {
            - count9: object {
                - size9: integer in [-3, 0]
                - offset9: number in [2, 4]
                - pair4: array of:
                    - order8: number in [10, 11]
                - items2: number in [-3, -1]
                - pair7: string
              }
            - temp3: object {
                - name2: number in [2, 3]
                - size8: string
                - limit4: object {
                    - min2: number in [-5, -1]
                    - count3: integer in [5, 6]
                    - sum1: array of:
                        - min4: integer in [2, 6]
                    - length2: integer in [4, 8]
                  }
                - mode7: string
                - rank1: string
              }
            - mean8: integer in [6, 7]
            - meta9: object {
                - order6: object {
                    - order7: number in [-3, 0]
                    - mean7: integer in [-4, -1]
                    - status8: integer in [7, 9]
                    - speed9: integer in [4, 7]
                  }
                - width8: integer in [3, 4]
                - unit6: string
                - name8: number in [0, 3]
                - node0: integer in [5, 10]
              }
            - node8: number in [8, 14]
            - temp5: object {
                - rank2: string
                - mode8: object {
                    - type3: object {
                        - length5: integer in [1, 7]
                        - sum2: integer in [9, 10]
                        - status0: number in [10, 16]
                      }
                    - price9: integer in [10, 12]
                    - length0: string
                    - type2: string
                  }
                - meta2: number in [10, 13]
                - list6: string
                - edge4: integer in [4, 10]
              }
          }
      }
  }
- score9: number in [2, 4]
- size2: integer in [0, 5]
- height7: number in [8, 11]
- unit9: array of:
    - weight7: integer in [8, 11]
- unit2: object {
    - limit3: number in [-5, -2]
    - score7: array of:
        - mode3: number in [-1, 0]
    - temp0: number in [3, 6]
    - mode2: object {
        - label2: integer in [2, 6]
        - height9: number in [1, 7]
        - f289: string
        - meta5: number in [0, 1]
        - weight0: integer in [1, 6]
        - price4: string
        - score4: string
      }
    - min6: array of:
        - score6: string
    - count4: number in [-5, -3]
    - weight3: number in [-5, -2]
    - key5: object {
        - height2: number in [8, 10]
        - temp8: array of:
            - pair8: number in [8, 13]
        - value6: object {
            - rate3: number in [3, 8]
            - max8: object {
                - type8: array of:
                    - list7: number in [4, 10]
                - edge8: number in [0, 1]
                - mode6: array of:
                    - label5: integer in [-1, 3]
                - max5: number in [6, 11]
                - status5: number in [0, 6]
              }
            - length6: number in [-5, -3]
            - key0: integer in [8, 9]
            - offset1: integer in [9, 13]
            - name7: string
          }
        - f316: object {
            - sum4: number in [2, 8]
            - label4: object {
                - f319: number in [4, 7]
                - sum6: array of:
                    - status4: integer in [4, 9]
                - col5: integer in [1, 2]
                - limit6: integer in [1, 5]
                - mean0: integer in [-4, -1]
              }
            - value0: object {
                - id0: string
                - rank8: string
                - offset5: integer in [9, 12]
                - f329: object {
                    - width2: integer in [0, 3]
                    - mode4: object {
                        - length4: integer in [-1, 2]
                        - min0: integer in [-4, -3]
                        - status3: number in [1, 2]
                      }
                    - temp2: number in [4, 8]
                    - size3: number in [2, 6]
                  }
                - limit5: string
              }
            - limit7: number in [-4, -2]
            - row9: string
            - f340: object {
                - f341: array of:
                    - order4: integer in [4, 9]
                - height4: number in [6, 9]
                - col6: object {
                    - edge1: integer in [10, 11]
                    - key9: string
                    - id2: string
                    - rank7: number in [3, 4]
                  }
                - count6: string
                - depth0: number in [-1, 1]
              }
          }
        - weight8: array of:
            - temp7: integer in [3, 9]
        - price1: object {
            - node7: array of:
                - f355: number in [-2, 0]
            - price5: object {
                - label8: integer in [2, 4]
                - max9: string
                - size1: integer in [2, 7]
                - weight9: integer in [7, 12]
                - edge9: string
              }
            - f362: object {
                - list3: array of:
                    - edge7: string
                - size7: integer in [7, 12]
                - f366: object {
                    - field8: integer in [6, 12]
                    - f368: array of:
                        - rate6: string
                    - offset4: integer in [2, 3]
                    - list5: string
                  }
                - meta6: number in [9, 14]
                - label9: array of:
                    - items0: number in [-2, 1]
              }
            - f375: string
            - rank9: object {
                - size6: integer in [7, 11]
                - f378: string
                - score0: number in [8, 9]
                - key4: number in [9, 11]
                - f381: integer in [0, 3]
              }
            - f382: string
          }
        - row5: number in [6, 11]
      }
  }
- col9: integer in [10, 11]
- f385: integer in [3, 8]

JSON document:
{
 "total": [
  5.0,
  6.0
 ],
 "min": {
  "size0": {
   "mode": 3,
   "flag": "s2",
   "min1": "s9",
   "col": "s0",
   "list": 1,
   "pair": "s5",
   "order": {
    "field": {
     "row": -1,
     "depth": 4.0,
     "status": "s3",
     "width": [
      11,
      10,
      11
     ],
     "node9": 4.0
    },
    "price": "s5",
    "limit0": "s2",
    "order2": {
     "speed": {
      "key6": "s2",
      "rate": 4,
      "value3": 4,
      "row2": 4
     },
     "meta3": 12.0,
     "mean2": {
      "node": 8,
      "height": {
       "status6": "s6",
       "rank": "s4",
       "name": "s1"
      },
      "unit": 6,
      "list4": "s2"
     },
     "mean5": [
      -3.0
     ],
     "label": 3.0
    },
    "meta": -4.0,
    "items": [
     "s1"
    ]
   }
  },
  "type": {
   "col8": 7.0,
   "name4": [
    10.0,
    7.0
   ],
   "max4": [
    -1.0
   ],
   "temp1": [
    "s0"
   ],
   "index8": {
    "flag7": 2.0,
    "id5": [
     "s2",
     "s8",
     "s4"
    ],
    "mode1": 0,
    "depth3": -1.0,
    "order1": 0,
    "type1": "s0"
   },
   "items1": [
    4.0,
    4.0
   ],
   "edge3": 13.0
  },
  "temp": {
   "row8": 9.0,
   "field3": 3.0,
   "type9": "s3",
   "rank4": "s3",
   "value": "s6",
   "offset": 3,
   "type0": {
    "index": 3,
    "id7": 0,
    "status1": 8,
    "max": {
     "mean9": 14.0,
     "score": [
      10.0,
      13.0
     ],
     "weight4": [
      "s7",
      "s9"
     ],
     "count": 12.0,
     "node1": "s5"
    },
    "value9": -1,
    "length": 8.0
   }
  },
  "id": [
   8.0,
   10.0,
   8.0
  ],
  "count1": {
   "mean": {
    "width3": "s9",
    "rate5": {
     "min7": 11,
     "label0": "s1",
     "weight1": {
      "limit2": [
       6,
       7,
       6
      ],
      "height6": 4,
      "weight": 10,
      "height8": 8
     },
     "edge": -4,
     "value2": "s1"
    },
    "depth5": 1,
    "value4": 9,
    "length7": "s3",
    "total0": 10
   },
   "type4": {
    "depth8": 9,
    "row7": [
     11.0,
     9.0,
     9.0
    ],
    "flag0": {
     "flag1": "s3",
     "meta8": {
      "id1": [
       2,
       2,
       0
      ],
      "total3": "s1",
      "row1": 0,
      "field0": {
       "min5": 6,
       "name1": 4.0,
       "id3": 4
      }
     },
     "status7": 6,
     "limit1": "s4",
     "score1": {
      "field5": "s4",
      "rate1": "s8",
      "rate8": [
       5,
       5,
       5
      ],
      "index2": 1.0
     }
    },
    "rank6": {
     "mean4": 1,
     "col7": [
      1,
      2
     ],
     "temp9": {
      "field2": 3.0,
      "index0": [
       "s4",
       "s4"
      ],
      "meta7": 5,
      "meta4": -4.0
     },
     "unit5": [
      8.0
     ],
     "mode0": 3.0
    },
    "row6": "s3",
    "price6": {
     "id9": [
      1.0
     ],
     "rate0": 9,
     "field1": 5.0,
     "sum7": "s7",
     "count7": 12
    }
   },
   "unit0": -1,
   "flag3": "s5",
   "unit8": [
    0,
    -4,
    0
   ],
   "node2": 13.0,
   "node3": {
    "length1": 4.0,
    "speed4": [
     8.0
    ],
    "rank0": {
     "offset8": {
      "height3": 7.0,
      "length9": {
       "edge2": "s4",
       "total9": "s5",
       "items4": "s8"
      },
      "max2": 6,
      "max6": 4
     },
     "rate9": 1,
     "size5": {
      "sum3": 4,
      "list0": "s2",
      "type6": 4.0,
      "f168": "s0"
     },
     "flag2": "s1",
     "min9": "s3"
    },
    "depth6": {
     "min8": 6,
     "offset0": [
      11.0
     ],
     "status9": "s4",
     "edge6": 7,
     "index6": "s8"
    },
    "price2": {
     "offset2": 5,
     "height1": "s3",
     "count8": -4,
     "node6": -2,
     "col2": 0.0
    },
    "index1": {
     "items3": 0,
     "speed0": {
      "pair5": "s7",
      "key3": "s0",
      "length8": [
       7.0,
       7.0,
       6.0
      ],
      "key8": {
       "total6": "s1",
       "id4": 7,
       "speed2": 2
      }
     },
     "max3": 6,
     "label6": 11.0,
     "depth2": 3.0
    }
   }
  },
  "items8": -2.0,
  "speed5": -4,
  "pair9": {
   "order9": "s4",
   "count2": 3.0,
   "value1": "s6",
   "row3": 5,
   "index5": {
    "offset3": "s3",
    "list1": {
     "price3": {
      "price0": "s3",
      "value7": -2.0,
      "order3": 8,
      "depth1": "s5"
     },
     "edge0": "s3",
     "pair3": 2.0,
     "flag5": {
      "weight5": 8.0,
      "rate4": 10,
      "field6": "s5",
      "id8": "s4"
     },
     "field9": 7
    },
    "col0": {
     "width6": 5,
     "col3": 6.0,
     "max0": -3,
     "total7": 9,
     "field4": "s8"
    },
    "flag9": 7.0,
    "row4": "s6",
    "offset6": [
     "s4",
     "s8",
     "s1"
    ]
   },
   "min3": 11.0,
   "total5": {
    "count9": {
     "size9": -2,
     "offset9": 4.0,
     "pair4": [
      10.0
     ],
     "items2": -2.0,
     "pair7": "s7"
    },
    "temp3": {
     "name2": 3.0,
     "size8": "s1",
     "limit4": {
      "min2": -3.0,
      "count3": 5,
      "sum1": [
       6
      ],
      "length2": 5
     },
     "mode7": "s0",
     "rank1": "s3"
    },
    "mean8": 7,
    "meta9": {
     "order6": {
      "order7": -3.0,
      "mean7": -4,
      "status8": 7,
      "speed9": 4
     },
     "width8": 4,
     "unit6": "s4",
     "name8": 2.0,
     "node0": 7
    },
    "node8": 11.0,
    "temp5": {
     "rank2": "s0",
     "mode8": {
      "type3": {
       "length5": 3,
       "sum2": 9,
       "status0": 12.0
      },
      "price9": 12,
      "length0": "s3",
      "type2": "s9"
     },
     "meta2": 10.0,
     "list6": "s0",
     "edge4": 6
    }
   }
  }
 },
 "score9": 4.0,
 "size2": 5,
 "height7": 8.0,
 "unit9": [
  11,
  8
 ],
 "unit2": {
  "limit3": -5.0,
  "score7": [
   0.0
  ],
  "temp0": 6.0,
  "mode2": {
   "label2": 6,
   "height9": 7.0,
   "f289": "s1",
   "meta5": 1.0,
   "weight0": 1,
   "price4": "s3",
   "score4": "s2"
  },
  "min6": [
   "s1"
  ],
  "count4": -5.0,
  "weight3": -3.0,
  "key5": {
   "height2": 8.0,
   "temp8": [
    8.0,
    12.0,
    8.0
   ],
   "value6": {
    "rate3": 3.0,
    "max8": {
     "type8": [
      4.0,
      5.0,
      9.0
     ],
     "edge8": 1.0,
     "mode6": [
      0,
      2,
      1
     ],
     "max5": 7.0,
     "status5": 2.0
    },
    "length6": -3.0,
    "key0": 9,
    "offset1": 10,
    "name7": "s5"
   },
   "f316": {
    "sum4": 3.0,
    "label4": {
     "f319": 4.0,
     "sum6": [
      9,
      7
     ],
     "col5": 1,
     "limit6": 4,
     "mean0": -4
    },
    "value0": {
     "id0": "s4",
     "rank8": "s7",
     "offset5": 9,
     "f329": {
      "width2": 3,
      "mode4": {
       "length4": 2,
       "min0": -4,
       "status3": 1.0
      },
      "temp2": 6.0,
      "size3": 6.0
     },
     "limit5": "s4"
    },
    "limit7": -2.0,
    "row9": "s1",
    "f340": {
     "f341": [
      7
     ],
     "height4": 9.0,
     "col6": {
      "edge1": 11,
      "key9": "s3",
      "id2": "s7",
      "rank7": 4.0
     },
     "count6": "s7",
     "depth0": -1.0
    }
   },
   "weight8": [
    3,
    3,
    9
   ],
   "price1": {
    "node7": [
     0.0,
     -2.0,
     -2.0
    ],
    "price5": {
     "label8": 4,
     "max9": "s0",
     "size1": 2,
     "weight9": 12,
     "edge9": "s9"
    },
    "f362": {
     "list3": [
      "s7",
      "s9",
      "s9"
     ],
     "size7": 12,
     "f366": {
      "field8": 10,
      "f368": [
       "s6",
       "s6",
       "s2"
      ],
      "offset4": 2,
      "list5": "s2"
     },
     "meta6": 12.0,
     "label9": [
      -2.0
     ]
    },
    "f375": "s4",
    "rank9": {
     "size6": 7,
     "f378": "s5",
     "score0": 9.0,
     "key4": 11.0,
     "f381": 1
    },
    "f382": "s0"
   },
   "row5": 9.0
  }
 },
 "col9": 11,
 "f385": 6
}

Validate the document against the schema. Keys are checked in the order the schema lists them, and array elements by index. Report the dotted path of the FIRST violation (e.g. 'k0.items.1'). If the document is fully valid, answer exactly 'valid'.
```

**Answer:**

```
valid
```

