import sqlite3
conn = sqlite3.connect("functions.db")
cur = conn.cursor()

cur.execute("""
    SELECT outputs, COUNT(*) as cnt 
    FROM functions 
    GROUP BY outputs 
    ORDER BY cnt DESC 
    LIMIT 20
""")


# cur.execute("""
#     SELECT inputs, COUNT(*) as cnt
#     FROM functions
#     GROUP BY inputs
#     ORDER BY cnt DESC
#     LIMIT 30
# """)
# conn.commit()


for row in cur.fetchall():
    print(row)
conn.close()