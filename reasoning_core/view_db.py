import sqlite3

conn = sqlite3.connect("functions.db")
cur = conn.cursor()

# 1. Remove test modules ONLY
cur.execute("""
DELETE FROM functions
WHERE module LIKE '%.tests.%'
   OR module LIKE '%.test_%'
   OR module LIKE '%conftest'
""")

# 2. Remove externals ONLY
cur.execute("""
DELETE FROM functions
WHERE module LIKE '%externals%'
""")

cur.execute("""
DELETE FROM functions
WHERE id NOT IN (
    SELECT MIN(id)
    FROM functions
    GROUP BY function_name, inputs, outputs, module
)
""")
print(f"Deleted {cur.rowcount} duplicate rows")


conn.commit()

# View the first 10 rows
cur.execute("SELECT * FROM functions;")
rows = cur.fetchall()

for row in rows:
    print(row)

conn.close()