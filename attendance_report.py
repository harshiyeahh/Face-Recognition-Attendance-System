import sqlite3
import pandas as pd

conn = sqlite3.connect('attendance.db')

df = pd.read_sql_query(
    "SELECT * FROM attendance",
    conn
)

print(df)

df.to_csv(
    "attendance_report.csv",
    index=False
)

print("Report Generated Successfully")