import sqlite3

conn = sqlite3.connect('data.db')
cursor = conn.cursor()

cursor.execute('DELETE FROM badge_ids')
cursor.execute('DELETE FROM uploaded_images')

conn.commit()
conn.close()

