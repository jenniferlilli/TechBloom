import sqlite3

def view_data():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    print("Badge IDs:")
    cursor.execute("SELECT session_id, badge_id FROM badge_ids")
    for row in cursor.fetchall():
        print("Session ID: " + row[0] + ", Badge ID: " + row[1])

    print("\nUploaded ZIPs:")
    cursor.execute("SELECT session_id, filename FROM uploaded_zips")
    for row in cursor.fetchall():
        print("Session ID: " + row[0] + ", Filename: " + row[1])

    print("\nSession Info:")
    cursor.execute("SELECT session_id, password FROM user_sessions")
    for row in cursor.fetchall():
        print("Session ID: " + row[0] + ", Password: " + row[1])

    conn.close()

if __name__ == "__main__":
    view_data()
