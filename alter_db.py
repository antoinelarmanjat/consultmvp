import sqlite3

# Connect to the database
conn = sqlite3.connect('patients.db')
cursor = conn.cursor()

# Create a temporary table with the new schema
cursor.execute('''
CREATE TABLE patients_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    date_of_birth DATE,
    weight REAL,
    allergies TEXT,
    smokes BOOLEAN,
    medications TEXT,
    last_visit_reason TEXT,
    last_visit_date DATE
)
''')

# Copy data from the old table to the new one
cursor.execute('''
INSERT INTO patients_new 
SELECT * FROM patients
''')

# Drop the old table
cursor.execute('DROP TABLE patients')

# Rename the new table to the original name
cursor.execute('ALTER TABLE patients_new RENAME TO patients')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database schema updated successfully!") 