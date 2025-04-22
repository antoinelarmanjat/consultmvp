import sqlite3
from datetime import datetime, timedelta
import random

# Connect to the database
conn = sqlite3.connect('patients.db')
cursor = conn.cursor()

# Sample data
first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emma', 'James', 'Olivia', 'William', 'Sophia']
last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
allergies = ['Penicillin', 'Peanuts', 'Shellfish', 'Latex', 'Aspirin', 'Pollen', 'Dust', 'Cat dander', 'Eggs', 'Soy']
medications = ['Lisinopril', 'Metformin', 'Atorvastatin', 'Levothyroxine', 'Amlodipine', 'Metoprolol', 'Omeprazole', 'Albuterol', 'Gabapentin', 'Hydrochlorothiazide']
visit_reasons = ['Annual checkup', 'Flu symptoms', 'Back pain', 'Headache', 'Fever', 'Cough', 'Sore throat', 'Joint pain', 'Rash', 'Fatigue']

# Function to generate a random date between 1950 and 2000
def random_date():
    start_date = datetime(1950, 1, 1)
    end_date = datetime(2000, 12, 31)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + timedelta(days=random_number_of_days)

# Function to generate a random date in the last 2 years
def random_recent_date():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years ago
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + timedelta(days=random_number_of_days)

# Generate 10 patients
for i in range(10):
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    date_of_birth = random_date()
    weight = random.uniform(50, 120)  # kg
    allergy = random.choice(allergies)
    smokes = random.choice([True, False])
    medication = random.choice(medications)
    visit_reason = random.choice(visit_reasons)
    last_visit_date = random_recent_date()

    # Randomly choose one field to be NULL
    null_field = random.choice(['weight', 'allergies', 'smokes', 'date_of_birth'])
    
    # Prepare the data with the chosen field as NULL
    data = {
        'first_name': first_name,
        'last_name': last_name,
        'date_of_birth': date_of_birth.strftime('%Y-%m-%d'),
        'weight': weight,
        'allergies': allergy,
        'smokes': smokes,
        'medications': medication,
        'last_visit_reason': visit_reason,
        'last_visit_date': last_visit_date.strftime('%Y-%m-%d')
    }
    
    # Set the chosen field to NULL
    data[null_field] = None

    # Insert the patient
    cursor.execute('''
        INSERT INTO patients (
            first_name, last_name, date_of_birth, weight, 
            allergies, smokes, medications, last_visit_reason, last_visit_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['first_name'], data['last_name'], data['date_of_birth'],
        data['weight'], data['allergies'], data['smokes'],
        data['medications'], data['last_visit_reason'], data['last_visit_date']
    ))

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database populated successfully!") 