import pandas as pd
import numpy as np
import datetime
import random
import os

def generate_synthetic_data(num_records=10000, output_path='data/synthetic_antibiotic_data.csv'):
    np.random.seed(42)
    random.seed(42)

    # Base categories
    genders = ['Male', 'Female', 'Other']
    regions = ['Region_Alpha', 'Region_Beta', 'Region_Gamma', 'Region_Delta', 'Region_Epsilon']
    hospital_ids = [f"HOSP_{i:03d}" for i in range(1, 21)]
    
    infection_types = ['UTI', 'Pneumonia', 'Skin Infection', 'Bloodstream', 'Respiratory Tract']
    bacterial_strains = ['E. coli', 'S. aureus', 'K. pneumoniae', 'P. aeruginosa', 'S. pneumoniae', 'MRSA']
    antibiotics = ['Amoxicillin', 'Ciprofloxacin', 'Azithromycin', 'Meropenem', 'Cephalexin', 'Vancomycin']

    # Generate dates over a 3 year period
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2023-12-31')
    days_between = (end_date - start_date).days

    records = []
    
    for i in range(num_records):
        patient_id = f"PAT_{i:06d}"
        age = int(np.random.normal(loc=45, scale=18))
        age = max(1, min(age, 95))
        gender = random.choice(genders)
        region = random.choice(regions)
        hospital_id = random.choice(hospital_ids)
        
        infection = random.choice(infection_types)
        
        # Correlate infection with bacterial strain
        if infection == 'UTI':
            strain = random.choices(['E. coli', 'K. pneumoniae', 'P. aeruginosa'], weights=[0.7, 0.2, 0.1])[0]
        elif infection == 'Pneumonia':
            strain = random.choices(['S. pneumoniae', 'S. aureus', 'K. pneumoniae'], weights=[0.6, 0.2, 0.2])[0]
        elif infection == 'Skin Infection':
            strain = random.choices(['S. aureus', 'MRSA', 'P. aeruginosa'], weights=[0.6, 0.3, 0.1])[0]
        else:
            strain = random.choice(bacterial_strains)
            
        # Select Antibiotic based on strain (some realistic correlations)
        if strain == 'MRSA':
            antibiotic = random.choices(['Vancomycin', 'Meropenem', 'Ciprofloxacin'], weights=[0.8, 0.1, 0.1])[0]
        elif strain == 'E. coli':
            antibiotic = random.choices(['Ciprofloxacin', 'Amoxicillin', 'Cephalexin'], weights=[0.5, 0.4, 0.1])[0]
        else:
            antibiotic = random.choice(antibiotics)
            
        dosage_mg = random.choice([250, 500, 750, 1000])
        duration_days = random.choices([3, 5, 7, 10, 14], weights=[0.1, 0.3, 0.4, 0.15, 0.05])[0]
        
        random_days = random.randint(0, days_between)
        treatment_date = start_date + datetime.timedelta(days=random_days)
        
        # Calculate Resistance Status based on rules
        # Base probability
        res_prob = 0.1
        
        # Time factor: Resistance increases over time (from 0 to 0.15 extra prob)
        time_factor = (random_days / days_between) * 0.15
        res_prob += time_factor
        
        # Antibiotic specific resistance
        if antibiotic == 'Amoxicillin' and strain == 'E. coli':
            res_prob += 0.2 # High resistance
        if strain == 'MRSA':
            if antibiotic != 'Vancomycin':
                res_prob += 0.6 # Extremely likely resistant
        
        # Region specific (Region_Delta might have poor stewardship)
        if region == 'Region_Delta':
            res_prob += 0.1
            
        # Age factor (Older patients might have more resistant strains)
        if age > 65:
            res_prob += 0.05

        # Decision
        is_resistant = 'Resistant' if random.random() < res_prob else 'Sensitive'
        
        # Treatment outcome depends heavily on resistance status
        if is_resistant == 'Resistant':
            outcome_prob = 0.2 # Low chance of recovery if resistant
        else:
            outcome_prob = 0.85 # High chance of recovery if sensitive
            
        outcome = 'Recovered' if random.random() < outcome_prob else 'Not Recovered'
        
        records.append({
            'Patient_ID': patient_id,
            'Age': age,
            'Gender': gender,
            'Region': region,
            'Hospital_ID': hospital_id,
            'Infection_Type': infection,
            'Bacterial_Strain': strain,
            'Antibiotic': antibiotic,
            'Dosage_mg': dosage_mg,
            'Duration_days': duration_days,
            'Treatment_Date': treatment_date,
            'Resistance_Status': is_resistant,
            'Treatment_Outcome': outcome
        })

    df = pd.DataFrame(records)
    
    # Sort by date
    df = df.sort_values(by='Treatment_Date').reset_index(drop=True)
    
    # Introduce ~5% missing values randomly in Age and Dosage
    np.random.seed(1)
    df.loc[df.sample(frac=0.05).index, 'Age'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'Dosage_mg'] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset with {num_records} records generated and saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
