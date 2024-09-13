import re
import pandas as pd

# Function to parse the text file
def parse_scenarios(txt_file):
    data = []
    scenario = None
    
    # Open and read the txt file
    with open(txt_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip()

            # Detect new Scenario (e.g., Name1, Name2, etc.)
            if re.match(r'^CHP\d+', line) or re.match(r'^WASTE\d+', line) or re.match(r'^PULP\d+', line):
                scenario = line
                continue

            # Detect and parse restriction details
            match = re.match(r'(\S+)\s+({.*}|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+({.*}|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', line)
            if match:
                restriction = match.group(1)
                min_value = match.group(2) if re.match(r'^[-+]?[0-9]*\.?[0-9]+', match.group(2)) else ''
                max_value = match.group(3) if re.match(r'^[-+]?[0-9]*\.?[0-9]+', match.group(3)) else ''
                categories = match.group(2) if min_value == '' else match.group(3) if max_value == '' else ''
                
                # Append the row to the data
                data.append([scenario, restriction, min_value, max_value, categories])
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Scenario', 'Restriction', 'Min Value', 'Max Value', 'Categories'])
    return df

# Read the scenarios from the txt file and export to Excel
def save_to_excel(txt_file, excel_file):
    df = parse_scenarios(txt_file)
    df.to_excel(excel_file, index=False)
    print(f'Saved parsed data to {excel_file}')

# Run the function
txt_file = 'scenarios.txt'  # Update this path to your text file
excel_file = 'parsed_scenarios.xlsx'  # Output Excel file

save_to_excel(txt_file, excel_file)
