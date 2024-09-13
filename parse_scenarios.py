# import re
# import pandas as pd

# def parse_restrictions(file_path):
#     with open(file_path, 'r') as file:
#         content = file.read()

#     # Split content into sections by Name
#     sections = re.split(r'(?<=\})\s*\n\s*\n(?=[A-Za-z])', content)

#     restrictions = []

#     for section in sections:
#         lines = section.strip().split('\n')
#         name = lines[0].strip()
#         numerical_restrictions = {}
#         categorical_restrictions = {}

#         for line in lines[3:]:  # Skip first 3 lines: Name, box, and header
#             parts = re.split(r'\s+', line.strip())
#             key = parts[0]
#             min_value = parts[1]
#             max_value = parts[2]
#             categories = re.findall(r'\{([^}]+)\}', parts[1])  # Find categories in {}

#             if categories:
#                 categorical_restrictions[key] = categories
#             else:
#                 min_value = float(min_value) if min_value != 'NaN' else None
#                 max_value = float(max_value) if max_value != 'NaN' else None
#                 numerical_restrictions[key] = (min_value, max_value)

#         restrictions.append({
#             'name': name,
#             'numerical': numerical_restrictions,
#             'categorical': categorical_restrictions
#         })

#     return restrictions

# def create_tables(restrictions):
#     table_list = []

#     for restriction in restrictions:
#         name = restriction['name']
#         numerical = restriction['numerical']
#         categorical = restriction['categorical']

#         rows = []
#         for key, (min_value, max_value) in numerical.items():
#             rows.append([name, key, min_value, max_value, ''])

#         for key, categories in categorical.items():
#             rows.append([name, key, '', '', ', '.join(categories)])

#         df = pd.DataFrame(rows, columns=['Scenario', 'Restriction', 'Min Value', 'Max Value', 'Categories'])
#         table_list.append(df)

#     return table_list

# def save_tables(tables, output_path):
#     with pd.ExcelWriter(output_path) as writer:
#         for idx, table in enumerate(tables):
#             table.to_excel(writer, sheet_name=f'Scenario_{idx+1}', index=False)

# if __name__ == "__main__":
#     file_path = 'scenarios.txt'  # Input file path
#     output_path = 'scenarios_parsed.xlsx'  # Output file path

#     restrictions = parse_restrictions(file_path)
#     tables = create_tables(restrictions)
#     save_tables(tables, output_path)
