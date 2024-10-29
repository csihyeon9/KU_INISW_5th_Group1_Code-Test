import requests
from bs4 import BeautifulSoup
import os  # Import os module to check for file existence

# Base URL for pagination
base_url = "https://www.fsc.go.kr/in090301?curPage="

# Define the XML file path
xml_path = 'scraped_titles_and_allow_directives.xml'

# Check if the file exists and remove it if it does
if os.path.exists(xml_path):
    os.remove(xml_path)

# Create a list to hold the terms
terms = []

# Loop through pages 1 to 23
for page in range(1, 24):
    response = requests.get(base_url + str(page))
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract financial terms
    page_terms = [a.text.strip() for a in soup.select('ul li .subject a')]
    terms.extend(page_terms)  # Append terms from the current page to the list

# Function to escape XML special characters
def escape_xml(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("'", "&apos;").replace('"', "&quot;")

# Write terms to the XML file with a root element
with open(xml_path, 'w', encoding='utf-8') as xml_file:
    xml_file.write('<?xml version="1.0" encoding="utf-8"?>\n')  # Write the XML declaration
    xml_file.write('<terms>\n')  # Start the root element
    for term in terms:
        escaped_term = escape_xml(term)  # Escape special characters
        xml_file.write(f'  <term>{escaped_term}</term>\n')  # Write each term with <term> tags
    xml_file.write('</terms>\n')  # End the root element

# Read the XML file to filter out unwanted entries
with open(xml_path, 'r', encoding='utf-8') as xml_file:
    lines = xml_file.readlines()

# Filter out unwanted terms (e.g., 'a1' and 'a2')
filtered_lines = [line for line in lines if '<term>' not in line or ('a1' not in line and 'a2' not in line)]

# Write back the filtered content to the XML file
with open(xml_path, 'w', encoding='utf-8') as xml_file:
    xml_file.writelines(filtered_lines)

print(f"Scraped data has been saved to {xml_path}")