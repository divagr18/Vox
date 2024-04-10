from googlesearch import search
from bs4 import BeautifulSoup
import requests

def get_service_website(service_name):
    query = f"{service_name} official website"
    try:
        # Perform a Google search and return the URL of the most relevant non-advertisement link
        for url in search(query, num=5, stop=5):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            ads = soup.find_all('div', class_='uEierd')
            if not ads:
                return url
    except Exception as e:
        print("An error occurred:", e)
        return None

service_name = input("Enter the name of the service: ")
website = get_service_website(service_name)
if website:
    print(f"The official website of {service_name} is: {website}")
else:
    print("Unable to retrieve the website.")
