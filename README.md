# LLM-Webscrapper

# GrowthX Company Information Extractor

This project utilizes web scraping and large language models (LLMs) to extract key information about a company from its website.

## Functionality

The `growthxproject.py` script performs the following:

1. **Web Scraping:** Reads the content of a company's website using the `requests` and `BeautifulSoup` libraries. 
   - Extracts the title, meta description, headings, and paragraphs from the HTML.

2. **Text Classification:** Uses the `transformers` library and a zero-shot classification model to identify "useful" paragraphs that likely contain relevant information.
   - This step filters out less relevant content.

3. **Text Generation:** Employs a text generation LLM (e.g., Llama-3.2-1B-Instruct) to generate a concise summary of the company's information.
   - The LLM takes the classified text as input and uses a carefully crafted prompt to extract specific information.

4. **JSON Output:** Formats the extracted information as a JSON object, including:
   - Company Name
   - Website
   - Target Audience
   - Industry
   - Type of Business
   - Size of the Company
   - Location
   - Contact Information
   - Summary 

## Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

2. **Run the script:**
    Create a file named `urls.txt` with each company URL on a separate line.
     ```bash
     python growthxproject.py 
     ``` 

## How it Works

1. **`growthxproject.py`**: Contains the core code for scraping, classifying, generating, and formatting company information.
2. **`company_summary.py`**: A separate script that demonstrates how to call the `extract_company_info` function and provides different ways to input the company URL (direct input, command line argument, file reading). 

## Requirements

- Python 3.7 or higher
- `requests`
- `BeautifulSoup4`
- `transformers`
- `torch` 

## Example Output (JSON)

```json
{
    "Company Name": "GrowthX Labs",
    "Website": "https://growthxlabs.com/",
    "Target Audience": "Businesses and teams looking to scale their AI-powered growth strategies",
    "Industry": "Technology",
    "Type of Business": "AI-Powered Growth Agency",
    "Size of the Company": "Very Small",
    "Location": "New York, NY",
    "Contact Information": "info@growthxlabs.com",
    "Summary": "We help teams build end-to-end AI-powered, human-guided automated content workflows that actually drive growth."
}
```

## Notes

- The accuracy of the extracted information depends on the quality of the website content and the capabilities of the LLM.
- The LLM may not always be able to extract all requested information, especially if the website content is ambiguous or incomplete.
- The project relies on the `transformers` library and may require updating the models for improved performance.


```

