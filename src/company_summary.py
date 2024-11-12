import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings 
# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, message="expandable_segments not supported on this platform")

import tensorflow as tf
from growthxproject import extract_company_info, classifier_model, summarize_model
import pandas as pd
import json
import re

tf.get_logger().setLevel('ERROR')

def main():
  with open("urls.txt", "r") as f:
    company_urls = f.readlines()
      # Load the classifier model
  classifier = classifier_model("facebook/bart-large-mnli")

  # Load the summarization model in a separate cell to avoid kernel crash
  model, tokenizer, device = summarize_model("meta-llama/Llama-3.2-1B-Instruct")

  # Initialize an empty DataFrame
  columns = ["Company Name", "Website", "Target Audience", "Industry", "Type of Business", "Size of the Company", "Location", "Contact Information", "Summary"]
  df = pd.DataFrame(columns=columns)
  
  for company_url in company_urls:
    print(company_url)
    company_url = company_url.strip() # remove potential whitespace
    company_info = extract_company_info(company_url, classifier, model, tokenizer, device)
    print(company_info)
    
    # Check if company_info is a dictionary
    if isinstance(company_info, dict):
        # Convert the dictionary to a JSON string
        json_output_dict = company_info
    else:
        
        json_match = re.search(r'\{.*\}', company_info, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            # Clean the JSON string
            json_str = re.sub(r',\s*}', '}', json_str)
            # Debugging: Print the cleaned JSON string
            print("Cleaned JSON string:", json_str)
        else:
            print("No JSON object found in company_info")
            continue 

        # Convert JSON to DataFrame and append to the main DataFrame
        try:
            json_output_dict = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            continue
    
    # Convert the dictionary to a DataFrame
    json_output_df = pd.DataFrame([json_output_dict])

    # Concatenate the DataFrame
    df = pd.concat([df, json_output_df], ignore_index=True)
    print(company_info)

  # Ensure the file is not open elsewhere
  try:
      with open("company_info.csv", "w") as f:
          pass
  except PermissionError:
      print("File is open elsewhere. Please close it and try again.")
      return

  # Save the DataFrame to a CSV file
  df.to_csv("company_info.csv", index=False)

if __name__ == "__main__":
  main()