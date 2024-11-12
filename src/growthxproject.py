
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
import os
from tqdm import tqdm


def read_company_site(url):

    """
    This function reads the content of a company's website and extracts the title, meta description, headings, and paragraphs.
    Args:
        url (str): The URL of the company's website.
    Returns:
        dict: A dictionary containing the extracted data.
    """
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract various parts of the webpage
    title = soup.find('title').get_text(strip=True) if soup.find('title') else "No title found"
    meta_description = soup.find('meta', attrs={'name': 'description'})
    meta_description = meta_description['content'].strip() if meta_description else "No description found"
    headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
    paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]

    # Combine extracted parts into a dictionary
    extracted_data = {
        "title": title,
        "meta_description": meta_description,
        "headings": headings,
        "paragraphs": paragraphs
    }
    return extracted_data

def classifier_model(classifier):
    """
    This function loads the summarization and classification models.
    Args:
        classifier (str): The name of the zero-shot classification model to load.
    Returns:
        the loaded classification model.
    """
     # Load the classifier model
    classifier = pipeline("zero-shot-classification", model=classifier, device=-1)

    return classifier

def summarize_model(summirizer):
    """
    This function loads the summarization and classification models.
    Args:
        summirizer (str): The name of the summarization model to load.
    Returns:
        tuple: A tuple containing the loaded summarization model, tokenizer and device.
    """
    # Set the device to GPU if available
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the summarization model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(summirizer).to(device)
    tokenizer = AutoTokenizer.from_pretrained(summirizer)


    return model, tokenizer, device

def classify_text(Text, classifier):
    #docstring
    """
    This function classifies the useful parts of the text data.
    Args:
        Text (list): A list of paragraphs to classify.
        classifier (pipeline): The zero-shot classification pipeline.
    Returns:
        dict: A dictionary containing the classification results.
    """
    labels = ["useful", "not useful"]
    if not Text:
        raise ValueError("The paragraphs list is empty. Please provide text data to classify.")

    # Predict useful parts
    useful_paragraphs = []
    for paragraph in Text:
        if paragraph.strip():  # Ensure the paragraph is not empty or just whitespace
            result = classifier(paragraph, candidate_labels=labels)
            if result['labels'][0] == "useful":
                useful_paragraphs.append(paragraph)
        else:
            print("Skipping empty paragraph.")
    return useful_paragraphs

class EndSequenceStoppingCriteria(StoppingCriteria):
    """
    This class defines a stopping criteria that checks if the generated text ends with a specific sequence.
    Args:
        stop_sequence (str): The sequence that the generated text should end with.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to decode the generated text.
    """

    def __init__(self, stop_sequence, tokenizer):
        self.stop_sequence = stop_sequence
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the generated tokens to text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # Check if the stop sequence is in the generated text and the JSON structure is completed
        if self.stop_sequence in generated_text and generated_text.endswith(self.stop_sequence):
            return True
        return False
    
def prep_prompt(input_text, stop_sequence, device, tokenizer):
    #docstring
    """
    This function prepares the input text and stopping criteria for the model.
    Args:
        input_text (str): The input text to generate from.
        stop_sequence (str): The sequence that the generated text should end with.
        device (torch.device): The device to run the model on.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to encode the input text.
    Returns:
        tuple: A tuple containing the inputs and stopping criteria.
    """

    # Encode the stop sequence
    stopping_criteria = StoppingCriteriaList([EndSequenceStoppingCriteria(stop_sequence, tokenizer)])

    # Set pad_token_id to eos_token_id if not already set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    # Move inputs to the appropriate device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return inputs, stopping_criteria

def generate_text(model, inputs, stopping_criteria, tokenizer):
    """
    This function generates text using the model and input text.
    Args:
        model (transformers.PreTrainedModel): The model to generate text with.
        inputs (dict): The input text encoded as input_ids and attention_mask.
        stopping_criteria (StoppingCriteria): The stopping criteria to use during generation.
    Returns:
        str: The generated text.
    """
    # Generate output with attention mask
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=1500,
        pad_token_id=tokenizer.pad_token_id,
        top_k=50, 
        top_p=0.95, 
        temperature=0.3,
        stopping_criteria=stopping_criteria 
        
    )
    return outputs

def summary(outputs, start_marker, end_marker, tokenizer):
    """
    This function extracts the JSON part from the generated text.
    Args:
        outputs (torch.Tensor): The generated text output.
        start_marker (str): The start marker for the JSON part.
        end_marker (str): The end marker for the JSON part.
    Returns:
        str: The extracted JSON part.
    """
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Find the second occurrence of the start marker
    first_start_index = generated_text.find(start_marker)
    second_start_index = generated_text.find(start_marker, first_start_index + len(start_marker))

    # Extract the JSON part from the generated text
    start_index = second_start_index + len(start_marker)
    end_index = generated_text.find(end_marker, start_index)
    json_output = generated_text[start_index:end_index].strip()
    return json_output

def extract_company_info(url, classifier, model, tokenizer, device):
    """
    This function takes a company URL and extracts relevant information using web scraping and LLM-based text generation.
    
    Args:
        url (str): The URL of the company's website.
    
    Returns:
        str: The JSON representation of the extracted company information.
    """
    try:
        stages = ["Reading website", "Classifying text", "Preparing prompt", "Generating summary", "Extracting JSON"]
        progress_bar = tqdm(stages, desc="Processing stages")

        progress_bar.set_description("Reading website")
        # Read the company's website
        extracted_data = read_company_site(url)
        progress_bar.update(1)

        progress_bar.set_description("Classifying text")
        # Classify the useful parts of the text
        useful_paragraphs = classify_text(extracted_data["paragraphs"], classifier)
        progress_bar.update(1)

        progress_bar.set_description("Preparing prompt")
        # creating the prompt
        input_text = f"""
        You are an assistant that provides concise company information. For each company, you will determine some attributes based on the provided information.
        In this regard, you will follow these rules:
        1. You will strictly follow the instructions described here.
        2. You will determine the attributes as accurately as possible based on the provided information.
        3. You will provide the response for these attributes strictly in JSON format, in a single structure, considering the fields \attribute\ and \value\. Do not include the question, just respond with attribute and value.
        4. The attributes you should respond to are:
        4.1 Company Name (attribute \"Company Name"\)
        4.2 Website (attribute \"Website"\)
        4.3 Target Audience (attribute \"Target Audience"\)
        4.4 Industry (attribute \"Industry"\)
        4.5 Type of Business (attribute \"Type of Business"\)
        4.6 Size of the Company (attribute Number of employees: Less than 1 - \"Not Found"\, more than 250 - \"Large"\, 51 to 250 - \"Medium"\, 11 to 50 - \"Small"\, 1 - 10 - \"Very Small"\ )
        4.7 Location (attribute \"Location"\)
        4.8 Contact Information (attribute \"E-mail"\)
        4.9 Summary (attribute \"Summary"\)
        5. If you cannot determine an attribute, or it has low probability, you will respond with the attribute and the value \"Not Found"\.
        6. The first field should be Attribute: \"Company Name\", value: the value will be provided along with the information.
        7. Example:
            7.1 Before the JSON format you will respond with ###START###
            7.2 The JSON format will be as follows:
        {{
            "Company Name": "Company Name",
            "Website": "Company URL",
            "Target Audience": "Target Audience",
            "Industry": "Industry",
            "Type of Business": "Type of Business",
            "Size of the Company": "Size of the Company",
            "Location": "Location",
            "Contact Information": "Contact Information",
            "Summary": "Summary"
        }}
            7.3 After generating the JSON, you will respond with ###END###
        8. Company information:
            8.1 Website title - {extracted_data["title"]}, 
            8.2 website url - {url} 
            8.3 Usefull text - {useful_paragraphs}
        9. Provide the requested information for the company.
        """

        # Defining a stopping criteria
        stop_sequence = "###END###"

        # preparing the prompt to model
        inputs, stopping_criteria = prep_prompt(input_text, stop_sequence, device, tokenizer)
        progress_bar.update(1)

        progress_bar.set_description("Generating summary")
        # summary of the company
        outputs = generate_text(model, inputs, stopping_criteria, tokenizer)
        progress_bar.update(1)

        progress_bar.set_description("Extracting JSON")
        # Extract the JSON part from the generated text
        start_marker = "###START###"
        end_marker = "###END###"
        json_output = summary(outputs, start_marker, end_marker, tokenizer)

        progress_bar.update(1)
    except Exception as e:
        json_output = {
            "Company Name": "Not Found",
            "Website": "Company URL",
            "Target Audience": "Not Found",
            "Industry": "Not Found",
            "Type of Business": "Not Found",
            "Size of the Company": "Not Found",
            "Location": "Not Found",
            "Contact Information": "Not Found",
            "Summary": "Could not extract information due to an error." + str(e)
        }
        progress_bar.close()
    return json_output

