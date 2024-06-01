# set environment variables
# https://platform.openai.com/account/api-keys
import os
os.environ["OPENAI_API_KEY"] = "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"
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
# from langchain_community.chat_models import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, ConfigDict, ValidationError
from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator, OPENAI_TEMPLATE
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_SUFFIX, SYNTHETIC_FEW_SHOT_PREFIX



# def read_variable_names_from_file(file_path):
#     # Read the first row of the Excel file to get variable names
#     df = pd.read_excel(file_path, nrows=1)  # Read only the first row from Excel file
#     return df.columns.tolist()  # Return list of column names as variable names

# Example usage:
# variable_names = read_variable_names_from_file('variables.xlsx')
# print(variable_names)  # This will print the list of variable names from the Excel file
print("Enter a prompt that explains what kind of data you want to generate?")
user_prompt = input()

class Channel4Viewers(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str
    day: int
    month: int
    year: int
    gender: str
    address: str
    city: str
    postcode: str
    country: str
    # def read_variable_names_from_file(file_path):
    # # Read the first row of the Excel file to get variable names
    #     df = pd.read_excel(file_path, nrows=1)  # Read only the first row from Excel file
    #     variable_names = df.columns.tolist()  # Return list of column names as variable names
    #     # return variable_names
    #     attributes= []
    #     for i in range(len(variable_names)):
    #         attributes.append({
    #         variable_names[i]: str
    #         })
    #     return attributes




def create_class_from_excel(file_path, class_name):
    # Read the Excel file to get variable names and data types
    df = pd.read_excel(file_path)
    
    # Initialize an empty dictionary to hold attribute definitions
    attributes = {}
    
    # Iterate over each row (variable) in the DataFrame
    for index, row in df.iterrows():
        # Extract variable name and data type from the DataFrame row
        variable_name = row['Variable']
        data_type = row['Data Type']
        
        # Map Excel data types to Pydantic types
        if data_type == 'str':
            attribute_type = str
        elif data_type == 'int':
            attribute_type = int
        else:
            attribute_type = str  # Default to string if data type is not recognized
        
        # Add the attribute to the dictionary with the corresponding data type
        attributes[variable_name] = (attribute_type, ...)
    
    # Use type() to dynamically create the class
    new_class = type(class_name, (BaseModel,), attributes)
    return new_class

# Specify the Excel file path and desired class name
# excel_file_path = 'variables.xlsx'
# desired_class_name = 'Channel4Viewers'

# Create the dynamic class based on data from Excel
# Channel4Viewers = create_class_from_excel(excel_file_path, desired_class_name)

# print(Channel4Viewers)


examples = [
    {"example": """email: john.calm@channel4.com, password: Testing1234, First name:John ,Last name: Calm ,day: 29, month: 05, year: 1970,
    gender: Male, address: Bishopsgate, city: London, postcode: E1 6AN, country: United Kingdom"""},
    {"example": """email: tony.sen@channel4.com, password: Testing1234, First name:Tony ,Last name: Sen ,day: 01, month: 05, year: 1990,
    gender: Male, address: Bishopsgate, city: London, postcode: E1 6AN, country: United Kingdom"""},
    {"example": """email: abru.bose@channel4.com, password: Testing1234, First name:Abru ,Last name: Bose ,day: 29, month: 05, year: 1992,
    gender: Female, address: Houghton, city: London, postcode: E1 6AN, country: United Kingdom"""}
]


# #craft a prompt template
OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject", "extra"],
    example_prompt=OPENAI_TEMPLATE,
)

# # #creating the data generator using langchain
synthetic_data_generator = create_openai_data_generator(
    output_schema= Channel4Viewers,
    llm=ChatOpenAI(temperature=1),
    prompt=prompt_template
)


synthetic_results = synthetic_data_generator.generate(
    subject="channel4_viewers",
    extra = user_prompt,
    # extra="the first name and last name must be chosen at random. Make it something you wouldn't normally choose. The email should be in the format first_name.last_name@channel4.com.",
    runs=5,
)

# # # print(type(synthetic_results), len(synthetic_results))
print('\n')
print(synthetic_results)


# Create a list of dictionaries from the objects
synthetic_data = []
for item in synthetic_results:
    synthetic_data.append({
        'email': item.email,
        'password': item.password,
        'first_name': item.first_name,
        'last_name': item.last_name,
        'day': item.day,
        'month': item.month,
        'year': item.year,
        'gender': item.gender,
        'address': item.address,
        'city': item.city,
        'country': item.country
    })

# # Create a Pandas DataFrame from the list of dictionaries
synthetic_df = pd.DataFrame(synthetic_data)

# # # Display the DataFrame
# # # print(type(synthetic_df))
print(synthetic_df)