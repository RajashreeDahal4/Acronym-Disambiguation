# ACRONYM DISAMBIGUATION


# Project Description: 
This purpose of this project is to find out which short form maps to long form and make sure that the right long form is chosen among the list of long forms for that short form. 

# to run the repository:
* python3 -m venv venv
* source venv/bin/activate
* pip install -r requirements.txt
* location for saved model in drive: https://drive.google.com/drive/u/2/folders/1KMsf-ozSSQwnt5VJE9vmTPVWMq3IPQzK
* saved weight_name: state_dict_scibert.pt
* train the model: python3 code/main.py train --config_file config.json
* prediction sample:python3 code/main.py predicts --config_file config.json --a "The <start> ISS <end> is the largest modular space station in low Earth orbit.The project involves five space agencies: those include the United States' NASA, Russia's Roscos, Japan's JAXA, Europe's ESA, and Canada's CSA" --b "International Space Station" "Information Systems Services" "Institute for Security Studies" "Issue" "Important"



For more details: contact rd0081@uah.edu
