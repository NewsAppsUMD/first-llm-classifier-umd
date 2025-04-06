# Bulk prompts

Our reusable prompting function is pretty cool. But requesting answers one by one across a big dataset could take forever. And it could cost us money to hit the Groq API so many times.

One solution is to submit your requests in batches and then ask the LLM to return its responses in bulk.

A common way to do that is to prompt the LLM to return its responses in JSON, a JavaScript data format that is easy to work with in Python. 

To try that, we start by adding the built-in `json` library to our imports in `classifier.py`

{emphasize-lines="2"}
```python
import os
import json
from retry import retry
from groq import Groq
```

Next, we make a series of changes to our function to adapt it to work with a batch of inputs. Get ready. It's a lot.

* We tweak the name of the function.
* We change our input argument to a list.
* We expand our prompt to explain that we will provide a list of team names.
* We ask the LLM to classify them individually, returning its answers in a JSON list.
* We insist on getting one answer for each input.
* We tweak our few-shot training to reflect this new approach.
* We submit our input as a single string with new lines separating each team name.
* We convert the LLM's response from a string to a list using the `json.loads` function.
* We check that the LLM's answers are in our list of acceptable answers with a loop through the list.
* We merge the team names and the LLM's answers into a dictionary returned by the function.

{emphasize-lines="2,17-27,36-43,46,53-54,62-66"}
```python
@retry(ValueError, tries=2, delay=2)
def classify_teams(name_list):
    prompt = """
    You are an AI model trained to classify text.

    I will provide list of professional sports team names separated by new lines

    You will reply with the sports league in which they compete.

    Your responses must come from the following list:
    - Major League Baseball (MLB)
    - National Football League (NFL)
    - National Basketball Association (NBA)

    If the team's league is not on the list, you should label them as "Other".

    Your answers should be returned as a flat JSON list.

    It is very important that the length of JSON list you return is exactly the same as the number of names you receive.

    If I were to submit:

    "Los Angeles Rams\nLos Angeles Dodgers\nLos Angeles Lakers\nLos Angeles Kings"

    You should return the following:

    ["National Football League (NFL)", "Major League Baseball (MLB)", "National Basketball Association (NBA)", "Other"]
    """

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": "Chicago Bears\nChicago Cubs\nChicago Bulls\nChicago Blackhawks",
            },
            {
                "role": "assistant",
                "content": '["National Football League (NFL)", "Major League Baseball (MLB)", "National Basketball Association (NBA)", "Other"]',
            },
            {
                "role": "user",
                "content": "\n".join(name_list),
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    answer_str = response.choices[0].message.content
    answer_list = json.loads(answer_str)

    acceptable_answers = [
        "Major League Baseball (MLB)",
        "National Football League (NFL)",
        "National Basketball Association (NBA)",
        "Other",
    ]
    for answer in answer_list:
        if answer not in acceptable_answers:
            raise ValueError(f"{answer} not in list of acceptable answers")

    return dict(zip(name_list, answer_list))
```

Try that with our team list. At the bottom, instead of the loop, just put this:

```python
classify_teams(team_list)
```

And re-run the classifier. You'll see that it works with only a single API call. The same technique will work for a batch of any size.

```python
{'Minnesota Twins': 'Major League Baseball (MLB)',
 'Minnesota Vikings': 'National Football League (NFL)',
 'Minnesota Timberwolves': 'National Basketball Association (NBA)',
 'Minnesota Wild': 'Other'}
```

Though, as you batches get bigger, one common problem is that the number of outputs from the LLM can fail to match the number of inputs you provide. This problem may lessen as LLMs improve, but for now it's a good idea to limit to batches to a few dozen inputs and to verify that you're getting the right number back.

{emphasize-lines="66-69"}
```python
@retry(ValueError, tries=2, delay=2)
def classify_teams(name_list):
    prompt = """
    You are an AI model trained to classify text.

    I will provide list of professional sports team names separated by new lines

    You will reply with the sports league in which they compete.

    Your responses must come from the following list:
    - Major League Baseball (MLB)
    - National Football League (NFL)
    - National Basketball Association (NBA)

    If the team's league is not on the list, you should label them as "Other".

    Your answers should be returned as a flat JSON list.

    It is very important that the length of JSON list you return is exactly the same as the number of names you receive.

    If I were to submit:

    "Los Angeles Rams\nLos Angeles Dodgers\nLos Angeles Lakers\nLos Angeles Kings"

    You should return the following:

    ["National Football League (NFL)", "Major League Baseball (MLB)", "National Basketball Association (NBA)", "Other"]
    """

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": "Chicago Bears,Chicago Cubs,Chicago Bulls,Chicago Blackhawks",
            },
            {
                "role": "assistant",
                "content": '["National Football League (NFL)", "Major League Baseball (MLB)", "National Basketball Association (NBA)", "Other"]',
            },
            {
                "role": "user",
                "content": "\n".join(name_list),
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    answer_str = response.choices[0].message.content
    answer_list = json.loads(answer_str)

    acceptable_answers = [
        "Major League Baseball (MLB)",
        "National Football League (NFL)",
        "National Basketball Association (NBA)",
        "Other",
    ]
    for answer in answer_list:
        if answer not in acceptable_answers:
            raise ValueError(f"{answer} not in list of acceptable answers")

    try:
        assert len(name_list) == len(answer_list)
    except AssertionError:
        raise ValueError(f"Number of outputs ({len(name_list)}) does not equal the number of inputs ({len(answer_list)})")

    return dict(zip(name_list, answer_list))
```

Okay. Naming sports teams is a cute trick, but what about something hard? 

We'll tackle that by pulling in our example dataset using `pandas`, a popular data manipulation library in Python.

First, we need to install it. Back to the terminal:

```bash
pip install pandas
```

Let's create a new file for our specific classifier, and call it `classifier_md.py`:

```bash
touch classifier_md.py
```

Let's edit that file, adding the imports and Groq setup:

{emphasize-lines="5"}
```python
import os
import json
from groq import Groq
from retry import retry
import pandas as pd

api_key = os.environ.get('GROQ_API_KEY')
client = Groq(api_key=api_key)
```

Now we're ready to load the Maryland grantees data prepared for the class. It contains the distinct list of all recipients listed as grantees in [this data](https://opendata.maryland.gov/Budget/State-of-Maryland-Grant-and-Loan-Data-FY2009-to-FY/absk-avps/data_preview). Let's take a look by starting a Python shell in the terminal and putting these lines in:

```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/dwillis/datajournalismbook-maryland/refs/heads/main/data/grantees.csv")
```

Have a look at a random sample to get a taste of what's in there.

```python
df.sample(10)
```

```
                                                Grantee
742             PHILLIPS WARFT ENVIRONMENTAL CENTER INC
4848                   HOUSING AUTHORITY ST MARYS CO MD
29                                     TURNER FARMS INC
4425                       ANNE ARUNDEL COUNTY MARYLAND
195                               ONE DAY AT A TIME INC
5818            FAMILY LEAGUE OF BALTIMORE CITY INC LMB
3571  UNIVERSITY OF MARYLAND COLLEGE PARK FOUNDATIONSTA
5036        HABITAT FOR HUMANITY OF WICOMICO COUNTY INC
6017         CHOPTANK COMMUNITY HEALTH SYSTEM  CAROLINE
5433              FAMILY AND MEDICAL COUNSELING SERVICE
```

Now let's adapt what we have to fit. Instead of asking for a sports league back, we will ask the LLM to classify each payee as a higher education institution, a museum, local government agency or other establishment. Add this function to your new classifier script.

{emphasize-lines="2-26,33-48,61-66"}
```python
@retry(ValueError, tries=2, delay=2)
def classify_grantees(name_list):
    prompt = """You are an AI model trained to categorize organizations based on their names.

    You will be given a list of organization names, each separated by a new line.

    Your task is to analyze each name and classify it into one of the following categories: Higher Education, Museum, Local Government, or Other.

    It is extremely critical that there is a corresponding category output for each organization name provided as an input.

    If a organization does not clearly fall into Higher Education, Museum, or Local Government categories, you should classify it as "Other".

    Even if the type of organization is not immediately clear from the name, it is essential that you provide your best guess based on the information available to you. If you can't make a good guess, classify it as Other.

    For example, if given the following input:

    "ANNE ARUNDEL COUNTY MARYLAND\nJOHN HOPKINS UNIVERSITY\nLEDOS PIZZA\nFREDERICK MEMORIAL HOSPITAL\nANNAPOLIS MARITIME MUSEUM\nCOLLEGE OF NOTRE DAME OF MARYLAND"

    Your output should be a JSON list in the following format:

    ["Local Government", "Higher Education", "Other", "Other", "Museum", "Higher Education"]

    This means that you have classified "ANNE ARUNDEL COUNTY MARYLAND" as a Local Government, "JOHN HOPKINS UNIVERSITY" as Higher Educatio, "LEDOS PIZZA" as Other, "FREDERICK MEMORIAL HOSPITAL" as Other, "ANNAPOLIS MARITIME MUSEUM" as Museum and "COLLEGE OF NOTRE DAME OF MARYLAND" as Higher Education.

    Ensure that the number of classifications in your output matches the number of organization names in the input. It is very important that the length of JSON list you return is exactly the same as the number of organization names you receive.
    """
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": "ANNE ARUNDEL COUNTY MARYLAND\nJOHN HOPKINS UNIVERSITY\nLEDOS PIZZA\nFREDERICK MEMORIAL HOSPITAL\nANNAPOLIS MARITIME MUSEUM\nCOLLEGE OF NOTRE DAME OF MARYLAND",
            },
            {
                "role": "assistant",
                "content": '["Local Government", "Higher Education", "Other", "Other", "Museum", "Higher Education"]',
            },
            {
                "role": "user",
                "content": "MD CENTER AT BOWIE STATE UNIVERSITY\nBALTIMORE CHILDRENS MUSEUM INCPORT DISCOVERY\nWMATA\nANNE ARUNDEL COUNTY COMMUNITY ACTION AGENCY",
            },
            {
                "role": "assistant",
                "content": '["Higher Education", "Museum", "Other", "Local Government"]',
            },
            {
                "role": "user",
                "content": "\n".join(name_list),
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    answer_str = response.choices[0].message.content
    answer_list = json.loads(answer_str)

    acceptable_answers = [
        "Local Government",
        "Higher Education",
        "Museum",
        "Other",
    ]
    for answer in answer_list:
        if answer not in acceptable_answers:
            raise ValueError(f"{answer} not in list of acceptable answers")

    try:
        assert len(name_list) == len(answer_list)
    except AssertionError:
        raise ValueError(f"Number of outputs ({len(name_list)}) does not equal the number of inputs ({len(answer_list)})")

    return dict(zip(name_list, answer_list))
```

Now pull out a random sample of grantees as a list and put that at the bottom of the script.

```python
df = pd.read_csv("https://raw.githubusercontent.com/dwillis/datajournalismbook-maryland/refs/heads/main/data/grantees.csv")
sample_list = list(df.sample(10).Grantee)
print(classify_grantees(sample_list))
```

Save `classifier_umd.py` and see how it does.

```bash
python classifier_umd.py
```

```python
{'SALISBURY STATE UNIVERSITY': 'Higher Education', 'LOCAL MANAGEMENT BOARD OF ALLEGANY CO INC': 'Local Government', 'WILLIS REDDEN': 'Other', 'ELB AUTOMOTIVE INC': 'Other', 'CATHOLIC CHARITIES ARCHDIOCESE OF WASHINGTON': 'Other', 'MARYLAND PARTNERSHIP FOR PREVENTION INC': 'Other', 'UNIVERSITY OF MARYLAND BALTIMORE COUNTY  CENTER FOR WOMEN  INFORMATION TECH': 'Higher Education', 'THE ARC OF HOWARD COUNTY INC': 'Other', 'METRO LAUNDRY CORPORATION': 'Other', 'EUI J CHOI': 'Other'}
```

That's nice for a sample. But how do you loop through the entire dataset and code them.

One way to start is to write a function that will split up a list into batches of a certain size.

```python
def get_batch_list(li, n=10):
    """Split the provided list into batches of size `n`."""
    batch_list = []
    for i in range(0, len(li), n):
        batch_list.append(li[i : i + n])
    return batch_list
```

Before we loop through our payees, let's add a couple libraries that will let us avoid hammering Groq and keep tabs on our progress. In the terminal, do this:

```bash
pip install rich
```

{emphasize-lines="4"}
```python
import os
import time
import json
from rich.progress import track
from retry import retry
from groq import Groq
import pandas as pd
```

That batching trick can then be fit into a new function that will accept a big list of grantees and classify them batch by batch. Add this to the script just after the `get_batch_list` function:

```python
def classify_batches(name_list, batch_size=10, wait=2):
    """Split the provided list of names into batches and classify with our LLM them one by one."""
    # Create a place to store the results
    all_results = {}

    # Batch up the list
    batch_list = get_batch_list(name_list, n=batch_size)

    # Loop through the list in batches
    for batch in track(batch_list):
        # Classify it with the LLM
        batch_results = classify_grantees(batch)

        # Add what we get back to the results
        all_results.update(batch_results)

        # Tap the brakes to avoid overloading groq's API
        time.sleep(wait)

    # Return the results
    return all_results
```

Now, let's take out a bigger sample. Update the script to use a bigger sample.

```python
bigger_sample = list(df.sample(100).Grantee)
```

And then replace the last line with this and run the script:

```python
print(classify_batches(bigger_sample))
```

Printing out to the console is interesting, but eventually you'll want to be able to work with the results in a more structured way. So let's convert the results into a `pandas` DataFrame by modifying our `classify_batches` function.

{emphasize-lines="20-23"}
```python
def classify_batches(name_list, batch_size=10, wait=2):
    # Store the results
    all_results = {}

    # Batch up the list
    batch_list = get_batch_list(name_list, n=batch_size)

    # Loop through the list in batches
    for batch in track(batch_list):
        # Classify it
        batch_results = classify_grantees(batch)

        # Add it to the results
        all_results.update(batch_results)

        # Tap the brakes
        time.sleep(wait)

    # Return the results
    return pd.DataFrame(
        all_results.items(),
        columns=["grantee", "category"]
    )
```

Results can now be stored as a DataFrame. Replace your last line with these:

```python
results_df = classify_batches(bigger_sample)
print(results_df.head())
```

```
                    grantee          category
0                 MINH VINH             Other
1     WALNUT HILL FARMS INC             Other
2    THE BENEDICTINE SCHOOL  Higher Education
3  THE UNIVERSITY OF KANSAS  Higher Education
4            GARRETT CO CAC  Local Government
```

Or a sum of all the categories.

```python
print(results_df.category.value_counts())
```

```
category
Other               81
Higher Education    11
Local Government     8
Name: count, dtype: int64
```
