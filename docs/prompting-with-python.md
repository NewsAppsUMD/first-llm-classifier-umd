# Prompting with Python

Now that you've got your Python environment set up, it's time to start writing prompts and sending them off to Groq.

First, we'll install the libraries we need. The `groq` package is the official client for Groq's API.

A common way to install packages from inside Codespaces is to use the `pip` command. Put the following command in the Terminal:

```bash
pip install groq
```

Remember storing your Groq API key as a GitHub secret? Good. You'll need it now. Let's create a Python script called classifier.py.

```bash
touch classifier.py
```

Open that file for editing, and at the top put our import statements and retrieve the API key and create a Groq client:

```python
import os
from groq import Groq

api_key = os.environ.get('GROQ_API_KEY')
client = Groq(api_key=api_key)
```

Let's make our first prompt. To do that, we submit a dictionary to Groq's `chat.completions.create` method. The dictionary has a `messages` key that contains a list of dictionaries. Each dictionary in the list represents a message in the conversation. When the `role` is "user" it is roughly the same as asking a question to a chatbot.

We also need to pick a model from [among the choices Groq gives us](https://console.groq.com/docs/models). We're picking Llama 3.3, the latest from Meta.

```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of data journalism in a concise sentence",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(response)
```

We'll save the response as a variable. Save the .py file, go to the terminal and run `python classifier.py` and print that Python object to see what it contains.

```bash
python classifer.py
```

You should see something like:

```python
ChatCompletion(
    id='chatcmpl-e219e15c-471f-468c-a0f7-69ba31c83da6',
    choices=[
        Choice(
            finish_reason='stop',
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content='Data journalism plays a crucial role in holding those in power accountable by providing
fact-based insights and analysis, enabling informed decision-making, and promoting transparency through the use of
data-driven storytelling.',
                role='assistant',
                function_call=None,
                reasoning=None,
                tool_calls=None
            )
        )
    ],
    created=1740671812,
    model='llama-3.3-70b-versatile',
    object='chat.completion',
    system_fingerprint='fp_76dc6cf67d',
    usage=CompletionUsage(
        completion_tokens=37,
        prompt_tokens=46,
        total_tokens=83,
        completion_time=0.134545455,
        prompt_time=0.00492856,
        queue_time=0.231341476,
        total_time=0.139474015
    ),
    x_groq={'id': 'req_01jn4200h0e4s8e12pj5d2e3ye'}
)
```

There's a lot here, but the `message` has the actual response from the LLM. Let's just print the content from that message. Note that your response probably varies from this guide. That's because LLMs mostly are probablistic prediction machines. Every response can be a little different. In the script, switch the last line to this and re-run the code.

```python
print(response.choices[0].message.content)
```

Is the response different this time?

Let's pick a different model from among [the choices that Groq offers](https://console.groq.com/docs/models). One we could try is qwen 2.5, an open model from Alibaba. Lets revise the code we already have and rerun it.

{emphasize-lines="8"}
```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of data journalism in a concise sentence",
        }
    ],
    model="qwen-2.5-32b",
)
```

Again, your response might vary from what's here. Let's find out.

```bash
python classifer.py
```

```text
Data journalism is crucial as it uses data analysis to uncover insights and tell compelling stories based on factual evidence.
```

:::{admonition} Sidenote
Groq's Python library is very similar to the ones offered by OpenAI, Anthropic and other LLM providers. If you prefer to use those tools, the techniques you learn here should be easily transferable.

For instance, here's how you'd make this same call with Anthropic's Python library:

```python
from anthropic import Anthropic

client = Anthropic(api_key=api_key)

response = client.messages.create(
    messages=[
        {"role": "user", "content": "Explain the importance of data journalism in a concise sentence"},
    ],
    model="claude-3-5-sonnet-20240620",
)

print(response.content[0].text)
```
:::


A well-structured prompt helps the LLM provide more accurate and useful responses.

One common technique for improving results is to open with a "system" prompt to establish the model's tone and role. Let's switch back to Llama 3.3 and provide a `system` message that provides a specific motivation for the LLM's responses.

{emphasize-lines="3-6,12"}
```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "you are an enthusiastic nerd who believes data journalism is the future."
        },
        {
            "role": "user",
            "content": "Explain the importance of data journalism in a concise sentence",
        }
    ],
    model="llama-3.3-70b-versatile",
)
```

Check out the results.

```bash
python classifer.py
```

```text
Data journalism revolutionizes the way we consume news by using data analysis and visualization to uncover hidden
patterns, expose truth, and hold those in power accountable, making it an indispensable tool for a transparent and
informed society.
```

Want to see how tone affects the response? Change the system prompt to something old-school.

{emphasize-lines="5"}
```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "you are a crusty, ill-tempered editor who hates math and thinks data journalism is a waste of time and resources."
        },
        {
            "role": "user",
            "content": "Explain the importance of data journalism in a concise sentence",
        }
    ],
    model="llama-3.3-70b-versatile",
)
```

Then re-run the code and summon J. Jonah Jameson.

```bash
python classifer.py
```

```text
If you must know, data journalism is supposedly important because it allows reporters to uncover hidden trends and patterns in complex issues by analyzing large datasets, but I still don't see the point of wasting all that time and resources on spreadsheets when a good instinct and a sharp eye for a story can get the job done just fine.
```
