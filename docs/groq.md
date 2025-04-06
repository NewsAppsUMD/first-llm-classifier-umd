# Getting started with Groq

In addition to the commercial chatbots that draw the most media attention, there are many other ways to access large-language models, including free variations that you can download and run directly on your computer.

For this demonstration, we will use an online service called [Groq](https://groq.com/). Not to be confused with Elon Musk’s [Grok](https://en.wikipedia.org/wiki/Grok_(chatbot)), an entirely different AI, Groq provides fast, inexpensive API access to various open-source models. It is stable and speedy. Its code is similar to other services, so the skills you learn here should be transferable to other platforms.

To get access to its services, go to [groq.com](https://groq.com/). Click on the "DEV CONSOLE" button on the top right. Create an account.

![Groq homepage](/_static/groq.png)

When you're logged in, click "API keys" on the top right.

![Groq panel](/_static/groq-panel.png)

Click on "Create API key" on the next page.

![Groq API key](/_static/groq-keys.png)

Go ahead and create an API key, name it. We think `first-llm-classifer` is a good choice. Hit "Submit."

![Groq key](/_static/groq-name.png)

A popup will appear with your key. Copy it to your clipboard. I've redacted my key in this screenshot, but you should see a long string of characters. It's important to keep this key secret. If someone else gets it, they can use your account to run up charges.

![Groq key](/_static/groq-copy.png)

Now, go to the Settings page of your GitHub repository and add your Groq API key as a new Secret in the Codespaces section under "Secrets and variables"

![Groq codespaces](/_static/groq-codespaces.png)
