# Getting started with GitHub Codespaces

If you have a GitHub account, you have access to Codespaces, the platform's way of providing virtual computing environments. It comes with a basic Linux setup, meaning Python and tools like pip are already installed.

This repository includes a configuration file, `.devcontainer/devcontainer.json`, that tells Codespaces how to set itself up. When you launch a codespace, it will:

* Start a computer with Python 3.12
* Install all of the packages we use in this class, listed in `requirements.txt`
* Load your `GROQ_API_KEY` secret as an environment variable

To start one, go to your copy of the repository on GitHub, click the green "Code" button, select the "Codespaces" tab and click "Create codespace on main."

If you haven't already saved your Groq API key as a Codespaces secret, GitHub will ask you for it before the codespace starts. Paste in the key you created in the previous chapter.

It can take a few minutes for the codespace to finish installing everything. When it's ready, open a Terminal and confirm your key is available by running:

```bash
python -c "import os; print('ready' if os.environ.get('GROQ_API_KEY') else 'missing key')"
```

If you see `missing key`, add the secret under the "Secrets and variables" > "Codespaces" section of your repository's settings, or at [github.com/settings/codespaces](https://github.com/settings/codespaces), then stop and restart your codespace.

:::{admonition} Note
The `pip install` commands in the chapters ahead are already taken care of by the codespace. Running them again won't hurt anything.
:::
