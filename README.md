# Reflex MCP Chat App

A starting point for a highly customizable Python web application that uses LLMs with access to MCP (Model context protocol) servers to build a highly capable chat application. (Please leave a ⭐ if you like it!)

The interface is based on the [Reflex chat app template](https://github.com/reflex-dev/reflex-chat).

The inner workings are completely new, utilizing the latest approaches to LLM application development with [LangGraph](https://www.langchain.com/langgraph) and [MCP](https://modelcontextprotocol.io/introduction) (Model Context Protocol).

[demo.webm](https://github.com/user-attachments/assets/1c24768b-9d23-423d-b81a-c760c0715b26)

## Key Technologies Used

- [Reflex](https://reflex.dev/) - A Python web framework for building interactive web applications.
- [LangGraph](https://www.langchain.com/langgraph) - A framework for building LLM applications
- [MCP](https://modelcontextprotocol.io/introduction) - Anthropics open source protocol for providing context to LLM applications.
  - Example integrations with:
    - [GitHubMCP](https://github.com/github/github-mcp-server)
    - [Git](https://github.com/modelcontextprotocol/servers/tree/main/src/git)
    - [Filesystem](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem)
    - [Minimal example SSE](https://github.com/timchild/test-mcp-server)
- [UV](https://docs.astral.sh/uv/) - An extremely fast Python project manager.
- [Taskfile](https://taskfile.dev/) - A task runner for automating common tasks in the development process.
- [GitHubActions](https://github.com/features/actions) - GitHub's automation tool for CI/CD.
- [Pyright](https://microsoft.github.io/pyright/#/) - A static type checker for Python.
- [Ruff](https://docs.astral.sh/ruff/) - A fast Python linter and code formatter.
- [Pre-commit](https://pre-commit.com/) - A framework for managing and maintaining multi-language pre-commit hooks.
- [Pytest](https://docs.pytest.org/en/latest/) - A testing framework for Python.
- [Dependency Injector](https://python-dependency-injector.ets-labs.org/) - A dependency injection framework for python that allow for easy management of app configuration and testing.

---

## Getting Started

### 🧬 1. Clone the Repo

```bash
git clone https://github.com/TimChild/reflex-mcp-chat.git
```

### 📦 2. Install UV

If you don't already have `uv` installed, you can follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

For macOS, linux, and wsl2

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 🛠️ 3. (Optionally) Install Taskfile

I recommend [installing Taskfile](https://taskfile.dev/installation/) to make it easier to run tasks in the project, but you can also run any necessary commands manually (using the `Taskfile.yml` as a reference).

On linux (Debian/Ubuntu):

```bash
sudo snap install task --classic
```

On macOS:

```bash
brew install go-task/tap/go-task
```

### ⚙️ 4. Setup

Create a `.env` file (or add directly to environment vars):

- `OPENAI_API_KEY` -- https://platform.openai.com/settings/
- `ANTHROPIC_API_KEY` -- https://console.anthropic.com/settings/keys
- `GITHUB_PERSONAL_ACCESS_TOKEN` -- https://github.com/settings/personal-access-tokens/new

[!TIP]
You can copy the `.env.template` file to `.env` and fill in the values there.
You can also set any non-empty string if you don't want to use that feature (the app checks that they are not-null for easier debugging).

You can also modify the `config.yml` file to add/remove MCP servers and change other app behavior.

Currently the config is set to allow the MCP file server to make changes in `~/mcp_allowed` (you'll need to create that directory first).

### 🚀 5. Run the application

```bash
task run
```

Then open [localhost:3000](http://localhost:3000) in your browser.

## App Features

- 100% Python-based, including the UI, using Reflex
- Selectable LLMs (OpenAI and Anthropic implemented, easily extendable) -- Dropdown selection in the UI
- Runs via [LangGraph](https://www.langchain.com/langgraph)'s standard [Graph](https://langchain-ai.github.io/langgraph/tutorials/introduction/) mode or new [Functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/) -- Dropdown selection in the UI (note: implemented behavior is identical -- allows you to extend either method)
- Custom [MCP](https://modelcontextprotocol.io/introduction) client for easy management of multiple MCP servers
- Multiple [MCP](https://modelcontextprotocol.io/introduction) severs included via 4 different modes for easy extension. Examples include:
  - http SSE (Server-Sent Events)
  - local python stdio via `uv`
  - docker container
  - stdio from github repo via `npx`
- The application is fully customizable without typical frontend development experience.
  - See https://reflex.dev/docs/getting-started/introduction/
- Responsive design for various devices

## Development Environment

There are many helpful tasks to help with development:

Then, to set up all development dependencies

```bash
task install
```

To run type checking, linting, and testing:

```bash
task test
```

To run tests on any changes:

```bash
task watch-tests
```

To run a jupyter lab with all dependencies:

```bash
task jupyter
```

CI/CD workflows are included for automated testing and deployment. Find them in the `.github/workflows` directory.

Uses [Dependency Injector](https://python-dependency-injector.ets-labs.org/) for easy management of app configuration and testing.

App configuration is managed via a `config.yml` file and `containers.py` file.

## TODOs:

A few things that could improve the app:

- [ ] Add a mkdocs site
- [ ] Add docker/docker-compose support
- [ ] Add deployment workflow template
- [ ] Add end-to-end tests using playwright
- [ ] Add postgres persistence (partially implemented)
- [ ] Better handling of long conversations
- [ ] Better handling of large content returned from tools
- [ ] Add MCP resource and prompt support
- [ ] Add MCP sampling support

## Contributing

If you'd like to contribute, please do the following:

- Fork the repository and make your changes.
- Once you're ready, submit a pull request for review.

## License

The following repo is licensed under the MIT License.
