# Reflex MCP Chat App

A starting point for a highly customizable Python web application that uses LLMs with access to MCP (Model context protocol) servers to build a highly capable chat application.

The interface is based on the [Reflex chat app template](https://github.com/reflex-dev/reflex-chat) for the interface.

The inner workings are completely new, utilizing the latest approaches to LLM application development with [LangGraph](https://www.langchain.com/langgraph) and [MCP](https://modelcontextprotocol.io/introduction) (Model Context Protocol).

<div align="center">
<img src="./docs/demo.gif" alt="icon"/>
</div>

## Key Technologies Used

- [Reflex](https://reflex.dev/) - A Python web framework for building interactive web applications.
- [LangGraph](https://www.langchain.com/langgraph) - A framework for building LLM applications
- [MCP](https://modelcontextprotocol.io/introduction) - Anthropics open source protocol for providing context to LLM applications.
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

<!-- icon for Run Setup -->

### ⚙️ 4. Run Setup

To set up all dependencies

```bash
task install
```

### 🚀 3. Run the application

```bash
task run
```

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

## Contributing

If you'd like to contribute, please do the following:

- Fork the repository and make your changes.
- Once you're ready, submit a pull request for review.

## License

The following repo is licensed under the MIT License.
