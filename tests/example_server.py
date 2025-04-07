from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test-server")


@mcp.tool(
    name="test-tool",
    description="A test tool description",
)
async def get_data() -> dict[str, str]:
    return {"data": "Hello World!"}


@mcp.resource(
    uri="data://example-{name}",
    name="Get name resource",
    description="Test resource description",
    mime_type="application/json",
)
async def get_resource(name: str) -> dict[str, str]:
    return {"name": name}


if __name__ == "__main__":
    print("Starting server...")
    mcp.run(transport="stdio")
