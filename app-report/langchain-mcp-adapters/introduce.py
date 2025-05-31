from mcp.server.fastmcp import FastMCP

mcp = FastMCP("introduce")


@mcp.tool()
def age(name: str) -> str:
    """Return age of a person given their name."""
    name_age_map = {
        "Alice": 29,
        "Bob": 35,
        "Charlie": 42,
        "Diana": 31,
        "Ethan": 26,
        "Fiona": 38,
        "George": 44,
        "Hannah": 30,
        "Ivan": 33,
        "Julia": 28,
    }
    age_value = name_age_map.get(name, 21)
    return f"{name} is {age_value} years old"


@mcp.tool()
def job(name: str) -> str:
    """Return the job title of a person given their name."""
    name_job_map = {
        "Alice": "Software Engineer",
        "Bob": "Product Manager",
        "Charlie": "Data Scientist",
        "Diana": "UX Designer",
        "Ethan": "DevOps Engineer",
        "Fiona": "Professor",
        "George": "Architect",
        "Hannah": "Doctor",
        "Ivan": "Photographer",
        "Julia": "Writer",
    }
    job = name_job_map.get(name, "unemployed")
    return f"{name} is a {job}."


@mcp.tool()
def hobby(name: str) -> str:
    """Return the hobby of a person given their name."""
    name_hobby_map = {
        "Alice": "hiking",
        "Bob": "playing guitar",
        "Charlie": "cooking",
        "Diana": "painting",
        "Ethan": "cycling",
        "Fiona": "reading",
        "George": "gardening",
        "Hannah": "swimming",
        "Ivan": "photography",
        "Julia": "yoga",
    }
    hobby = name_hobby_map.get(name, "doing nothing")
    return f"{name}'s hobby is {hobby}."


if __name__ == "__main__":
    mcp.run(transport="stdio")
