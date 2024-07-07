import re

# Copy the release notes from the GitHub release page
markdown_text = """
## What's Changed
* Update pace_chla_to_image_function by @giswqs in https://github.com/opengeos/HyperCoast/pull/69


**Full Changelog**: https://github.com/opengeos/HyperCoast/compare/v0.6.2...v0.6.3
"""

# Regular expression pattern to match the Markdown hyperlinks
pattern = r"https://github\.com/opengeos/HyperCoast/pull/(\d+)"


# Function to replace matched URLs with the desired format
def replace_url(match):
    pr_number = match.group(1)
    return f"[#{pr_number}](https://github.com/opengeos/HyperCoast/pull/{pr_number})"


# Use re.sub to replace URLs with the desired format
formatted_text = re.sub(pattern, replace_url, markdown_text)

for line in formatted_text.splitlines():
    if "Full Changelog" in line:
        prefix = line.split(": ")[0]
        link = line.split(": ")[1]
        version = line.split("/")[-1]
        formatted_text = (
            formatted_text.replace(line, f"{prefix}: [{version}]({link})")
            .replace("## What's Changed", "**What's Changed**")
            .replace("## New Contributors", "**New Contributors**")
        )


with open("docs/changelog_update.md", "w") as f:
    f.write(formatted_text)

# Print the formatted text
print(formatted_text)

# Copy the formatted text and paste it to the CHANGELOG.md file
