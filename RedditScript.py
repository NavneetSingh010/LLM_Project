# Import required libraries
from huggingface_hub import InferenceClient
import praw
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face API token from environment
HF_TOKEN = os.getenv("HF_API_TOKEN")

# Initialize Hugging Face inference client for LLM chat model
client = InferenceClient(token=HF_TOKEN)

# Load Reddit API credentials from environment
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "PersonaScraper by /u/Potential-Win-4655")

# Create and return an instance of the Reddit API
def reddit_instance():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

# Extract Reddit username from a profile URL
def extract_username(url):
    match = re.search(r'reddit\.com/user/([A-Za-z0-9_-]+)/?', url)
    return match.group(1) if match else None

# Scrape the user's latest posts and comments using PRAW
def scrape_user_data(username, limit=5):
    reddit = reddit_instance()
    user = reddit.redditor(username)
    posts = []
    comments = []
    try:
        # Collect user's latest submissions
        for submission in user.submissions.new(limit=limit):
            posts.append({
                "title": submission.title,
                "text": submission.selftext,
                "url": submission.url
            })
        # Collect user's latest comments
        for comment in user.comments.new(limit=limit):
            comments.append({
                "body": comment.body,
                "url": f"https://www.reddit.com{comment.permalink}"
            })
    except Exception as e:
        print("Error scraping user:", e)
    return posts, comments

# Generate a persona description from scraped text using LLM
def generate_persona(text_posts, text_comments, username):
    # Construct prompt to instruct the model
    prompt = f"""You are a persona-building assistant.
Use the following Reddit posts and comments by u/{username} to create a detailed user persona.
Include:
- Name (can be invented),
- Age range,
- Interests,
- Personality traits,
- Occupation (guess if not mentioned),
- Writing style,
- Typical subreddit activity

Cite specific posts/comments under each characteristic.

=== POSTS ===
{text_posts}

=== COMMENTS ===
{text_comments}
"""

    # Format prompt into messages for chat-style LLMs
    messages = [
        {"role": "system", "content": "You are a helpful assistant who builds detailed user personas from social media data."},
        {"role": "user", "content": prompt}
    ]

    # Send request to the LLM model for persona generation
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",  # You can replace with other supported chat model
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )

    # Extract and return generated content from response
    return response.choices[0].message.content

# Save the generated persona to a local text file
def save_persona(text, username):
    filename = f"user_persona_{username}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Persona saved to {filename}")

# Entry point of the script
if __name__ == "__main__":
    # Ask user for Reddit profile URL
    profile_url = input("Enter Reddit profile URL: ")

    # Extract the username from the URL
    username = extract_username(profile_url)

    # Exit if URL format is invalid
    if not username:
        print("Invalid Reddit URL.")
        exit()

    # Scrape user data (posts and comments)
    posts, comments = scrape_user_data(username)

    # Format posts and comments for prompt input
    text_posts = "\n".join([f"- {p['title']}\n  {p['text']}\n  URL: {p['url']}" for p in posts])
    text_comments = "\n".join([f"- {c['body']}\n  URL: {c['url']}" for c in comments])

    # Generate persona using the LLM
    persona = generate_persona(text_posts, text_comments, username)

    # Save the generated persona to file
    save_persona(persona, username)
