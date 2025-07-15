from huggingface_hub import InferenceClient
import praw
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")  # Your Huggingface API token here

# Initialize client for chat (conversational) model
client = InferenceClient(token=HF_TOKEN)

# Setup Reddit API credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "PersonaScraper by /u/Potential-Win-4655")

# Create Reddit instance
def reddit_instance():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

# Extract username from Reddit URL
def extract_username(url):
    match = re.search(r'reddit\.com/user/([A-Za-z0-9_-]+)/?', url)
    return match.group(1) if match else None

# Scrape Reddit user data (posts and comments)
def scrape_user_data(username, limit=5):
    reddit = reddit_instance()
    user = reddit.redditor(username)
    posts = []
    comments = []
    try:
        for submission in user.submissions.new(limit=limit):
            posts.append({
                "title": submission.title,
                "text": submission.selftext,
                "url": submission.url
            })
        for comment in user.comments.new(limit=limit):
            comments.append({
                "body": comment.body,
                "url": f"https://www.reddit.com{comment.permalink}"
            })
    except Exception as e:
        print("❌ Error scraping user:", e)
    return posts, comments

# Generate persona using Huggingface chat method
def generate_persona(text_posts, text_comments, username):
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

    # Conversational input expects messages list
    messages = [
        {"role": "system", "content": "You are a helpful assistant who builds detailed user personas from social media data."},
        {"role": "user", "content": prompt}
    ]


    response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",  # specify model here or rely on default set in client
    messages=messages,
    max_tokens=500,
    temperature=0.7,
    
 )

    # Extract the assistant's message text from response
    return response.choices[0].message.content


# Save persona to file
def save_persona(text, username):
    filename = f"user_persona_{username}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"✅ Persona saved to {filename}")

# Main execution
if __name__ == "__main__":
    profile_url = input("Enter Reddit profile URL: ")
    username = extract_username(profile_url)

    if not username:
        print("❌ Invalid Reddit URL.")
        exit()

    posts, comments = scrape_user_data(username)
    text_posts = "\n".join([f"- {p['title']}\n  {p['text']}\n  URL: {p['url']}" for p in posts])
    text_comments = "\n".join([f"- {c['body']}\n  URL: {c['url']}" for c in comments])

    persona = generate_persona(text_posts, text_comments, username)
    save_persona(persona, username)
