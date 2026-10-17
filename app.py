
import os
import json
import random
import smtplib

from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from google import genai


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

MODEL_NAME = "gemini-3.8-flash"

STORY_FILE = BASE_DIR / "life_story.txt"
HISTORY_FILE = BASE_DIR / "motivation_history.json"
OUTPUT_FILE = BASE_DIR / "today_motivation.txt"

INDIA_TZ = ZoneInfo("Asia/Kolkata")


def check_configuration():
    required = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "GMAIL_ADDRESS": GMAIL_ADDRESS,
        "GMAIL_APP_PASSWORD": GMAIL_APP_PASSWORD,
        "RECIPIENT_EMAIL": RECIPIENT_EMAIL,
    }

    missing = [
        name for name, value in required.items()
        if not value or not value.strip()
    ]

    if missing:
        raise ValueError(
            "Missing configuration: " + ", ".join(missing)
        )

    if not STORY_FILE.exists():
        raise FileNotFoundError(
            "life_story.txt was not found."
        )


# ==========================================
# 2. READ YOUR STORY AND PREVIOUS HISTORY
# ==========================================

def load_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        data = json.loads(
            HISTORY_FILE.read_text(encoding="utf-8")
        )
        return data if isinstance(data, list) else []

    except (json.JSONDecodeError, OSError):
        print("Warning: Could not read history; starting fresh.")
        return []


def load_story():
    story = STORY_FILE.read_text(
        encoding="utf-8"
    ).strip()

    if not story:
        raise ValueError("life_story.txt is empty.")

    return story


# ==========================================
# 3. CHOOSE A DIFFERENT FOCUS
# ==========================================

FOCUS_AREAS = [
    "career and getting your first job",
    "discipline and procrastination",
    "health and self-respect",
    "financial independence",
    "emotional growth and regret",
    "consistency when motivation disappears",
    "coding skills and employability",
    "taking control of your daily routine",
    "fear of failure and wasted potential",
    "building a better future",
    "stopping excuses and taking responsibility",
    "becoming independent",
]


def choose_focus(history):
    last_focus = (
        history[-1].get("focus")
        if history else None
    )

    choices = [
        focus for focus in FOCUS_AREAS
        if focus != last_focus
    ]

    return random.choice(choices or FOCUS_AREAS)


# ==========================================
# 4. GENERATE PERSONALIZED MOTIVATION
# ==========================================

def generate_motivation(story, history, focus, now):
    client = genai.Client(api_key=GEMINI_API_KEY)

    recent = history[-10:]

    previous_content = {
        "titles": [
            item.get("title", "") for item in recent
        ],
        "quotes": [
            item.get("quote", "") for item in recent
        ],
        "messages": [
            item.get("letter", "") for item in recent
        ],
    }

    prompt = f"""
You are my best friend. You know everything about my life and you are not going to sugarcoat anything. You talk to me like a real person, not like a life coach or a motivational speaker. No big fancy words. No poetic language. No cliches. Just straight, plain, simple English like a friend who is genuinely fed up watching me waste my potential.

DATE AND TIME IN INDIA:
{now.strftime("%A, %d %B %Y, %I:%M %p")}

TODAY'S FOCUS:
{focus}

MY PERSONAL STORY:
<personal_story>
{story}
</personal_story>

PREVIOUS CONTENT TO AVOID REPEATING:
{json.dumps(previous_content, ensure_ascii=False)}

Write a fresh motivational message for me.

STYLE AND TONE RULES — READ CAREFULLY:
- Talk like a blunt close friend, not a motivational speaker. Short sentences. Everyday words.
- Do NOT use fancy or literary words like: resilience, tenacity, forge, transcend, illuminate, pivotal, adversity, embrace, empower, unwavering, profound, ignite, catalyst, caliber, constellation, epitome. Use plain words instead.
- Do NOT use dramatic poetic phrases. Just say things directly.
- Be HARD and NEGATIVE in a real way. Use the breakup as fuel. Example approach: "She left because you were broke, unfit, and had no direction. That is the truth. Not because she was a bad person. Because you gave her reasons to leave. Stop pretending otherwise."
- Do not invent new facts about the ex. Only use what is in the story. But do not be gentle about the real facts.
- Make me face my procrastination, laziness, excuses, and wasted time directly.
- Challenge my choices and habits. Do not comfort me.
- Never blame me for childhood abuse.
- Skip famous quotes, cliches, and empty pep talk lines.
- Use only 1 or 2 short paragraphs, around 100 to 150 words.
- Include one original quote that sounds like something a real harsh friend would say, not a poster.
- End with one specific action I can do today, stated as a direct instruction.
- Make this message completely different from previous messages.
- Do not reuse previous titles, quotes, openings, or advice.

Return valid JSON only, using these fields:
{{
  "title": "Short, blunt title",
  "letter": "One or two short paragraphs in plain friend-talk",
  "quote": "One original quote, no fancy words",
  "action": "One specific action for today, stated directly",
  "closing": "One short plain closing line"
}}

Do not include Markdown code fences.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "temperature": 1.2,
                "response_mime_type": "application/json",
            },
        )
    except Exception as api_err:
        raise RuntimeError(
            f"Gemini API call failed: {api_err}"
        ) from api_err

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    result = json.loads(response.text)

    fields = ["title", "letter", "quote", "action", "closing"]

    for field in fields:
        if (
            not isinstance(result.get(field), str)
            or not result[field].strip()
        ):
            raise ValueError(
                f"Gemini returned an invalid '{field}' field."
            )

    # Check for an exact duplicate quote.
    old_quotes = {
        item.get("quote", "").strip().casefold()
        for item in history
        if item.get("quote")
    }

    if result["quote"].strip().casefold() in old_quotes:
        retry = client.models.generate_content(
            model=MODEL_NAME,
            contents=(
                "Write one short, completely new original "
                "hard-hitting motivational quote. Do not reuse "
                "these previous quotes: "
                + json.dumps(list(old_quotes), ensure_ascii=False)
                + "\nToday's focus: " + focus
                + "\nReturn only the quote."
            ),
            config={"temperature": 1.2},
        )

        if retry.text and retry.text.strip():
            result["quote"] = retry.text.strip().strip('"“”')

    return result


# ==========================================
# 5. FORMAT AND SAVE THE MESSAGE
# ==========================================

def format_message(result, focus, now):
    return f"""{result['title']}

{now.strftime("%A, %d %B %Y — %I:%M %p IST")}
Today's focus: {focus}

{result['letter']}

ORIGINAL QUOTE
"{result['quote']}"

TODAY'S ACTION
{result['action']}

{result['closing']}
"""


# ==========================================
# 6. SEND THE EMAIL
# ==========================================

def send_email(subject, body):
    message = EmailMessage()
    message["From"] = GMAIL_ADDRESS
    message["To"] = RECIPIENT_EMAIL
    message["Subject"] = subject
    message.set_content(body)

    # Gmail SMTP over a secure SSL connection.
    with smtplib.SMTP_SSL(
        "smtp.gmail.com",
        465,
        timeout=30,
    ) as server:
        server.login(
            GMAIL_ADDRESS,
            GMAIL_APP_PASSWORD.replace(" ", ""),
        )
        server.send_message(message)


# ==========================================
# 7. MAIN PROGRAM
# ==========================================

def send_error_email(error_message):
    """Send a plain error notification email when the API or generation fails."""
    now = datetime.now(INDIA_TZ)
    subject = "Motivation Bot ERROR — Something went wrong"
    body = (
        f"Motivation Bot failed to generate your message.\n"
        f"Time: {now.strftime('%A, %d %B %Y, %I:%M %p IST')}\n\n"
        f"Error:\n{error_message}\n\n"
        "Possible reasons:\n"
        "  - The Gemini API key is wrong or expired\n"
        "  - The API is not responding right now\n"
        "  - There was a network issue\n\n"
        "Check your .env file and try again."
    )
    try:
        send_email(subject, body)
        print("Error notification email sent.")
    except Exception as mail_err:
        print(f"Could not send error email either: {mail_err}")


def main():
    check_configuration()

    now = datetime.now(INDIA_TZ)
    story = load_story()
    history = load_history()

    focus = choose_focus(history)

    print("Generating your motivation...")
    print("Focus:", focus)

    try:
        result = generate_motivation(
            story,
            history,
            focus,
            now,
        )
    except RuntimeError as api_err:
        print(f"\nAPI error: {api_err}")
        send_error_email(str(api_err))
        return
    except Exception as err:
        print(f"\nUnexpected error during generation: {err}")
        send_error_email(str(err))
        return

    body = format_message(result, focus, now)

    # Save the latest message locally or in the CI workspace.
    OUTPUT_FILE.write_text(
        body,
        encoding="utf-8",
    )

    # Record this message for future uniqueness checks.
    history.append({
        "generated_at": now.isoformat(),
        "focus": focus,
        "title": result["title"],
        "letter": result["letter"],
        "quote": result["quote"],
        "action": result["action"],
        "closing": result["closing"],
    })

    HISTORY_FILE.write_text(
        json.dumps(
            history,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    subject = f"Reality Check: {result['title']}"

    # Send only after generation and saving succeed.
    send_email(subject, body)

    print("\nEmail sent successfully!")
    print("To:", RECIPIENT_EMAIL)
    print("Saved:", OUTPUT_FILE.name)
    print("History:", HISTORY_FILE.name)


if __name__ == "__main__":
    main()
