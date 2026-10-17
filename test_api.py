
import os
import sys

from dotenv import load_dotenv
from google import genai

# Load the API key from the .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Check whether the key was loaded
if not api_key:
    print("FAILED: GEMINI_API_KEY was not found.")
    print("Check that .env exists in your project folder.")
    sys.exit(1)

print("API key found in .env.")
print("Connecting to Gemini...")

try:
    # Create the Gemini client
    client = genai.Client(api_key=api_key)

    # Send a small test request
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Reply with exactly this text: "
                 "Gemini API connection successful!"
    )

    # A successful response verifies that the request worked
    if response.text:
        print("\nSUCCESS: Gemini responded!")
        print("Response:", response.text)
    else:
        print("\nThe request completed, but no text was returned.")
        print("Check the response or try again.")

except Exception as error:
    print("\nFAILED: The Gemini request did not succeed.")
    print("Error type:", type(error).__name__)
    print("Details:", str(error))
    print("\nCheck the API key, model availability,")
    print("API access, usage limits, and network connection.")
