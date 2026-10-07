from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("Error: HF_TOKEN not found in .env file.")
    exit()


# -----------------------------
# 2. Create Hugging Face client
# -----------------------------
client = InferenceClient(
    token=HF_TOKEN
)


# -----------------------------
# 3. Get user input
# -----------------------------
topic = input("Enter topic: ")
number = input("How many flashcards? ")


# -----------------------------
# 4. Create prompt
# -----------------------------
prompt = f"""
You are an educational flashcard generator.

Create {number} flashcards about:

Topic: {topic}

Use exactly this format:

CARD 1
Question: ...
Answer: ...

CARD 2
Question: ...
Answer: ...

CARD 3
Question: ...
Answer: ...

Rules:
- Create exactly {number} flashcards.
- Keep questions short and clear.
- Keep answers simple and easy to understand.
- Focus on important concepts.
- Do not add unnecessary explanations.
"""


# -----------------------------
# 5. Send request to Hugging Face
# -----------------------------
try:

    response = client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="Qwen/Qwen2.5-7B-Instruct-1M",
        max_tokens=1000
    )


    # -----------------------------
    # 6. Display flashcards
    # -----------------------------
    print("\n==============================")
    print("       GENERATED FLASHCARDS")
    print("==============================\n")

    print(response.choices[0].message.content)

except Exception as e:

    print("\nSomething went wrong!")
    print("Error:", e)