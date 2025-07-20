import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your key in env variables

def generate_retention_email(customer_profile):
    prompt = (
        f"Write a friendly and personalized retention email to prevent customer churn "
        f"based on this customer profile:\n{customer_profile}\n"
        f"Keep it concise and encouraging."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150,
    )
    return response['choices'][0]['message']['content'].strip()
