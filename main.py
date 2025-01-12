from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import json
import uvicorn
import asyncio

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Person(BaseModel):
    id: str
    firstName: str
    lastName: str
    dateOfBirth: str
    age: int
    upcomingBirthday: str
    cityOfResidence: str
    email: str
    phoneNumber: str
    nationality: str
    mainInterests: list
    socialMediaLinks: list
    loyaltyPrograms: list
    passions: list
    lifestyle: list
    travelDocuments: list
    typeOfTravel: list
    travelSpan: list
    travelBucketList: list
    specialRequirements: list


class Group(BaseModel):
    id: str
    groupName: str
    userName: str
    password: str
    mainUser: Person
    dependents: list[Person]
    augmentedData: str


# OpenAI client setup
openai_client = OpenAI(
    api_key="")  # Add your openai api key here, we can't share it here as github is public.
# We use gpt-4o for now, but we can change it to any oepnai model you want.
openai_gpt_model = "gpt-4o"


@app.post("/process_data/")
async def process_data(group: Group):
    json_body = group.dict()

    prompt_getter = f"""
    Analyze the provided JSON data and generate detailed insights similar to a professional travel and lifestyle analyst. Your task is to make specific, evidence-based deductions from the data provided.

    DEDUCTION REQUIREMENTS:
    1. Professional Context:
       - Analyze email domains for career insights
       - Consider loyalty programs for professional travel patterns
       - Deduce work-life patterns from interests and travel preferences

    2. Family & Social Analysis:
       - Identify age-appropriate activities and interests
       - Analyze family dynamics and age gaps
       - Consider educational implications for children
       - Deduce lifestyle patterns from family composition

    3. Travel & Lifestyle Patterns:
       - Connect loyalty programs to travel frequency
       - Analyze travel timing based on family circumstances
       - Consider seasonal preferences and constraints
       - Identify travel style based on interests and demographics

    4. Behavioral & Preference Analysis:
       - Link interests to potential activities
       - Connect lifestyle choices to travel preferences
       - Analyze fitness and wellness patterns
       - Deduce meal and schedule preferences

    5. Cultural & Geographic Insights:
       - Consider nationality and residence implications
       - Analyze cultural preferences and limitations
       - Identify location-based opportunities

    RESPONSE FORMAT:
    - In augmentedData field, provide numbered insights where each insight:
      * Makes a specific deduction
      * Explains the reasoning ("because of...")
      * Links multiple data points
      * Provides actionable implications
      * Follows format: "Deduction (because of specific data points)"

    Example insight format:
    1. "Principal works at [Company] (because of the email domain)"
    2. "Family likely travels during [specific times] (because of children's age and school system)"
    3. "Lifestyle indicates [specific pattern] (because of multiple interests and preferences shown)"

    Input Data: {json.dumps(json_body)}
    """

    sys_prompt = """You are an expert travel and lifestyle analyst specializing in deriving detailed, evidence-based insights from personal and family data. Your deductions should be specific, well-reasoned, and always supported by data points from the input."""

    async def generate_response():
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": sys_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt_getter,
                    },
                ],
                stream=True
            )

            response_json = json_body.copy()
            response_json['augmentedData'] = []
            accumulated_insights = []

            # Stream the chunks
            for chunk in completion:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content

                    if '```json' in content:
                        content = content.replace('```json', '')
                    if '```' in content:
                        content = content.replace('```', '')

                    accumulated_insights.append(content)

                    try:
                        full_text = ''.join(accumulated_insights)
                        # Extract insights from the text
                        insights = [
                            line.strip()
                            for line in full_text.split('\n')
                            if line.strip() and not line.strip().startswith('{') and not line.strip().startswith('}')
                            and not line.strip().startswith('[') and not line.strip().startswith(']')
                            and '"augmentedData"' not in line
                        ]

                        response_json['augmentedData'] = insights
                        yield f"data: {json.dumps(response_json)}\n\n"
                    except Exception as json_error:
                        print(f"JSON encoding error: {json_error}")
                        continue

        except Exception as e:
            error_response = {"error": str(e)}
            yield f"data: {json.dumps(error_response)}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
