from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel
from openai import OpenAI
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# OpenAI client setup
openai_client = OpenAI(api_key="")


@app.route("/process_data/", methods=["POST"])
def process_data():
    try:
        logger.debug("Received request")
        json_body = request.get_json()
        if not json_body:
            logger.error("No JSON data received")
            return jsonify({"error": "No data provided"}), 400

        logger.debug(f"Received data: {json_body}")

        # Prepare the system and user prompts
        prompt_getter = f"""
Step 1: Data Extraction and Preliminary Analysis
        Analyze the JSON data to extract key attributes, identify relationships between data points, and assign initial weights.
        Instructions:
        Extract data points under the categories Passions, Interests, Lifestyle, and Travel Preferences (e.g., preferred destinations, travel frequency).
        Assign weights:
        Passions = 3–5
        Interests = 1–3
        Lifestyle = 0.5–2
        Travel Preferences = Based on relevance to the task (1–5 range).
        Identify direct relationships (e.g., Opera → Music → Cultural Interest → European Travel).
        Output Format:
        Data Point: [Label]  
        - Weight: [Value]  
        - Correlation: [List of related points]  
        - Reasoning: [Explanation of weight and correlations]  

        Step 2: Embedding-Based Clustering
        Cluster related data points using embeddings to group semantically similar attributes.
        Instructions:
        Generate embeddings for each data point, including travel-related data.
        Group semantically related points into clusters.
        Name each cluster descriptively and include members.
        Output Format:
        Cluster: [Cluster Name]  
        - Members: [List of data points]  
        - Reasoning: [Why these points are grouped together]  

        Step 3: Hypothesis Generation (Chain-of-Thought Reasoning)
        Generate hypotheses for each cluster, focusing on travel-related interests and preferences.
        Instructions:
        Expand beyond direct relationships, exploring nuanced travel possibilities (e.g., "Opera → Classical Music → Vienna’s Music Scene").
        Connect hypotheses to potential travel themes, itineraries, or experiences.
        Output Format:
        Cluster: [Cluster Name]  
        - Hypothesis: [Hypothesis about preferences or travel interests]  
        - Reasoning: [Why this hypothesis is plausible based on data]  


        Step 4: Personality Profiling
        Analyze clusters using the Personality framework and relate personality traits to travel behavior.
        Instructions:
        Align clusters with Personality traits:
        Honesty-Humility
        Emotionality
        Extraversion
        Agreeableness
        Conscientiousness
        Openness
        Use a 1–5 scale (Low–High) for each trait.
        Relate traits to travel preferences (e.g., high Openness → Preference for cultural exploration).
        Output Format:
        Cluster: [Cluster Name]  
        - Personality Analysis:  
          - Honesty-Humility: [Score]  
          - Emotionality: [Score]  
          - Extraversion: [Score]  
          - Agreeableness: [Score]  
          - Conscientiousness: [Score]  
          - Openness: [Score]  
          - Reasoning: [Explanation for Personality alignment and travel behavior]  


        Step 5: Insight Generation and Travel Recommendations
        Generate actionable insights and travel suggestions based on Personality traits and clusters.
        Instructions:
        Provide nuanced travel insights that consider clusters and personality traits.
        Suggest travel experiences, destinations, and itineraries aligned with preferences.
        Include recommendations for:
        Destinations matching interests (e.g., "Italy for wine lovers").
        Themed travel (e.g., "Luxury travel for relaxation").
        Group travel (for households or couples).
        Output Format:
        Insight: [Detailed insight about preferences and travel behaviors]  
        - Travel Suggestion: [Recommended destination, theme, or itinerary]  
        - Reasoning: [How this aligns with clusters and Personality traits]  

        Step 6: Multi-Person Analysis (Optional)
        Compare profiles to generate shared travel insights and group recommendations.
        Instructions:
        Perform Steps 1–5 for each individual.
        Identify convergences (shared interests) and divergences (individual preferences).
        Output Format:
        Convergence: [Shared travel preferences or traits]  
        Divergence: [Differing preferences or traits]  
        - Group Travel Suggestion: [Suggestions for shared experiences]  

        Step 7: Travel Radius Analysis
        Generate location-based travel insights if a city of residence is provided.
        Instructions:
        Create travel radius categories:
        Local experiences (<1 hour by car or train).
        Short getaways (1–3 hours by flight).
        Long getaways (3–7 hours by flight).
        Recommend destinations fitting each category.
        Output Format:
        Travel Radius: [Category]  
        - Destinations: [List of destinations]  
        - Reasoning: [How these align with user preferences]  


        Step 8: Augmented Insight and Final Travel Plan
        Synthesize all insights into a comprehensive travel plan that considers individual or group preferences.
        Instructions:
        Combine clusters, Personality traits, and travel radius analysis.
        Suggest an overarching travel strategy, including destinations, itineraries, and key experiences.
        Output Format:
        Augmented Insight: [Comprehensive insight integrating all analysis]  
        - Final Travel Plan: [Suggested destinations, itinerary, and experiences]  
        - Justification: [How this ties together the entire analysis]  


        Apart from the above response, create a merged, detailed summary combining all the information from the sections provided above into a single, cohesive document titled 'GLOBAL AUGMENT Note.' Ensure the summary captures key insights and organizes the content logically to provide a comprehensive overview of the context and data. This summary should serve as a robust foundation for designing a custom Retrieval-Augmented Generation (RAG) system for travel advisory purposes. Highlight connections, patterns, or potential applications of the summarized details that would enhance the RAG's functionality in delivering accurate and contextually relevant travel advisories.
        """

        sys_prompt = """You are an advanced AI model specializing in analyzing, clustering, and generating contextual insights from structured JSON data. Your goal is to deduce meaningful hypotheses, psychological traits, and actionable recommendations with a focus on travel and tour planning. Leverage embeddings and Personality-based reasoning to provide nuanced insights and enriched travel suggestions tailored to individual or group preferences."""

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
            stream=False
        )

        content = completion.choices[0].message.content
        logger.debug(f"OpenAI response: {content}")

        if '```json' in content:
            content = content.replace('```json', '')
        if '```' in content:
            content = content.replace('```', '')

        insights = [
            line.strip()
            for line in content.split('\n')
            if line.strip() and not line.strip().startswith('{') and not line.strip().startswith('}')
            and not line.strip().startswith('[') and not line.strip().startswith(']')
            and '"augmentedData"' not in line
        ]

        ordered_response = {
            "id": json_body["id"],
            "groupName": json_body["groupName"],
            "userName": json_body["userName"],
            "password": json_body["password"],
            "customers": json_body["customers"],
            "augmentedData": insights
        }

        logger.debug(f"Sending response: {ordered_response}")
        return jsonify(ordered_response)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

# For PythonAnywhere, we don't need the if __name__ == '__main__' block
