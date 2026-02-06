

import base64
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from PIL import Image
import requests
from typing import Tuple, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()



# ============================================================================
# PYDANTIC MODEL
# ============================================================================

class CivicIssue(BaseModel):
    """Model for civic issue detection results"""
    valid: bool = Field(description="return true if image is valid else false")
    justification: str = Field(description="Short justification for the  validity")
    description: str = Field(description="Short description of the problem")
    severity:str = Field(description="Severity of the issue like low, medium, high")


# ============================================================================
# CONFIGURATION
# ============================================================================

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Sightengine Configuration
SIGHTENGINE_API_USER = os.getenv("SIGHTENGINE_API_USER")
SIGHTENGINE_API_SECRET = os.getenv("SIGHTENGINE_API_SECRET")
# ============================================================================
# INITIALIZE LLM AND PROMPT
# ============================================================================

# Initialize the OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=OPENAI_API_KEY
)

# Initialize the Pydantic parser
parser = PydanticOutputParser(pydantic_object=CivicIssue)

# Create the prompt template
prompt1 = ChatPromptTemplate.from_messages([
(
"system",
"""
You are an expert AI system for Sanitation & Waste Management issue analysis using images.

Your task is to analyze the provided image and return a structured JSON following exactly the given schema.

-----------------------------
OBJECTIVES
-----------------------------

1. Determine whether the image is a VALID sanitation-related civic issue.
2. If valid, describe what sanitation problem is visible.
3. Justify clearly why the image is valid or invalid.
4. Assign severity based on public health risk.

-----------------------------
VALIDITY RULES
-----------------------------

valid = True ONLY IF:
- Image clearly shows at least one sanitation issue such as:
  Missed garbage collection,
  Littering on streets,
  Open dumping,
  Public toilet dirty,
  Dead animal on road,
  Drain cleaning needed.

valid = False IF:
- Image is unrelated (people selfies, landscapes, random objects, or other problem except sanitation and waste detection)
- Image is too blurry or dark
- image dont show sanitation and waste detection issue
- No sanitation problem is visible

If valid = False:
- description = reason why valid is false
- severity = "low"

-----------------------------
SEVERITY DEFINITIONS
-----------------------------

LOW:
- Small litter
- Dry waste
- No insects, no decay, no standing water

MEDIUM:
- Overflowing garbage bins
- Dirty toilets
- Stagnant water
- Mild odor or insects

HIGH:
- Dead animals
- Medical or biological waste
- Decomposing garbage
- Sewage mixed with waste
- Heavy flies/rodents
- Situations likely to spread disease

-----------------------------
OUTPUT RULES
-----------------------------

Return ONLY valid JSON.
Do NOT include explanations outside JSON.
Follow exactly this schema:

{format_instructions}

-----------------------------
THINKING PROCESS
-----------------------------

1. Check if sanitation issue exists.
2. Decide valid true/false.
3. Identify main problem.
4. Evaluate health risk.
5. Generate short justification.
6. Assign severity.

Be concise and factual.
"""
),
(
"human",
[
 {"type": "text", "text": "Analyze this image."},
 {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}}
]
)
])



def detect_civic_issue_from_image(image_path: str) -> CivicIssue:
    
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Read and convert image to base64
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create the processing chain
        chain = prompt1 | llm | parser
        
        # Invoke the chain with the image
        result = chain.invoke({
            "image_base64": image_base64,
            "format_instructions": parser.get_format_instructions()
        })
        
        return result
        
    except Exception as e:
        raise Exception(f"Error detecting civic issue: {str(e)}")


def detect_civic_issue_from_bytes(image_bytes: bytes) -> CivicIssue:
    """
    Detect civic issues from image bytes.
    Used directly by FastAPI.
    """
    try:
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        chain = prompt1 | llm | parser

        result = chain.invoke({
            "image_base64": image_base64,
            "format_instructions": parser.get_format_instructions()
        })

        return result

    except Exception as e:
        raise Exception(f"Error detecting civic issue: {str(e)}")




def check_if_ai_generated(image_path: str, threshold: float = 0.5) -> Tuple[bool, Dict[str, Any]]:

    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Sightengine API endpoint
        url = "https://api.sightengine.com/1.0/check.json"
        
        # Make API request
        with open(image_path, "rb") as image_file:
            response = requests.post(
                url,
                files={"media": image_file},
                data={
                    "models": "genai",
                    "api_user": SIGHTENGINE_API_USER,
                    "api_secret": SIGHTENGINE_API_SECRET
                }
            )
        
        # Parse response
        result = response.json()
        
        # Check if API call was successful
        if result.get("status") == "success":
            # Get AI generation confidence score
            ai_score = result.get("type", {}).get("ai_generated", 0)
            
            # Determine if image is AI-generated based on threshold
            is_ai_generated = ai_score > threshold
            
            return is_ai_generated, result
        else:
            # API call failed
            error_message = result.get("error", {}).get("message", "Unknown error")
            raise Exception(f"Sightengine API error: {error_message}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error calling Sightengine API: {str(e)}")
    except Exception as e:
        raise Exception(f"Error checking if image is AI-generated: {str(e)}")


def check_if_ai_generated_bytes(
    image_bytes: bytes,
    threshold: float = 0.5
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if image bytes are AI-generated using Sightengine.
    """

    try:
        url = "https://api.sightengine.com/1.0/check.json"

        response = requests.post(
            url,
            files={"media": ("image.jpg", image_bytes, "image/jpeg")},
            data={
                "models": "genai",
                "api_user": SIGHTENGINE_API_USER,
                "api_secret": SIGHTENGINE_API_SECRET
            }
        )

        result = response.json()

        if result.get("status") == "success":
            ai_score = result.get("type", {}).get("ai_generated", 0)
            is_ai = ai_score > threshold
            return is_ai, result

        else:
            error_message = result.get("error", {}).get("message", "Unknown error")
            raise Exception(f"Sightengine API error: {error_message}")

    except Exception as e:
        raise Exception(f"Error checking AI image: {str(e)}")



# ============================================================================
# COMBINED ANALYSIS FUNCTION
# ============================================================================

def analyze_image_complete(image_path: str, ai_threshold: float = 0.5) -> Dict[str, Any]:
    
    
    try:
        # Detect civic issue
        civic_issue = detect_civic_issue_from_image(image_path)
        
        # Check if AI-generated
        is_ai, ai_result = check_if_ai_generated(image_path, ai_threshold)
        
        # Get confidence score
        confidence = ai_result.get("type", {}).get("ai_generated", 0) if ai_result.get("status") == "success" else 0
        
        # Combine results
        return {
            "success": True,
            "civic_issue": {
                "issue_type": civic_issue.issue_type,
                "description": civic_issue.description
            },
            "ai_detection": {
                "is_ai_generated": is_ai,
                "confidence": confidence,
                "status": ai_result.get("status", "unknown")
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


