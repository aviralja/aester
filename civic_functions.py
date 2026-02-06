"""
Standalone Functions for Civic Issue Detection and AI Image Checking

This module provides two main functions:
1. detect_civic_issue_from_image() - Detects civic issues from an image file
2. check_if_ai_generated() - Checks if an image is AI-generated

Both functions can be used independently without the FastAPI framework.
"""

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
    issue_type: str = Field(description="Type of civic issue like pothole, garbage, water leakage")
    description: str = Field(description="Short description of the problem")


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
You are an AI vision system for civic issue detection.

TASK:
1. First determine whether the image clearly contains a civic issue.
2. If NO civic issue is visible, output:
   issue_type = "cant_find"
   description = "No civic issue detected"

3. If YES, choose ONLY ONE issue_type from the following allowed list:

Road & Transportation:
Potholes, Broken roads, Damaged footpaths, Speed breaker damage,
Road waterlogging, Traffic signal not working, Missing signboards,
Bus stop shelter damage

Sanitation & Waste Management:
Garbage overflow, Missed garbage collection, Littering on streets,
Open dumping, Public toilet dirty, Dead animal on road,
Drain cleaning needed

Water Supply & Sewerage:
Water pipe leakage, No water supply, Low water pressure,
Sewer overflow, Open manhole, Blocked drainage, Contaminated water

Electricity & Street Lighting:
Streetlight not working, Power outage, Fallen electric pole,
Exposed wires, Transformer issue

Public Health & Safety:
Stray animals, Mosquito breeding, Unsafe building, Fire hazard,
Chemical spill, Gas leak, Accident prone area

Environment & Parks:
Tree fallen, Illegal tree cutting, Park maintenance issue,
Air pollution, Water pollution, Noise pollution

Traffic & Law Enforcement:
Illegal parking, Signal jumping, Wrong side driving,
Encroachment on road, Overloaded vehicles

Public Infrastructure:
Broken benches, Damaged playground equipment,
Bus stop damaged, Broken railing, Public building maintenance

RULES:
- Never invent new issue types.
- Choose one main issue:
    Road & Transportation  
    Sanitation & Waste Management  
    Water Supply & Sewerage  
    Electricity & Street Lighting  
    Public Health & Safety  
    Environment & Parks  
    Traffic & Law Enforcement  
    Public Infrastructure  
- Choose the closest matching issue.

- Output must strictly follow the given JSON format.

{format_instructions}
"""
    ),
    (
        "human",
        [
            {"type": "text", "text": "Analyze this image and identify the civic issue if present."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}}
        ]
    )
])


# ============================================================================
# FUNCTION 1: DETECT CIVIC ISSUE FROM IMAGE
# ============================================================================
prompt2=ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an AI vision system for civic issue detection.

TASK:
1. Use BOTH the user description and the image.
2. Determine whether a civic issue is present.

IF NO civic issue is visible:
- issue_type = "cant_find"
- description = "No civic issue detected"

IF YES:
- Choose ONLY ONE issue_type from the allowed list.
- Write a short, clear description of the issue.

DESCRIPTION RULES:
- One sentence only.
- Describe what is wrong and where.
- Do NOT mention camera, image, or photo.

ALLOWED ISSUE TYPES:

Road & Transportation:
Potholes, Broken roads, Damaged footpaths, Speed breaker damage,
Road waterlogging, Traffic signal not working, Missing signboards,
Bus stop shelter damage

Sanitation & Waste Management:
Garbage overflow, Missed garbage collection, Littering on streets,
Open dumping, Public toilet dirty, Dead animal on road,
Drain cleaning needed

Water Supply & Sewerage:
Water pipe leakage, No water supply, Low water pressure,
Sewer overflow, Open manhole, Blocked drainage, Contaminated water

Electricity & Street Lighting:
Streetlight not working, Power outage, Fallen electric pole,
Exposed wires, Transformer issue

Public Health & Safety:
Stray animals, Mosquito breeding, Unsafe building, Fire hazard,
Chemical spill, Gas leak, Accident prone area

Environment & Parks:
Tree fallen, Illegal tree cutting, Park maintenance issue,
Air pollution, Water pollution, Noise pollution

Traffic & Law Enforcement:
Illegal parking, Signal jumping, Wrong side driving,
Encroachment on road, Overloaded vehicles

Public Infrastructure:
Broken benches, Damaged playground equipment,
Bus stop damaged, Broken railing, Public building maintenance

RULES:
- Never invent new issue types.
- Choose the closest match.
- Output must strictly follow JSON.

{format_instructions}
"""
    ),
    (
        "human",
        [
            {
                "type": "text",
                "text": "User description: {user_description}"
            },
            {
                "type": "text",
                "text": "Analyze the image and description together."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,{image_base64}"
                }
            }
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

def detect_civic_issue_from_image_description(image_path: str, user_description: str) -> CivicIssue:
    
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Read and convert image to base64
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create the processing chain
        chain = prompt2 | llm | parser
        
        # Invoke the chain with the image
        result = chain.invoke({
            "image_base64": image_base64,
            "user_description": user_description,
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

def detect_civic_issue_from_bytes_description(image_bytes: bytes, user_description: str) -> CivicIssue:
    """
    Detect civic issues from image bytes and user description.
    Used directly by FastAPI.
    """
    try:
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        chain = prompt2 | llm | parser

        result = chain.invoke({
            "image_base64": image_base64,
            "user_description": user_description,
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


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the functions
    """
    
    # Example image path - replace with your actual image
    image_path = "test_image.jpg"
    
    print("=" * 70)
    print("CIVIC ISSUE DETECTION & AI IMAGE CHECKER - STANDALONE FUNCTIONS")
    print("=" * 70)
    
    # Check if example image exists
    if not os.path.exists(image_path):
        print(f"\n⚠️  Example image not found: {image_path}")
        print("Please update the 'image_path' variable with an actual image file.")
    else:
        # Test 1: Detect Civic Issue
        print("\n1. DETECTING CIVIC ISSUE:")
        print("-" * 70)
        try:
            civic_result = detect_civic_issue_from_image(image_path)
            print(f"✅ Issue Type: {civic_result.issue_type}")
            print(f"✅ Description: {civic_result.description}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 2: Check if AI-Generated
        print("\n2. CHECKING IF AI-GENERATED:")
        print("-" * 70)
        try:
            is_ai, ai_details = check_if_ai_generated(image_path)
            confidence = ai_details.get("type", {}).get("ai_generated", 0)
            print(f"✅ Is AI Generated: {is_ai}")
            print(f"✅ Confidence Score: {confidence:.2%}")
            print(f"✅ API Status: {ai_details.get('status')}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 3: Complete Analysis
        print("\n3. COMPLETE ANALYSIS:")
        print("-" * 70)
        try:
            complete_result = analyze_image_complete(image_path)
            if complete_result["success"]:
                print(f"✅ Civic Issue: {complete_result['civic_issue']['issue_type']}")
                print(f"✅ Description: {complete_result['civic_issue']['description']}")
                print(f"✅ AI Generated: {complete_result['ai_detection']['is_ai_generated']}")
                print(f"✅ AI Confidence: {complete_result['ai_detection']['confidence']:.2%}")
            else:
                print(f"❌ Error: {complete_result['error']}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("FUNCTION SIGNATURES:")
    print("=" * 70)
    print("""
1. detect_civic_issue_from_image(image_path: str) -> CivicIssue
2. detect_civic_issue_from_bytes(image_bytes: bytes) -> CivicIssue
3. check_if_ai_generated(image_path: str, threshold: float = 0.5) -> Tuple[bool, Dict]
4. check_if_ai_generated_bytes(image_bytes: bytes, threshold: float = 0.5) -> Tuple[bool, Dict]
5. analyze_image_complete(image_path: str, ai_threshold: float = 0.5) -> Dict[str, Any]
    """)