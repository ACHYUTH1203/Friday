import json
import os
import pdfplumber
import io
import uuid
import logging
from datetime import datetime
from pymongo import MongoClient
from tavily import TavilyClient
from llm import generate_llm_response
from state import AgentState
from dotenv import load_dotenv
from typing import Optional, Any, List, Dict
import re
load_dotenv()

logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["friday-database"]
user_info_col = db["user_info"]
threads_col = db["threads"]

TAVILY_KEY = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_KEY)


class ResumeParser:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file
        self.full_text = self._extract_text_from_pdf()
        
        self.basic_info: Optional[Dict[str, Any]] = None
        self.work_experience: Optional[List[Dict[str, Any]]] = None
        self.skills: Optional[Dict[str, Any]] = None
        self.education: Optional[List[Dict[str, Any]]] = None
        self.mentioned_projects: Optional[List[Dict[str, Any]]] = None
        
        self.full_info: Optional[str] = None 

    def _extract_text_from_pdf(self):
        text_content = ""
        try:
            with pdfplumber.open(self.pdf_file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_content += extracted + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return ""
            
        return text_content.strip()
    
    def extract_details(self):
        
        system_prompt = """
        You are an expert Resume Parser. Your job is to extract structured information from resume text into a strict JSON format.

        ### RULES ###
        1. Return ONLY valid JSON.
        2. Do not infer or hallucinate information not present in the text.
        3. If a field is missing, use null (not "N/A" or "Not Mentioned").
        4. Dates must be in "YYYY-MM" format. If only the year is known, use "YYYY".
        5. Tech stack must be a list of strings, not a single comma-separated string.

        ### NEGATIVE EXAMPLE (DO NOT DO THIS) ###
        {
            "basic_info": {
                "phone": "Call me at 987-654-3210",
                "email": "N/A"
            },
            "work_experience": [
                {
                    "start_date": "Jan 20",
                    "end_date": "Current"
                }
            ],
            "skills": {
                "technical": "Python, Java, C++"
            }
        }

        ### POSITIVE EXAMPLE (DO THIS) ###
        {
            "basic_info": {
                "full_name": "John Doe",
                "email": "johndoe@example.com",
                "phone": "9876543210",
                "linkedin_url": "https://linkedin.com/in/johndoe",
                "github_url": "https://github.com/johndoe",
                "location": "New York, USA",
                "portfolio_url": null
            },
            "summary": "Experienced Backend Engineer specializing in scalable Python applications.",
            "work_experience": [
                {
                    "job_title": "Senior Software Engineer",
                    "company": "Tech Corp",
                    "start_date": "2020-01",
                    "end_date": "Present",
                    "description": "Led a team of 5 to migrate legacy monolith to Microservices."
                }
            ],
            "education": [
                {
                    "degree": "B.Tech Computer Science",
                    "university": "State University",
                    "graduation_year": "2019"
                }
            ],
            "skills": {
                "technical": ["Python", "FastAPI", "Docker", "PostgreSQL"],
                "soft": ["Team Leadership", "Agile Methodology"]
            },
            "projects": [
                {
                    "title": "E-Commerce Chatbot",
                    "description": "Built a chatbot using OpenAI API.",
                    "tech_stack": ["Python", "LangChain", "React"]
                }
            ]
        }
        """

        focus_prompt = f"""
        Here is the resume text content to process:
        
        {self.full_text}
        
        Extract the details following the Positive Example structure exactly.
        """

        try:
            response_str = generate_llm_response(
                system_prompt=system_prompt,
                focus_prompt=focus_prompt,
                temperature=0.0,
                json_mode=True
            )
            
            data = None
            if isinstance(response_str, str):
                data = json.loads(response_str)
            elif isinstance(response_str, dict):
                data = response_str
            
            if data:
                if data.get("basic_info"):
                    self.basic_info = data["basic_info"]
                    
                if data.get("work_experience"):
                    self.work_experience = data["work_experience"]
                    
                if data.get("skills"):
                    self.skills = data["skills"]
                    
                if data.get("education"):
                    self.education = data["education"]
                    
                if data.get("projects"):
                    self.mentioned_projects = data["projects"]

                self.full_info = json.dumps(data, indent=4)

            return data

        except Exception as e:
            logger.error(f"Error during LLM extraction: {e}")
            return None

def router_node(state: AgentState):
    query = state.get("user_query", "")
    
    system_prompt = """
    You are a routing classifier for a career assistant.

    Your task:
    Classify the user's query into EXACTLY ONE of the following categories.
    Return ONLY the category name. Do NOT explain your reasoning.

    Categories:

    1. gap_analysis
    - The user wants feedback on their resume or profile
    - The user asks what skills, projects, or experience they are missing
    - The user asks how to improve chances for a role
    - The user shares a resume, skills, or background and asks "what next"

    2. job_search
    - The user asks for job roles, job listings, or career options
    - The user asks "what jobs can I get", "which roles suit me"
    - The user asks about companies, openings, or where to apply
    - The user wants job recommendations based on skills or experience

    3. hiring_guide
    - The user asks about interviews, hiring process, or selection rounds
    - The user asks about salary, compensation, or career growth
    - The user asks for preparation tips, exam patterns, or interview questions
    - The user asks how hiring works at a company or role

    Rules:
    - Choose the single best category based on the PRIMARY intent.
    - If the query mentions multiple topics, pick the MOST dominant one.
    - If uncertain, choose gap_analysis.
    - Output ONLY one of:
    gap_analysis
    job_search
    hiring_guide
    """

    focus_prompt = f"User Query: {query}"
    
    route = generate_llm_response(
        system_prompt=system_prompt, 
        focus_prompt=focus_prompt, 
        temperature=0.0
    ).strip().lower()

    valid_routes = ["gap_analysis", "job_search", "hiring_guide"]
    if route not in valid_routes:
        return "gap_analysis"
        
    return route

def load_user_context_node(state: AgentState):
    logger.info("--- NODE: Load Context ---")
    
    current_user_id = state.get("user_id")
    if not current_user_id or current_user_id == "default_user":
        current_user_id = str(uuid.uuid4())
        logger.info(f"Generated new User ID: {current_user_id}")
    else:
        logger.info(f"Using existing User ID: {current_user_id}")

    user_doc = user_info_col.find_one({"user_id": current_user_id})
    resume_updates = {}
    
    if user_doc and "resume_data" in user_doc:
        data = user_doc["resume_data"]
        resume_updates = {
            "full_info": json.dumps(data),
            "skills": data.get("skills"),
            "work_experience": data.get("work_experience"),
            "mentioned_projects": data.get("projects"),
        }
        logger.info("Resume found in DB.")
    else:
        logger.warning("No resume found in DB.")

    thread_doc = threads_col.find_one({"user_id": current_user_id})
    history = thread_doc.get("history", []) if thread_doc else []
    
    recent_history = history[-10:] 
    
    return {
        **resume_updates, 
        "chat_history": recent_history,
        "user_id": current_user_id 
    }

def context_refinement_node(state: AgentState):
    logger.info("--- NODE: Context Refinement ---")
    
    user_query = state.get("user_query", "")
    history = state.get("chat_history", [])

    if not history:
        return {"refined_query": user_query}

    history_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history])

    system_prompt = """
    You are a Conversation Context Manager.
    Your goal is to output a "Refined Query" that is fully standalone.
    
    Rules:
    1. If the User Query refers to past context (e.g., "What about the second one?", "How do I learn that?"), rewrite it using the History to make it specific (e.g., "Tell me more about the Data Analyst job listed previously").
    2. If the User Query is already standalone (e.g., "Find python jobs"), return it exactly as is.
    3. Output ONLY the refined query string.
    """

    focus_prompt = f"""
    ### CHAT HISTORY ###
    {history_text}

    ### CURRENT USER QUERY ###
    {user_query}

    Refined Query:
    """

    refined_query = generate_llm_response(
        system_prompt=system_prompt,
        focus_prompt=focus_prompt,
        temperature=0.0
    )
    
    logger.debug(f"Original: {user_query} | Refined: {refined_query}")
    return {"refined_query": refined_query}


def resume_ingestion_node(state: AgentState):
    logger.info("--- NODE: Ingesting Resume ---")
    pdf_path = state.get("file_path")
    user_id = state.get("user_id", "default_user")
    
    if not pdf_path or not os.path.exists(pdf_path):
        return {"final_response": "Error: No resume file found."}

    parser = ResumeParser(pdf_path)
    extracted_data = parser.extract_details()
    
    if not extracted_data:
        return {"final_response": "Error: Failed to extract data."}

    try:
        user_info_col.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "resume_data": extracted_data,
                    "last_updated": datetime.utcnow()
                }
            },
            upsert=True
        )
        logger.info("Resume data saved to MongoDB.")
    except Exception as e:
        logger.error(f"DB Save Error: {e}")

    return extracted_data

def gap_analysis_node(state: AgentState):
    logger.info("--- NODE: Gap Analysis ---")
    
    skills_text = str(state.get("skills", {}))
    projects_text = str(state.get("mentioned_projects", []))
    
    system_prompt = """
    You are a Senior Technical Career Coach. 
    Analyze the candidate's skills and projects to identify "Resume Gaps" for a modern tech role.
    
    Output structured advice:
    1. **Missing Critical Skills**: What standard industry tools are missing? (e.g., specific clouds, CI/CD, testing).
    2. **Project Recommendations**: Suggest 2 complex "Capstone Projects" that would fill these gaps.
    """
    
    focus_prompt = f"""
    Candidate Skills: {skills_text}
    Candidate Projects: {projects_text}
    
    Perform the Gap Analysis.
    """

    analysis = generate_llm_response(
        system_prompt=system_prompt,
        focus_prompt=focus_prompt,
        temperature=0.3
    )
    
    return {"gap_analysis_report": analysis}

def hiring_guide_node(state: AgentState):
    logger.info("--- NODE: Hiring Guide ---")
    
    full_profile = state.get("full_info", "")
    
    system_prompt = """
    You are a Recruitment Specialist. Based on the candidate's profile, provide:
   
    1. **Job Platforms**: List 5 platforms best for their specific domain (e.g., LinkedIn, Wellfound for startups, Toptal, etc.).
    2. **Exams/Certifications**: Recommend 2 relevant exams (e.g., AWS Cloud Practitioner, Google Pro Developer) that add value to this specific profile.
    """
    
    focus_prompt = f"""
    Candidate Profile JSON:
    {full_profile}
    
    Generate the hiring guide.
    """

    guide = generate_llm_response(
        system_prompt=system_prompt,
        focus_prompt=focus_prompt,
        temperature=0.2
    )
    
    return {"hiring_tips": guide}

def job_search_node(state: AgentState):
    logger.info("--- NODE: Job Search (Tavily) ---")
    
    skills = state.get("skills", {}).get("technical", [])
    experience = state.get("work_experience", [])
    target_role = state.get("target_role", "Software Engineer")
    
    years_exp = len(experience) * 1.5
    seniority = "Junior" if years_exp < 3 else "Senior" if years_exp > 5 else "Mid-Level"

    query_gen_system = "You are a Search Query Optimizer. Output ONLY the raw search query string."
    
    query_gen_prompt = f"""
    Create a single, highly effective job search query for this candidate.
    
    Role: {target_role}
    Seniority Level: {seniority}
    Key Skills: {', '.join(skills[:5])}
    Location: Remote or India (preferred)
    
    Rules:
    - Include the tech stack (e.g., "Python", "Django").
    - Include "hiring" or "jobs".
    - Exclude generic terms.
    - Output ONLY the query (e.g., "Junior Python Django Developer jobs remote").
    """
    
    search_query = generate_llm_response(
        system_prompt=query_gen_system,
        focus_prompt=query_gen_prompt,
        temperature=0.0
    ).strip().replace('"', '')
    
    logger.info(f"Executing Search Query: {search_query}")

    formatted_results = ""
    
    try:
        tavily = TavilyClient(api_key=TAVILY_KEY)
        
        response = tavily.search(
            query=search_query,
            search_depth="advanced",
            topic="general", 
            max_results=5,
            include_domains=["linkedin.com", "wellfound.com", "ycombinator.com", "indeed.com", "naukri.com"],
        )
        
        if response.get('results'):
            formatted_results += f"### 🎯 Live Job Matches for: *{search_query}*\n\n"
            
            for job in response['results']:
                title = job.get('title', 'Unknown Role')
                url = job.get('url', '#')
                content = job.get('content', 'No description available.')[:2000]
                
                formatted_results += f"**[{title}]({url})**\n"
                formatted_results += f"> {content}...\n\n"
        else:
            formatted_results = f"No direct job listings found for query: {search_query}."

    except Exception as e:
        logger.error(f"Tavily Search Error: {e}")
        formatted_results = f"Error during job search: {str(e)}"
        
    return {
        "job_search_results": formatted_results
    }


def final_answer_node(state: AgentState):
    logger.info("--- NODE: Final Answer & Save ---")
    
    parts = []
    if "I need your resume" in str(state.get("gap_analysis_report")) or \
       "I need your resume" in str(state.get("job_search_results")):
       parts.append("Please upload your resume so I can help you with that!")
    else:
        if state.get("gap_analysis_report"): parts.append(f"### Gap Analysis\n{state['gap_analysis_report']}")
        if state.get("hiring_tips"): parts.append(f"### Hiring Guide\n{state['hiring_tips']}")
        if state.get("job_search_results"): parts.append(f"### Matching Jobs\n{state['job_search_results']}")
        if state.get("refined_query") and not parts:
            parts.append("I've processed your request. Is there anything specific about your resume you'd like to discuss?")

    final_text = "\n\n".join(parts)
    
    user_id = state.get("user_id", "default_user")
    user_q = state.get("user_query", "")
    
    new_messages = [
        {"role": "user", "content": user_q, "timestamp": datetime.utcnow()},
        {"role": "bot", "content": final_text, "timestamp": datetime.utcnow()}
    ]
    
    try:
        threads_col.update_one(
            {"user_id": user_id},
            {"$push": {"history": {"$each": new_messages}}},
            upsert=True
        )
        logger.info("Conversation saved to Threads.")
    except Exception as e:
        logger.error(f"Thread Save Error: {e}")

    return {"final_response": final_text}