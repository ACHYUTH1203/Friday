from typing import TypedDict, List, Annotated, Optional, Dict, Any
class AgentState(TypedDict):
    """
    Represents the state of the Resume Parsing Agent.
    All fields are optional to handle missing data gracefully.
    Values are kept as simple strings or lists of strings.
    """
    file_path: Optional[Any]
    user_id: Optional[str]
    basic_info: Optional[Any]
    work_experience: Optional[Any]
    skills: Optional[Any]
    education: Optional[Any]
    mentioned_projects: Optional[Any]
    full_info:Optional[Any]

    user_query: str           
    target_role: Optional[str] 
    
    gap_analysis_report: Optional[str]
    hiring_tips: Optional[str]
    job_search_results: Optional[str]
    final_response: Optional[str]