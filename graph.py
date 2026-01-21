import logging
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import (
    load_user_context_node,
    context_refinement_node,
    resume_ingestion_node,
    router_node,
    gap_analysis_node,
    hiring_guide_node,
    job_search_node,
    final_answer_node
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("LangGraph-Workflow")

logger.info("Initializing LangGraph workflow")

def custom_router_logic(state: AgentState):
    """
    Wrapper around the standard router to handle file uploads priority.
    """
    logger.info("Routing decision started")

    if state.get("file_path"):
        logger.info("File path detected → Routing to ingest_resume")
        return "ingest_resume"
    route = router_node(state)
    logger.info(f"NLP router selected route → {route}")

    return route

workflow = StateGraph(AgentState)
logger.info("StateGraph initialized")

logger.info("Registering nodes")

workflow.add_node("load_context", load_user_context_node)
workflow.add_node("refine_query", context_refinement_node)
workflow.add_node("ingest_resume", resume_ingestion_node)
workflow.add_node("gap_analysis", gap_analysis_node)
workflow.add_node("hiring_guide", hiring_guide_node)
workflow.add_node("job_search", job_search_node)
workflow.add_node("finalize", final_answer_node)

logger.info("Setting entry point → load_context")
workflow.set_entry_point("load_context")


logger.info("Adding standard edges")
workflow.add_edge("load_context", "refine_query")

logger.info("Adding conditional routing logic")

workflow.add_conditional_edges(
    "refine_query",
    custom_router_logic,
    {
        "ingest_resume": "ingest_resume",
        "gap_analysis": "gap_analysis",
        "hiring_guide": "hiring_guide",
        "job_search": "job_search",
        "comprehensive": "gap_analysis"
    }
)

logger.info("Adding convergence edges → finalize")

workflow.add_edge("ingest_resume", "finalize")
workflow.add_edge("gap_analysis", "finalize")
workflow.add_edge("hiring_guide", "finalize")
workflow.add_edge("job_search", "finalize")

logger.info("Adding END edge")
workflow.add_edge("finalize", END)

logger.info("Compiling workflow graph")
app = workflow.compile()

logger.info("LangGraph workflow compiled successfully")
