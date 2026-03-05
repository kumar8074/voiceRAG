from typing import TypedDict, List, Any, Dict

class RAGState(TypedDict):
    query:        str
    user_id:      str
    session_id:   str
    db:           Any          

    # Populated by nodes
    is_general:   bool
    history:      List[Dict]
    contexts:     List[Dict]    # current best search results
    active_query: str           # original query or current sub-query batch (JSON array string on retries)
    retry_count:  int

    # Output — set by terminal nodes
    final_response: str

