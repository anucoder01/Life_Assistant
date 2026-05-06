# decision/decision_support.py
# ============================================================
# DECISION SUPPORT LAYER — Goes beyond Q&A to active planning.
#
# This layer detects when a query is asking for:
# - Study planning ("help me plan my week")
# - Task prioritization ("what should I do first?")
# - Habit tracking ("am I keeping up with my gym schedule?")
# - Conflict detection ("do I have anything at 3pm tomorrow?")
# - Goal progress ("how am I doing on my Python learning?")
#
# It enriches the query with structured analysis before
# passing to the LLM generator.
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List
from datetime import datetime


# Keywords that trigger decision support mode
DECISION_KEYWORDS = {
    "planning": ["plan", "schedule", "organize", "week", "today", "tomorrow",
                 "priority", "prioritize", "roadmap"],
    "task_management": ["task", "todo", "deadline", "finish", "complete",
                        "overdue", "backlog"],
    "habit_tracking": ["habit", "streak", "gym", "exercise", "study hours",
                       "keep up", "consistent"],
    "conflict_detection": ["conflict", "clash", "overlap", "free", "available",
                           "busy", "when can"],
    "goal_progress": ["progress", "goal", "target", "achieving", "how am i doing",
                      "on track"]
}


class DecisionSupport:
    """Enhances queries that need structured planning/decision help."""

    def __init__(self, username: str):
        self.username = username

    def detect_intent(self, query: str) -> str:
        """
        Detect if the query needs decision support and what type.

        Returns:
            One of: "planning", "task_management", "habit_tracking",
                    "conflict_detection", "goal_progress", or "general"
        """
        query_lower = query.lower()

        for intent, keywords in DECISION_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return intent

        return "general"

    def enrich_query(self, query: str, intent: str) -> str:
        """
        Add decision support instructions to the query based on intent.
        This tells the LLM to be more structured and actionable.
        """
        current_time = datetime.now().strftime("%A, %B %d %Y, %I:%M %p")

        enrichments = {
            "planning": f"""[Current date/time: {current_time}]
[Mode: Study/Schedule Planning]
Please help create a structured plan. Consider:
- What tasks are due soon
- Optimal time blocks for deep work
- Breaks and transitions
- Any conflicts in the schedule""",

            "task_management": f"""[Current date/time: {current_time}]
[Mode: Task Management]
Please analyze tasks and provide:
- Priority ranking (urgent vs important)
- Estimated time needed
- Suggested order of completion
- Any tasks that can be batched""",

            "habit_tracking": f"""[Current date/time: {current_time}]
[Mode: Habit Analysis]
Please analyze habit/routine data and provide:
- Consistency patterns
- Streaks or gaps
- Suggestions for improvement
- Encouragement where warranted""",

            "conflict_detection": f"""[Current date/time: {current_time}]
[Mode: Schedule Analysis]
Please check for:
- Time conflicts
- Overlapping commitments
- Available free windows
- Buffer time between tasks""",

            "goal_progress": f"""[Current date/time: {current_time}]
[Mode: Goal Tracking]
Please assess progress and provide:
- Current status toward goals
- What's on track vs lagging
- Recommended next actions
- Motivational insight if appropriate"""
        }

        enrichment = enrichments.get(intent, "")
        if enrichment:
            return f"{enrichment}\n\nUser query: {query}"
        return query

    def process(self, query: str) -> Dict:
        """
        Process a query through the decision support layer.

        Returns:
        {
            "enriched_query": "...",
            "intent": "planning",
            "needs_decision_support": True
        }
        """
        intent = self.detect_intent(query)
        needs_support = intent != "general"

        enriched_query = self.enrich_query(query, intent) if needs_support else query

        return {
            "enriched_query": enriched_query,
            "intent": intent,
            "needs_decision_support": needs_support
        }
