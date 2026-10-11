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
import numpy as np
from embeddings.embedder import get_embedder
from config import INTENT_CLASSIFIER_THRESHOLD


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

    # Label descriptions and examples act as class prototypes in embedding
    # space. This gives semantic routing without training or shipping a model.
    INTENT_EXAMPLES = {
        "planning": ["Help me plan my week", "organize my study schedule", "make a roadmap"],
        "task_management": ["What task should I finish first?", "show my deadlines and todo list", "prioritize my backlog"],
        "habit_tracking": ["Am I consistent with exercise?", "review my study habit streak", "how often have I gone to the gym"],
        "conflict_detection": ["Do I have a meeting conflict?", "when am I free tomorrow", "check overlapping events"],
        "goal_progress": ["How am I progressing toward my goal?", "am I on track with my target", "review my learning progress"],
        "general": ["Explain a concept from my notes", "find information in my documents", "answer a general question"],
    }
    _intent_prototypes = None

    @classmethod
    def _get_intent_prototypes(cls):
        if cls._intent_prototypes is None:
            phrases, labels = [], []
            for label, examples in cls.INTENT_EXAMPLES.items():
                for example in examples:
                    phrases.append(example)
                    labels.append(label)
            vectors = get_embedder().embed(phrases)
            prototypes = {}
            for label in cls.INTENT_EXAMPLES:
                prototypes[label] = np.mean(
                    [v for v, item_label in zip(vectors, labels) if item_label == label],
                    axis=0,
                )
                prototypes[label] /= np.linalg.norm(prototypes[label]) + 1e-12
            cls._intent_prototypes = prototypes
        return cls._intent_prototypes

    def detect_intent(self, query: str) -> str:
        """
        Detect if the query needs decision support and what type.

        Returns:
            One of: "planning", "task_management", "habit_tracking",
                    "conflict_detection", "goal_progress", or "general"
        """
        query_lower = query.lower()

        # Semantic intent classification using the project's local
        # sentence-transformer. Fall back to the keyword rules if unavailable.
        confidence = 0.0
        try:
            vector = get_embedder().embed(query)
            scores = {
                label: float(np.dot(vector, prototype))
                for label, prototype in self._get_intent_prototypes().items()
            }
            intent = max(scores, key=scores.get)
            confidence = scores[intent]
            if confidence >= INTENT_CLASSIFIER_THRESHOLD:
                self._last_intent_confidence = confidence
                return intent
        except Exception as exc:
            print(f"Semantic intent classifier unavailable; using keyword rules: {exc}")

        for intent, keywords in DECISION_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                self._last_intent_confidence = confidence
                return intent

        self._last_intent_confidence = confidence
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
            "intent_confidence": round(getattr(self, "_last_intent_confidence", 0.0), 3),
            "needs_decision_support": needs_support
        }
