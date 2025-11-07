"""
Story Quality Evaluation Module

This module provides functionality to evaluate the quality of generated stories
using a language model to assess relevance and coherence.
"""
from groq import Groq
import os
from typing import Dict, List, Tuple, Optional
import json
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())


def get_default_client():
    """Get a default Groq client instance."""
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


# Default model for evaluation
DEFAULT_EVALUATION_MODEL = "llama-3.3-70b-versatile"


def evaluate_story_quality(
        story: str,
        genre: str,
        characters: List[str],
        writing_style: str,
        evaluation_criteria: Optional[Dict] = None,
        client: Optional[Groq] = None,
        model: str = DEFAULT_EVALUATION_MODEL
) -> Tuple[bool, Dict]:
    """
    Evaluate the quality of a generated story using a language model.

    Args:
        story: The generated story to evaluate
        genre: The target genre of the story
        characters: List of characters that should be included
        writing_style: The expected writing style
        evaluation_criteria: Optional custom criteria for evaluation
        client: Optional Groq client instance (will use default if not provided)
        model: The model to use for evaluation (defaults to DEFAULT_EVALUATION_MODEL)

    Returns:
        Tuple of (is_acceptable: bool, evaluation: dict)
        where evaluation contains detailed feedback and scores
    """
    # Use provided client or create a default one
    if client is None:
        client = get_default_client()
    """
    Evaluate the quality of a generated story using a language model.

    Args:
        story: The generated story to evaluate
        genre: The target genre of the story
        characters: List of characters that should be included
        writing_style: The expected writing style
        evaluation_criteria: Optional custom criteria for evaluation

    Returns:
        Tuple of (is_acceptable: bool, evaluation: dict)
        where evaluation contains detailed feedback and scores
    """
    if evaluation_criteria is None:
        evaluation_criteria = {
            "relevance_to_genre": 0.25,
            "character_inclusion": 0.25,
            "writing_style_match": 0.2,
            "coherence": 0.15,
            "creativity": 0.15
        }

    # Prepare the evaluation prompt
    system_prompt = """You are a professional story editor with expertise in multiple genres. 
    Your task is to evaluate the quality of a generated story based on specific criteria.
    Be critical but fair in your assessment."""

    evaluation_guidelines = """
    Evaluate the story based on these criteria (score 1-5 for each):
    1. Relevance to Genre: Does the story match the specified genre?
    2. Character Inclusion: Are all specified characters included and relevant?
    3. Writing Style: Does it match the requested writing style?
    4. Coherence: Is the story logically consistent and easy to follow?
    5. Creativity: Is the story original and engaging?

    Return a JSON object with:
    - scores: object with criteria scores (1-5)
    - overall_score: average score (1-5)
    - feedback: brief explanation of the evaluation
    - is_acceptable: boolean indicating if the story meets minimum quality standards
    """

    user_prompt = f"""
    Please evaluate the following story:

    GENRE: {genre}
    CHARACTERS: {', '.join(characters)}
    WRITING STYLE: {writing_style}

    --- STORY ---
    {story}
    --- END STORY ---

    {evaluation_guidelines}

    Respond ONLY with a valid JSON object, no other text.
    """

    try:
        # Call the model for evaluation
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent evaluations
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        # Parse the response
        if response.choices and len(response.choices) > 0:
            evaluation = json.loads(response.choices[0].message.content)

            # Map between our criteria keys and the model's response keys
            criteria_mapping = {
                'relevance_to_genre': ['Relevance to Genre', 'relevance_to_genre', 'genre_relevance'],
                'character_inclusion': ['Character Inclusion', 'character_inclusion', 'character_presence'],
                'writing_style_match': ['Writing Style', 'writing_style_match', 'style_match'],
                'coherence': ['Coherence', 'coherence', 'narrative_flow'],
                'creativity': ['Creativity', 'creativity', 'originality']
            }
            
            # Calculate weighted score
            weighted_score = 0.0
            for criterion, weight in evaluation_criteria.items():
                # Try to find a matching score in the response using all possible key variations
                score = 0
                for possible_key in criteria_mapping.get(criterion, [criterion]):
                    score = evaluation.get('scores', {}).get(possible_key, 0)
                    if score:  # If we found a match, use it
                        break
                
                weighted_score += score * weight
                print(f"Criterion: {criterion}, Score: {score}, Weight: {weight}, Partial Weighted: {score * weight}")
            
            # Ensure the score is within valid range (1-5)
            weighted_score = max(1.0, min(5.0, weighted_score))
            
            # Get the model's decision (if available)
            model_decision = evaluation.get("is_acceptable", True)  # Default to True if not specified
            score_based_decision = weighted_score >= 3.0  # 3.0 out of 5.0 is the threshold
            
            # Final decision is a combination of model's decision and our score
            is_acceptable = model_decision and score_based_decision
            
            # Update the evaluation with our calculated score and decision details
            evaluation['weighted_score'] = weighted_score
            evaluation['evaluation_decision'] = {
                'model_decision': model_decision,
                'score_based_decision': score_based_decision,
                'final_decision': is_acceptable,
                'criteria_weights': evaluation_criteria
            }
            
            print(f"Final weighted score: {weighted_score}, Model decision: {model_decision}, Final decision: {is_acceptable}")

            print(is_acceptable, evaluation, model_decision, score_based_decision)

            return is_acceptable, evaluation

    except Exception as e:
        return False, {"error": f"Evaluation failed: {str(e)}"}


def is_story_acceptable(
        story: str,
        genre: str,
        characters: List[str],
        writing_style: str,
        client: Optional[Groq] = None,
        model: str = DEFAULT_EVALUATION_MODEL
) -> bool:
    """
    Simplified interface to just get a boolean result of story evaluation.

    Args:
        story: The generated story to evaluate
        genre: The target genre of the story
        characters: List of characters that should be included
        writing_style: The expected writing style

    Returns:
        bool: True if the story meets quality standards, False otherwise
    """
    is_acceptable, _ = evaluate_story_quality(
        story=story,
        genre=genre,
        characters=characters,
        writing_style=writing_style,
        client=client,
        model=model
    )
    return is_acceptable
