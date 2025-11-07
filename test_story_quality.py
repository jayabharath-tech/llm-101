"""
Story Quality Test Suite

This module contains test cases to evaluate the quality of stories generated
by the story generator using the story evaluator.

Note: These tests make actual API calls to the LLM and may incur costs.
"""
import unittest
import os
from story_generator import generate_story
from story_evaluator import evaluate_story_quality

# Skip tests if API key is not set
SKIP_TESTS = not bool(os.getenv("GROQ_API_KEY"))
skip_if_no_api_key = unittest.skipIf(SKIP_TESTS, "GROQ_API_KEY not set")

class TestStoryQuality(unittest.TestCase):
    """Test cases for story quality evaluation."""

    def setUp(self):
        """Set up test fixtures."""
        self.genre = "Fantasy"
        self.characters = ["Aria", "Kael", "Lyra"]
        self.num_paragraphs = 3
        self.writing_style = "Descriptive"

    @skip_if_no_api_key
    def test_story_meets_quality_standards(self):
        """Test that generated stories meet minimum quality standards."""
        
        # Generate the story
        story, paragraphs = generate_story(
            genre=self.genre,
            characters=self.characters,
            num_paragraphs=self.num_paragraphs,
            writing_style=self.writing_style
        )
        
        # Evaluate the story
        is_acceptable, evaluation = evaluate_story_quality(
            story=story,
            genre=self.genre,
            characters=self.characters,
            writing_style=self.writing_style
        )
        
        # Assert the story meets quality standards
        self.assertTrue(is_acceptable, "Story should meet minimum quality standards")
        self.assertGreaterEqual(
            evaluation.get('overall_score', 0), 
            3.0,
            "Overall score should be at least 3.0"
        )
        
        # Check that all required characters are included
        for character in self.characters:
            self.assertIn(
                character.lower(), 
                story.lower(), 
                f"Character {character} should be included in the story"
            )
    
    @skip_if_no_api_key
    def test_story_follows_genre_conventions(self):
        """Test that generated stories follow genre conventions."""
        
        # Generate and evaluate the story
        story, _ = generate_story(
            genre="Fantasy",
            characters=["hero", "dragon"],
            num_paragraphs=1,
            writing_style="Dramatic"
        )
        
        is_acceptable, evaluation = evaluate_story_quality(
            story=story,
            genre="Fantasy",
            characters=["hero", "dragon"],
            writing_style="Dramatic"
        )
        
        # Assert the story follows fantasy genre conventions
        self.assertTrue(is_acceptable, "Story should follow genre conventions")
        self.assertGreaterEqual(
            evaluation.get('scores', {}).get('Relevance to Genre', 0),
            3,
            "Genre relevance score should be at least 3"
        )

if __name__ == "__main__":
    if SKIP_TESTS:
        print("Warning: GROQ_API_KEY not set. Tests will be skipped.")
        print("Please set the GROQ_API_KEY environment variable to run the tests.")
    unittest.main()
