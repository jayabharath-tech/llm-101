"""
AI-Powered Story Generator

This application generates personalized stories based on user inputs using Groq's LLM.
"""
import os
import requests
import streamlit as st
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv, find_dotenv
from groq import Groq

# Load environment variables
_ = load_dotenv(find_dotenv(), override=True)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
TEXT_MODEL = "llama-3.3-70b-versatile"

# Stability AI configuration for image generation
# TODO: Purchase required
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
STABILITY_API_URL = "https://api.stability.ai/v2beta/stable-image/generate/sd3"

# Initialize session state
if 'story_generated' not in st.session_state:
    st.session_state.story_generated = False
    st.session_state.story_content = ""
    st.session_state.paragraphs = []
    st.session_state.images = []


def generate_story(genre, characters, num_paragraphs, writing_style):
    """Generate a story using Groq's LLM"""
    system_prompt = """You are a creative storyteller. Generate an engaging and coherent story based on the user's inputs.
    The story should be well-structured and follow the specified genre and style."""

    user_prompt = f"""Generate a {genre} story with the following details:
    - Number of paragraphs: {num_paragraphs}
    - Characters: {', '.join(characters) if isinstance(characters, list) else characters}
    - Writing Style: {writing_style}
    Make sure to:
    1. Create a compelling narrative with a clear beginning, middle, and end
    2. Include the specified characters in meaningful roles
    3. Follow the chosen genre's conventions
    4. Maintain the specified writing style throughout
    5. Format the output with clear paragraph breaks (use double newlines between paragraphs)
    """

    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
            stream=False # Todo: Set to True and handle the chunks/streams
        )
        # Extract the story content from the response
        if response.choices and len(response.choices) > 0:
            story = response.choices[0].message.content
            # Split the story into paragraphs
            paragraphs = [p.strip() for p in story.split('\n\n') if p.strip()]
            return story, paragraphs

        raise ValueError("No story was generated. Please try again.")

    except Exception as e:
        st.error(f"Error generating story: {str(e)}")
        return None, []


def generate_image(prompt, width=1024, height=1024):
    """Generate an image using Stability AI's SD3 model"""
    if not STABILITY_API_KEY:
        st.warning("Stability API key not found. Image generation will be skipped.")
        return None

    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*"
    }

    data = {
        "prompt": prompt,
        "output_format": "png",
        "model": "sd3",
        "width": width,
        "height": height,
        "samples": 1,
    }

    try:
        response = requests.post(
            STABILITY_API_URL,
            headers=headers,
            files={"none": ''},
            data=data,
            timeout=60
        )
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None


def generate_image_prompt(paragraph):
    """Generate a prompt for image generation based on story paragraph"""
    system_prompt = """You are a creative prompt engineer. Create a detailed, visual description 
    suitable for generating an image that captures the essence of the story paragraph.
    Focus on the key visual elements, characters, setting, and mood."""

    user_prompt = f"""Create a detailed, visual description based on this story paragraph:
    {paragraph}
    
    The description should be vivid and suitable for generating a high-quality image. 
    Focus on the key visual elements, characters, setting, and mood.
    Keep it under 100 words."""

    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=200,
            stream=False
        )

        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()

        return "A beautiful illustration of the story scene"

    except Exception as e:
        print(f"Error generating image prompt: {e}")
        return "A beautiful illustration of the story scene"


def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(
        page_title="AI Story Generator",
        page_icon="📖",
        layout="wide"
    )

    st.title("📖 AI Story Generator")
    st.markdown("Create your own personalized story with AI!")

    with st.sidebar:
        st.header("Story Settings")

        # Genre selection
        genre = st.selectbox(
            "Choose a genre",
            ["Fantasy", "Science Fiction", "Mystery", "Adventure", "Romance", "Horror", "Fairy Tale"]
        )

        # Character input
        characters = st.text_input(
            "Enter character names (comma separated)",
            "hero, villain, sidekick"
        )
        characters = [c.strip() for c in characters.split(",") if c.strip()]

        # Number of paragraphs
        num_paragraphs = st.slider("Number of paragraphs", 1, 10, 5)

        # Writing style
        writing_style = st.selectbox(
            "Writing Style",
            ["Descriptive", "Concise", "Poetic", "Humorous", "Dramatic", "Mysterious"]
        )

        # Generate button
        if st.button("Generate Story"):
            with st.spinner("Crafting your story..."):
                # Generate story
                story, paragraphs = generate_story(
                    genre, characters, num_paragraphs, writing_style
                )
                if story and paragraphs:
                    st.session_state.story_generated = True
                    st.session_state.story_content = story
                    st.session_state.paragraphs = paragraphs
                    st.session_state.images = []

                    # Generate images for each paragraph
                    if STABILITY_API_KEY:
                        with st.spinner("Generating images..."):
                            for i, para in enumerate(st.session_state.paragraphs):
                                image_prompt = generate_image_prompt(para)
                                image = generate_image(image_prompt)
                                if image:
                                    st.session_state.images.append(image)
                st.rerun()

    # Display the generated story
    if st.session_state.story_generated:
        st.markdown(f"**Genre:** {genre}  |  **Style:** {writing_style}")

        for i, para in enumerate(st.session_state.paragraphs):
            st.markdown(para)

            # Display corresponding image if available
            if i < len(st.session_state.images):
                st.image(
                    st.session_state.images[i],
                    use_container_width=True,
                    caption=f"Illustration for paragraph {i + 1}"
                )

            st.markdown("---")
    else:
        # Show instructions if no story generated yet
        st.markdown("""
        ### Welcome to the AI Story Generator!
        
        To get started:
        1. Select your preferred genre and writing style
        2. Enter character names (separated by commas)
        3. Choose the number of paragraphs
        4. Click "Generate Story"
        
        The AI will create a unique story just for you, complete with illustrations!
        """)

        # Example story preview
        with st.expander("See an example"):
            st.markdown("""
            **Genre:** Fantasy  
            **Characters:** Elara, King Aldric, Shadow Dragon  
            **Style:** Descriptive
            
            ---
            
            The ancient forest of Eldermere stretched endlessly before Elara, its towering trees whispering secrets of ages past. As the first light of dawn filtered through the dense canopy, she adjusted the leather strap of her satchel and took a deep breath of the crisp morning air. The map she had stolen from the royal archives had led her here, to the edge of the Forbidden Expanse, where legend spoke of a power that could save her dying kingdom...
            """)


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        st.error("Error: GROQ_API_KEY environment variable is not set. Please set it in your .env file.")
    else:
        main()
