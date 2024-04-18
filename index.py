import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import base64
from io import BytesIO
from openai import OpenAI
from transformers import BlipProcessor, BlipForConditionalGeneration

load_dotenv()


OPENAPI_KEY_VALUE = st.secrets.OPENAPI_KEY

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large")


client = OpenAI(api_key=OPENAPI_KEY_VALUE)

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #0F1958;
    }
    button[title="View fullscreen"]{
        visibility: hidden;
    }
    label[data-testid="stWidgetLabel"] {
        p{
            font-size: 24px;
            font-weight: bold;
            color: #FCB929;
        }
    }
    div.stFileUploader > div:first-child {
    font-size: 24px;
    font-weight: bold;
}
    </style>
    """,
    unsafe_allow_html=True,
)
st.image("titlepage.svg", use_column_width=True)
st.image("secondpage.svg", use_column_width=True)
# File upload handling
uploaded_file = st.file_uploader(
    "Upload your next social media post!", type=['png', 'jpg', 'jpeg'])


def generate_social_media_post(image_conditional_caption, image_unconditional_caption, company_info, social_media_posts):
    prompt = f"""Generate a creative social media post based on the following inputs:\nConditional Caption: {image_conditional_caption}\nUnconditional Caption: {
        image_unconditional_caption} \n Company Information: {company_info} \n Recent Social Media Posts: {social_media_posts} \n Just give the caption that I can post on social media sites please."""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a pro social media marketing caption generator. Create engaging and relevant posts."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-3.5-turbo",
    )
    return response.choices[0].message.content


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def evaluate_social_media_post(image, caption, company_information):
    base64_image = image_to_base64(image)
    prompt = f"""
    Given an image and its caption, evaluate the caption's effectiveness based on the following criteria:

    1. **Engagement Potential**: Consider factors such as the caption's ability to capture attention, provoke thought, or encourage interaction (likes, comments, shares). Assess whether the caption uses language that is likely to engage the target audience, including any use of humor, questions, or call-to-actions.

    2. **Alignment with Company Values**: Examine if the caption accurately reflects the company's values and branding. The company is committed to [describe company values briefly, e.g., sustainability, innovation, customer focus]. Determine if the caption supports these values, either directly through the content or indirectly through tone and approach.

    3. **Rating for the post**: Finally rate the post on a scale of 1 to 10.

    ### Image Description
    I am sending the image in the api call

    ### Caption
    {caption}

    ### Company Information
    {company_information}

    Please provide a detailed evaluation of the caption based on the above criteria, highlighting its strengths and suggesting any improvements if necessary.
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a pro social media marketing caption evaluator. Evaluate my caption."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')

    with col2:
        st.image(image, caption='Uploaded Image', width=300)

    with col3:
        st.write(' ')

    # Conditional image captioning
    text = "a photograph of"
    inputs = processor(image, text, return_tensors="pt")
    out = model.generate(**inputs)
    conditional_caption = processor.decode(out[0], skip_special_tokens=True)
    st.write("Conditional Caption:", conditional_caption)

    # Unconditional image captioning
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    unconditional_caption = processor.decode(out[0], skip_special_tokens=True)
    st.write("Unconditional Caption:", unconditional_caption)

    # Text input for company info and social media posts
    company_info = st.text_area(
        "Company Info", value="Example - Fenty Beauty by Rihanna was created with the promise of inclusion for all women. With an unmatched offering of shades and colors for ALL skin tones, you'll never look elsewhere for your beauty staples.")
    social_media_posts = st.text_area(
        "Recent Social Media Posts", value=""""Mirror mirror on the wall who's the baddest of them all..."
        "In the mood for sum soft smooth skin 😉 Grab a spoon for this #CookiesNClean Face Scrub..."
        "The cherry on top? #GlossBombHeat in 'Hot Cherry' 🔥🍒..."
        "All that glitters is gold ✨creating the perfect canvas for this gold AND bold look..."
        "Double the gloss double the glam 💦 Are you ready to #DoubleGloss fam?..."
        """)
    social_post = ""
    if st.button("Generate and Evaluate Social Media Post"):
        social_post = generate_social_media_post(
            conditional_caption, unconditional_caption, company_info, social_media_posts)
        post_evaluation = evaluate_social_media_post(
            image, social_post, company_info)
        st.write("Generated Social Media Post:", social_post)
        st.write("Social Media Post Evaluation:", post_evaluation)
