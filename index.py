import os

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI
from transformers import BlipProcessor, BlipForConditionalGeneration

load_dotenv()

# And the root-level secrets are also accessible as environment variables:
st.write(
    "Has environment variables been set:",
    os.environ["OPENAPI_KEY"] == st.secrets["OPENAPI_KEY"],
)

OPENAPI_KEY = os.getenv("OPENAPI_KEY")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large")


client = OpenAI(api_key=OPENAPI_KEY)

# File upload handling
uploaded_file = st.file_uploader(
    "Upload an image", type=['png', 'jpg', 'jpeg'])


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


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image')

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

    if st.button("Generate Social Media Post"):
        social_post = generate_social_media_post(
            conditional_caption, unconditional_caption, company_info, social_media_posts)
        st.write("Generated Social Media Post:", social_post)
