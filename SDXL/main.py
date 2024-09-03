from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import datetime
import random
import boto3
import json
import base64

def claude_invoke_model(prompt, image_media_type=None, image_data_base64=None, model_params={}):
    
    llm = ChatBedrock(
        region_name='us-east-1',
        model_id='anthropic.claude-3-sonnet-20240229-v1:0',
        # model_id='anthropic.claude-3-haiku-20240307-v1:0',
    )

    # llm = ChatOpenAI(model="gpt-4o")

    messages = [
        # SystemMessage(content="ユーザーから与えられたプロンプトをSDXLで画像を生成するためのプロンプトに変換してください。"),
        HumanMessage(content=prompt),
    ]

    chain = llm | StrOutputParser()

    output = chain.invoke(messages)

    return output

def revise_prompt(original_prompt, claude_revise_params):
    input_prompt = f"""
Revise the following image generation prompt to optimize it for Stable Diffusion, incorporating best practices:
    {original_prompt}
    Please consider the following guidelines in your revision:
    1. Be specific and descriptive, using vivid adjectives and clear nouns.
    2. Include details about composition, lighting, style, and mood.
    3. Mention specific artists or art styles if relevant.
    4. Use keywords like "highly detailed", "4k", "8k", or "photorealistic" if appropriate.
    5. Separate different concepts with commas.
    6. Place more important elements at the beginning of the prompt.
    7. Use weights (e.g., (keyword:1.2)) for emphasizing certain elements if necessary.
    8. If the original prompt is not in English, translate it to English.
    9. please do not use the '(keyword:1.2)' format in the prompt.
    Your goal is to create a clear, detailed prompt that will result in a high-quality image generation with Stable Diffusion.
    Please provide your response in the following JSON format:
    {{"revised_prompt":"<Revised Prompt>"}}
    Ensure your response can be parsed as valid JSON. Do not include any explanations, comments, or additional text outside of the JSON structure.
"""

    output = claude_invoke_model(input_prompt,{})
    return output

def generate_image_from_prompt(prompt, revision_no ,model_params={}):
    seed = random.randint(0, 4294967295)

    body = {
        "text_prompts":[{"text": prompt},{"text": "low quality, blurry, poorly lit, unrefined, pixelated, distorted, unnatural, amateurish, overexposed, underexposed, noisy, grainy, unrealistic, poorly composed, unprofessional, lack of detail, lack of clarity, dull, boring, unappealing, unattractive"}],
        "cfg_scale": 30,
        "steps": 150,
        "seed": seed,
        "height": 1024,
        "width": 1024,
    }

    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'
    )

    response = bedrock.invoke_model(
        modelId="stability.stable-diffusion-xl-v1",
        body=json.dumps(body)
    )

    response_body = json.loads(response.get("body").read())

    base64_data = response_body.get("artifacts")[0]['base64']

    binary_data = base64.b64decode(base64_data)

    dt_str = revision_no
    with open(f"{dt_str}.png", "wb") as f:
        f.write(binary_data)

    print(f"Image generated successfully with seed: {seed}")
    return f"{dt_str}.png"

if __name__ == "__main__":
    prompt = "白人で金髪で目が大きな美しい女性のアップの画像"
    max_prompt_revisions = 2

    start_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"Original Prompt: {prompt}")

    for revision in range(max_prompt_revisions + 1):
        print(f"Prompt Revision {revision + 1}")
        prompt = revise_prompt(prompt, {})
        ## promptがjson形式で返ってくるので、jsonからpromptを取り出す
        prompt = json.loads(prompt)["revised_prompt"]
        print(f"Revised Prompt: {prompt}")

        revision_no = start_timestamp + f"_{revision + 1}"
        generate_image_from_prompt(prompt,revision_no ,{})

