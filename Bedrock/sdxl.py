import boto3
import random
import json
import base64
from datetime import datetime

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

prompt = "A beautiful Japanese woman walking in Yokohama City. She is dressed in fashionable clothes, captured in a close-up shot under natural light. The image has a 32k resolution, cinematic composition, professional color grading, film grain, and an atmospheric feel"
style_preset = "photographic"
seed = random.randint(0, 4294967295)

## SD用のリクエストbodyを作成

response = bedrock.invoke_model(
    modelId="stability.stable-diffusion-xl-v1",
    body=json.dumps({
        "text_prompts":[{
            "text": prompt,
            "weight": 1.0
        },
        {
            "text": "ow quality, blurry, poorly lit, unrefined, pixelated, distorted, unnatural, amateurish, overexposed, underexposed, noisy, grainy, unrealistic, poorly composed, unprofessional, lack of detail, lack of clarity, dull, boring, unappealing, unattractive",
            "weight": 1.0
        }
        
        ],
        "samples": 1,
        "cfg_scale": 7,
        "seed": seed,
        "style_preset": style_preset,
        "steps": 30,
        "height": 512,
        "width": 512,
    })
)

# StreamingBodyからbodyコンテンツを取得し、JSONをPythonオブジェクトに変換
response_body = json.loads(response.get("body").read())

# Pythonオブジェクトから、Base64エンコードされた画像データを取り出し
base64_data = response_body.get("artifacts")[0]['base64']

# 画像データをBase64デコード
binary_data = base64.b64decode(base64_data)

# 現在時刻のファイル名でファイル保存
dt_str = str(datetime.now())
with open(f"{dt_str}.png", "wb") as f:
  f.write(binary_data)
