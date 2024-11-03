from flask import Flask, render_template, request, redirect, url_for, jsonify
from google.cloud import storage
import replicate
import os

app = Flask(__name__)


bucket_name = "store-ytes"
replicate.api_token = "r8_3XfTRKMjfKHUxcNLGEWkRAMzn7mHKMr3G0K4i"
service_account_path = "causal-destiny-425314-p0-4c33dc2ed62d.json"

# 預設的 LoRA 模型 ID
lora_model = "ostris/flux-dev-lora-trainer"  # LoRA 模型的 ID

@app.route('/')
def home():
    # 返回簡單的首頁，顯示歡迎消息
    return render_template('home.html')  # home.html 是首頁模板

@app.route('/train')
def train():
    return render_template('train.html')  # 确保渲染 train.html

@app.route('/index')
def index():
    return render_template('index.html')  # 确保渲染 index.html

def upload_to_gcs(file):
    client = storage.Client.from_service_account_json(service_account_path)
    bucket = client.bucket(bucket_name)
    
    file_name = "lewis.zip"
    blob = bucket.blob(file_name)
    blob.upload_from_file(file)
    
    return f"https://storage.googleapis.com/{bucket_name}/{file_name}"

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    trigger_word = request.form.get("triggerWord")
    autocaption_prefix = request.form.get("autocaptionPrefix")

    if file:
        file_url = upload_to_gcs(file)
        
        try:
            training = replicate.trainings.create(
                model="ostris/flux-dev-lora-trainer",
                version="e440909d3512c31646ee2e0c7d6f6f4923224863a6a10c494606e79fb5844497",
                destination="ytes10419/lewis",
                input={
                    "steps": 1000,
                    "lora_rank": 16,
                    "optimizer": "adamw8bit",
                    "batch_size": 1,
                    "resolution": "512,768,1024",
                    "autocaption": True,
                    "input_images": file_url,
                    "trigger_word": trigger_word,
                    "learning_rate": 0.0004,
                    "wandb_project": "flux_train_replicate",
                    "wandb_save_interval": 100,
                    "caption_dropout_rate": 0.05,
                    "cache_latents_to_disk": False,
                    "wandb_sample_interval": 100,
                }
            )
            return jsonify({"message": "訓練已啟動", "training_id": training.id})
        except Exception as e:
            return jsonify({"message": f"訓練啟動失敗: {str(e)}"}), 500
    else:
        return jsonify({"message": "請提供 ZIP 檔案"}), 400

@app.route('/check_status/<training_id>', methods=['GET'])
def check_status(training_id):
    try:
        training_status = replicate.trainings.get(training_id)
        return jsonify({"status": training_status.status, "trained_version": training_status.trained_version})
    except Exception as e:
        return jsonify({"message": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    # 從表單中獲取模型 ID
    model = request.form['version']  # 從前端的 version 值獲取模型 ID
    prompt = request.form['prompt']
    story = request.form['story']

    # 設置其他參數的默認值（若沒有提供的話）
    lora_scale = float(request.form.get('lora_scale', 0.5))  # 默認為 0.5
    guidance_scale = float(request.form.get('guidance_scale', 7.5))  # 默認為 7.5
    num_inference_steps = int(request.form.get('num_inference_steps', 50))  # 默認為 50
    num_outputs = int(request.form.get('num_outputs', 1))  # 默認為 1
    aspect_ratio = request.form.get('aspect_ratio', '1:1')  # 默認為 1:1

    print(f"Received prompt: {prompt}")
    print("Calling replicate.run...")

    try:
        # 呼叫 Replicate API 生成圖像
        output = replicate.run(
            model,  # 使用從前端獲取的模型 ID
            input={
                "prompt": prompt,
                "lora_scale": lora_scale,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "num_outputs": num_outputs,
                "aspect_ratio": aspect_ratio
            }
        )

        print(f"Generated image URL: {output}")
        
        # 將生成的圖像 URL 和故事描述傳遞到 result 頁面
        return redirect(url_for('result', image_url=output, story=story))

    except Exception as e:
        print(f"Error occurred: {e}")
        return "Error occurred while generating the image."

@app.route('/result')
def result():
    # 從 generate 路由中獲取圖像 URL 和故事描述
    image_url = request.args.get('image_url')
    story = request.args.get('story')
    
    # 渲染結果頁面並傳遞圖像和故事
    return render_template('result.html', image_url=image_url, story=story)

if __name__ == '__main__':
    app.run(debug=True)
