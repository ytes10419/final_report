from flask import Flask, render_template, request, redirect, url_for
import replicate

app = Flask(__name__)

# 定义模型和版本
model = "ytes10419/lewis:a8750b2e4d3a0b29c1adeac3205506d9bc2ccf1da7759050d38796fd753ce5a9"  # 主模型 ID
lora_model = "ostris/flux-dev-lora-trainer"  # LoRA 模型的 ID

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    global model, lora_model  

    prompt = request.form['prompt']
    story = request.form['story']
    print(f"Received prompt: {prompt}")  # 打印提示词
    print("Calling replicate.run...")

    try:
        print("About to call replicate.run...")
        # 调用 Replicate 生成图像
        output = replicate.run(
            model,  # 使用主模型 ID
            input={
                "prompt": prompt,
                "lora_model": lora_model,  # 传入 LoRA 模型 ID 来调整输出
                "lora_scale": 1.0,  # 控制 LoRA 模型对输出的影响强度
                "guidance_scale": 7.5,  # 控制模型指导强度
                "num_inference_steps": 50,  # 推理步骤数
                "num_outputs": 1,  # 输出图片数量
                "aspect_ratio": "1:1",  # 图像的宽高比
            }
        )

        print(f"Generated image URL: {output}")  # 打印生成的图像 URL
        print("Replicate.run completed.")  # 打印完成信息
        
        # 传递生成的图像 URL 和故事描述到 result 页面
        return redirect(url_for('result', image_url=output, story=story))

    except Exception as e:
        print(f"Error occurred: {e}")  # 打印错误信息
        return "Error occurred while generating the image."

@app.route('/result')
def result():
    # 获取从 generate 路由传递过来的图像 URL 和故事描述
    image_url = request.args.get('image_url')
    story = request.args.get('story')
    
    # 渲染结果页面并传递图像和故事
    return render_template('result.html', image_url=image_url, story=story)

if __name__ == '__main__':
    app.run(debug=True)
