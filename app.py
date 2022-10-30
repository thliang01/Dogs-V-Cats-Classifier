from fastai.vision.all import *
import gradio as gr

def is_cat(x):
    return x[0].isupper()

learn = load_learner('model.pkl')

categories = ('Dog', 'Cat')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['Dog.jpg', 'cat.jpg', 'dunno.jpg']
title = "Dogs V Cats Classifier"
description = "A classifier trained on the Oxford Pets dataset with fastai. Created as a demo for Gradio and HuggingFace Spaces."
interpretation='default'
enable_queue=True

intf = gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    examples=examples,
    title=title,
    description=description,
    interpretation=interpretation,
    enable_queue=enable_queue
)

intf.launch(inline=False)