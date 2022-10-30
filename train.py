from fastai.vision.all import *
path = untar_data(URLs.PETS)
files = get_image_files(path/"images")
def is_cat(x): return x[0].isupper() 
dls = ImageDataLoaders.from_name_func(
    path,
    files,
    pat='(.+)_\d+.jpg',
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    label_func=is_cat, 
    item_tfms=Resize(192),
    batch_tfms=aug_transforms(size=224, min_scale=0.75))

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(1)
learn.path = Path('.')
learn.export('model.pkl')