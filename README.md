# datasets-utils

### Installation

```bash
git clone https://github.com/boomb0om/datasets-utils
cd datasets-utils/
pip install -r requirements.txt```
```

### Usage
---

Checkout [notebook](example/process_laion_example.ipynb) with basic usage on laion dataset

### API

#### Watermarks detection

Watermark classifier model. There are two classifiers: 
- resnext50_32x4d-small - fast watermark classifier
- resnext101_32x8d-large - slower but more accurate version

Model returns 1 if there is a watermark on image and 0 otherwise.

```python
from filters import ResnetWatermarksPredictor, get_watermarks_detection_model

device = torch.device("cuda:0")

wdetection_model = get_watermarks_detection_model(
    'resnext101_32x8d-large', # or 'resnext50_32x4d-small'
    device=device,
)

dataset_predictor = ResnetWatermarksPredictor(
    wdetection_model,
    save_parquets=True,
    save_parquets_dir='data/',
    device=device,
    workers=16,
    bs=32
)

df_result = dataset_predictor.run(
    task_name='watermarks_detection',
    files=files, # list of paths to images
)
```

#### CLIP predictor on list of labels

```python
from filters import CLIPPredictor
import clip

device = torch.device("cuda:0")

clip_model, clip_processor = clip.load(
    "ViT-L/14@336px", 
    device=device,
)

dataset_predictor = CLIPPredictor(
    clip_model=clip_model, clip_processor=clip_processor,
    save_parquets=True,
    save_parquets_dir='data/',
    device=device,
    workers=16,
    bs=32,
    templates=['{}']
)

labels_for_clip = [
    "picture has watermark",
    "slide of presentation with text",
    "document with text",
    "web site with text"
]

df_result = dataset_predictor.run(
    task_name='clip_labels',
    files=files, # list of paths to images
    labels=labels_for_clip
)
```

#### ruCLIP similarity between image and caption

```python
from filters import RuCLIPPredictor
import ruclip

device = torch.device("cuda:0")

ruclip, ruclip_processor = ruclip.load(
    'ruclip-vit-base-patch32-384', 
    device=device, 
)

dataset_predictor = RuCLIPPredictor(
    ruclip, ruclip_processor,
    save_parquets=True,
    save_parquets_dir='data/',
    device=device,
    workers=16,
    bs=32
)

df_result = dataset_predictor.run(
    task_name='ruclip_similarity',
    files=files, # list of paths to images
    texts=texts # list of captions to images
)
```

#### Text area on image

```python
from filters import FastCRAFTPredictor, get_text_detection_model

device = torch.device("cuda:0")

craft_model = get_text_detection_model(
    'CRAFT-MLT',
    device=device,
)
dataset_predictor = FastCRAFTPredictor(
    craft_model,
    save_parquets=True,
    save_parquets_dir='data/',
    device=device,
    workers=16,
    bs=32,
)

df_result = dataset_predictor.run(
    task_name='text_detection',
    files=files, # list of paths to images
)
```

