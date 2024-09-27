
how_many=30000
ref_data="coco2014"
ref_dir="/path/to/coco/"
ref_type="val2014"
eval_res=256
batch_size=128
clip_model="ViT-G/14"

caption_file='assets/captions.txt'
fake_dir='eval_results/sdxl'

python3 evaluation.py --how_many $how_many --ref_data $ref_data --ref_dir $ref_dir --ref_type $ref_type --fake_dir $fake_dir --eval_res $eval_res --batch_size $batch_size --clip_model $clip_model
