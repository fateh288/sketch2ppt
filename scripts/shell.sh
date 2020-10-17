python model_main_tf2.py --model_dir=models/my_efficientdet_d1_coco17_tpu-32 --pipeline_config_path=models/my_efficientdet_d1_coco17_tpu-32/pipeline.config

python exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/my_efficientdet_d1_coco17_tpu-32/pipeline.config --trained_checkpoint_dir ./models/my_efficientdet_d1_coco17_tpu-32 --output_directory /home/fs/Tensorflow/workspace/training_demo/models/my_efficientdet_d1_coco17_tpu-32/exported-models/my_model
