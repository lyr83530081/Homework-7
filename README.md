# Homework-7
U-Net for Semantic Segmentation of fisheye images

Train:
1. Create a folder 'data'
  put training data in 'data/rgb_images'
  put masks in 'data/semantic_annotations/gtLabels'
  put test data in 'data/test_set'
  
2. Create a folder 'checkpoints' to store weights

3. train in cmd: python -e epoch_num -b batch_size -l learning_rate -f weight.pth
    
    For example: python -e 100 -b 32 -l 0.0001 -f CP_Epoch100.pth
