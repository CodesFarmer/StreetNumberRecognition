#### MTCNN TF Version  

*Here we try to reduce the time consuming during hand detection*  

For output net, we replace the original convolutional and pooling layer with mobilenet unit.  
Second, we will try to quantization the weights within the neural network to compress the model.  
  
##### Different Net Comparasion  
|  NeuralNetwork  |  Time(ms)  |  IoU  | IoU > 0.6 | IoU > 0.7 | IoU > 0.8 | IoU > 0.9 |
|:---------------:|:----------:|:-----:|:---------:|:---------:|:---------:|:---------:|
|  AlexNet        |   9.3      |78.07  |95.30%|84.85%|48.57%|24.24%|
|  MobileNet      |   4.4      |75.31  |93.46%|75.54%|35.51%|17.31%|
|  XCeption       |   4.5      |75.03  |91.80%|75.00%|34.50%|15.98%|
|  MN Distilling  |   3.9      |73.02  |87.64%|67.69%|32.24%|14.96%|
  
##### Different Hyper Parameters for MobileNet  
| LayerOrder | RegressionWeights | Minimum lr |hard samples| IoU | Time(ms) |
|:----------:|:-----------------:|:----------:|:-----------:|:---:|:--------:|
|CPMMPMFF|1.0|1e-6|0.0%|75.31|4.4|
|CPMMPMFF|2.0|1e-6|0.0%|77.23|4.4|
|CPMMPMFF|0.5|1e-6|0.0%|73.14|4.2|
|CPMPMMFF|1.0|1e-6|0.0%|75.75|4.3|
|CPMMPMFF|2.0|1e-6|25.0%|70.39|4.4|  
|CPMMPMFF|1.0|1e-6|50.0%|74.02|4.5|
|CPMMPMFF|1.0|1e-6|62.5%|74.91|4.4|
|CPMMPMFF|1.0|1e-6|75.0%|77.51|4.5|
|CPMMPMFF|2.0|1e-6|75.0%|77.61|4.5|
|CPMMPMFF|3.0|1e-6|75.0%|78.08|4.3|
|CPMPMMFF|2.0|1e-6|75.0%|76.52|4.3|
| ReLU   |3.0|1e-6|75.0%|77.82|7.7|
#### Tracking Net
We employ the output of ONet as the initialized input for tracking, and ONet is used for tracking  
Here we used it as baseline and compare it to our designed TNet  

|  NeuralNetwork  |  Time(ms)  |  IoU  | IoU > 0.6 | IoU > 0.7 | IoU > 0.8 | IoU > 0.9 |
|:---------------:|:----------:|:-----:|:---------:|:---------:|:---------:|:---------:|
| MobileNet       |  48.26     | 78.76 | 96.41%    |  83.99%   | 50.22%    |  26.59    |
| Tracking-ONet   | 3.6        |75.23  | 92.65%    |  71.72%   | 35.02%    |  16.11    |


