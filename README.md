# DPNet: Dynamic Pyramid Neural Network for Salient Object Detection


## Abstract
Recent advances in salient object detection (SOD) achieve consistent performance gains that can be attributed to stronger multi-scale representation ability. We observed that in certain types of scenarios are best predicted with small kernel sizes and other scenarios better handled with large kernel sizes. However, most existing saliency models address this dilemma by extracting the multi-scale features from the encoder in a level-wise manner, and then averagely combining features with different scale of features for all input images without considering this differences, resulting in suboptimal performance. In this paper, we present a novel network design tailored for SOD, dubbed as dynamic pyramid neural network (DPNet), which generates eligible scale of features based on the input and adopts a weighted combination across these multi-scale features for the final prediction. Specifically, we design a dynamic pyramid convolution (DPConv), which allows to adaptively generate eligible features representation based on the input. DPConv can be used as a drop-in replacement for vanilla convolution layers in state-of-the-art saliency models, achieving consistent performance gains over the baseline model. Moreover, we propose a weighted bidirectional decoder to capture all useful information from all feature levels for SOD, avoiding arbitrarily feature combinations. Without any bells and whistles, DPNet outperforms various state-of-the-art salient object detection baselines on all tested datasets, making it a robust, general architecture that can be beneficial to future research in SOD.

## Network architecture



 <img src="./img/pipeline.jpg" width = "800" height = "400" alt="Framework" align=center />


	


## Saliency Maps

We provide the [saliency maps](https://pan.baidu.com/s/1M73-wrHnoFOaLhADjDjs4A) (Fetch Code: iirk) for comparisions,  including DUTS-OMRON, DUTS-TE, ECSSD, HKU-IS, PASCAL-S. 
To obtain the same score with our paper, we recommend the [evaluation code](https://github.com/ArcherFMY/sal_eval_toolbox) provided by Feng Mengyang.


| Backbone | # Params | #FLOPs | Saliency maps | Pre-trained model|
| :---: | :---: |  :---: |  :---: |  :---: |
| DPNet-50| 27.1M | 9.2G | [maps](https://pan.baidu.com/s/1M73-wrHnoFOaLhADjDjs4A) (Fetch Code: iirk) | [model](https://pan.baidu.com/s/1XUjvDCj-G6akY_rmbCOqfg) (Fetch Code: 6unj)| 
| DPNet-101| 44.7M | 12.6G |[maps](https://pan.baidu.com/s/1FzdrxWG9hE0svuVJabgH1g) (Fetch Code: izwv) |[model](https://pan.baidu.com/s/1yghNd3kyY-Fu6Ic80p5GIg) (Fetch Code: x8h4)| 
| DPNet-152| 59.1M | 16G | [maps](https://pan.baidu.com/s/17QJTNN53-xaxfI6ldZZFHA) (Fetch Code: xsx5)  | [model](https://pan.baidu.com/s/12cB6dVrneBifzxOdYYjaqA) (Fetch Code: vh5j)|


We also provid the saliency maps of DPNet-50 without pruning.

## SOC Saliency Maps
In the paper, we compare DPNet with 12 methods on SOC test set (1200 images). The SOC saliency maps of previous methods is borrowed from [SRCN project](https://github.com/wuzhe71/SCRN), including [DSS](https://openaccess.thecvf.com/content_cvpr_2017/papers/Hou_Deeply_Supervised_Salient_CVPR_2017_paper.pdf)、[NLDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Luo_Non-Local_Deep_Features_CVPR_2017_paper.pdf)、[SRM](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_A_Stagewise_Refinement_ICCV_2017_paper.pdf)、[Amulet](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Amulet_Aggregating_Multi-Level_ICCV_2017_paper.pdf)、[DGRL](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Detect_Globally_Refine_CVPR_2018_paper.pdf)、[BMPM](https://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Zhang_A_Bi-Directional_Message_CVPR_2018_paper.pdf)、[PiCANet-R](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_PiCANet_Learning_Pixel-Wise_CVPR_2018_paper.pdf)、[R3Net](https://www.ijcai.org/Proceedings/2018/0095.pdf)、[C2S-Net](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xin_Li_Contour_Knowledge_Transfer_ECCV_2018_paper.pdf)、[RANet](https://openaccess.thecvf.com/content_ECCV_2018/papers/Shuhan_Chen_Reverse_Attention_for_ECCV_2018_paper.pdf)、[CPD](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Cascaded_Partial_Decoder_for_Fast_and_Accurate_Salient_Object_Detection_CVPR_2019_paper.pdf)、[AFN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Feng_Attentive_Feedback_Network_for_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.pdf)、[BASNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.pdf)、[PoolNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_A_Simple_Pooling-Based_Design_for_Real-Time_Salient_Object_Detection_CVPR_2019_paper.pdf)、[SCRN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Stacked_Cross_Refinement_Network_for_Edge-Aware_Salient_Object_Detection_ICCV_2019_paper.pdf)、[SIBA](https://openaccess.thecvf.com/content_ICCV_2019/papers/Su_Selectivity_or_Invariance_Boundary-Aware_Salient_Object_Detection_ICCV_2019_paper.pdf)、[EGNet](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhao_EGNet_Edge_Guidance_Network_for_Salient_Object_Detection_ICCV_2019_paper.pdf)、[F3Net](https://aaai.org/ojs/index.php/AAAI/article/view/6916)、[GCPANet](https://aaai.org/ojs/index.php/AAAI/article/view/6633)、[MINet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_Multi-Scale_Interactive_Network_for_Salient_Object_Detection_CVPR_2020_paper.pdf).

Here, we also share [our SOC saliency maps](https://pan.baidu.com/s/1ogydA8WRLIIDlem_irNeDQ) (Fetch code:rnsm) for comparison. To obtain the same score with our paper, we recommend the [evaluation code](https://github.com/mczhuge/SOCToolbox) provided by Fan Dengping.

## Acknowledgement

Our work is based on [F3Net](https://github.com/weijun88/F3Net). We fully thank their open-sourced code.


