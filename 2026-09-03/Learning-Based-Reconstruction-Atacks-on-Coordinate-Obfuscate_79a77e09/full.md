# Learning-Based Reconstruction Atacks on Coordinate-Obfuscated Point Clouds

Mohammad Waquas

Usmani   
mohammadwaqu@umass.edu   
University of Massachusetts   
Amherst   
Massachusetts, USA   
Susmit Shannigrahi   
sshannigrahi@tntech.edu   
Tennessee Technological University Tennessee, USA   
Michael Zink   
zink@ecs.umass.edu   
University of Massachusetts   
Amherst   
Massachusetts, USA

## Abstract

Volumetric video based on point cloud representations enables immersive virtual and augmented reality applications but introduces significant challenges for eficient and secure content delivery. Prior work proposed a selective coordinate encryption framework for point clouds that encrypts only a subset of coordinates, reducing computational costs while visually degrading unauthorized content. However, it remains unclear whether the remaining unencrypted information is suficient to enable content reconstruction.

In this paper, we evaluate the robustness of selective coordinate encryption against machine learning-based reconstruction attacks. We consider an attacker with access to selectively encrypted point clouds attempting to recover encrypted coordinates without decryption by exploiting spatial and geometric correlations in the unencrypted data. We eval uate PointNet and Random Forest models under two encryption granularities: X, where all � coordinates are encrypted, and 2X, where every second � coordinate is encrypted

Our results show that reconstructing fully encrypted � coordinates remains challenging, whereas the 2X scheme leaks suficient information through neighboring coordinates to enable accurate reconstruction. These findings demonstrate that the security of selective coordinate encryption depends strongly on encryption granularity.

## Keywords

Machine Learning, Point Clouds, Volumetric Video, Selective Encryption, Coordinate Encryption, Reconstruction Attacks, Deep Learning, Digital Rights Management

## 1 Introduction

Volumetric video and point cloud content are increasingly important for immersive virtual reality, augmented reality, and collaborative 3D applications. Unlike traditional 2D video, point clouds consist of large collections of spatial coordinates and associated attributes describing a 3D scene. Their large data volume introduces substantial computational and bandwidth overheads, increasing latency and motion-to-photon delay, contributing to motion sickness [3].

Protecting volumetric content from unauthorized access and redistribution is another challenge. While transportlayer encryption secures communication channels, it does not provide eficient protection tailored for point cloud streaming. Although several studies have investigated encryption of individual point clouds [6], comparatively little attention has been given to securing volumetric video streams.

To address this gap, our prior work proposed a selective coordinate encryption framework for point cloud Digital Rights Management (DRM) [10]. Rather than encrypting entire point clouds, the framework encrypts only selected coordinates, reducing computational overhead while degrading the visual quality of unauthorized content. However, because only a subset of the point cloud information is encrypted, an important security question emerges: can an attacker reconstruct the missing encrypted coordinates using the remaining unencrypted, visible information?

Modern machine learning (ML) models can efectively learn spatial and geometric relationships in point clouds for tasks such as classification, segmentation, reconstruction, and completion [4, 5, 8, 9, 14, 16–19]. Consequently, selectively encrypted point clouds may leak suficient information through visible coordinates and attributes to enable recovery of protected coordinates without decryption.

In this work, we evaluate the robustness of selective coordinate encryption against ML-based reconstruction attacks. We consider an attacker with access to selectively encrypted point clouds who attempts to reconstruct encrypted coordinates by exploiting spatial and geometric relationships in the visible data. To this end, we evaluate PointNet [8] and Random Forest models using diferent combinations of spatial, geometric, and color features under two encryption granularities: X, where all � coordinates are encrypted, and 2X, where every second � coordinate is encrypted.

For the X scheme, reconstructing all encrypted � coordinates remains challenging, resulting in relatively high prediction errors and noticeable geometric deviations. Feature importance analysis further indicates that the surface normal component $n _ { x } .$ followed by the visible � and � coordinates, provides the most useful information for reconstruction.

In contrast, the 2X scheme leaks substantial information through neighboring visible � coordinates, enabling significantly accurate reconstruction of point cloud geometry. These findings show that the security of selective coordinate encryption depends critically on encryption granularity and highlight the importance of carefully balancing computational eficiency with resistance to reconstruction attacks.

## 2 Background and Related Work

3D Point Clouds: Point clouds represent 3D objects or scenes as collections of points with spatial coordinates (�, �, �) and optional attributes such as color or surface normals. They are commonly stored in the Polygon File Format (.PLY). A sequence of point cloud frames forms a volumetric video, enabling 6DoF immersive experiences.

Securing 3D Point Clouds: Numerous studies have investigated protecting point clouds using chaotic mapping, holographic methods, permutation and rotation-based encryption [6]. While these approaches provide strong protection for individual point clouds, they often target standalone content rather than real-time volumetric video streaming.

Our prior work [10] proposed a selective coordinate encryption framework for eficient point cloud video streaming. Rather than encrypting an entire point cloud, the framework selectively encrypts subsets of spatial coordinates while leaving the remaining coordinates, attributes, and metadata unencrypted, balancing visual distortion with computational eficiency. Encrypting only the � coordinate reduced encryption and decryption times by 37% and 46%, respectively, compared to full point cloud encryption.

This framework utilizes Attribute-Based Encryption [1], motivated by our prior streaming studies [11, 12]. Building upon this, ABE-VVS [13] integrated attribute-based selective coordinate encryption for volumetric video delivery, reducing server and cache CPU usage while improving client QoE.

However, the resilience of selective coordinate encryption against machine learning-based reconstruction attacks remains unexplored. In particular, it is unknown whether the remaining visible coordinates and attributes leak suficient information to recover encrypted geometry. This work addresses that gap by evaluating reconstruction attacks using modern machine learning models.

Machine Learning-based Reconstruction: Machine learning has been widely applied to point cloud understanding and geometric reconstruction. Comprehensive surveys of point cloud completion categorize existing approaches into traditional geometric methods and deep learning-based methods, including PointNet-based, GAN-based, Transformer-based, and point upsampling techniques [20].

PointNet and PointNet++ [8, 9] demonstrated the ability to directly learn spatial and geometric features from unordered point clouds, enabling accurate classification, segmentation, and object recognition. Subsequently, learning-based methods such as FoldingNet [17], PCN [19], Xie et al. [16], and PU-Net [18] extended these ideas to point cloud completion, reconstruction, and upsampling. More recently [14, 15] proposed a Random Forest-based point cloud upsampling framework integrated with selective coordinate encryption.

Unlike conventional point cloud completion, which reconstructs missing geometry from incomplete observations, our work investigates a reconstruction attack in which selectively encrypted coordinates are inferred from the remaining unencrypted information within a protected point cloud. To quantify the information leakage, we evaluate both Point-Net and Random Forest models for reconstructing encrypted coordinates from visible coordinates, neighborhood information, and other point attributes.

## 3 Methodology

## 3.1 Threat Model

Figure 1 illustrates the threat model. A content provider selectively encrypts point cloud coordinates before distributing the content through a CDN-based streaming infrastructure. Authorized users possessing valid decryption keys can decrypt the protected coordinates and reconstruct the original point cloud for volumetric video playback.

We consider an unauthorized attacker with access to selectively encrypted point cloud frames but without the required decryption keys. Because only a subset of spatial coordinates is encrypted, while the remaining coordinates, surface normals, color attributes, and metadata remain visible, the attacker exploits this information leakage to reconstruct the protected geometry using ML techniques. We assume the attacker knows the encryption granularity and methodology and has access to publicly available point cloud datasets for training ML model capable of learning spatial and geometric relationships between visible and encrypted coordinates.

Unlike traditional cryptographic attacks aimed at breaking the encryption scheme itself, the objective ofthe attacker in our model is to infer or predict the encrypted coordinate values using machine learning-based reconstruction techniques. By leveraging correlations among neighboring points, unencrypted coordinates, surface normals, and color attributes, the attacker attempts to reconstruct the original point cloud geometry and recover meaningful visual information without performing decryption. Our work evaluates whether selectively encrypted point clouds leak suficient information to enable such reconstruction attacks.

![](images/85e0be37020bbe4a18ff090c0e1d960863fb761a3388577f79c6a016d5330c1f.jpg)  
Figure 1: Illustration of the threat model.

## 3.2 Data Preparation

Before training, each point cloud frame is independently normalized to a unit sphere by centering it at its centroid and scaling by its maximum Euclidean distance from the centroid. The point cloud frames used in our dataset contain varying numbers of points. To provide a consistent input size for PointNet (Sect. 3.3), all frames are truncated to 100,000 points, corresponding to the smallest point count in the dataset. In contrast, the Random Forest model (Sect. 3.4) operates on individual point-level feature vectors and therefore utilizes all available points from each frame.

We evaluate two granularities of coordinate encryption: X, where the � coordinates of each point are encrypted, and 2X, where the � coordinate of every second point is encrypted, leaving the alternating � coordinates visible.

For the X scenario, reconstruction models use the visible coordinates (�, �) together with surface normals � = $( n _ { x } , n _ { y } , n _ { z } )$ , color attributes $C o l = \left( r , g , b \right)$ , and additional geometric descriptors including local surface curvature (����) and average neighborhood distance (��), computed using the � = 16 nearest neighbors.

For the 2X scenario, the partially observed coordinate sequence takes the form $X _ { 2 } = ( x _ { 1 } , 0 , x _ { 3 } , 0 , \dots )$ . Models use the same features as the X scenario, with additional configurations incorporating neighboring coordinate features �� and ��, representing the previous and next unencrypted � coordinates in the point cloud ordering.

## 3.3 PointNet Reconstruction Model

To evaluate the feasibility of reconstructing selectively encrypted coordinates using deep learning, we employ a PointNetbased regression architecture [8] that learns local and global geometric features directly from point cloud data.

Shared MLP layers with dimensions $d _ { i n }  6 4  6 4 $ $6 4 \to 1 2 8 \to 2 0 4 8$ , where $d _ { i n }$ denotes the input feature dimensionality, together with batch normalization and ReLU activations, extract point-level feature representations. To capture global geometric context, Symmetric max- and meanpooling aggregate the 2048-dimensional point features into a 4096-dimensional global representation, which is combined with 64-dimensional local features. The combined features are passed through regression layers of size 4160 → 512 → 256→1 to predict the encrypted � coordinates. The model contains about 2.55 million trainable parameters.

The model is trained using Mean Absolute Error (MAE) loss and the Adam optimizer (learning rate = 0.001, batch size = 8) for up to 100 epochs with early stopping (patience = 15). The model with the lowest training loss is used for evaluation on the held-out test set.

## 3.4 Random Forest Reconstruction Model

Furthermore, we evaluate Random Forest regression [2] as a traditional machine learning baseline. Random Forests construct an ensemble of decision trees and aggregate their predictions to model complex non-linear relationships between input features and encrypted coordinate values.

Unlike PointNet, which processes entire point cloud frames, Random Forest operates on individual point-level feature vectors, each consisting of an encrypted coordinate and its associated visible features. The model uses 100 decision trees with a maximum depth of 20 and bootstrap aggregation. Feature configurations vary with encryption granularity, and model is evaluated on same held-out test set as PointNet.

## 4 Evaluation

## 4.1 Point Cloud Dataset

The evaluation was conducted using the Open3D [7] Ofice and Living Room point cloud datasets. Each point cloud contains coordinates (�, �, �), surface normals (�), and color attributes (���). These datasets were selected as they represent realistic indoor scenes with complex geometric structures.

For all experiments, a subset of frames was reserved as a held-out test set and excluded from training. The Ofice dataset was split into 49 training and 4 test frames, while Living Room dataset was split into 53 training and 4 test frames. To simulate the X and 2X encryption schemes, and assuming the attacker knows the encryption granularity, encrypted � coordinates were replaced with zero as placeholders to preserve the point cloud format. Reconstruction performance was evaluated exclusively on the held-out test frames.

## 4.2 Evaluation Metrics

We evaluate reconstruction quality using following metrics:

Point-to-Point Mean Absolute Error (MAE): directly measures the absolute diference between reconstructed and original � coordinates. Since point correspondence is preserved, MAE is computed using the original point ordering.

Chamfer Distance (CD): Measures geometric similarity between reconstructed and original point clouds by computing the average nearest-neighbor distance in both directions. Lower CD values indicate higher geometric similarity.

HausdorfDistance (HD): Measures the maximum nearestneighbor distance between two point clouds, capturing the largest geometric deviation. HD is therefore particularly sensitive to outliers and localized reconstruction errors. Lower HD values indicate smaller worst-case geometric deviations.

Since all point clouds are normalized, MAE, CD, and HD are reported in normalized coordinate units.

Table 1: Diferent feature configurations evaluated
<table><tr><td colspan="2">X Encryption</td><td colspan="2">2X Encryption</td></tr><tr><td>Model</td><td>Features</td><td>Model</td><td>Features</td></tr><tr><td>M1</td><td> $\overline { { Y , Z + N + C o l + N D + C u r v } }$ </td><td>M1</td><td> $\overline { { { X _ { 2 } , Y , Z + N + C o l } } }$ </td></tr><tr><td>M2</td><td> $Y , Z + N + C o l + N D$ </td><td>M2</td><td> $X _ { 2 } , Y , Z + N$ </td></tr><tr><td>M3</td><td> $Y , Z + N + C o l$ </td><td>M3</td><td> $X _ { 2 } , Y , Z$ </td></tr><tr><td>M4</td><td> $Y , Z + N$ </td><td>M4</td><td> $P X + N X + Y , Z + N + C o l$ </td></tr><tr><td>M5</td><td> $Y , Z$ </td><td>M5</td><td> $P X + N X + Y , Z + N$ </td></tr><tr><td></td><td></td><td>M6</td><td> $P X + N X + Y , Z$ </td></tr><tr><td></td><td></td><td>M7</td><td> $P X + N X$ </td></tr></table>

## 4.3 Attack on X Encryption Granularity

We first evaluate the X encryption scenario, where the attacker attempts to reconstruct the � coordinate of every point using the remaining visible information. Table 1 summarizes the feature configurations used for the PointNet and Random Forest models. For comparison, we also include a Zeroed-X baseline, in which encrypted � coordinates are set to zero and no reconstruction is performed, providing a lower bound for evaluating reconstruction performance.

Point-to-Point Reconstruction Accuracy: Figure 2 (left) shows the average MAE of the reconstructed � coordinates across the test set with 95% confidence interval. PointNet and Random Forest achieve similar performance, with PointNet providing up to 4% lower MAE across most configurations. Models M3 and M4 achieve the lowest error, while M1 and M2 perform slightly worse, indicating that local curvature and neighborhood distance provide little benefit. M5 exhibits the highest error, suggesting that surface normals contribute useful information beyond the visible � and � coordinates.

Compared to the Zeroed-X baseline (MAE = 35.0%), both models reduce reconstruction error, with the best-performing PointNet M3, achieving an MAE of 27.9%. However, the error remains substantial, indicating that accurate reconstruction of fully encrypted � coordinates remains challenging.

Geometric Reconstruction Accuracy: Figure 2 (middle and right) presents the average CD and HD with 95% confidence intervals. Random Forest consistently achieves lower CD and HD than PointNet, indicating better geometric reconstruction. Consistent with the MAE results, M1 and M2 show no benefit from local curvature and neighborhood distance. Instead, M4, which uses only visible coordinates and surface normals, achieves the lowest CD and HD, suggesting that surface normals provide useful geometric information, while their removal in M5 produces the largest degradation.

Compared to the Zeroed-X baseline $( \mathrm { C D } = 0 . 4 2 9 5 , \mathrm { H D } =$ 0.8540), all models substantially improve geometric similarity, with Random Forest M4 achieving the lowest CD (0.11) and HD (0.52). Nevertheless, HD remains consistently higher than CD, reflecting localized reconstruction outliers due to its sensitivity to worst-case errors. Overall, although reconstruction improves geometric fidelity relative to the baseline, accurately recovering the original geometry remains challenging when all � coordinates are encrypted.

Impact of Feature Selection: Figure 4 (left) shows the Random Forest feature importance scores. The $Y , Z ,$ and $n _ { x }$ features contribute most to inferring hidden � coordinates, while the remaining normal components $( n _ { y } \mathrm { a n d } n _ { z } )$ , color attributes $( r , g , b )$ , local curvature, and neighborhood distance contribute considerably less.

## 4.4 Attack on 2X Encryption Granularity

In the 2X encryption scenario, every second � coordinate is hidden while neighboring � coordinates remain visible. This direct coordinate leakage through adjacent points may enable interpolation-based reconstruction attacks. Table 1 summarizes the evaluated feature configurations.

Models M1–M3 evaluate whether half of the unencrypted � coordinates (� ), with combinations of other coordinates, surface normals, and color attributes, provide suficient information to reconstruct encrypted � values. Models M4–M7 additionally incorporate neighboring visible coordinates, �� and ��, to evaluate impact of local coordinate leakage. The ���� and �� features are omitted because they provided little benefit in the X scenario. For comparison, we also include a Zeroed-2X baseline and a simple interpolation baseline.

Point-to-Point Reconstruction Accuracy: Figure 3 (left) presents the MAE results for the 2X encryption scenario. The Zeroed-2X baseline already exhibits substantially lower error than the X baseline (17.5% vs. 35.0%), since only half of the � coordinates are encrypted. Models M1–M3, which rely solely on the target point’s visible attributes, provide only modest improvements, with MAE values ranging from 12–18%. In contrast, Models M4–M7, which incorporate the neighboring visible coordinates �� and ��, achieve substantially lower errors. Simple interpolation using only �� and �� reduces the MAE to 4.1%, outperforming M1–M3 and achieving performance comparable to Random Forest M7 (4.2%), which also utilizes only �� and ��. However, the best-performing configuration, Random Forest M4, further reduces the MAE to approximately 3%, demonstrating that machine learning can exploit additional point attributes beyond simple interpolation to achieve more accurate reconstruction.

![](images/06cdce94b86e2126a623b5b5e32309548f5153438363bfb7dbb6099b4f129964.jpg)

![](images/a21adea611c382f90edef8c7d5611b5bf6bdba27810597b99d0ac25159e33e6f.jpg)

![](images/68ef2109a2135455e8a962402e0f2b93f1c1e21368923482c04d0f2e34d95600.jpg)  
Figure 2: Reconstruction performance for the X encryption attack across diferent models with 95% confidence interval.

![](images/c7d9bafe05b41fbedc51c58a1248db9d01847890497faa85115a586ec1c5f039.jpg)

![](images/f6c2aa89e4587087cf35455c737b5613255ad2a979785720872065049072bfe3.jpg)

![](images/9163de0e220534e82b65cec046a2ea7f15d7e0dba24bad53595f9b8a121e9587.jpg)

Figure 3: Reconstruction performance for the 2X encryption attack across diferent models with 95% confidence interval.  
![](images/1762b2c555430b44b30f7915eacfb347eec109945583ac1c3810459173bebfde.jpg)

![](images/6ace16482583045834e4dfb0d394a8827f2dfc6496c5008789463d85b594b837.jpg)

![](images/c058ff7a54974014e4fdd31de01f0a232d777c3c11f7cc46d1bdcbe342e78f97.jpg)  
Figure 4: Feature importance (%) reported by Random Forest models for X and 2X encryption granularities.

Geometric Reconstruction Accuracy: Figure 3 (middle and right) presents the CD and HD results. Consistent with the MAE results, Models M1–M3 provide only modest improvements over the baseline (CD = 0.04). In contrast, Models M4–M7 substantially reduce the CD by incorporating �� and ��. Simple interpolation achieves a CD of 0.013, whereas Random Forest M4 further reduces it to 0.009, showing that machine learning leverages additional point attributes beyond simple interpolation.

Interestingly, HD is higher than the baseline across most ML configurations and simple interpolation. Unlike CD, HD measures the maximum geometric deviation and is therefore highly sensitive to outliers. In the Zeroed-2X baseline, assign ing all encrypted coordinates to $X = 0$ produces a relatively uniform distortion. In contrast, reconstruction methods occasionally generate large errors for a small number of points, creating localized outliers that disproportionately increase the HD despite substantial improvements in MAE and CD.

Impact of Feature Selection: Figure 4 (middle) shows the feature importance for M1. Similar to the X reconstruction scenario, the � and � coordinates together with the normal $n _ { x }$ contribute most to reconstruction. The $X _ { 2 }$ feature exhibits negligible importance because reconstruction targets correspond to points whose � coordinates are encrypted and replaced with zeros, leaving no useful $X _ { 2 }$ information available at the target location.

Figure 4 (right) presents the feature importance for M4. Here, the neighboring visible coordinates �� and �� dominate all other features, while the contribution of remaining attributes becomes negligible. These results confirm that neighboring visible coordinates are the primary source of information leakage in the 2X encryption scenario.

## 5 Discussion and Conclusions

Our results show that reconstruction attacks depend strongly on encryption granularity. While reconstructing fully encrypted � coordinates remains dificult, the 2X scheme leaks suficient information through neighboring visible coordi nates to enable accurate reconstruction. Although simple interpolation provides a strong baseline, machine learning further exploits additional point attributes to improve reconstruction accuracy. Feature importance analysis further identifies neighboring coordinates and the $n _ { x }$ surface normal component as the primary sources of information leakage. These findings suggest that protecting additional coordinate dimensions or surface normals may further improve resistance against reconstruction attacks.

Several directions remain for future work. More advanced reconstruction models, such as PointNet++ [9], larger datasets, and object-specific datasets may further improve attack performance. Alternative attacks based on point cloud upsampling [14] may also be efective against the 2X scheme. Evaluating additional encryption granularities also remains an interesting future direction.

## 6 Acknowledgments

This work is supported by NSF under Grants 1901137, 2106463, 2319962, 2019163, and 2126148.

## References

[1] John Bethencourt, Amit Sahai, and Brent Waters. 2007. Ciphertext-Policy Attribute-Based Encryption. In 2007 IEEE Symposium on Security and Privacy (SP ’07). 321–334. doi:10.1109/SP.2007.11

[2] Leo Breiman. 2001. Random Forests. Mach. Learn. 45, 1 (Oct. 2001), 5–32. doi:10.1023/A:1010933404324

[3] Serhan Gül, Dimitri Podborski, Thomas Buchholz, Thomas Schierl, and Cornelius Hellge. 2020. Low-latency cloud-based volumetric video streaming using head motion prediction. In Proc. NOSSDAV ’20 (Istan bul, Turkey) (NOSSDAV’20). Association for Computing Machinery, New York, NY, USA, 27–33. doi:10.1145/3386290.3396933

[4] Ying Li, Lingfei Ma, Zilong Zhong, Fei Liu, Michael A. Chapman, Dongpu Cao, and Jonathan Li. 2021. Deep Learning for LiDAR Point Clouds in Autonomous Driving: A Review. IEEE Transactions on Neural Networks and Learning Systems 32, 8 (2021), 3412–3432. doi:10.1109 TNNLS.2020.3015992

[5] Daniel Maturana and Sebastian Scherer. 2015. VoxNet: A 3D Convolutional Neural Network for real-time object recognition. In 2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 922–928. doi:10.1109/IROS.2015.7353481

[6] Manal Mizher, Salam Hamdan, Manar Mizher, Ahmad A. Mazhar, Hani Mimi, and Abdullah Mubarak Heyari. 2023. Three Dimensional Objects Encryption Algorithms: A Review. In Proc. 2023 Int. Conf. Information Technology (ICIT). 435–440. doi:10.1109/ICIT58056.2023.10226134

[7] open3d.org. 2025. Dataset. https://www.open3d.org/docs/release tutorial/data/index.html

[8] Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. 2017. PointNet: Deep Learning on Point Sets for 3D Classification and Seg mentation. In Proceedings of the IEEE Conference on Computer Vision

and Pattern Recognition (CVPR).

[9] Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. 2017. PointNet++: deep hierarchical feature learning on point sets in a metric space. In Proceedings ofthe 31st International Conference on Neural Information Processing Systems (Long Beach, California, USA) (NIPS’17). Curran Associates Inc., Red Hook, NY, USA, 5105–5114.

[10] Mohammad Waquas Usmani, Susmit Shannigrahi, and Michael Zink. 2025. Lightweight DRM for Volumetric Point Clouds through Attribute-Based Selective Coordinate Encryption. In Proceedings of the Twenty-Sixth International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing (Rice University, Houston, TX, USA) (MobiHoc ’25). Association for Computing Machinery, New York, NY, USA, 456–461. doi:10.1145/3704413.3765298

[11] Mohammad Waquas Usmani, Susmit Shannigrahi, and Michael Zink. 2025. Secure the Stream, Not the Hosts: Attribute-Based Encryption for DRM Enabled Video Streaming. In Proceedings ofthe 16th ACMMultimedia Systems Conference (Stellenbosch, South Africa) (MMSys ’25). Association for Computing Machinery, New York, NY, USA, 190–200. doi:10.1145/3712676.3714450

[12] Mohammad Waquas Usmani, Susmit Shannigrahi, and Michael Zink. 2025. Securing Immersive 360 Video Streams through Attribute-Based Selective Encryption. arXiv:2505.04466 [cs.MM] https://arxiv.org/abs/ 2505.04466

[13] Mohammad Waquas Usmani, Susmit Shannigrahi, and Michael Zink. 2026. ABE-VVS: Attribute-Based Encrypted Volumetric Video Streaming. arXiv:2601.08987 [cs.CR] https://arxiv.org/abs/2601.08987

[14] Mohammad Waquas Usmani, Sankalpa Timilsina, Michael Zink, and Susmit Shannigrahi. 2025. Secure AI-Driven Super-Resolution for Real-Time Mixed Reality Applications. In 2025 International Symposium on Multimedia (ISM). 116–123. doi:10.1109/ISM66958.2025.00035

[15] Mohammad Waquas Usmani, Sankalpa Timilsina, Michael Zink, and Susmit Shannigrahi. 2026. Secure Point Cloud Streaming with Learning-Based Super-Resolution. International Journal of Semantic Computing (2026). doi:10.1142/S1793351X2644006X

[16] Jianwen Xie, Yifei Xu, Zilong Zheng, Song-Chun Zhu, and Ying Nian Wu. 2021. Generative PointNet: Deep Energy-Based Learning on Unordered Point Sets for 3D Generation, Reconstruction and Classifi cation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 14976–14985.

[17] Yaoqing Yang, Chen Feng, Yiru Shen, and Dong Tian. 2018. FoldingNet: Point Cloud Auto-Encoder via Deep Grid Deformation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

[18] Lequan Yu, Xianzhi Li, Chi-Wing Fu, Daniel Cohen-Or, and Pheng-Ann Heng. 2018. PU-Net: Point Cloud Upsampling Network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

[19] Wentao Yuan, Tejas Khot, David Held, Christoph Mertz, and Martial Hebert. 2018. PCN: Point Completion Network. In 2018 International Conference on 3D Vision (3DV). 728–737. doi:10.1109/3DV.2018.00088

[20] Zhiyun Zhuang, Zhiyang Zhi, Ting Han, Yiping Chen, Jun Chen, Cheng Wang, Ming Cheng, Xinchang Zhang, Nannan Qin, and Lingfei Ma. 2024. A Survey of Point Cloud Completion. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 17 (2024), 5691–5711. doi:10.1109/JSTARS.2024.3362476