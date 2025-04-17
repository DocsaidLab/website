---
slug: pedestrian-paper-roadmap
title: Pedestrian Detection 技術地圖
authors: Z. Yuan
image: /img/2025/0414.jpg
tags: [pedestrian-detection]
description: 行人偵測的 40 篇論文導讀。
---

最近想做個相關的專案，先從文獻回顧做起吧。

我去翻了一下過去的論文，過去十幾年間大概有近千篇論文，這次我們先挑個幾篇來讀一讀，看看行人偵測的技術脈絡與發展。

<!-- truncate -->

## 第一章：馬路上的神經網路

> **入門必讀：了解行人偵測在自動駕駛或其他應用領域的核心議題與技術脈絡**

1. [**2021 - Design guidelines on deep learning–based pedestrian detection methods for supporting autonomous vehicles**]
   從自動駕駛應用出發，系統性歸納深度學習式行人偵測的方法、考量與設計準則，包括資料增強策略與網路架構選擇建議。

2. [**2022 - Performance evaluation of cnn-based pedestrian detectors for autonomous vehicles**]
   評估多種 CNN-based 行人偵測器在自動駕駛場景下的效能與時間複雜度，提供對模型選型與硬體部署的重要參考。

3. [**2022 - A robust pedestrian detection approach for autonomous vehicles**](https://arxiv.org/pdf/2210.10489)
   著眼於自動駕駛對準確率與可靠性的高要求，透過融合複數特徵層以及後處理策略，提出兼具精度與穩定度的偵測方法。

4. [**2021 - Deep neural network based vehicle and pedestrian detection for autonomous driving: A survey**]
   行人偵測與車輛偵測常被同時討論，本文從整合視角出發，系統回顧 DNN 技術在自動駕駛下的偵測方案，涵蓋多種卷積網路與先進架構。

5. [**2022 - Deep learning-based pedestrian detection in autonomous vehicles: Substantial issues and challenges**](https://www.mdpi.com/2079-9292/11/21/3551)
   聚焦深度學習在自動駕駛行人偵測上碰到的顯著挑戰，如夜間/惡劣天候、資料偏差、運算成本等，並總結未來可能的研究方向。

---

## 第二章：標註世界的基石

> **從經典 KITTI 到夜間、多光譜基準，這些 Dataset 與 Benchmark 塑造了行人偵測的研究生態**

6. [**2012 - Are we ready for autonomous driving? the kitti vision benchmark suite**](https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR2012/data/papers/424_O3C-04.pdf)
   KITTI 是自動駕駛場景的經典資料集，提供多感測器（RGB、LiDAR）輸入與完整標註，被廣泛用於行人偵測與車輛辨識評測。

7. [**2011 - Pedestrian detection: An evaluation of the state of the art**](https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/01Ped.pdf)
   早期經典綜述，針對當時各種傳統特徵（如 HOG、Haar、LBP）及機器學習技術進行評估與比較，至今仍是理解行人偵測歷程的必備讀物。

8. [**2017 - Citypersons: A diverse dataset for pedestrian detection**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_CityPersons_A_Diverse_CVPR_2017_paper.pdf)
   衍生自 Cityscapes，特別標註與設計以評估行人偵測，包含擁擠場景與各種遮蔽情形，成為研究深度學習行人偵測的重要基準。

9. [**2019 - Nightowls: A pedestrians at night dataset**](https://ora.ox.ac.uk/objects/uuid:48f374c8-eac3-4a98-8628-92039a76c17b/download_file?file_format=pdf&safe_filename=neumann18b.pdf&type_of_work=Conference+item)
   針對夜間行人打造的大型資料集，解決夜間行人能見度低、成像品質差的痛點，推動各種昏暗場景下的偵測技術演進。

10. [**2015 - Multispectral pedestrian detection: Benchmark dataset and baseline**](https://openaccess.thecvf.com/content_cvpr_2015/papers/Hwang_Multispectral_Pedestrian_Detection_2015_CVPR_paper.pdf)
    首度提出可見光 + 紅外線的多光譜 Benchmark，為研究者提供跨波段的行人偵測資料，奠定多光譜偵測評估體系的基礎。

---

## 第三章：多光譜的視線

> **可見光不足時，紅外線/熱成像融合成解方：行人偵測從單波段邁入多波段**

11. [**2018 - Multispectral pedestrian detection based on deep convolutional neural networks**]
    在可見光與紅外線上同時訓練深度卷積網路，結合多通道特徵學習來提升對低光與背光場景的偵測性能。

12. [**2019 - Benchmarking a large-scale fir dataset for on-road pedestrian detection**](https://thefoxofsky.github.io/files/dataset.pdf)
    提出大規模 FIR（熱紅外）資料集，並對多種偵測器進行性能評估，系統性說明在高溫或夜間偵測的挑戰。

13. [**2023 - Multispectral pedestrian detection via two-stream yolo with complementarity fusion for autonomous driving**](https://www.researchgate.net/profile/Chan-Hung-Tse-2/publication/372212101_Multispectral_Pedestrian_Detection_Via_Two-Stream_YOLO_With_Complementarity_Fusion_For_Autonomous_Driving/links/64ba0799c41fb852dd887152/Multispectral-Pedestrian-Detection-Via-Two-Stream-YOLO-With-Complementarity-Fusion-For-Autonomous-Driving.pdf)
    採用雙路 YOLO 模型分別處理 RGB 與熱成像，再在中後期進行特徵融合，實驗顯示可大幅提升在極端光源差異時的檢測率。

14. [**2020 - Attention based multi-layer fusion of multispectral images for pedestrian detection**](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9187824)
    將注意力機制（attention）引入多光譜融合，透過多層次特徵的引導式權重分配，有效強化行人邊緣與細節訊號。

15. [**2021 - Cross-modality fusion transformer for multispectral object detection**](https://arxiv.org/pdf/2111.00273)
    雖然標題泛指物件偵測，但文中針對行人（尤其夜間）特別做了深度實驗，利用 Transformer 在跨模態注意力機制的優勢，精準提取可見光與紅外互補資訊。

---

## 第四章：像素邊緣的旅人

> **多數城市環境下常見的難題：大範圍或局部遮蔽、遠距離行人縮小等挑戰**

16. [**2023 - Occlusion and multi-scale pedestrian detection a review**](https://www.sciencedirect.com/science/article/pii/S2590005623000437)
    系統梳理近年行人遮蔽與多尺度偵測研究方法，對於想了解這兩大難題的脈絡與解法趨勢相當受用。

17. [**2022 - Occlusion handling and multi-scale pedestrian detection based on deep learning: a review**](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9718221)
    重於深度學習方法之間的差異與挑戰，特別討論目標分割、關鍵點定位、姿勢估計等技術在遮蔽情況下的有效性。

18. [**2020 - Pedestrian detection: The elephant in the room**](https://d1wqtxts1xzle7.cloudfront.net/82239902/2003.08799v2-libre.pdf?1647444211=&response-content-disposition=inline%3B+filename%3DPedestrian_Detection_The_Elephant_In_The.pdf&Expires=1744617541&Signature=aoYOmdL4KXL2ldxXLTSI~wthVshxFwcwG8pWOnLeQ5S2iNus0fJjYpp7zmoPvqhWJF3QRHKkNoNjvETt5DT2wukKxy3UUwV7av66qMXxmQ6qj~5rCidGCLrGHG5SDeftbchdMhYhCFkwXFFjtV6yE3y0VNGg524BKMZ-p061syN~jDk-mBL1WYXSpJ13twbqnk4PHnu4EideOFSHXiHPF2ys0D8cS4LHxo9OrMuIhRPf7RpYDiV0fvK5XZFbs9GllaSk6NezHnmy0KDJW-WjxvAg61ySBW0UEvw1IQ0MzwYyDPQEuGyhD~ymnYHMWsrzZiNWsmH~1oyb7TY8JlUaXQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
    作者針對各資料集中「被忽略或難處理」的 occlusion case 做分析，直指當前演算法在嚴重遮擋下的痛點，呼籲社群重視此「房間裡的大象」。

19. [**2021 - Robust small-scale pedestrian detection with cued recall via memory learning**](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Robust_Small-Scale_Pedestrian_Detection_With_Cued_Recall_via_Memory_Learning_ICCV_2021_paper.pdf)
    面對遠距離或影像中極度縮小的行人，提出「記憶學習」機制，讓模型能重新提取被忽略的細微特徵，顯著增強偵測率。

20. [**2022 - Towards versatile pedestrian detector with multisensory-matching and multispectral recalling memory**](https://ojs.aaai.org/index.php/AAAI/article/view/20001)
    進一步結合多模態匹配與記憶網路，針對小目標與遮蔽同時應對，展現如何在單一模型下擴增偵測器的適用範圍。

---

## 第五章：域之裂縫

> **同一個偵測模型如何在日夜、紅外、低品質、不同城市場景都能穩健運作？**

21. [**2019 - Domain-adaptive pedestrian detection in thermal images**](https://www.amazon.science/publications/domain-adaptive-pedestrian-detection-in-thermal-images)
    以對抗學習將可見光資料的偵測特徵遷移到熱成像領域，無需大規模標註 IR 資料即可達到不錯的檢測效能。

22. [**2020 - Task-conditioned domain adaptation for pedestrian detection in thermal imagery**](https://www.researchgate.net/profile/Kieu-My/publication/343167450_Task-conditioned_Domain_Adaptation_for_Pedestrian_Detection_in_Thermal_Imagery/links/5f19f8a2a6fdcc9626ad1e77/Task-conditioned-Domain-Adaptation-for-Pedestrian-Detection-in-Thermal-Imagery.pdf)
    更進一步考慮任務需求，透過條件約束增強可見光到 IR 的跨域對齊效果，縮小日夜場景的性能落差。

23. [**2019 - Domain adaptation for privacy-preserving pedestrian detection in thermal imagery**](https://www.researchgate.net/profile/Kieu-My/publication/335603374_Domain_Adaptation_for_Privacy-Preserving_Pedestrian_Detection_in_Thermal_Imagery/links/5d95aeeb458515c1d38efa58/Domain-Adaptation-for-Privacy-Preserving-Pedestrian-Detection-in-Thermal-Imagery.pdf)
    以隱私保護為出發點，強調使用熱成像替代可見光，同時利用跨域學習避免標註成本，是領域自適應在實際應用的一大案例。

24. [**2019 - Unsupervised domain adaptation for multispectral pedestrian detection**](https://openaccess.thecvf.com/content_CVPRW_2019/papers/MULA/Guan_Unsupervised_Domain_Adaptation_for_Multispectral_Pedestrian_Detection_CVPRW_2019_paper.pdf)
    結合多光譜與無監督學習，自動對齊源域（可見光）與目標域（紅外）特徵分布，以最小的人工成本達成跨波段人形偵測。

25. [**2023 - Cross modality knowledge distillation for robust pedestrian detection in low light and adverse weather conditions**]
    以知識蒸餾（knowledge distillation）架構，讓紅外線路徑的模型教導可見光路徑，在夜間或強烈天候環境下依舊保持高偵測率。

---

## 第六章：霧中之影

> **霧霾、暴雨、昏暗街頭都是常見卻棘手的實際場景**

26. [**2019 - Deep learning approaches on pedestrian detection in hazy weather**](https://d1wqtxts1xzle7.cloudfront.net/64964371/Deep_Learning_Approaches_on_Pedestrian_Detection_in_Hazy_Weather-libre.pdf?1605650125=&response-content-disposition=inline%3B+filename%3DDeep_Learning_Approaches_on_Pedestrian_D.pdf&Expires=1744617719&Signature=Ew95oMy5WCXwuYQzPUw5pO1i5AbeervP5OaLKm1fQ4QZH4I9kxpwAGeUf5c3DwpzMwA5XQCHthwM9RZz9GaI~LTnRV5TUjevH4PNrRFuodW5aOOgmYhgcnln3XHY4~wF2eeZxTC7kigOz75hPeDFKwFoEnYYdcfXUV9S0BOAHNDaBbmzaXtWktDo-NQCYAFRJH~-oZILL5azgNRg7JGYFQXH111Z1rxVQcz92pn5S00h3cGfH9VD9VPD0K7biMkYfF45j7Kfz0~HcqsWPhL-mWouY-j2bLzK1-2D5u~tHwuSUXuls1ZGtppQDM1WMCVBfRWMHNdHOINX0vvyoHTUMw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
    提出針對霧霾環境的失真補償與深度學習架構，協助模型克服圖像清晰度下降的問題，有效提升在陰霾天氣的檢測表現。

27. [**2020 - Pedestrian detection in severe weather conditions**](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9044295)
    系統分析雨、雪、霧、低光源對偵測結果的影響，並提出多任務學習框架，利用輔助任務（如天氣分類、對比度校正）增強行人辨識。

28. [**2018 - Pedestrian detection at night time in fir domain: Comprehensive study about temperature and brightness and new benchmark**]
    聚焦夜晚熱成像（FIR）特性，提出溫度與亮度對比分析並釋出新基準，徹底釐清夜間環境下行人溫度梯度的偵測意涵。

29. [**2022 - Pedestrian detection at daytime and nighttime conditions based on yolo-v5**]
    雖基於 YOLO 架構，但全程針對行人偵測進行優化，並特別討論白天與夜晚在可見光下的資料差異，給出多種引導調參策略。

30. [**2023 - All-weather pedestrian detection based on double-stream multispectral network**](https://www.mdpi.com/2079-9292/12/10/2312)
    進一步深化至「全天候」場景，將可見光與紅外線整合為雙流式網路，在雨、霧、夜間等不利條件下顯示出高穩定度。

---

## 第七章：深度與雷射交響曲

> **在自動駕駛與機器人領域，除了相機外，LiDAR、深度攝影機同樣扮演關鍵角色**

31. [**2022 - Active pedestrian detection for excavator robots based on multi-sensor fusion**]
    針對工程機械的安全需求，整合 RGB 與感測器訊號，將「主動偵測」策略引入挖掘機系統，提升機器人週邊環境感知。

32. [**2022 - Pedestrian detection and tracking based on 2d lidar and rgb-d camera**]
    單純依靠 2D LiDAR 難以判斷行人輪廓，故引入 RGB-D 融合對應，以動態聚類方式辨識並追蹤行人，是室內外應用的參考示例。

33. [**2019 - An efficient 3d pedestrian detector with calibrated rgb camera and 3d lidar**]
    利用同步標定後的 RGB + 3D LiDAR 資料，提出可在空間中回歸立體包圍盒的偵測器，顯示多感測器能顯著降低誤檢與漏檢。

34. [**2022 - Lidar-based dense pedestrian detection and tracking**](https://www.mdpi.com/2076-3417/12/4/1799)
    專注於「密集行人」環境，透過高密度 LiDAR 點雲與前景估計模型結合，解決人群中互相遮擋與重疊的定位困境。

35. [**2018 - Robust camera lidar sensor fusion via deep gated information fusion network**](https://rasd3.github.io/assets/publications/2018_iv_robust_fusion/paper.pdf)
    將「深度閘門機制」應用於攝影機與雷射點雲的融合流程，依據場景動態自適應調整權重，顯示在夜間、高速移動等極端情況下的抗干擾能力。

---

## 第八章：像素叢林的行者

> **從交通要道到公共場所，行人偵測在監控場景中仍是主力需求**

36. [**2019 - Fast pedestrian detection in surveillance video based on soft target training of shallow random forest**](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8610075)
    強調監控影片的多幀冗餘性，提出「軟標籤訓練」與淺層隨機森林，加速監控端的行人偵測並保持足夠精度。

37. [**2019 - Pedestrian detection and behaviour characterization for video surveillance systems**]
    不只做偵測，還結合行為分析，透過局部動作特徵理解人群在監控影片中的互動模式，是監控應用的進一步延伸。

38. [**2020 - Pedestrian detection and tracking in video surveillance system: issues, comprehensive review, and challenges**](https://books.google.com.tw/books?hl=zh-TW&lr=&id=MJn8DwAAQBAJ&oi=fnd&pg=PA163&dq=Pedestrian+detection+and+tracking+in+video+surveillance+system:+issues,+comprehensive+review,+and+challenges&ots=51-oqHj4Pa&sig=gDtUli58fCA2bLvUQx9p4fB-oQY&redir_esc=y#v=onepage&q=Pedestrian%20detection%20and%20tracking%20in%20video%20surveillance%20system%3A%20issues%2C%20comprehensive%20review%2C%20and%20challenges&f=false)
    從大量監控場景難點（擁擠、部份遮蔽、光源變化）著手，對行人偵測與追蹤方法作全面討論並列舉未來挑戰。

39. [**2015 - Video processing algorithms for detection of pedestrians**](https://cmst.eu/wp-content/uploads/files/10.12921_cmst.2015.21.03.005_Piniarski.pdf)
    較早期的監控場景方法，多以背景分割、光流與傳統機器學習特徵為主，回顧這篇能了解深度模型普及前的基礎思路。

40. [**2021 - A smart surveillance system for pedestrian tracking and counting using template matching**](https://www.researchgate.net/profile/Ahmad-Jalal-9/publication/384675317_A_Smart_Surveillance_System_for_Pedestrian_Tracking_and_Counting_using_Template_Matching/links/67026403f599e0392fbc3885/A-Smart-Surveillance-System-for-Pedestrian-Tracking-and-Counting-using-Template-Matching.pdf)
    在姿勢或深度模型資源不足的情況下，嘗試以樣板匹配與輕量級特徵實現基本行人偵測與計數功能，展現監控端對即時性和硬體限制的考量。

---

## 結語

行人偵測是電腦視覺領域持續發展的重要課題，從最初的傳統影像特徵到今日的深度學習框架，再延伸到紅外、多光譜、LiDAR 等多種感測器，都為了在真實世界中辨識並追蹤人類行為。

在這張地圖裡，我們看到了：

- **多樣感知技術的融合**：從可見光到熱成像與點雲，充分利用不同感測器的互補性。
- **跨域與泛化的強化**：面對日夜差異、惡劣天候、不同場景，要做的並不只是增減資料，而是如何自動遷移與對齊特徵。
- **遮蔽與群體挑戰**：人群中重疊、人形被部份掩蓋，小目標在畫面中只剩下幾個像素等問題，永遠是行人偵測的難題。
- **多場景應用的擴張**：從自動駕駛、都市監控到機器人安全，行人偵測同時考驗即時性、穩定性與隱私安全。

如果你正打算進入這個領域，以上 40 篇論文提供了從入門到進階的關鍵方向。

行人偵測的未來，將不只侷限於單一波段或特定場景，也不斷帶動安全、交通、零售、機器人等多元領域的研究與發展。

期望這份導讀能協助你快速找到合適的參考與切入點，一起為更可靠、更智慧的行人偵測技術持續努力。
