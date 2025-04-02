# COVID-MedNet-A-Novel-Framework-for-COVID-19-X-ray-Detection


Summary: The early and accurate detection of COVID-19 through medical imaging remains a critical challenge. In this study, we propose an innovative approach leveraging a modified ResNet-101 architecture for detecting COVID-19 from chest X-ray images. Our model integrates cutting-edge technologies such as Vision-Language Models (VLMs) and Large Language Models (LLMs) to not only classify X-ray images but also generate comprehensive medical reports. These models enhance diagnostic accuracy and provide a more interpretable and detailed understanding of the findings. The network architecture is fine-tuned with a custom classification layer, trained using augmented image datasets, and evaluated against real-world COVID-19 X-ray data. Additionally, we utilize advanced explainable AI (XAI) techniques to improve the transparency of model predictions. The proposed method achieves high classification accuracy, and the integration of LLMs and VLMs facilitates the generation of textual medical insights, bridging the gap between image classification and clinical reporting. This dual capability offers a promising solution for the automatic and interpretable detection of COVID-19 in healthcare settings, enhancing both clinical decision-making and patient outcomes.

INTRODUCTION

COVID-19, caused by the SARS-CoV-2 virus, emerged as a global pandemic that drastically altered the world’s healthcare systems, economies, and daily life. First identified in December 2019 in Wuhan, China, the virus quickly spread worldwide, leading to millions of confirmed cases and fatalities (Sawicka, et al., 2022). Its rapid transmission and varied clinical presentations posed significant challenges to healthcare systems and research communities. The pandemic has underscored the importance of rapid diagnostic techniques to control the spread of the virus and minimize its impact on global health (Gilliland et al., 2024). Early detection plays a crucial role in preventing widespread transmission, and as such, diagnostic tools for COVID-19 have become essential for timely intervention and treatment. Among the various diagnostic methods, Polymerase Chain Reaction (PCR) testing has been the gold standard for confirming COVID-19 infections. However, while PCR tests are highly accurate, they are time-consuming and require significant resources. This has led to a global push for alternative methods to detect the virus more efficiently and at scale.
One such alternative method that has gained attention is the use of chest X-ray (CXR) imaging, which is commonly employed in diagnosing pneumonia and other respiratory conditions. Chest X-rays offer several advantages, including being non-invasive, relatively inexpensive, and faster than PCR tests, making them ideal for use in both clinical and resource-limited settings. The use of CXR imaging has proven effective in detecting characteristic patterns of pneumonia associated with COVID-19, such as bilateral infiltrates and ground-glass opacities. Research has shown that certain radiological features present in X-ray images can be highly indicative of COVID-19, helping clinicians make quicker decisions (Simpson et al., 2020). Despite its potential, interpreting chest X-rays manually can be challenging, particularly when large volumes of images must be reviewed in a short period. This has prompted the development of automated systems that leverage artificial intelligence (AI) to assist in image interpretation, improving both speed and accuracy.
Artificial intelligence, specifically deep learning, has shown remarkable promise in medical imaging, particularly in the automated detection of diseases from radiological images. Convolutional Neural Networks (CNNs), a type of deep learning model, have been widely applied in medical image analysis and have demonstrated success in classifying chest X-ray images to detect COVID-19 (Rajpurkar et al., 2017). CNNs, such as ResNet, have been shown to excel in tasks requiring image recognition, including the classification of pneumonia and COVID-19. The ResNet architecture, with its deep layers and residual connections, has been particularly useful for detecting subtle patterns in X-ray images, enabling high levels of diagnostic accuracy. Studies have demonstrated the efficacy of ResNet-based models for COVID-19 detection from chest X-rays, and their ability to learn from large datasets has further enhanced their performance (He et al., 2016). ResNet-101, a deeper variant of the ResNet architecture, has been applied in numerous studies and has proven effective in classifying complex image datasets, including those related to COVID-19 (Zhou et al., 2020).
While the accuracy of deep learning models such as ResNet-101 is promising, the lack of interpretability remains a major barrier to their widespread clinical adoption. The “black-box” nature of these models makes it difficult for healthcare professionals to trust and understand the reasoning behind predictions, which is crucial in medical applications. This challenge has led to the development of Explainable AI (XAI) techniques, which aim to provide transparency into how models make their decisions. Methods such as Gradient-weighted Class Activation Mapping (Grad-CAM) and Local Interpretable Model-agnostic Explanations (LIME) are increasingly used to interpret the decisions made by deep learning models (Selvaraju et al., 2017; Ribeiro et al., 2016). These techniques allow clinicians to visualize the regions of interest in X-ray images that contribute to a model’s decision, thereby enhancing trust and aiding in clinical decision-making.
In addition to traditional image analysis, the integration of Vision-Language Models (VLMs) and Large Language Models (LLMs) has the potential to further enhance COVID-19 diagnostic workflows. VLMs are capable of processing both visual data (in example: X-ray images) and textual data (e.g., medical reports), enabling the generation of context-aware descriptions of medical images. This can provide clinicians with a more comprehensive understanding of the diagnostic findings, which can be essential for treatment decisions. By combining VLMs with deep learning-based image classification, it is possible to generate detailed and accurate reports that describe the observed radiological features, such as the presence of ground-glass opacities or lung consolidation. Moreover, LLMs, such as GPT-3, have shown promise in generating medically relevant text, further enhancing the capabilities of AI systems in clinical settings (Brown et al., 2020). These models can generate medical reports that are coherent, contextually relevant, and linguistically appropriate, providing healthcare professionals with essential insights for patient care.
The combination of deep learning, XAI, VLMs, and LLMs represents an innovative approach to improving the detection and diagnosis of COVID-19. These technologies not only enhance diagnostic accuracy but also facilitate the generation of comprehensive, interpretable, and context-aware medical reports. The integration of these AI techniques into the diagnostic process holds the potential to transform healthcare by providing timely and accurate information to clinicians, improving patient outcomes, and enabling more efficient healthcare delivery. As AI and machine learning continue to evolve, their role in detecting COVID-19 and other infectious diseases will undoubtedly expand, providing new solutions to global health challenges.

THEORETICAL BACKGROUND AND RELATED WORK

Pneumonia is an infection that inflames the air sacs in one or both lungs. These air sacs, called alveoli, can fill with fluid or pus, causing symptoms like cough with phlegm or pus, fever, chills, and difficulty breathing. Pneumonia can be caused by various infectious agents, including bacteria, viruses, fungi, or other microorganisms. Pneumonia is an acute infection affecting the lung tissue caused by various pathogens, excluding bronchiolitis. Defining pneumonia as a distinct set of co-infections with unique traits is an aspirational goal but is presently of limited practicality due to challenges in identifying the specific causative organisms in individuals. However, while the advantages of categorizing pneumonia into more specific and uniform subtypes are noteworthy, they should be cautiously weighed when formulating research studies. It is feasible to explore more uniform groupings of pneumonia, which holds the potential for faster progress in this field (Mackenzie, 2016).
Bacterial Pneumonia it's caused by bacteria such as Streptococcus pneumoniae and Haemophilus influenzae. It can range from mild to severe and often occurs after a bout of respiratory illness or when the immune system is weakened.
Viral Pneumonia can be caused by viruses, like influenza (flu) viruses, respiratory syncytial virus (RSV), or the COVID-19 virus (causing SARS-CoV-2), can lead to viral pneumonia. It tends to be less severe than bacterial pneumonia in healthy individuals but can be serious in older adults or those with weakened immune systems. Over 20 viruses have been associated with community-acquired pneumonia (CAP). Symptoms, test results, biomarkers, and X-ray patterns don't distinctly point to a particular viral cause. Presently, the primary method for lab confirmation involves detecting viral genetic material through reverse transcription-PCR in respiratory secretions (Dandachi & Rodriguez-Barradas, 2018).
Fungal Pneumonia based on Fungi such as Pneumocystis jirovecii can cause pneumonia, especially in people with weakened immune systems, such as those with HIV/AIDS or undergoing chemotherapy.
Aspiration Pneumonia occurs when food, drink, saliva, or vomit is breathed into the lungs, leading to irritation and infection. It's more common in people with swallowing difficulties or those who have lost consciousness.
Pneumonia can affect people of all ages but is often more severe in infants, young children, the elderly, and those with underlying health conditions or compromised immune systems. Diagnosis involves a combination of physical exams, medical history, imaging tests (like X-rays or CT scans), blood tests, and sometimes sputum cultures to identify the specific cause of infection.
Treatment typically involves antibiotics for bacterial pneumonia, antiviral medications for viral cases (if available), and antifungal drugs for fungal pneumonia. Supportive care, such as rest, staying hydrated, and sometimes supplemental oxygen, is essential for recovery.
Preventive measures, including vaccination against certain bacterial and viral causes of pneumonia, good hygiene practices (like handwashing), and maintaining a healthy lifestyle, can reduce the risk of contracting pneumonia.
Diagnosing pneumonia is a blend of simplicity and complexity. Recent research is challenging our long-standing perceptions of pneumonia and the radiological benchmarks that have guided studies for years. Specifically, the reliability of chest X-rays in diagnosing pneumonia is now heavily doubted when compared to computed tomography scans. Pneumonia can be defined in various ways like clinically, pathologically, radiologically, or microbiologically, often a mix of these. Yet, while the field evolves, until new studies redefine pneumonia criteria, clinicians can rely on existing guidelines rooted in traditional standards, which remain as valid as ever (Waterer, 2021).
DL, a subset of artificial intelligence (AI), involves training neural networks to recognize patterns and features in data. In the context of pneumonia detection on X-rays, algorithms have shown promising results in aiding radiologists by automating the process of identifying pneumonia-related abnormalities.
The work of (Sharma & Guleria, 2023) shows a DL model employing VGG16 to detect and categorize pneumonia using two sets of chest X-ray images. When paired with Neural Networks (NN), the VGG16 model achieves an accuracy of 92.15%, a recall of 0.9308, a precision of 0.9428, and an F1-Score of 0.937 for the first dataset. Additionally, the NN-based experiment utilizing VGG16 is conducted on another CXR dataset comprising 6,436 images of pneumonia, normal cases, and COVID-19 instances. The outcomes for the second dataset indicate an accuracy of 95.4%, a recall of 0.954, a precision of 0.954, and an F1-score of 0.954.
The research findings demonstrate that employing VGG16 with NN yields superior performance compared to utilizing VGG16 with Support Vector Machine (SVM), K-Nearest Neighbour (KNN), Random Forest (RF), and Naïve Bayes (NB) for both datasets. Furthermore, the proposed approach showcases enhanced performance results for both dataset 1 and dataset 2 in contrast to existing models.
In the analysis of (Reshan et al., 2023) a DL model is showcased to distinguish between normal and severe pneumonia cases. The entire proposed system comprises eight pre-trained models: ResNet50, ResNet152V2, DenseNet121, DenseNet201, Xception, VGG16, EfficientNet, and MobileNet. These models were tested on two datasets containing 5856 and 112,120 chest X-ray images. The MobileNet model achieves the highest accuracy, scoring 94.23% and 93.75% on the respective datasets. Various crucial hyperparameters such as batch sizes, epochs, and different optimizers were carefully considered when comparing these models to identify the most suitable one.
To distinguish pneumonia cases from normal instances, the capabilities of five pre-trained CNN models namely ResNet50, ResNet152V2, DenseNet121, DenseNet201, and MobileNet have been assessed. The most favourable outcome is achieved by MobileNet using 16 batch sizes, 64 epochs, and the ADAM optimizer. Validation of predictions has been conducted on publicly accessible chest radiographs. The MobileNet model exhibits an accuracy of 94.23%. These metrics serve as a foundation for devising potentially more effective CNN-based models for initial solutions related to Covid-19 (Reshan et al., 2023).
The work of (Wang et al., 2023) introduce PneuNet, a diagnostic model based on Vision Transformer (VIT), aiming for precise diagnosis leveraging channel-based attention within lung X-ray images. In this approach, multi-head attention is employed on channel patches rather than feature patches. The methodologies proposed in this study are tailored for the medical use of deep neural networks and VIT. Extensive experimental findings demonstrate that our approach achieves a 94.96% accuracy in classifying three categories on the test set, surpassing the performance of prior DL models.
. 
DATASETS

There are several publicly available datasets to detect Covid-19 and Pneumonia. Covid-19 Radiography Database (Rahman, 2022) (Dataset 1) released chest X-ray images for Covid-19 positive cases along with normal and viral Pneumonia images, i.e., 3,616 Covid-19, 10,192 normal, 6,012 Lung Opacity (Non-Covid lung infection), and 1345 viral Pneumonia images and corresponding lung masks. The extensive Covid-19 X-Ray and CT chest images dataset CoV-Healthy-6k (El-Shafai, 2020) (Dataset 2 for X-rays and Dataset 5 for CT scans) contains X-rays and CT scans of healthy individuals and people with the Covid-19 virus, where several X-rays and CT scans can be found with and without Covid-19. Concerning the CT scans were used 5427 with Covid-19 and 2628 without, X-ray 4,044 with Covid and 5,500 noncovid.
The Curated Dataset for COVID-19 Posterior-Anterior Chest Radiography Images (X-Rays) (Sait, 2021) (Dataset 3) has available contains 1281 COVID-19 X-Rays, 3270 Normal X-Rays, 1656 viral-pneumonia X-Rays, and 3001 bacterial-pneumonia X-Rays. The Covid-19 lung CT Scans dataset (Aria, 2021) (Dataset 4) has more than 8,000 images of CT scans with Covid-19 and without. The images in this dataset were gathered from radiology departments at teaching hospitals in Tehran, Iran. The Covid-19 status of patients in the dataset was verified using reverse transcription-polymerase chain reaction (RT-PCR) tests. Table 1 shows the characteristics of the public datasets used on this research mentioning the number of X-rays with and without COVID-19 for training, test and validation. 
The datasets utilized in this study have played a critical role in advancing AI-based diagnostic techniques for the detection of COVID-19 from chest X-rays. Through careful preprocessing and quality assurance measures, these datasets have been optimized for the development and validation of deep learning models. They are meticulously divided into training, validation, and test subsets, ensuring comprehensive model evaluation and performance benchmarking. Furthermore, the inclusion of expert-annotated labels, with a secondary validation process, ensures the datasets' integrity and helps to reduce the risk of annotation bias. These datasets form the foundation for training a deep convolutional neural network (CNN), which incorporates cutting-edge transfer learning techniques to enhance model performance.
By employing transfer learning, the CNN model benefits from the knowledge learned by pre-trained models on vast, diverse datasets like ImageNet. This approach is particularly advantageous when dealing with smaller, specialized datasets, such as those containing medical images of COVID-19. Fine-tuning these pre-trained networks for COVID-19 detection allows the model to adapt and apply broad feature recognition capabilities to specific medical contexts, improving its ability to distinguish between subtle COVID-19-related abnormalities and normal lung tissue. This fine-tuning process not only bolsters the model’s accuracy but also enhances its generalization, making it an effective tool for detecting COVID-19 in chest X-rays. By improving diagnostic accuracy, this method accelerates the detection process and supports clinicians in making timely and informed decisions. Furthermore, the combination of AI-driven methods with medical expertise offers a promising avenue for enhancing healthcare systems' capabilities, particularly in resource-constrained settings where rapid and reliable diagnostics are critical for managing the spread of COVID-19. The results of this study highlight the transformative potential of AI in medical imaging and its significant role in the fight against the global pandemic.

Table 1. The Datasets used on the research concerning X-rays and the identification of cases with COVID-19 and normal cases.


Dataset	
	Covid	Noncovid	Pneumonia
Covid-19 Radiography Database (Rahman, 2022)  
3616	10192	6012
The extensive Covid-19 X-Ray and CT chest images dataset CoV-Healthy-6k (El-Shafai, 2020)  
4044	5500	
Curated Dataset for COVID-19 Posterior-Anterior Chest Radiography Images (X-Rays) (Sait, 2021) 
1281	3270	4657



COVID-MEDNET MODEL


The COVID-MedNet model exemplifies a meticulously engineered deep learning framework tailored for the precise classification of COVID-19 X-ray images, leveraging the robust ResNet-101 architecture as its foundational backbone. This sophisticated implementation begins by designating a datapath, specified as 'dataset1', which houses the COVID-19 X-ray dataset. An image datastore is subsequently constructed, incorporating hierarchical subfolder organization and utilizing folder names to assign labels, ensuring that the dataset is systematically structured with clinically relevant COVID-19 annotations.
To support a comprehensive evaluation, the dataset is carefully partitioned into training, validation, and testing subsets through a stratified splitting methodology. Initially, seventy percent of the data is allocated to the training set, with the remaining thirty percent reserved for testing, employing a deliberate shuffling technique to preserve label distribution consistency. From the training subset, fifteen percent is further designated for validation, enhancing the model’s ability to generalize to unseen samples. Image preprocessing constitutes a critical phase, wherein each X-ray undergoes a custom function to standardize its format and dimensions. Grayscale images are transformed into RGB by tripling the single channel, and all images are resized to a uniform 224 by 224 pixel resolution, aligning seamlessly with the input requirements of the ResNet-101 backbone.
The pretrained ResNet-101 network is instantiated, and its architecture is refined by generating a layer graph, providing a versatile representation of the network’s topology. Detailed inspection of the layer composition identifies 'pool5' as a key pooling layer, necessitating the strategic removal of the final three layers: the fully connected layer 'fc1000', the softmax layer, and the classification layer. This adaptation reorients the backbone for the specialized task of COVID-19 detection, with the excision process executed iteratively and reinforced by error-handling protocols to maintain structural robustness. To enhance transfer learning efficiency, convolutional layers’ weights and biases are assigned a learning rate factor of 0.1, stabilizing early feature extraction while enabling subtle refinements in deeper layers.
A custom sequence of layers is introduced to replace the removed components, meticulously designed to classify X-ray images into categories matching the unique labels within the training dataset. This sequence begins with a fully connected layer of 512 units, named 'fc1_covid', with weights and biases assigned a learning rate factor of 50 to expedite adaptation. This is followed by a batch normalization layer 'bn1_covid' to normalize activations, a ReLU layer 'relu1_covid' to introduce nonlinearity, a dropout layer 'dropout1_covid' with a 0.5 probability to curb overfitting, a second fully connected layer 'fc2_covid' sized to the number of classes with matching learning rate factors of 50, a softmax layer 'softmax_covid' to compute class probabilities, and a classification layer 'classification_covid' to deliver the final output. These layers are seamlessly incorporated into the layer graph and interconnected, originating from the 'pool5' layer, with conditional checks to prevent duplicate connections, ensuring a fluid data flow through the modified backbone.
Training is orchestrated with a refined configuration utilizing the Adam optimizer, featuring a mini-batch size of 128 to optimize computational throughput, a maximum of ten epochs to ensure thorough convergence, and an initial learning rate of 0.0001. The learning rate follows a piecewise schedule, decreasing by a factor of 0.1 every five epochs to fine-tune parameter adjustments. Data augmentation is strategically implemented to enhance model robustness, incorporating horizontal reflections and rotational adjustments between -10 and 10 degrees, applied via an augmented image datastore that enforces an RGB format and adheres to the 224 by 224 by 3 input size. Validation and test sets are similarly standardized to this dimensional specification, and the network is trained with the curated training data, monitored through validation assessments every 100 iterations and a patience threshold of six to halt training if performance stagnates.
Following training, the model’s performance is rigorously evaluated on the test set, producing predicted labels and posterior probabilities. A confusion matrix is generated to illustrate classification outcomes, annotated with absolute counts for row and column summaries, and enhanced with distinct color coding—blue for accurate predictions and orange for discrepancies—to improve interpretability. Additional evaluation includes the computation of a receiver operating characteristic (ROC) curve, with the area under the curve (AUC) determined to gauge discriminative ability, alongside comprehensive metrics such as precision, recall, and F1-score, averaged across classes to provide a holistic performance overview.
To elucidate the model’s decision-making process, interpretability is explored using Grad-CAM and LIME techniques on a carefully selected set of four test images. Grad-CAM utilizes gradients of the predicted class score relative to the feature maps of the last convolutional layer 'res5c_branch2c', located before 'pool5', to produce heatmaps highlighting influential regions, overlaid on the original 224 by 224 images with a jet colormap. Concurrently, LIME assesses feature importance by perturbing the input across 1000 samples and employing grid-based segmentation, visualized with a parula colormap, offering a complementary perspective on the model’s focal areas. These visualizations are thoughtfully arranged, presenting the original image, Grad-CAM, and LIME outputs side by side for each sample, resulting in a detailed figure that emphasizes the interpretability of the COVID-MedNet model in the context of COVID-19 X-ray classification.
This implementation showcases a precise adaptation of the ResNet-101 backbone, integrating targeted layer modifications, specifically replacing 'fc1000', the softmax layer, and the classification layer with a bespoke classification head coupled with a carefully calibrated learning rate strategy and extensive technical enhancements, delivering a powerful and interpretable diagnostic tool for medical imaging applications.

 



  



















Dataset 1

 

 

 





 
Accuracy: 99.31%
Mean Precision: 0.9923
Mean Recall: 0.9885
Mean F1-Score: 0.9904
AUC: 0.9996

Dataset 2
 

 



 

Accuracy: 97.1%
Mean Precision: 0.9661
Mean Recall: 0.9624
Mean F1-Score: 0.9640
AUC: 0.9916

 


 


Dataset 3 




 
 

 


 


Accuracy: 99.79 %
Mean Precision: 0.9977
Mean Recall: 0.9969
Mean F1-Score: 0.9973
AUC: 1.0000



