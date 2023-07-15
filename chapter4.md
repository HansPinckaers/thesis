---
author:
- Hans Pinckaers^a^\*, Jolique van Ipenburg^a^, Jonathan Melamed^b^, Angelo De Marzo^c^, Elizabeth A. Platz^d^, Bram van Ginneken^a^, Jeroen van der Laak^a,e^, Geert Litjens^a^
bibliography: library.bib
title: Predicting biochemical recurrence of prostate cancer with artificial intelligence
abstract: |
  **Background\:** The first sign of metastatic prostate cancer after
  radical prostatectomy is rising PSA levels in the blood, termed
  biochemical recurrence. The prediction of recurrence relies mainly on
  the morphological assessment of prostate cancer using the Gleason
  grading system. However, in this system, within-grade morphological
  patterns and subtle histopathological features are currently omitted,
  leaving a significant amount of prognostic potential unexplored.

  **Methods\:** To discover additional prognostic information using
  artificial intelligence, we trained a deep learning system to predict
  biochemical recurrence from tissue in H&E-stained microarray cores
  directly. We developed a morphological biomarker using convolutional
  neural networks leveraging a nested case-control study of 685 patients
  and validated on an independent cohort of 204 patients. We use
  concept-based explainability methods to interpret the learned tissue
  patterns.

  **Results\:** The biomarker provides a strong correlation with
  biochemical recurrence in two sets (n=182 and n=204) from separate
  institutions. Concept-based explanations provided tissue patterns
  interpretable by pathologists.

  **Conclusions\:** These results show that the model finds predictive
  power in the tissue beyond the morphological ISUP grading.
---

Corresponding author: Hans Pinckaers, Radboud University Medical Center,
Postbus 9101, 6500 HB Nijmegen, The Netherlands (tel +31 634 856 950,
hans.pinckaers@radboudumc.nl)

^a^ Department of Pathology, Radboud Institute for Health Sciences,
Radboud University Medical Center, Nijmegen, The Netherlands

^b^ Department of Pathology, New York University Langone Medical Center,
New York, USA

^c^ Departments of Pathology, Urology and Oncology, The Brady Urological
Research Institute and the Sidney Kimmel Comprehensive Cancer Center at
Johns Hopkins, Baltimore, Maryland, USA

^d^ Department of Epidemiology, Johns Hopkins Bloomberg School of Public
Health, Baltimore, Maryland, USA

^e^Center for Medical Image Science and Visualization, Linköping
University, Linköping, Sweden

#  Introduction

Prostate cancer is a common malignancy among men, affecting 1.4 million
per year.[@Sung2021-iz] A significant proportion of these men will
receive the primary curative treatment of a prostatectomy. This
surgery's success can partly be judged by the concentration of
prostate-specific antigen (PSA) in the blood. While it has a dubious
role in prostate cancer
screening[@US_Preventive_Services_Task_Force2018-ro,Heijnsdijk2018-yn],
this protein is a valuable biomarker in PCa patients' follow-up
post-prostatectomy. In a successful surgery, the concentration will
mostly be undetectable (\<0.1 ng/mL) after four to six
weeks[@Goonewardene2014-nu].

However, in approximately 30% of the patients
[@Amling2000-ty,@Freedland2005-yu,@Han2001-mx], PSA will rise again
after surgery, called biochemical recurrence, pointing to regrowth of
prostate cancer cells. Biochemical recurrence is a prognostic indicator
for subsequent progression to clinical metastases and prostate cancer
death.[@Van_den_Broeck2019-xb] Estimating chances of biochemical
recurrence could help to better stratify patients for specific adjuvant
treatments.

The risk of biochemical recurrence of prostate cancer is currently
assessed in clinical practice through a combination of the ISUP
grade[@Epstein2016-im], the PSA value at diagnosis and the TNM
staging criteria. In a recent European consensus guideline, these
factors were proposed to separate the patients into a low-risk,
intermediate-risk and high-risk group.[@Mottet2021-uu] A high ISUP
grade independently can, independently of other factors, assign a
patient to the intermediate (grade 2/3) or high-risk group (grade 4/5).

Based on the distribution of the Gleason growth
patterns[@Epstein2010-au], which are prognostically predictive
morphological patterns of prostate cancer, pathologists assign cancerous
tissue obtained via biopsy or prostatectomy into one of five groups.
They are commonly referred to as International Society of Urological
Pathology (ISUP) grade groups, the ISUP grade, Gleason grade groups, or
just grade groups.
[@Epstein2016-im,@Pierorazio2013-sq,@Epstein2016-pf,@Van_Leenders2020-fy].
Throughout this paper we will use the term *ISUP grade*. The ISUP grade
suffers from several well-known limitations. For example, there is
substantial disagreement in the grading using the Gleason
scheme.[@Ozkan2016-dq,@Van_Leenders2020-fy]. Furthermore,
although the Gleason growth patterns have seen significant updates and
additions since their inception in the 1960s, they remain relatively
coarse descriptors of tissue morphology. As such, the prognostic
potential of more fine-grained morphological features has been
underexplored. We hypothesize that artificial intelligence, and more
specifically deep learning, has the potential to discover such
information and unlock the true prognostic value of morphological
assessment of cancer. Specifically, we developed a deep learning system
(DLS), trained on H&E-stained histopathological tissue sections,
yielding a score for the likelihood of early biochemical recurrence.

Deep learning is a recent new class of machine learning algorithms that
encompasses models called neural networks. These networks are optimized
using training data; images with labels, such as recurrence information.
From the training data, relevant features to predict the labels are
automatically inferred. During development, the generalization of these
features is tested on separated training data, which is not used for
learning. Afterwards, a third independent set of data, the test set, is
used to ensure generalization. Since features are inferred, handcrafted
feature engineering is not needed anymore to develop machine learning
models. Neural networks are the current state-of-the-art in image
classification[@Krizhevsky2012-zi].

Deep learning has previously been shown to find visual patterns to
predict genetic mutations from morphology, for example, in
lymphoma[@Swiderska-Chadaj2020-uy] and lung
cancer[@Coudray2018-fh]. Additionally, deep learning has been used
for feature discovery in colorectal cancer[@Wulczyn2021-zw] and
intrahepatic cholangiocarcinoma[@Muhammad2021-an] using survival
data. Although deep learning has been used with biochemical recurrence
data on prostate cancer, Leo *et al.[@Leo2021-lj]* assumed manual
feature selection beforehand, strongly limiting the extent of new
features to be discovered. Yamamoto et al.[@Yamamoto2019-vy] used
whole slide images and a deep-learning-based encoding of the slides to
tackle the slides' high resolution. They leverage classical regression
techniques and support-vector machine models on these encodings. The
deep learning model was not directly trained on the outcome, limiting
the feature discovery in this work as well.

A common critique of deep learning is its black-box nature of the
inferred features.[@Rudin2019-fp] Especially in the medical field,
decisions based on these algorithms should be extensively validated and
be explainable. Besides making the algorithms' prediction trustworthy
and transparent, from a research perspective, it would be beneficial to
visualize the data patterns which the model learned, allowing insight
into the inferred features. We can visualize the patterns learned by the
network leveraging a new technique called Automatic Concept Explanations
(ACE)[@Ghorbani2019-sy]. ACE clusters patches of the input image
using their intermediate inferred features showing common patterns
inferred by the network. We were interested in finding these common
concepts over a range of images to unravel patterns that the model has
identified.

This study aimed to use deep learning to develop a new prognostic
biomarker based on tissue morphology for recurrence in patients with
prostate cancer treated by radical prostatectomy. As training data, we
used a nested case-control study[@Toubaji2011-og]. This study
design ensured we could evaluate whether the network learned
differentiating patterns independent of Gleason patterns.

# Methods

*Cohorts*

Two independent cohorts of patients who underwent prostatectomy for
clinically localized prostate cancer were used in this study. Patients
were treated at either the Johns Hopkins Hospital in Baltimore or New
York Langone Medical Center. Both cohorts were accessed via the Prostate
Cancer Biorepository Network[@noauthor_undated-cy].

For the development of the novel deep-learning-based biomarker (further
referred to as DLS biomarker), we used a nested case-control study of
patients from Johns Hopkins. This study consists of 524 matched pairs
(724 unique patients) containing four tissue spots per patient. They
were sampled from 4,860 prostate cancer patients with clinically
localized prostate cancer who received radical retropubic prostatectomy
between 1993 and 2001. Men were routinely checked after prostatectomy at
3 months and at least yearly thereafter. Surveillance for recurrence was
conducted using digital rectal examination and measurement of serum PSA
concentration. Patients were followed for outcome until 2005, with a
median follow-up of 4.0 years. The outcome was defined as recurrence,
based on biochemical recurrence (serum PSA \>0.2 ng/mL on 2 or more
occasions after a previously undetectable level after prostatectomy), or
events indicating biochemical recurrence before this was measured; local
recurrence, systemic metastases, or death from prostate cancer. Controls
were paired to cases with recurrence using incidence density
sampling[@Wang2009-te]. For each case, a control was selected who
had not experienced recurrence by the date of the case's recurrence and
was additionally matched based on age at surgery, race, pathologic
stage, and Gleason sum in the prostatectomy specimen based on the
pathology reports. Given the incidence density sampling of controls,
some men were used as controls for multiple cases, and some controls
developed recurrence later and became cases for that time period.

  ----------------------------------------------------------------------------------------------------------
  **Table 1**: Baseline                                                                          
  characteristics of                                                                             
  test set and                                                                                   
  development set from                                                                           
  the John Hopkins                                                                               
  Hospital, prostate                                                                             
  cancer recurrence                                                                              
  cases and controls,                                                                            
  men who underwent                                                                              
  radical prostatectomy                                                                          
  for clinically                                                                                 
  localized disease                                                                              
  between 1993 to 2001.                                                                          
  ---------------------- --------------- --------- ---------- -- -------------- ---------------- -----------
                         **Development                           **Test set**                    
                         set**                                                                   

                         **Recurrence    **No      **P**         **Recurrence   **Controls\***   **P**
                         cases**         events                  cases**                         
                                         cases**                                                 

  **N**                  368             135                     91             91               

  **Age, mean (SD)**     58.9 (6.2)      59.3      p=0.540       58.4 (6.1)     58.3 (6.3)       *Matched*
                                         (6.3)                                                   

  **preop. PSA (ng/mL),  12.3 (10.0)     10.1      p=0.010       12.3 (10.8)    10.5 (7.7)       p=0.195
  mean (SD)**                            (7.5)                                                   

  **Race, n (%)**                                  p=0.599                                       *Matched*

  White                  327 (88.9)      120                     72 (79.1)      75 (82.4)        
                                         (88.9)                                                  

  Black or African       32 (8.7)        14 (10.4)               12 (13.2)      10 (11.0)        
  American                                                                                       

  Other                  9 (2.4)         1 (0.7)                 7 (7.7)        6 (6.6)          

  **Pathological stage**                           p=0.107                                       *Matched*

  pT2                    43 (11.7)       25 (18.5)               20 (22.0)      19 (20.9)        

  pT3a                   199 (54.1)      63 (46.7)               50 (54.9)      51 (56.0)        

  pT3b or N1             126 (34.2)      47 (34.8)               21 (23.1)      21 (23.1)        

  **Gleason sum                                    p=0.179                                       *Matched*
  prostatectomy (%)**                                                                            

  6                      38 (10.3)       25 (18.5)               20 (22.0)      23 (25.3)        

  7                      233 (63.3)      76 (56.3)               51 (56.0)      50 (54.9)        

  8+                     97 (26.4)       34 (25.2)               20 (22.0)      18 (19.8)        

  **ISUP grade, n (%)**                            p=0.002                                       p=0.851

  1                      38 (10.3)       25 (18.5)               20 (22.0)      23 (25.3)        

  2                      140 (38.0)      61 (45.2)               35 (38.5)      38 (41.8)        

  3                      93 (25.3)       15 (11.1)               16 (17.6)      12 (13.2)        

  4                      49 (13.3)       21 (15.6)               13 (14.3)      10 (11.0)        

  5                      48 (13.0)       13 (9.6)                7 (7.7)        8 (8.8)          

  **Positive surgical    140 (38.1)      24 (17.8) p\<0.001      36 (39.6)      20 (22.0)        p=0.016
  margins**                                                                                      

  **Mean year of         1997.0 (2.3)    1995.5    p\<0.001      1997 (2.3)     1995 (2.1)       p\<0.001
  surgery**                              (2.3)                                                   

  \* due to the nested                                                                           
  case-control nature,                                                                           
  some controls could                                                                            
  have a biochemical                                                                             
  recurrence, but always                                                                         
  later than their                                                                               
  matched case.                                                                                  
  ----------------------------------------------------------------------------------------------------------

  -------------------------------------------------------------------------
  **Table 2:** Baseline                                         
  characteristics of the cohort                                 
  from New York Langone hospital,                               
  prostate cancer recurrence                                    
  cases and controls, men who                                   
  underwent radical prostatectomy                               
  between 2001 to 2003                                          
  ------------------------------- -------------- -------------- -----------
                                  **Recurrence   **Controls**   **P**
                                  cases**                       

  **N**                           38             166            

  **preop. PSA (ng/mL), mean      11.6 (11.5)    6.7 (3.9)      p=0.014
  (SD)**                                                        

  **Age**, mean (SD)              61.7 (8.9)     60.3 (6.6)     p=0.359

  **Race**, n (%)                                               p=0.401

  African-American                2 (5.3)        4 (2.4)        

  Asian                           2 (5.3)        3 (1.8)        

  Caucasian                       33 (86.8)      144 (86.7)     

  Not reported                    0 (0)          2 (1.2)        

  Other                           1 (2.6)        13 (7.8)       

  **Pathological stage**, n (%)                                 p\<0.001

  pT2a                            0 (0)          12 (7.2)       

  pT2b                            3 (7.9)        5 (3.0)        

  pT2c                            16 (42.1)      114 (68.7)     

  pT3a                            10 (26.3)      27 (16.3)      

  pT3b                            9 (23.7)       8 (4.8)        

  **ISUP grade**, n (%)                                         p\<0.001

  1                               3 (7.9)        67 (40.4)      

  2                               13 (34.2)      76 (45.8)      

  3                               6 (15.8)       13 (7.8)       

  4                               5 (13.2)       3 (1.8)        

  5                               11 (28.9)      7 (4.2)        

  **Surgical Margins**, n (%)                                   p=0.060

  Focal                           10 (26.3)      20 (12.0)      

  Free of tumour                  27 (71.1)      144 (86.7)     

  Widespread                      1 (2.6)        2 (1.2)        
  -------------------------------------------------------------------------

The TMA spots were cores (0.6 mm in diameter) from the highest-grade
tumour nodule. Random subsamples were taken in quadruplicate for each
case. The whole slides were scanned using a Hamamatsu NanoZoomer-XR
slide scanner at 0.23 μ/px. TMA core images were extracted using QuPath
(v0.2.3, [@Bankhead2017-tf]). We discarded analysis of cores with
less than 25% tissue. The cores were manually checked (HP) for prostate
cancer, excluding 535 cores without clear cancer cells present in the
TMA cross-section, resulting in a total of 2343 TMA spots. The nested
case-control set was split based on the matched pairs into a development
set (268 unique pairs), and a test set (91 pairs); the latter was used
for evaluation only. We leveraged cross-validation by subdividing the
development into three folds to tune the models on different parts of
the development set. We divided paired patient, randomly, keeping into
account the distribution of the matched variables. The random assignment
was done using the scikit-multilearn package[@Szymanski2019-mf],
specifically the 'IterativeStratification' method in
'skmultilearn.model_selection'. After splitting the dataset into
training and test, we split the training dataset into three folds using
the same method for the cross-validation.

To validate the DLS biomarker on a fully independent external set, we
used the cohort from New York Langone Medical Center. This external
validation cohort consists of 204 patients with localized prostate
cancer treated with radical prostatectomy between 2001 and 2003.
Patients were followed for outcome until 2019, with a median follow-up
of 5 years. Biochemical recurrence was defined as either a single PSA
measurement of ≥ 0.4 ng/m or PSA level of ≥ 0.2 ng/ml followed by
increasing PSA values in subsequent follow-up. Cores were sampled from
the largest tumour focus or any higher-grade focus (\> 3mm). Subsamples
were taken in quadruplicate for each case. Images were scanned using a
Leica Aperio AT2 slide scanner at 0.25 μ/px.

*Model details*

For developing the convolutional neural networks (CNNs) we used
PyTorch[@Paszke2019-ic]. As an architecture, we used
ResNet50-D[@He2019-lm] pretrained on ImageNet from PyTorch Image
Models[@Wightman2021-an]. We used the Lookahead
optimizer[@Zhang2019-hr] with RAdam[@Liu2019-pg], with a
learning rate of 2e-4 and mini-batch size of 16 images. We used weight
decay (7e-3), and a drop-out layer (p=0.15) before the final
fully-connected layer. We used EfficientNet-style[@Tan2019-ke]
dropping of residual connections (p=0.3) as implemented in PyTorch Image
Models. We used Bayesian Optimization to find the optimal values.

We resized the TMAs to 1.0 mu/pixel spacing and cropped to 768x768
pixels. Extensive data augmentations were used to promote
generalization. The transformations were: flipping, rotations, warping,
random crop, HSV color augmentations, jpeg compression, elastic
transformations, Gaussian blurring, contrast alterations, gamma
alterations, brightness alterations, embossing, sharpening, Gaussian
noise and cutout[@DeVries2017-ah]. Augmentations were implemented
by albumentations[@Buslaev2020-pn] and
fast.ai[@Howard2020-wy].

TMA spots from cases experiencing recurrence were assigned a value of
0-4, depending on the year on which the first event, either biochemical
recurrence, metastases, or prostate cancer-related death, was recorded,
with 0 meaning recurrence within a year, 4 meaning after 4+ years. TMA
spots from cases without an event were also assigned the label 4.

We validated the model on the development validation fold each epoch
with a moving average of the weights from 5 subsequent epochs. We used
the concordance index as a metric to decide which model performed the
best.

As the final prediction at the patient level, the TMA spot with the
highest score was used. The final DLS consists of an ensemble of 15
convolutional neural networks. Using cross-validation as described
above, 15 networks were trained for each fold, of which the five best
performing were used for the DLS.

![](media/image3.png){width="6.5in" height="3.4444444444444446in"}

**Figure 1.** Overview of the methods summarizing the biomarker
development and the Automatic Concept Explanations (ACE) process. Cores
were extracted from TMA slides and used to train a neural network to
predict the years to biochemical recurrence. On the nested case-control
test set, a matched analysis was performed. For ACE, patches were
generated from the cores, inferenced through the network and clustered
based on their intermediate features.

*Statistical analysis*

For primary analysis of the nested case-control study, odds ratios (OR)
and 95% confidence intervals (CI) were calculated using conditional
logistic regression, following Dluzniewski et
al.[@Dluzniewski2012-sk]. Due to the study design, calculating
hazard ratios using a Cox proportional hazard regression is not
appropriate. For the primary analysis, the continuous DLS marker was
given as the only variable. For a secondary analysis, we added the
non-matched variables PSA, positive surgical margins, and a binned
indicator variable for year of surgery. Since matching was done on
Gleason sum, and our goal was to identify patterns beyond currently used
Gleason patterns, we corrected for the residual differences of the ISUP
grade between cases and control (see Table 1). A correction was
performed by adding a continuous covariate since, due to the small
differences, an indicator covariate did not converge. Analysis was done
using the lifelines Python package (v.
0.25.10)[@Davidson-Pilon2021-uq] with Python (v. 3.7.8). Since the
DLS predicts the time to recurrence, high values indicate a low
probability of recurrence. We multiplied the DLS output by -1 to make
the analysis more interpretable. For three patients (1 from the Johns
Hopkins cohort and 2 from the New York Langone cohort), PSA values were
missing and were therefore replaced by the median.

For primary analysis of the New York Langone cohort, we calculated
hazard ratios (HR) using a Cox proportional hazards regression. We
report a secondary multivariable analysis including indicator variables
for relevant clinical covariates, Gleason sum, pathological stage, and
surgical margin status. We tested the proportional hazards assumption as
satisfactory (every p-value above 0.01) using the Pearson correlation
between the residuals and the rank of follow-up time. Kaplan Meier plots
were generated for the New York Langone cohort. Due to the nested
case-control design for the Johns Hopkins set, this set could not be
visualized in a Kaplan Meier plot.

*Automatic Concept Explanations*

To generate concepts, we picked the best performing single CNN from the
DLS based on its validation set fold. We used a combination of the
methods of Yeh *et al.*, 2020[@Yeh2020-bq] and Ghorbani *et al.*,
2019[@Ghorbani2019-sy].

We tiled the TMA images into 256x256 patches within the tissue,
discarding patches with more than 50% whitespace. These patches were
padded to the original input shape of the CNN (768x768 pixels). The
latent space of layer 42 of 50 was saved for each tile. Afterwards, we
used PCA (50 components) to lower the dimensionality and then performed
k-means (k=15) to cluster the latent spaces.

In contrast to Yeh *et al.* and Ghorbani *et al.*, we did not sort the
concepts on completeness of the explanations or importance for
prediction of individual samples. We sorted the concepts to find
interesting new patterns related to recurrence across images by ranking
the concepts based on the DLS score of the TMA spot from which they
originated.

For each concept, 25 examples were randomly picked and visually
inspected by a pathologist (JvI), with a special interest in
uropathology, blinded to the case characteristics and prediction of the
network.

# Results

The DLS system was developed on the Johns Hopkins cohort with 2343 TMA
spots of 685 included unique patients (39 patients were excluded due to
insufficient tumour amount in the cores). 492 patients were recurrence
cases (72%). The 685 included patients were split into a development set
of 503 unique patients and a test set of 91 matched pairs of cases and
controls (182 unique patients).

In the external validation cohort, 38 out of the 204 patients (19%) had
biochemical recurrence after complete remission, PSA nadir after 3
months post-prostatectomy. From the 204 patients, 620 TMA spots were
included. Clinical characteristics of the cohorts can be found in Table
1 and Table 2.

The DLS marker showed a strong association in the primary analyses on
the test set of the Johns Hopkins cohort with an OR of recurrence of
3.28 (95% CI 1.73-6.23; p\<0.005) per unit increase, with DLS system
continuous output ranging from 0-3, with two cases below 0 (-0.27 and
-0.24) (Table 3).

In addition, for the John Hopkins cohort, we checked for confounding by
ISUP grade, PSA level at diagnosis, positive surgical margins, and year
of prostatectomy. Neither covariate was found to bias the estimates of
effect substantially. The biomarker maintained a strong correlation of
OR 3.32 (CI 1.63 - 6.77; p=0.001) per unit increase, adjusting for these
factors and the continuous term for the residual difference between
cases and controls in the ISUP grade.

In the univariable analysis, the DLS marker was strongly associated with
recurrence in the New York Langone external validation cohort with an HR
of 5.78 (95% CI 2.44-13.72; p\<0.005) per unit increase. In the
multivariate model, including ISUP grade and the other prognostic
indicators in addition to the DLS biomarker, the DLS biomarker was still
strongly associated with recurrence with an HR of 3.02 (CI 1.10 - 8.29;
p=0.03) per unit increase. Kaplan Meier curves based on a median
cut-off, and four-group categorization, show a clear separation of the
low-risk and high-risk groups (Figure 3).

+---------------------+-----------------------+------------------------+
| **Table 3**:        |                       |                        |
| Conditional         |                       |                        |
| logistic regression |                       |                        |
| analyses of the     |                       |                        |
| Johns Hopkins test  |                       |                        |
| set.                |                       |                        |
+=====================+=======================+========================+
| **Covariate**       | **Matched analysis\   | **Multivariate         |
|                     | Johns Hopkins         | analysis\              |
|                     | (OR)^1^**             | Johns Hopkins (OR)**   |
+---------------------+-----------------------+------------------------+
| **Biomarker**       | [3.28]{.underline}    | [3.32]{.underline} (CI |
|                     | (CI 1.73 - 6.23;      | 1.63 - 6.77; p=0.001)  |
|                     | p\<0.005)             |                        |
+---------------------+-----------------------+------------------------+
| **preop. PSA        |                       | 1.04 (CI 0.99 - 1.10;  |
| (ng/mL)**           |                       | p=0.10)                |
+---------------------+-----------------------+------------------------+
| **Surgical margins  |                       | 1.69 (CI 0.69 - 4.18;  |
| (pos)**             |                       | p=0.25)                |
+---------------------+-----------------------+------------------------+
| **ISUP grade        |                       | 1.34 (CI 0.64 - 2.82;  |
| (cont.)\***         |                       | p=0.44)                |
+---------------------+-----------------------+------------------------+
| **Mean year of      |                       |                        |
| surgery**           |                       |                        |
+---------------------+-----------------------+------------------------+
| 1992 - 1994 (n=75)  |                       | *1.0*                  |
+---------------------+-----------------------+------------------------+
| 1994 - 1997 (n=55)  |                       | 3.35 (CI 1.13 - 9.91;  |
|                     |                       | p=0.03)                |
+---------------------+-----------------------+------------------------+
| 1997 - 2001 (n=52)  |                       | 8.22 (CI 2.38 - 28.37; |
|                     |                       | p=0.0009)              |
+---------------------+-----------------------+------------------------+
| ^1^ Cases and       |                       |                        |
| controls were       |                       |                        |
| matched on age at   |                       |                        |
| surgery, race,      |                       |                        |
| pathologic stage,   |                       |                        |
| and Gleason sum in  |                       |                        |
| the prostatectomy   |                       |                        |
| specimen.           |                       |                        |
|                     |                       |                        |
| **^2^** The ISUP    |                       |                        |
| grade covariate was |                       |                        |
| added to correct    |                       |                        |
| for the residual    |                       |                        |
| differences left    |                       |                        |
| after matching      |                       |                        |
| cases with controls |                       |                        |
| on prostatectomy    |                       |                        |
| Gleason sum.        |                       |                        |
+---------------------+-----------------------+------------------------+

  -----------------------------------------------------------------------
  **Table 4**: Cox                             
  proportional                                 
  hazard analyses of                           
  New York Langone                             
  external                                     
  validation cohort.                           
  ------------------ ------------------------- --------------------------
  **Covariate**      **Univariate analysis\    **Multivariate analysis\
                     NYU (HR)**                NYU (HR)**

  **Biomarker**      [4.79]{.underline} (CI    [3.02]{.underline} (CI
                     2.09 - 10.96; p=0.0002)   1.10 - 8.29; p=0.03)

  **preop. PSA                                 1.07 (CI 1.02 - 1.12;
  (ng/mL)**                                    p=0.004)

  **ISUP grade**                               

  1                                            1.0

  2                                            2.64 (CI 0.73 - 9.58;
                                               p=0.14)

  3                                            8.74 (CI 2.16 - 35.30;
                                               p=0.00)

  4                                            12.78 (CI 2.82 - 57.91;
                                               p=0.00)

  5                                            9.60 (CI 2.32 - 39.69;
                                               p=0.00)

  **Pathological                               
  stage**                                      

  pT2a + b                                     1.0

  pT2c                                         1.02 (CI 0.27 - 3.80;
                                               p=0.98)

  pT3a                                         1.26 (CI 0.28 - 5.67;
                                               p=0.77)

  pT3b                                         2.77 (CI 0.66 - 11.62;
                                               p=0.16)

  **Surgical                                   
  margins**                                    

  Free                                         1.0

  Focal                                        2.13 (CI 0.76 - 5.96;
                                               p=0.15)

  Widespread                                   0.20 (CI 0.01 - 3.39;
                                               p=0.27)
  -----------------------------------------------------------------------

Automatic Concept Explanations provided semantically meaningful concepts
(Figure 1). Concepts were identified that correlated with either a
relatively rapid or slow biochemical recurrence. Visual inspection by
JvI reveals that generally, the concepts with adverse behaviour show
mainly Gleason pattern 4 and some Gleason pattern 5, with cribriform
configuration in TMAs within the concepts with most adverse behaviour.
The two intermediate concepts show mainly stroma and less aggressive
growth patterns. The two concepts predicted to be part of late
recurrence cases show mainly Gleason 3 patterns, with readily
recognizable well-formed glands. See the supplementary materials for a
detailed analysis.

![](media/image4.png){width="6.5in" height="4.097222222222222in"}

**Figure 2.** Automatic Concepts Explanations. Sorted by their average
score for the cores in which the pattern occurs. Showing the two most
benign concepts, two intermediate and two aggressive concepts. Green,
yellow and red shaded areas indicate 33%, 66% percentiles.\
See the supplementary materials for all concepts.

![](media/image1.png){width="2.464772528433946in"
height="2.5104166666666665in"}

**Figure 3.** Kaplan Meier plot for New York Langone external validation
cohort, Groups were separated using the median DLS biomarker score in
this cohort (left) and using four thresholds
(right).![](media/image2.png){width="2.569327427821522in"
height="2.9965726159230095in"}

# 

# Discussion

We have developed a deep-learning-based morphological biomarker for the
prediction of prostate cancer biochemical recurrence based on
prostatectomy tissue microarrays. Using a nested case-control study, we
trained convolutional neural networks end-to-end with biochemical
recurrence data. The DLS marker provides a continuous score based on the
speed of biochemical recurrence it perceived. The DLS marker had an OR
of 3.32 (CI 1.63 - 6.77; p=0.001) per unit increase for the test set,
and an HR of 3.02 (CI 1.10 - 8.29; p=0.03) per unit increase for the
external validation set. These findings support our hypothesis that
there is more morphological information in the tissue besides the ISUP
grade.

In the Kaplan Meier plot (Figure 3) the biomarker especially seems able
to separate men with relatively rapid recurrence from men without (\<5
years). However, we hypothesize that the decreased long-term separation
in those survival curves is less due to the training cohort containing a
median follow-up for four years. Furthermore, we choose to group
patients together with more than four years of no biochemical
recurrence, This limits the model\'s capabilities to differentiate
patients with very late recurrence. Additionally, due to the limitations
of the morphology of the present tumour to inform about long-term
outcomes (e.g., cells that escaped the primary tumour may subsequently
acquire genomic changes that influence recurrence). Furthermore, it
should be noted that the number of at-risk patients was small at these
long-term time points.

The nested case-control study contained follow-up information in
timespans of years, this limited the use of survival based loss
functions[@Kvamme2019-pr]. When more granular follow-up information
is at hand, future work could investigate usage of Cox regression based
loss functions to better leverage the information of the clinical
cohort.

The DLS marker showed strong and similar association in both cohorts
prepared at different pathology laboratories, which supports the
robustness to differences in tissue preparation, staining protocols and
scanners.

We showed that Automatic Concept Explanation may be helpful to find
concepts correlated with good and poor prognosis. The most
discriminatory concepts followed the morphological patterns of Gleason
grading. Well-defined prostate cancer glands were predicted to undergo
biochemical recurrence later than disorganized sheets of prostate cancer
cells. These concepts support the DLS system capturing the expected
morphological patterns in support of the validity of the DLS approach.

This study focused on the use of deep learning to automatically discover
features relevant for biochemical recurrence prediction. Compared to
before-mentioned studies on prostate cancer prognostics
models[@Leo2021-lj,@Yamamoto2019-vy], we are the first paper to
directly optimize a neural network from prostatectomy tissue towards
biochemical recurrence. Additionally, we report that training towards
the biochemical recurrence endpoint results in patterns in the networks'
features aligning with the ISUP grading.

In the increasing digitalisation of pathology labs, our DLS marker may
be applied on digitally chosen regions of interest. Our marker is
trained on tissue microarray spots that were selected at the highest
grade cancer focus. Furthermore, it has to be noted that a TMA core
allows for only limited assessment of the overall prostate cancer growth
patterns. Since these tissue cores represent only limited samples from
what is usually a much larger tumour lesion, the potential more
aggressive patterns may still be present outside of the chosen regions,
including regions of potential extraprostatic extension and perineural
invasion. Validation will need to be done on entire prostatectomy
sections and across cancer foci.

There have been improvements to prostate cancer
grading[@Epstein2016-pf,@Epstein2010-au], and recently the
cribriform pattern is suggested to be important for
prognostics[@Hollemans2021-rd,@Van_Leenders2020-fy]. However, the
evaluation of this pattern can show a range of inter-observer
variability[@Van_der_Slot2021-xy], although a recent consensus
approach could help decrease this
variability[@Van_der_Kwast2021-kn]. Although we certainly have to
keep in mind all the before-mentioned limitations, our findings are in
line with outcomes concerning adverse behaviour in earlier work. The DLS
system identified a concept that consisted of fields with
cribriform-like growth patterns. This cribriform-like growth pattern was
found to be part of the concept that was most associated with early
recurrent cases.

The results in this study are limited to newer insights of prostate
cancer growth, information on cribriform-growth and intraductal
carcinoma were not readily available for use in the multivariate
analysis, although the external validation cohort was graded using the
2005 ISUP consensus [@Epstein2005-hw] partly encoding the presence
of cribriform growth inside the ISUP grade.

Although biochemical recurrence is a common end-point to study prostate
cancer progression, a clinical utility would be mostly found in
assessing time-to-metastases or death. However, time-wise, they are
typically significantly further separated from the surgical event,
making it harder to identify relationships between tissue morphology and
these end-points. Nevertheless, we would like to investigate them in the
future.

# Conclusions

In summary, we have developed a deep-learning-based visual biomarker for
prostate cancer recurrence based on tissue microarray hotspots of
prostatectomies. The DLS marker provides a continuous score predicting
the speed of biochemical recurrence. We obtained an odds ratio of 3.32
(CI 1.63 - 6.77; p=0.001) for a nested case-control study from Johns
Hopkins Hospital, matched on Gleason sum on other factors. Additionally,
we obtained an HR of 3.02 (CI 1.10 - 8.29; p=0.03) for an external
validation cohort from the New York Langone hospital, adjusted for ISUP
grade, pathological stage, preoperative PSA concentration, and surgical
margins status. Thus, this visual biomarker may provide prognostic
information in addition to the current morphological ISUP grade.

**Acknowledgments**

*This work was supported by the Dutch Cancer Society under Grant KUN
2015-7970.*

*This work was additionally supported by the Department of Defense
Prostate Cancer Research Program, DOD Award No W81XWH-18-2-0013,
W81XWH-18-2-0015, W81XWH-18-2-0016, W81XWH-18-2-0017, W81XWH-18-2-0018,
W81XWH-18-2-0019 PCRP Prostate Cancer Biorepository Network (PCBN),
DAMD17-03-1-0273, and supported by Prostate Cancer NCI-NIH grant (P50
CA58236).*

**References**

**Appendix can be found here:**
