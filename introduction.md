This thesis will focus on the most common form of prostate cancer, prostatic
adenocarcinoma. For simplicity, we will refer to this type as just 'prostate
cancer.' This disease affects one in .. men every year, making it the most
prevalent common type of cancer in men (excluding skin cancers). Prostate
cancer is a disease of the epithelial cells of the prostate. Epithelial cells
line our body cavities, our hollow organs, and glands. They undergo rapid
proliferation, primarily due to damage. This proliferation increases the risk
of genetic mutations, ultimately increasing the risk of a cell uncontrollably
dividing. Together with enabling factors of its tissue environment, this can
give rise to cancer.

In general, the more aggressive cancerous cells are, the less they will behave
and morphologically appear like their original function. The prostate is a
gland that produces prostatic fluid. The fluid is transported to the urethra by
small tubes. These tubes, called prostatic glands, are lined with epithelium.
Low-grade cancer will thus mimic those gland structures. High-grade prostate
cancer loses its structural morphology, forming sheets of cells or even
quasi-randomly dispersed individual cancerous cells.

American pathologist Donald Floyd Gleason systematically wrote down the
correlation between growth patterns and prognosis in prostate cancer in the
1960s[@cite_original]. Pathologists still use this Gleason grading, albeit several
revisions later[@Epstein2016-im], to classify prostate cancer.

[Insert image Gleason grown patterns]

# Prognostic biomarkers

To decide on a treatment plan, clinicians divide patients into risk groups
according to traditional baseline characteristics, such as PSA blood level,
Gleason grade, tumor location, tumor size, and lymph node
status.[lam2019] This information is gathered from histopathological,
radiological assessment, and lab assessments. These assessments can be
considered biomarkers as they indicate the prognosis of a patient[@chen2011].
The more precise these assessments are, the better we can tailor the treatment
to the specific patient; this is known as personalized medicine.[@citation_needed] 

To make treatment more tailored to the patient, researchers try to develop new
biomarkers. There is a demand for new biomarkers because most prostate cancers
progress so slowly that they are unlikely to threaten the affected individual's
survival. However, treatments for prostate cancer obviously have adverse
effects (#tab:adverse){reference-type="ref" reference="tab:adverse"}). To
prevent adverse effects and increase treatment response, researchers are
developing new markers in genomics, radiology, and pathology, the latter of
which is the subject of this thesis.

::: {#tab:adverse}
  ------------------------ ------------------------------- ----------------------------------------------------------------------------------------------------
  **Treatment Option**     **Disease Progression**         **Potential Adverse Effects**
  Active surveillance      Localized                       Illness uncertainty
  Radical prostatectomy    Localized                       Erectile dysfunction
                                                           Urinary incontinence
  External beam radiation  Localized and advanced disease  Urinary urgency and frequency, dysuria, diarrhea and proctitis
                                                           Erectile dysfunction
                                                           Urinary incontinence
  Brachytherapy            Localized                       Urinary urgency and frequency, dysuria, diarrhea and proctitis
                                                           Erectile dysfunction
                                                           Urinary incontinence
  Cryotherapy              Localized                       Erectile dysfunction
                                                           Urinary incontinence and retention
                                                           Rectal pain and fistula
  Hormone therapy          Advanced                        Fatigue
                                                           Hot flashes, and flare effect
                                                           Hyperlipidemia
                                                           Insulin resistance
                                                           Cardiovascular disease
                                                           Anemia
                                                           Osteoporosis
                                                           Erectile dysfunction
                                                           Cognitive deficits
  Chemotherapy             Advanced                        Myelosuppression
                                                           Hypersensitivity reaction
                                                           Gastrointestinal upset
                                                           Peripheral neuropathy
  ------------------------ ------------------------------- ----------------------------------------------------------------------------------------------------

  : Common Prostate Cancer Treatment Options and Potential Adverse Effects, reproduced from Dunn et al.[@dunn2011]
:::

## Recent genomic biomarkers (mutations & prostateDx etc)

Besides these standard clinical assessments, there are increasingly genetic
markers used. Genomic alterations, such as mutations, amplifications,
deletions, and fusions can be indicative of prognosis or treatment response. 

## Recent radiology-based biomarkers

## Recent pathology-based biomarkers

```
Genomic biomarkers 
Besides traditional pathology, molecular biomarkers are
gaining traction in precision medicine. Genomic alterations, such as mutations,
amplifications, deletions, and fusions can be indicative of diagnosis or
prognosis. For example, molecular tests like Oncotype DX for breast cancer or
ProstateDX for prostate cancer are used in clinical practice.

For prostate cancer, the Prolaris test by Myriad Genetics looks at the
expression of cell cycle progression genes. The Decipher test by GenomeDX
Biosciences investigates the whole transcriptome. Both tests aim to predict
prognosis after prostatectomy or radiotherapy.

However, the costs of these tests can be prohibitive for wide adoption.
Research aims to find cheaper solutions by investigating whether computational
analysis of traditional pathology slides could uncover similar information. For
example, Coudray et al. [1] showed that CNNs can find EGFR mutations in lung
cancer slides.

Overall, molecular information provides opportunities for precision medicine.
Ideally, we can leverage computational approaches applied to traditional
pathology to unlock this information in a cheap and accessible manner.

## "Visual" biomarkers 

Besides molecular biomarkers, morphology and tissue
architecture can also hold prognostic information. As pathologists gain
experience, they unconsciously pick up these visual patterns and take them into
account during diagnosis.

However, these visual biomarkers are hard to quantify and explicitly specify,
especially compared to molecular biomarkers. Still, they have great potential
for computational pathology algorithms. Deep learning has shown promise in
finding these patterns directly from histopathology images in an end-to-end
manner.

For example, Wulczyn et al. [2] predict survival in colorectal cancer using
tissue slides and patient data. Coudray et al. [1] find histological patterns
predictive of EGFR and KRAS mutation status in lung cancer. Pinckaers et al.
[3] show that CNNs can find prognostic morphologic patterns in prostate cancer.

In summary, besides known molecular biomarkers, subtle visual patterns likely
hold additional prognostic information. Advanced machine learning techniques
may help unlock the prognostic potential of traditional pathology slides.

<!-- References: -->

<!-- [1] Coudray, N., Ocampo, P.S. & Sakellaropoulos, T. et al. Classification and -->
<!-- mutation prediction from non-small cell lung cancer histopathology images using -->
<!-- deep learning. Nat Med 24, 1559–1567 (2018). -->
<!-- https://doi.org/10.1038/s41591-018-0177-5 -->

<!-- [2] Wulczyn, E., Steiner, D.F., Moran, M. et al. Interpretable survival -->
<!-- prediction for colorectal cancer using deep learning. npj Digit. Med. 4, 71 -->
<!-- (2021). https://doi.org/10.1038/s41746-021-00431-3 -->

<!-- [3] Pinckaers, H., van Ipenburg, J., Melamed, J. et al. Predicting biochemical -->
<!-- recurrence of prostate cancer with artificial intelligence. Nature Machine -->
<!-- Intelligence (2022). https://doi.org/10.1038/s42256-021-00454-0 -->
```
# Convolutional neural networks

Convolutional neural networks (CNNs) have emerged among the state-of-the-art
machine learning algorithms for various computer vision tasks, such as image
classification and segmentation. 

The central component of a convolutional neural network is often represented as
a sliding kernel (or filter) over an input matrix, producing an output matrix.
This has several advantages; we can use a smaller kernel than the whole making
the network less complex while exploiting the fact that objects in the image
are translation invariant. A cat in the upper-left corner is still a cat in the
lower right corner. We introduce this inductive bias to the network by using
convolutions.

Most convolutional neural network architectures have alternating blocks of
layers consisting of a convolutional operation, a non-linear activation
function, and often a normalization operation. The non-linearities are
essential, as they make the networks able to represent more complex
(non-linear) functions. Normalization layers bound the output of the block to
be within a specific range which helps during the optimization of the network.

Even though sliding kernels are less complex than having one parameter per
input value, the network architectures have evolved to become deeper and wider
to enhance their accuracy further. Training larger CNNs demands larger amounts
of computer memory, which increases exponentially with the size of input
images. Consequently, most natural image datasets in computer vision, such as
ImageNet and CIFAR-10, contain sub-megapixel images to circumvent memory
limitations. 

In specific domains like remote sensing and medical imaging, there is a need to
train CNNs on high-resolution, where most of the information is contained.
Ideally we want to combine the high-resolution information with more global
context, as pathologists can do during daily practive. However, computer memory
becomes a limiting factor. The memory requirements of CNNs increase
proportionally to the input size of the network, quickly filling up memory with
multi-megapixel images. As a result, only small CNNs can be trained with such
images, rendering state-of-the-art architectures unattainable even on large
computing clusters.

## Computation Pathology 

```
Pathology is undergoing a digital revolution. With whole slide scanners, glass
slides can be digitized resulting in gigapixel digital images, commonly
referred to as whole slide images (WSIs). This enables new opportunities for
computational analysis and assistance. The field leveraging computational
techniques on digital pathology images is referred to as computational
pathology.

Some early successes in the field focused on segmentation and tissue
classification. For example, training neural networks to find tumor areas or
specific tissue structures. However, ultimately the goal is not a precise
segmentation or localization, but to mimic and enhance the diagnostic
capabilities of pathologists. 

Researchers have shown the potential of using deep learning on histopathology
for diagnosis and prognosis. For example, Litjens et al. [1] gave an overview
of deep learning applications in computational pathology up to 2016. They
highlight areas as mitosis counting, tissue classification, detection of lymph
node metastases, and prognosis prediction. 

More recent work focuses on clinical implementation and validation. Campanella
et al. [2] propose a system for prostate cancer detection trained with weak
slide-level labels. Bulten et al. [3] validate a system to determine the
Gleason grade of prostate cancer biopsies, showing human-level performance.   

Especially interesting for clinical implementation is the ability to learn with
weaker labels, circumventing expensive precise annotations. Although labels
directly from pathology reports are noisy, the information is good enough to
guide treatment decisions. As such, recent focus, like in this thesis, is to
train deep learning models directly using these readily available labels.

Overall, deep learning shows potential to enhance and assist pathology using
the digital pathology slides. The following chapters will dive deeper into the
specific computational pathology tasks tackled in this thesis.
```
<!-- References: -->

<!-- [1] Litjens G, Kooi T, Bejnordi BE, et al. A survey on deep learning in medical image analysis. Med Image Anal. 2017;42:60-88. doi:10.1016/j.media.2017.07.005 --> 

<!-- [2] Campanella G, Hanna MG, Geneslaw L, et al. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. Nat Med. 2019;25(8):1301-1309. doi:10.1038/s41591-019-0508-1 -->

<!-- [3] Bulten W, Pinckaers H, van Boven H, et al. Automated deep-learning system for Gleason grading of prostate cancer using biopsies: a diagnostic study. Lancet Oncol. 2020;21(2):233-241. doi:10.1016/S1470-2045(19)30739-9 -->


# Weakly supervised methods 

For others, several authors have suggested approaches to train convolutional
neural networks (CNNs) with large input images while preventing memory
bottlenecks. Their methods can be roughly grouped into three categories: (A)
altering the dataset, (B) altering usage of the dataset, and (C) altering the
network or underlying implementations.

```
As a result of the memory limitations, most networks are trained on small
patches from the WSI. This patch-based training requires detailed pixel-level
annotations of the classes, such as outlines of tumor regions by an expert
pathologist. However, pixel-level labeling is expensive, time-consuming, and
not routinely performed in clinical practice.

To overcome this, researchers have developed weakly supervised methods that can
learn from readily available slide-level labels in pathology reports, without
expensive patch-level annotations. A popular approach is multiple instance
learning (MIL). In MIL, a CNN is trained on patches under the assumption that a
positive slide contains at least one positive patch. Only the most informative
patch per slide is used for backpropagation. However, this limits the
field-of-view and contextual information.

Another weakly supervised approach is to compress the WSI into a
lower-dimensional latent space using autoencoders. The embedding can be used to
train a model per patch. However, useful information may be lost during
compression. Also, relationships between patches are disregarded.

Methods like recurrent attention networks try to increase the field-of-view by
analyzing multiple patches. However, memory constraints limit the patch size,
and patches must be kept in memory to calculate attention weights. Overall,
most current weakly supervised methods rely on patches, constraining the
network's field-of-view.

In this thesis, a novel streaming method is proposed to train CNNs end-to-end
on entire WSIs with slide-level labels. By reconstructing activations
tile-by-tile, streaming removes the need to crop images. A CNN can learn from
full contextual information at high resolution, without relying on patches.
Experiments show streaming reaches performance on par with patch-based methods
needing more supervision. Thus, streaming enables direct learning from
morphology to aid histopathology analysis using readily available slide-level
labels. Chapter 2 will go more into depth on this.
```

#  Thesis overview

```
Chapter 2 proposes a method called "streaming" to train convolutional neural
networks end-to-end on multi-megapixel histopathology images, circumventing
memory limitations. It tiles the input image and reconstructs activations,
allowing the use of entire high-resolution images during training without
cropping. Experiments show streaming enables the use of larger images,
improving performance on public datasets.

Chapter 3 applies streaming to train models on whole prostate biopsy images
using only slide-level labels from pathology reports. It shows a modern CNN can
learn from high-resolution images without patch-level annotations. The method
reaches similar performance to state-of-the-art patch-based and multiple
instance learning techniques.

Chapter 4 demonstrates a deep learning system to predict biochemical recurrence
of prostate cancer using tissue morphology. Trained on a nested case-control
study and validated on an independent cohort, the system finds patterns
predictive of recurrence beyond standard Gleason grading. Concept-based
explanations show tissue features aligned with pathologist interpretation.

In summary, the thesis explores computational pathology methods to analyze
entire high-resolution histopathology images despite memory constraints. It
shows neural networks can learn from morphology to aid prostate cancer
diagnosis and prognosis when trained end-to-end on whole images using readily
available slide-level labels.
```
