This thesis will focus on the most common form of prostate cancer, prostatic
adenocarcinoma. For simplicity, we will refer to this type as just 'prostate
cancer.' This disease affects one in 1.4 million men every year, making it the
most prevalent common type of cancer in men (excluding skin
cancers).[@Sung2021-iz] Prostate cancer is a disease of the epithelial cells of
the prostate. Epithelial cells line our body cavities, hollow organs, and
glands. They undergo rapid proliferation, primarily due to damage. This
proliferation increases the risk of genetic mutations, ultimately increasing
the risk of a cell uncontrollably dividing. Together with enabling factors of
its tissue environment, this can give rise to cancer.

In general, the more aggressive cancerous cells are, the less they will behave
and morphologically appear like their original function. The prostate is a
gland that produces prostatic fluid. The fluid is transported to the urethra by
small 0ubes. These tubes, called prostatic glands, are lined with epithelium.
Low-grade cancer will thus mimic those gland structures. High-grade prostate
cancer loses its structural morphology, forming sheets of cells or even
quasi-randomly dispersed individual cancerous cells.

American pathologist Donald Floyd Gleason systematically wrote down the
correlation between growth patterns and prognosis in prostate cancer in the
1960s[@cite_original]. Pathologists still use this Gleason grading, albeit
several revisions later[@Epstein2016-im], to classify prostate cancer.

![](chpt1_imgs/Gleasonscore.jpg)

**Gleason's growth patterns.** Image of the Gleason score for prostate cancer
grading based on original description in 1977. From: Morphology & Grade.
ICD-O-3 Morphology Codes. National Institutes of Health.[@zotero-512]

# Prognostic biomarkers

To decide on a treatment plan, clinicians divide patients into risk groups
according to traditional baseline characteristics, such as PSA blood level,
Gleason grade, tumor location, tumor size, and lymph node
status.[lam2019] This information is gathered from histopathological,
radiological assessment, and lab assessments. These assessments can be
considered biomarkers as they indicate the prognosis of a patient[@chen2011].
The more precise these assessments are, the better we can tailor the treatment
to the specific patient; this is known as personalized medicine.

To make treatment more tailored to the patient, researchers try to develop new
biomarkers. There is a demand for new biomarkers because most prostate cancers
progress so slowly that they are unlikely to threaten the affected individual's
survival, and patients with the same histological and clinical characteristics,
can have strikingly different outcomes [@cucchiara2018]. Being able to pick out
patients with good prognostis would improve their quality of life since
treatments for prostate cancer obviously have adverse effects
(#tab:adverse){reference-type="ref" reference="tab:adverse"}). Equally so for
patients for which we can find out the treatment will not contribute to their
health. To prevent adverse effects and increase treatment response, researchers
are developing new markers in genomics[@cucchiara2018], radiology[@roest2023],
and pathology, the latter of which is the subject of this thesis.

\pagebreak
::: {#tab:adverse}
  ------------------------ ------------------------------- --------------------------------------
  **Treatment Option**     **Disease Progression**         **Potential Adverse Effects**
  Active surveillance      Localized                       Illness uncertainty
  Radical prostatectomy    Localized                       Erectile dysfunction
                                                           Urinary incontinence
  External beam radiation  Localized and advanced disease  Urinary urgency and frequency
                                                           Dysuria, diarrhea and proctitis
                                                           Erectile dysfunction
                                                           Urinary incontinence
  Brachytherapy            Localized                       Urinary urgency and frequency
                                                           Dysuria, diarrhea and proctitis
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
  ------------------------ ------------------------------- --------------------------------------

  : Common Prostate Cancer Treatment Options and Potential Adverse Effects,
  reproduced from Dunn et al.[@dunn2011] ::: \pagebreak

## Biomarkers based on histopathology

We know that histopathology holds prognostic information. Commonly, pathologist
also report extra-capsular extension of the tumor and perineural invasion, both
signs of poor prognosis. As mentioned earlier, the Gleason patterns were
discovered by recording patient prognosis. Gleason growth patterns are grouped
into five different groups, of which current pathologists mainly use the last
three. It's not hard to imagine there being more clues in the morphology of the
behavior of the tumor, if only because the landscape of prostate cancer growth
patterns is certainly more complex than the three groups we divide them in. Of
note, recently, the 'subpattern' cribriform-like growth was discovered to be an
aggressive pattern. 

However, these visual biomarkers are hard to explicitly specify and quantify
manually. Luckily, machine learning can help. The first chapter will discuss
this approach further. However, it makes sense to introduce this research
field, computational pathology, first.

## Computation Pathology 

Pathology is undergoing a digital revolution. More and more labs are purchasing
whole-slide scanners, with some already reading most slides digitally. Glass
slides are digitized, resulting in gigapixel digital images, commonly referred
to as whole-slide images (WSIs). Once the data is digital, opportunities for
computational analysis and assistance arise. 

Litjens et al. [1] gave an overview of deep learning applications in
computational pathology up to 2016. Some early successes in the field focused
on segmentation, tissue classification, and disease classification. Often
reaching comparable results on the manual performance of the tasks by
pathologists. Notably, the vast majority of these tasks are not on prognosis or
treatment response prediction. Likely due to the fact these tasks are
relatively easier and the kind of data needed is relatively cheap to obtain
compared to survival data. 

All state-of-the-art methods use some flavor of deep learning. A method where
we train a model with multiple layers of computations, interwoven with
non-linearities. A decade ago, optimizing these neural networks on GPU
accelerators became common. The use of GPU made us able to develop models with
a lot of layers (hence 'deep' learning) on large datasets. From the start, we
have been using a type of neural network, termed convolutional neural networks,
in vision applications. 

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

# Weakly supervised methods 

For others, several authors have suggested approaches to train convolutional
neural networks (CNNs) with large input images while preventing memory
bottlenecks. Their methods can be roughly grouped into three categories: (A)
altering the dataset, (B) altering usage of the dataset, and (C) altering the
network or underlying implementations.

<!--
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
-->

#  Thesis overview

Chapter 2 proposes a method called "streaming" to train convolutional neural
networks end-to-end on multi-megapixel histopathology images, circumventing
memory limitations. We tiles the input image and reconstructs activations,
allowing the use of entire high-resolution images during training without
cropping. 

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

\pagebreak
