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

# The Gleason grading system

In general, the more aggressive cancerous cells are, the less they will behave
and morphologically appear like their original function. The prostate is a
gland that produces prostatic fluid. The fluid is transported to the urethra by
small tubes. These tubes, called prostatic glands, are lined with epithelium.
Low-grade cancer will thus mimic those gland structures. High-grade prostate
cancer loses its structural morphology, forming sheets of cells or even
quasi-randomly dispersed individual cancerous cells.

American pathologist Donald Floyd Gleason systematically wrote down the
correlation between growth patterns and prognosis in prostate cancer in the
1960s[@cite]. Pathologists still use this Gleason grading, albeit several
revisions later[@cite], to classify prostate cancer.

[Insert image Gleason grown patterns]

# Prognostic biomarkers

The Gleason grading system can be considered a biomarker as it indicates the
prognosis of a patient with prostate cancer[@chen2011]. Traditionally
clinicians use nomograms to decide treatment protocols, which combine the
clinical characteristics of the patients with other diagnostic tests, such as
histopathological, radiological assessment, and lab assessments. The more
precise these assessments are, the better we can tailor the treatment to the
specific patient, this is known as personalized medicine.[@cite] 

## Genomic biomarkers (mutations & prostateDx etc)

- trying to leverage machine learning to make radiology/pathology reports more precise

## "Visual" biomarkers

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

# Introduce Computation Pathology 

- examples of pathology research (Wouter's gleason papers etc)

Some computational pathology tasks can be achieved with patch-based methods. 

# Weakly supervised methods 

For others, several authors have suggested approaches to train convolutional
neural networks (CNNs) with large input images while preventing memory
bottlenecks. Their methods can be roughly grouped into three categories: (A)
altering the dataset, (B) altering usage of the dataset, and (C) altering the
network or underlying implementations.

Chapter 2 will go more into depth on this.

