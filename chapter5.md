---
title: Discussion
---

In this PhD thesis, we addressed the main problem of the dimensionality of
whole slide images in computational pathology and prostate cancer prognosis.
These images are so large that previous works typically focused on the patch
level or employed pre-trained mechanisms to predict clinically relevant
endpoints based on the whole slides. In the first chapter of this thesis, we
proposed a new method called "streaming," which aims to solve the
dimensionality problem with a memory-efficient implementation of convolutions.
We demonstrated that this approach is numerically equivalent to a convolutional
neural network, without batch normalization, on a subset of ImageNet.
Additionally, we showed that increasing the resolution leads to higher
performance due to the increase in detail. The second chapter of the thesis
employed this new method to predict prostate cancer on whole slide images. When
compared to a top multiple instance learning method, the streaming method
outperformed it and demonstrated better generalization to an unseen scanner. We
also showed that using gradient saliency, the method can provide some
explainability by localizing where the network is focusing its attention. These
gradient saliency maps correspond to the predictions of a multiple instance
learning network, indicating their correlation to the classification endpoint.
In the third chapter, we demonstrated that our method can predict prostate
cancer recurrence in patients after prostatectomy. By using histology data, we
could further improve our prognostication for these patients beyond typical
clinical variables, such as Gleason scores. This chapter laid the foundation
for extracting these features from whole slide images since, in this work, we
used predefined extracted areas from the original slide based on the highest
grade tumor. However, we also considered the possibility that there might be
more information on the slide to predict prognostics. This hypothesis was
explored in the final chapter, where we analyzed whole slide images of a
population cohort of patients who had undergone prostatectomy. We attempted to
predict the time to biochemical recurrence for these patients. Although we
could not achieve statistically significant results due to the size of the
dataset, there was a discernible signal in the slides. Using explainability
methods, we demonstrated that the network focuses on the tumor and other
relevant areas. This lays the groundwork for fully learning end-to-end,
clinically interesting endpoints from histology images while harnessing the
full potential of neural networks to find relevant features without manual
feature engineering. One could argue that working on patches adds assumptions
to the task and lacks context due to cropping the slide. However, deep learning
has shown that neural networks can learn these assumptions and signals
themselves, given enough data and appropriate labels. Interestingly, chapter 2
revealed that the streaming method is data-efficient, even compared to multiple
instance learning, which utilizes patches. We hypothesize that this is because
streaming employs very large feature maps that are harder to overfit.
Furthermore, we use an aggregation layer later in the network while the feature
map is still relatively large, making it more difficult to overfit to specific
noise.


## Generalization

One of the challenges in this field is the problem of model generalization.
Many publications focus on creating a variation of a model for a specific
dataset, which may not generalize well to other datasets, containing slides of
different labs or different scanners. This may be the result of differences in
color distribution, compression, or image properties in various datasets. To
bring these algorithms into the clinic, it is essential to address the bias in
the networks.


## Academic and industry overlap

Unfortunately, the current academic incentives means that a lot of effort in
the research community is not focused on these core problems in the field.
Several factors contribute to this, such as funding concerns, the need for PhD
students to publish, and the competitive nature of the field. This leads to an
unfair competition where larger AI firms with more resources drive the field
forward, while academia often lags behind. [TODO fix dictation]So another class
that I see of the problem of tracking smallest plural fast pathologist was
using deep learning as a alright, let's switch and put them in it's still
fairly easy to get a tornado set for certain Mainly in the event of a task with
the baton of this and trading them off the shelf for small variation of
existing model on them.This is almost guaranteed to work and it might be almost
the wonders of finally creating a publication, given the incentives of a PhD,
in this country, everyone needs to publish a certain number of papers.It is
very appealing to go this route.Since it is still happening in radiology of
fields, which has been digitized way earlier than pathology. I'm afraid that
these publications will keep getting written and worked on in the near future.
It is telling that when asking around, people rarely read each other's research
in this field, attempting to read the papers of the bigger AI discussing
methods. In my opinion, this leads to research that reinvents the wheel all of
the time and doesn't build upon each other. And given the hype of deep
learning, given a big enough dataset, it is still possible to publish in a very
high impact journal.We published a Gleason grading algorithm in a very high
impact journal. And, cynically, we could say that we already knew it was going
to work if we did not do that. AI researchers in this field will not learn that
much from this paper, physicians may learn that it's possible to do these tasks
at an expert level. But I hope that this is soon not a surprise anymore and
everybody knows that given enough data, these algorithms can find the right
patterns. Another issue is the career ladder in academia, where personal
publications are very important. This may lead to researchers focusing on
short-term projects that are not as impactful in the long run. In my opinion,
academia, funded by public money and without profit incentives, should focus on
long-term goals and moon-shot ideas, leaving short-term projects to companies
that can develop and implement them more quickly. On short term research, as
mentioned above, like smaller papers, smaller tasks. In my opinion, when you
can create an algorithm that is commercially interesting, meaning it can
actually be implemented and sold in the clinic and you can develop this
algorithm within a year, it may not make sense to develop those in academia,
but let companies develop them. Since academia is funded by public money and
doesn't have any profit incentive they should focus on the long-term work, on
moonshot ideas. With the risk of not getting funding and the fact that PhD
students have to publish, make this unlikely to happen. Meaning that right now,
academia is doing a lot of similar work as companies do, often with way less
funding. This leads to an unfair advantage or unfair competition. The companies
often have way more hardware. Especially in the field of image analysis in
general. After language processing you see that bigger AI firms with large sums
of money drive the field forward. And academia often hobbles  after those
companies. Nonetheless, fundamental research remains crucial, and the
importance of computational pathology and prostate cancer prognosis should not
be underestimated. It is essential to continue exploring new methods and
techniques to improve patient outcomes and advance the field.


## Bigger picture

The computational pathology field has started working on predicting
human-defined features, such as mitotic count, tumor grading, and regular
disease classification. What is happening now is that we are deriving more
features using deep learning in cohorts of patients where clinical endpoints
are available. Since deep learning works on input images, it has the potential
to identify interesting features from histology that could predict treatment
response. These solutions can assist oncologists in helping patients, providing
broader benefits than just automating small tasks for the pathologist. There
are still plenty of papers published where one can quickly compile  a dataset
for a very specific, narrow task, perhaps using just a couple of hundred
patients. The approach is to take this task, develop a model, and then attempt
to predict outcomes for this task. What we have seen recently in natural
language processing (NLP) and other fields is that given enough data and
correlations, large-scale models can be successful even when tackling tasks
that are not specifically designed for them. We should learn from these NLP
foundational models and aim for bigger models, rather than focusing on
publishing papers for small, narrow tasks. By expanding our scope and
ambitions, we can make greater strides in computational pathology and related
fields. One of the really exciting projects in this field is the EU's Big
Picture project, where a total of one million slides will be collected. Given
enough data and the right techniques, such as self-supervised learning, models
can predict or estimate prognosis better than tumor grading. Self-supervised
learning networks can discover patterns on their own when provided with
sufficient data. These networks have the potential to automatically generate
grading schemes and extract valuable information from the data. It is not
surprising that self-supervised learning networks can provide more prognostic
information than traditional methods used by oncologists. Instead of relying on
discrete grades (e.g., 3-5 grades), these models can operate in a continuous
manner, enabling a more nuanced understanding of the data. As a result,
statistical models can predict or estimate diagnoses more effectively than the
human-based approaches currently in use. In our study, a couple of hundred
patients were analyzed using a specific model. We found that by clustering the
latent space we could find the cribiform growth pattern, even if they are not
specifically targeted. However, we must also be careful with interpreting
patterns or, in this case, clustering of data. It is crucial not to overstate
the importance of certain patterns, especially since we may not know their
relevance in the larger context. The data we can derive from biological tissue,
as well as the knowledge encoded in medical literature, can be used to build
self-supervised foundational models that can potentially discover relationships
we are not yet aware of. Extend.. (all the omics etc) (Although interobserver
variability is a significant problem in computational pathology, the model's
predictions do not stand alone in determining a patient's treatment. Instead,
they contribute to predicting whether a patient will benefit from a specific
treatment.)
