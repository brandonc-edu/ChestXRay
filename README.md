# ChestXRay

In medicine, when a patient presents with issues within the chest, a typical procedure is to perform a chest x-ray on them. This chest x-ray is brought to radiologists, who will attempt to locate any abnormalities, analyze what they represent, and report their findings to the requesting physician.

Therein lies the problem: a radiologist must look over the chest x-ray themselves and spot any issues. While this may be a simple enough task for an experienced radiologist, the issue is that the radiologist, as a human being, is prone to many forms of human error. Long shifts with limited breaks, an inherently stressful environment, potentially tedious work, and other factors external to work; all contribute to the chance that a radiologist may make the very human error of misanalyzing a chest x-ray, which could have a major impact on the diagnosis (and indirectly, the prognosis) of the affected patient.

But how could this problem be fixed? There could be adjustments to ease the psychological burden, but if we’re unable to “fix” the radiologist, then perhaps we can provide a safety net: a machine learning model — unaffected by emotions, stress, or other human limitations — that can examine a chest x-ray and provide its diagnosis.
By producing a machine learning model that can diagnose chest X-rays, we essentially provide the radiologist with an immediate second opinion on the chest X-ray. This second opinion would reinforce a radiologist’s confidence in correct diagnoses, alert them to potentially misdiagnosed chest X-rays, and ultimately reduce human error's impact on chest X-ray diagnoses.

As mentioned previously, chest X-rays can be a critical component of a patient’s prognosis. A misdiagnosis could lead the patient down a useless or even harmful treatment plan and waste valuable time which the patient may not have. This measure, though it's only one component in a patient’s treatment, could very well tip the balance in their favor in life-or-death situations, ultimately saving human lives.

## Dataset

The chest X-ray dataset has been retrieved from [Kaggle](https://www.kaggle.com/datasets/pritpal2873/chest-x-ray-dataset-4-categories/data), with the dataset being sourced on Kaggle, and created and used for a [research paper](https://ijnrd.org/papers/IJNRD2311166.pdf) by Pritpal Singh. The dataset comprises 7132 samples with 4273 samples of Pneumonia, 576 samples of COVID-19, 700 samples of Tuberculosis, and 1583 samples of normal chest X-rays. The feature set is composed of all the pixel values for each image, however, there are varying sizes of images within the entire dataset, so no exact number of features can be given. However, our preprocessing step reduces them to a uniform size of 224 x 224, which yields 50,176 features.

## Report

Please open the pdf file in the repo to examine our report, which includes our development methodology (and how it changes over time), as well as our final results!
