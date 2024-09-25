## Project Options
- CT Segmentation
- X-Ray Segmentation
## Chest X-Ray Overview
- Also called "Chest Radiograph"
- Two types
	- PA -- back to the XRay machine
	- AP -- front to the XRay machine
	![[Screenshot 2024-09-10 at 1.06.02 PM.png]]
- They look mostly similar, but there are small differences that could cause sensitivity to the model
- What we can see on the xray
  ![[Screenshot 2024-09-10 at 1.07.12 PM.png]]
	  - The right and left sides are flipped.
- One of the biggest issues with AI is detecting the nodules
	- A nodule is less than 3cm and if it is bigger, then it is a mass
		- These are types of tumors
- Detecting Lung Cancer earlier using AI would be one of the most important challenges to do this
![[Screenshot 2024-09-10 at 1.18.44 PM.png]]
- Companies doing this:
	- Lunit Insight CXR
	- Kakao's Kara CXR
## Data
![[Screenshot 2024-09-10 at 1.20.35 PM.png]]
- We will be using the **NIH-CXR** for our project
### DICOM Format
- Data standard
![[Screenshot 2024-09-10 at 1.21.38 PM.png]]
- We can use `pydicom` to read them in python
- These files contain both the image and metadata
# Project Guidelines
- We will be given skeleton code for supervised training using a vision transformer model
- Our task:
	  Beat the TA's model using the same architecture and training data
	  Strategy is up to us
		  - Data augmentations
		  - Pre-training methods
- CXR classification is multilabel (not multiclass) classification
	- I.E. binary classification for each label. Different from n-way classification such as the ImageNet challenge
	- So your output needs to be a vector with as many dimensions as target lesions
- CXR datasets are imbalanced
	- Dealing with this imbalance can be important for model performance
- Some lesions are harder to detect than others
	- For example, pleural effusion is easier than nodules
	- Lesions that are more easily visible for the human eye tend to be easier for machines to detect as well
- Different from natural images... as you will see
	- Lower inter-class variability ("label sharpness")
	- Small, detailed features tend to be more important
![[Screenshot 2024-09-10 at 1.26.30 PM.png]]
- **ONLY USE PRETRAINING OR DATA AUGMENTATION TO IMPROVE PERFORMANCE**
	- We should not change the model size, and only augment the 
	- 