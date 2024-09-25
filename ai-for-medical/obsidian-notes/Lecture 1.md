**Date: 2024-09-03**

# Grading
Attendance: 10% (0 for more than 3 absences)
Two Projects: 70%
Scribe: 20%
- Scribe performs note-taking about class contents and submit them in LaTeX format for review on KLMS.
- You can sign up for it and will take notes on that day, including any relevant figures.

# Project Logistics
Midterm Project (35%)
- Individual project
- Topics:
	- CXR Classification
	- CT Segmentation
- We will be provided some benchmark datasets and references
Final Project (35%)
- Team Project (2 people)
- Topics:
	- CT Reconstruction
	- MR Reconstruction
	- Ultrasound Beamformer
# Course Overview
- There is a barrier of entry:
	- You need to know image formation and underlying physics of the technologies in order to enter the field of AI-accelerated Medical Imaging.
# Advantages of Medical Imaging
- Typically, there are three ways to see inside the human body
	- Surgery
	- Endoscopy
	- Medical Imaging
- Biomedical Imaging allows us to achieve this while being completely noninvasive
- Examples include:
	- X-ray, MRI, CT, SPECT, PET, Ultrasound
- While there is some radiation exposure, it is usually negligible
- We can see things clearly, such as physiology, metabolism, and bodily functions
# Biomedical Imaging Modalities
**Radiographic Imaging**: transmission of x-rays through the body.
- Examples: X-ray or CT
**Nuclear medicine:** emission of gamma rays from radio-tracers in the body
- Examples: planar scintigraphy, SPECT, PET
**Ultrasound**: Reflection of ultrasonic waves within the body
**MRI:** Precession of spin systems in a large magnetic field

## Projection and Tomographic Imaging
- Projection imaging - -essentially a shadow
	- E.g. X-ray
![[Pasted image 20240903134647.png]]

- Tomographic Imaging
	- E.g. CT, SPECT, PET, MRI

![[Pasted image 20240903134602.png]]
# Imaging Types
## Projection Radiography (X-Ray)
- A form of electromagnetic radiation
- Wavelength: 0.01~10nm
- It can penetrate solid objects
- First form of imaging

## Computed Tomography (CT)
- A series of 2D X-Ray images taken along a single axis of rotation
- Uses a special tomographic reconstruction algorithm
	- Filtered back projection, Radon transform
- Reduction of radiation dose while maintaining image quality

## Positron Emission Tomography (PET)
- Detecting pairs of gamma rays emitted by a positron-emitting radionuclide (tracer)
- Flourodeoxyglucose (FDG), an analogue of glucose, in combination with PET provides information of regional metabolic activity (e.g. imaging of cancer metastasis)
- PET-CT or PET-MRI combinations
## Single Photon Emission Computed Tomography (SPECT)
- Planar scitigraphy: imaging performed with gamma camera after injection of radionuclide
## Ultrasound
- Cyclic sound pressure with a frequency greater than the upper limit of human hearing (20kHz)
## Magnetic Resonance Imaging (MRI)
- Aligning hydrogen atoms along a strong magnetic field
- Providing radio frequency (RF) energy through an RF coil to cause magnetic resoance and to recieve RF signals.
- Distinct from other imaging modalities, MRI provides various soft tissue contrast depending on scan parameters and imaging methods (MRI pulse sequences)
# Electromagnetic Spectrum of Imaging
![[Pasted image 20240903135810.png]]

# X-Ray Physics
![[Pasted image 20240903140225.png]]

CT energy range
- Soft x-ray: 10nm (124eV) ~ 0.1nm (12.4keV)
- Diagnostic x-ray: 0.1nm (12.4keV)~0.01nm(124keV)
- Higher energy: more  penetration, less contrast
