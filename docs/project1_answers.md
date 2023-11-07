# Project 1 answers

### 1 - Registration and staff meeting
- We acknowledge that we submitted the registration form 
- We met with Ravi on his office hour on October 16th (at approximately 7:50pm)

### 2 - Dataset selection

The dataset we are going to use is the following : https://www.kaggle.com/datasets/diraf0/sloan-digital-sky-survey-dr18

This dataset is the result of the Sloan Digital Sky Survey, a study that archives and gathers the visible stellar objects and classifies them as Galaxies, Quasars, or Stars. 
- It has 43 columns and more than a hundred thousand rows.
- It contains many continuous covariates, and one categorical for the classification of the stellar object.
- It is not a time series.
- The data is gloablly clean and only a few to no values are missing. 

This dataset is publicly accessible and under the following license : https://creativecommons.org/licenses/by-sa/4.0/
We can use it for research purpose, and will only need to give appropriate credit if we publicly share our model.

### 3 - Holdout set

We acknowledge that we have reserved 20% of the rows of our dataset.

### 4 - Outcome
For our continuous outcome, we will predict redshift, which is interesting because it determines how galaxies are distributed throughout the universe, informs us about the expansion rate of the cosmos, and contributes to our understanding of dark matter and dark energy. Additionally, redshift data can reveal galaxy properties and evolution, making it ideal for applying machine learning techniques to large datasets like the SDSS for insightful astrophysical discoveries.

For the categorical outcome, we'll be classifying whether a stellar object is a star or not. This classification is essential for astrophysical research, allowing scientists to study the properties and distribution of stars relative to other celestial objects. With a dataset like SDSS, which provides extensive data on various objects, machine learning can efficiently differentiate stars, aiding in large-scale cosmic surveys and contributing to our broader knowledge of stellar evolution and galaxy formation.

### 5 - 

(a) Space studies have picked up a good pace for the past ten years with a massive expansion of earth telescopes like Hubble and space telescopes like the newly launched James Webb Space Telescope. Thus, massive amounts of data are collected everyday for us to understand better the contents of our universe. It is then important to know how to classify these objects in the best way possible, and how to characterize them using different properties. This is where machine learning comes into play: as stated before, classifiers and redshift predictions will contribute to a better understanding of the distant universe, which makes this dataset exciting to dive into.

(b) The data collection process for SDSS involves using a combination of optical and infrared imaging and spectroscopy to study celestial objects. Different phases of SDSS have used more and more advanced telescopes, instruments and techniques to improve data quality and expand the survey's scope. For example, the introduction of the Sloan 2.5 m telescope and the implementation of robotic fiber positioners (focal plane system) in SDSS-V (the current phase) aimed to improve efficiency and data quality.

A few issues with data collection can be outlined:

•	Coordination: Coordinating data collection from multiple telescopes and instruments in different locations is a complex task. It requires precise synchronization to ensure the accuracy and reliability of the data.

•	Cross-Matching Catalogs: Cross-matching of different input catalogs to ensure each object in the sky has a unique identifier, whose accuracy is crucial, as it affects the quality of the catalog and the identification of targets.

•	Instrumentation: Ensuring proper calibration, maintenance, and performance of new instruments and telescopes for accurate data collection.

•	Data Accessibility: Ensuring that data are accessible to the scientific public, including proper documentation. 

(c) Engineered features

(d) 

(e)

(f)

(g) Little to no data is missing, the dataset is rather complete. We plan on deleting the few lines where data is missing.