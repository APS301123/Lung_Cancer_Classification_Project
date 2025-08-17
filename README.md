# Lung_Cancer_Classification_Project
Lung Cancer accounts for about 1 in 5 cancer deaths. Many papers discuss the need to non-invasively identity Lung Cancer. One such paper, *Noninvasive Urinary Metabolomic Profiling Identifies Diagnostic and Prognostic Markers in Lung Cancer* applies a Random Forest Classifier(n_estimators = 5000) to an untargeted metabolomic dataset (Metabolights MTBLS28) the researchers created from human urine samples(1005 samples and 3166 metabolites -> 1807 ESI+, 1359 ESI-). This form of identification had an accuracy of 0.781 in the original paper. This project attempts to try and achieve higher accuracy by using Gradient Boosting and Ensemble Models.

---

# Table of Contents

**Section 1: Setup<br>**

  1.1 Imports and Settings<br>
  1.2 Read in and Preprocess Dataset<br>
  1.3 Create Test-Training Test split<br>

**Section 2: ML Models with Non-Dimensionally Reduced Data<br>**

  2.1 RF Model Initial Test and Analysis<br>
  2.2 RF Model Tuned Test, Analysis, and Comparison<br>
  2.3 LGBM Model Initial Test and Analysis<br>
  2.4 LGBM Model Tuned Test, Analysis, and Comparison<br>
  2.5 XGB Model Initial Test and Analysis<br>
  2.6 XGB Model Tuned Test, Analysis, and Comparison<br>
  

**Section 3: Ensemble Model<br>**
  3.1 Add additional model CatBoost

---
# Models Used
  Random Forest Classifier<br>
  LightGBM<br>
  XGBoost<br>
  CatBoost<br>
  **Voting Classifier Models:** <br>
  Ensemble 1: XGBoost + CatBoost  <br>
  Ensemble 2: LightGBM + XGBoost + CatBoost <br>


---
# Best Models
  **XGBoost**<br>
  <img width="293" height="412" alt="XGBoost Diagram" src="https://github.com/user-attachments/assets/43e69161-cc00-4dd1-a635-287e0b2a3ec8" />

  **Ensemble 1: XGBoost + CatBoost**  <br>
  <img width="276" height="512" alt="Ensemble 1 Diagram" src="https://github.com/user-attachments/assets/2baaaa9a-23a9-4a44-840a-21a5da75248c" />

---
# ML Model Metrics
<img width="689" height="282" alt="ML Model Metrics" src="https://github.com/user-attachments/assets/607de985-5c21-4799-bcb8-388d020dad69" />


---
# Python Packages Used

* Numpy
* Sklearn
* Matplotlib
* Pandas
* UMAP
* PCA

---
# Citations

Uses data uploaded to MetaboLights with the identifier MTBLS28 <br>

Ewy A. Mathé, Andrew D. Patterson, Majda Haznadar, Soumen K. Manna, Kristopher W. Krausz, Elise D. Bowman, Peter G. Shields, Jeffrey R. Idle, Philip B. Smith, Katsuhiro Anami, Dickran G. Kazandjian, Emmanuel Hatzakis, Frank J. Gonzalez, Curtis C. Harris; Noninvasive Urinary Metabolomic Profiling Identifies Diagnostic and Prognostic Markers in Lung Cancer. Cancer Res 15 June 2014; 74 (12): 3259–3270. https://doi.org/10.1158/0008-5472.CAN-14-0109 <br>

Statista. Lung cancer accounts for nearly 1 in 5 cancer deaths: Estimated number of cancer deaths worldwide in 2022, by type of cancer. Based on International Agency for Research on Cancer data. 2022. Accessed August 6, 2025. https://www.statista.com/chart/33881/number-of-cancer-deaths-by-type-of-cancer/ <br>

Dataiku Blog. Tree-based models: how they work (in plain English!). Published April 6, 2020. Accessed August 6, 2025. https://blog.dataiku.com/tree-based-models-how-they-work-in-plain-english <br>

---
# Acknowledgements
**I would like to acknowledge the following people:** <br>
* Mentor: Dr. Xiangping Lin for his time and advice throughout this internship; my work would not have been possible without his help
* PI: Dr. Michael P. Snyder for giving me this opportunity 
* Coordinators: Ms. Dawn Billman and Dr. Ruth Schade for being available and helping to navigate the program


