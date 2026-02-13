## **Graphite Technical Consulting \- Project Scope & Timeline** 

*Spring Semester: January 26, 2026 \- May 1, 2026 (14 weeks)* 

*Team: 3-5 technical/SWE members* 

**Project Goal:** Build an ML-powered system to classify Partial Discharge (PD) waveforms from electromagnetic (EM) sensor data, enabling accurate identification of power transformer failure modes (internal discharge, corona, surface discharge, noise).

**Phase 1: Project Initiation, Data Access & Research** **Weeks 1-2 (Jan 26 \- Feb 8\)**

*Quantifiable Deliverables:*

* Obtain IEEE access or identify alternative access paths to LEE dataset  
* Contact list for IEEE access (minimum 3 contacts/pathways identified)  
* Complete Derrick's PD classification training lesson  
* Document outlining 5-7 critical PD failure modes with technical specifications:  
  * Internal discharge  
  * Corona discharge  
  * Surface discharge  
  * Noise patterns  
  * Other relevant failure modes  
* List of minimum 10 public domain EM data sources identified  
* Technical specification document defining:  
  * Expected EM sensor data formats (HFCT, coupling capacitor, TEV)  
  * ADC output structures  
  * Waveform characteristics for each failure mode  
* Initial project architecture diagram

**Phase 2: Data Collection & Preparation** **Weeks 3-5 (Feb 9 \- Mar 1\)**

*Quantifiable Deliverables:*

* LEE dataset successfully accessed and downloaded  
* Minimum dataset targets:  
  * 50+ two-second EM data samples from LEE dataset  
  * Additional 50+ samples from other public sources (if available)  
  * Coverage across multiple failure modes  
* Cleaned, deduplicated dataset with documented quality metrics  
* Data validation report showing:  
  * Number of samples per failure mode  
  * Data quality metrics (completeness, consistency)  
  * ADC value ranges and characteristics  
* Scripts for data extraction and preprocessing

**Phase 3: Middleware Development & Noise Floor Analysis** **Weeks 6-8 (Mar 2 \- Mar 22\)**

*Quantifiable Deliverables:*

* Noise floor analysis implementation:  
  * Algorithm to automatically set optimal noise floor thresholds  
  * Comparison of minimum 3 different approaches (e.g., standard deviation pulse rate, histogram knee, wavelet energy)  
  * Performance metrics for each approach  
* Data formatting middleware:  
  * Convert raw ADC values to standardized waveforms  
  * Handle different sensor types (HFCT, coupling capacitor, TEV)  
  * Trigger threshold implementation (both hardware-style and software-style)  
* Test suite covering:  
  * Minimum 10 test cases across different noise conditions  
  * Edge cases with high noise-to-signal ratios  
* Technical documentation of middleware architecture

**Phase 4: PD Waveform Analysis & Feature Extraction** **Weeks 9-10 (Mar 23 \- Apr 5\)**

*Quantifiable Deliverables:*

* Feature extraction system that processes waveforms and extracts minimum 10 features:  
  * Pulse width  
  * Amplitude (peak values)  
  * Phase angle  
  * Rise time / fall time  
  * Oscillation characteristics  
  * Other relevant features identified during research  
* Waveform clustering implementation:  
  * Group similar waveforms into clusters  
  * Visualization tools to inspect clusters  
  * Minimum 3-5 distinct clusters identified  
* Feature importance analysis:  
  * Ranking of which features are most discriminative  
  * Documentation of feature selection rationale  
* Code repository with feature extraction pipeline

*Checkpoint (Week 10 \- Apr 5):*

* Working feature extraction and clustering demo ready

**Phase 5: ML Model Selection & Implementation** **Weeks 11-13 (Apr 6 \- Apr 26\)**

*Quantifiable Deliverables:*

* ML model evaluation report:  
  * Minimum 5 models researched and compared  
  * Performance metrics for each (accuracy, precision, recall, F1)  
  * Recommendation for top 3 models  
* Implementation of top 3 ML models:  
  * Trained on extracted features  
  * Classification accuracy across failure modes  
  * Confusion matrices showing performance  
* Iterative improvement loop:  
  * Document minimum 2 iterations of feature selection refinement  
  * Model performance improvement from iteration 1 to final  
* Test coverage:  
  * Minimum 20 test cases across different failure scenarios  
  * Cross-validation results  
* Final trained model capable of classifying:  
  * Internal discharge  
  * Corona discharge  
  * Surface discharge  
  * Noise  
  * With documented accuracy metrics for each class

**Phase 6: Demo Preparation, Documentation & Final Recommendations** **Week 14 (Apr 27 \- May 1\)**

*Quantifiable Deliverables:*

* Working demo showing:  
  * Raw EM data input → waveform processing → feature extraction → ML classification  
  * Minimum 5 end-to-end scenarios demonstrating different failure modes  
  * Classification confidence scores  
* Final recommendations report including:  
  * Optimal noise floor analysis method  
  * Best performing feature set  
  * Top ML model recommendation with justification  
  * Accuracy metrics and performance analysis  
  * Limitations and areas for improvement  
  * Recommendations for multimodal data fusion (future project 2\)  
* Complete documentation package:  
  * Technical architecture documentation  
  * Feature extraction methodology  
  * ML model training procedures  
  * API/integration documentation  
  * User guide for demo scenarios  
  * Data sources bibliography  
* Final presentation deck  
* Code handoff with setup instructions and reproducibility guide

**Project Expectations:**

1. **Weekly check-in meetings with Derrick Du** for technical guidance and progress review  
2. **Minimum viable deliverable:** Phases 1-5 (Data collection through ML implementation) must be complete and functional  
3. **Code quality standards:** Version-controlled repository, documented code, reproducible results  
4. **Handoff requirements:** Complete documentation and final recommendations report enabling Magnefy to continue development post-project  
5. **Future roadmap awareness:** This project (single-mode EM classification) sets foundation for:  
   * Project 2: Multimodal data fusion  
   * Project 3: Maintenance action chatbot/recommendations

