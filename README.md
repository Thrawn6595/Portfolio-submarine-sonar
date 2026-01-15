![Sonar Mine Detection](./notebooks/images/00_Sonar_Mine_detection_Cover.png)

# Underwater mine detection using sonar signals


### Motivation - why this project exists

Some machine learning projects fail not because the algorithms are wrong, but because teams dive straight into modelling without understanding what problem they need to solve for, they optimise for accuracy when the business needs something else, or build brilliant models that are impractical to operationalise at scale.

This project follows the ML development cycle closely from business problem framing to translation into clear ML objectives and maps out deployment considerations in the early design stage.

### Situation

Naval vessels need to distinguish between underwater mines and rocks on the seabed using active sonar. Sound waves bounce back as echoes that must be interpreted quickly. Miss a mine and you lose the vessel and crew. Flag too many false alarms and operators stop trusting the system.

### Complication

This is precisely why I chose this dataset, treating mine detection as a simple binary classification problem where you maximise accuracy misses the point entirely. Missing a mine is catastrophic, whilst a false alarm is manageable. This asymmetric cost makes it less of a vanilla classification problem and forces careful thinking about what metrics actually matter.

### Solution:

A tiered decision framework: high-confidence detections are automatically flagged as mines, very low-confidence readings dismissed as rocks, and uncertain cases sent for human review. This respects operator capacity whilst ensuring every genuine threat gets attention, we want to avoid a situation of overwhelming an operator with volume, they should only be called upon to review a decision on non trivial cases.

Success means recall approaching 100 per cent (targeting 98 per cent or better on test data) and precision above 70 per cent to maintain operator trust.

### Production perspective

Beyond model performance, this project will demonstrate an end-to-end ML workflow. The final model is exposed as a containerised API designed to accept sonar signal arrays (60 frequency bands) and return classification probabilities with confidence scores, enabling integration into operational detection pipelines.


---
### License

MIT Licence

### Acknowledgements

**Dataset:** UCI Machine Learning Repository — Connectionist Bench (Sonar, Mines vs. Rocks)  
**Citation:** Gorman, R. P., & Sejnowski, T. J. (1988). *Analysis of Hidden Units in a Layered Network Trained to Classify Sonar Targets*. Neural Networks, 1, 75–89.

### Author
Adama Abanteriba Richards  
**Portfolio:** https://github.com/Thrawn6595/Portfolio-submarine-sonar
