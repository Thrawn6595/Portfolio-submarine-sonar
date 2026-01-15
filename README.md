![Sonar Mine Detection](./notebooks/images/00_Sonar_Mine_detection_Cover.png)

# Underwater mine detection using sonar signals

### Motivation

I'm drawn to problems where technical decisions and human factors intersect, particularly where standard approaches like optimising for accuracy miss the underlying business constraints. Having seen projects struggle from poor problem framing, I wanted to work through the complete ML development cycle on a dataset that forces careful thinking from the start. One of the Key motto's at SEEK which has stayed with me is "Do the right amount of thinking upfront"

The sonar mine detection problem is ideal for this: it deals with asymmetric cost (missed mines are catastrophic, false alarms are manageable), requires designing for operator workload, and demands thinking through deployment considerations before building models. This project follows the ML development cycle closely from business problem framing to translation into clear ML objectives, mapping out deployment considerations in the early design stage.

### Situation

Naval vessels need to distinguish between underwater mines and rocks on the seabed using active sonar. Sound waves bounce back as echoes that must be interpreted quickly. Miss a mine and you may lose the vessel and crew, flag too many false alarms and it becomes a nuisance to operators.

### Complication

This asymmetric cost makes it less of a vanilla classification problem. A missed mine is catastrophic whilst a false alarm is operationally manageable, meaning accuracy alone is insufficient as a success metric. The challenge is designing a system that handles this cost imbalance whilst remaining practical for operators to use.

### Solution

A tiered decision framework: high-confidence detections are automatically flagged as mines, very low-confidence readings dismissed as rocks, and uncertain cases sent for human review. This respects operator capacity whilst ensuring every genuine threat gets attention, we want to avoid overwhelming an operator with volume so they should only be called upon to review decisions on non-trivial cases.

Success means recall approaching 100 per cent (targeting 98 per cent or better on test data) and precision above 70 per cent to maintain operator trust.

### Production perspective

Beyond model performance, this project will demonstrate an end-to-end ML workflow. The final model is exposed as a containerised API designed to accept sonar signal arrays (60 frequency bands) and return classification probabilities with confidence scores, enabling integration into operational detection pipelines.

### Acknowledgements

Dataset from the UCI Machine Learning Repository, originally published by Gorman and Sejnowski (1988).

Adama Abanteriba | [Portfolio](https://github.com/Thrawn6595/Portfolio-submarine-sonar)

---

MIT Licence
