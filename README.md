![Sonar Mine Detection](./notebooks/images/00_Sonar_Mine_detection_Cover.png)

## Sonar Mine vs Rock Classification

### Context

Naval vessels operate in environments where underwater mines may be concealed among natural rock formations on the sea floor. Active sonar provides the only advance warning: emitted sound waves and their returning echoes must be interpreted to distinguish mines from benign objects.

These decisions are made in real time and under uncertainty, where a single missed mine can result in the loss of a vessel, its crew, and the mission.

This project uses the **UCI Sonar dataset** to assess whether machine learning classification can support this task, and which data processing and modelling choices are best suited.

### Business Objective

Enable safe submarine operations by reliably detecting underwater mines before they pose a threat.

### Translating the Business Objective into ML Performance

In this domain, not all errors are equal. Missing a mine represents catastrophic failure, while false alarms are operationally tolerable. As a result, model performance is evaluated under **asymmetric error costs**, where recall is prioritised over raw accuracy.

- **Recall ≈ 100%** — Every mine must be detected  
  *(Operational target: ≥98% on the held-out test set, tuned toward 100%)*

- **Precision ≥ 70%** — False alarms must remain within operator capacity  
  *(≤30% false positive rate to maintain alert credibility)*

### What Good Looks Like

- Zero missed mines on the held-out test set  
- Precision high enough that operators trust the alerts  
- A champion model suitable for deployment in a real-time detection system

### Production Perspective

Beyond model performance, this project will demonstrate an end-to-end ML workflow. The final model is exposed as a containerised API designed to accept sonar signal arrays (60 frequency bands) and return classification probabilities with confidence scores, enabling integration into operational detection pipelines.

---

## License

MIT License — see the `LICENSE` file for details.

---

## Acknowledgements

**Dataset:** UCI Machine Learning Repository — Connectionist Bench (Sonar, Mines vs. Rocks)  
**Citation:**  
Gorman, R. P., & Sejnowski, T. J. (1988). *Analysis of Hidden Units in a Layered Network Trained to Classify Sonar Targets*. Neural Networks, 1, 75–89.

---

## Contact

**Author:** Adama Abanteriba R  
**Portfolio:** https://github.com/Thrawn6595/Portfolio-submarine-sonar