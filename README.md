![Sonar Mine Detection](./notebooks/images/00_Sonar_Mine_detection_Cover.png)

# Sonar Mine vs Rock Classification

## Context

Naval vessels navigate waters where underwater mines pose existential threats. Active sonar returns provide the only advance warning, but distinguishing mines from naturally occurring rocks requires real-time classification under uncertainty.

This project explores the UCI Sonar dataset (208 records, 60 frequency-band features) to determine whether ML classification can add value to mine detection systems and what data processing and modeling approaches deliver on the business objective: keeping vessels and crew safe.

**The Difference:** Most implementations optimise for accuracy. We take a different route, recognising that missing a mine (false negative) means catastrophic loss, while false alarms (false positives) merely waste investigation time. This asymmetric cost structure fundamentally shapes our design, development, and deployment decisions.

---

## Key Objectives

### Business Objective
Enable safe submarine operations by reliably detecting underwater mines before they pose a threat.

### ML Performance Targets

- **Recall ≈ 100%** — Detect every mine (operational target: ≥98% on test set)
- **Precision ≥ 70%** — Maintain alert credibility (≤30% false positive rate)

### Production Objective
End-to-end ML pipeline from research to containerised API deployment, 

**What Good Looks Like:**
- Zero missed mines on held-out test set (100% recall)
- Precision high enough that operators trust the alerts (≥70%)
- Champion model deployed as production-ready API

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

**Dataset:** UCI Machine Learning Repository - Connectionist Bench (Sonar, Mines vs. Rocks)  
**Citation:** Gorman, R. P., and Sejnowski, T. J. (1988). "Analysis of Hidden Units in a Layered Network Trained to Classify Sonar Targets" in Neural Networks, Vol. 1, pp. 75-89.

---

## Contact

**Author:** Adama Abanteriba R  
**Portfolio:** [GitHub Repository](https://github.com/Thrawn6595/Portfolio-submarine-sonar)