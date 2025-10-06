# InstaDrive: Instance-Aware Driving World Models for Realistic and Consistent Video Generation

## Project Links

| Category              | Link                                                         | Badge                                                        |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Demo**  | [Live Demo](https://shanpoyang654.github.io/InstaDrive/page.html) | ![Demo](https://img.shields.io/badge/Demo-Live-green)        |
| **Paper**    | [arXiv Paper](https://www.researchgate.net/publication/394053515_InstaDrive_Instance-Aware_Driving_World_Models_for_Realistic_and_Consistent_Video_Generation)              | ![Paper](https://img.shields.io/badge/PDF-arXiv-blue)        |
 




## Abstract
Autonomous driving relies on robust models trained on high-quality, large-scale multi-view driving videos for tasks like perception, tracking, and planning. While world models offer a cost-effective solution for generating realistic driving videos, they struggle to maintain instance-level temporal consistency and spatial geometric fidelity.  

To address these challenges, we propose **InstaDrive**, a novel framework that enhances driving video realism through two key advancements:  

1. **Instance Flow Guider module** — extracts and propagates instance features across frames to enforce temporal consistency.  
2. **Spatial Geometric Aligner module** — improves spatial reasoning, ensures precise instance positioning, and models occlusion hierarchies.  

By incorporating these instance-aware mechanisms, InstaDrive achieves state-of-the-art video generation quality and enhances downstream autonomous driving tasks on the nuScenes dataset.  
Additionally, we utilize **CARLA's autopilot** to procedurally simulate rare but safety-critical scenarios.

![InstaDrive Overview](./data/teaser.png)  
![InstaDrive method](./data/flow.png)

---

## 1. Multimodal Condition Controllability

### 1.1 Layout Controllability
InstaDrive responds precisely to control conditions like box projection, map projection, and instance flow.  

- **Example 1**: Oncoming bus, parked car, pedestrian, roundabout, parked trucks.  
  ![Video](./data/Layout_controlability/ex2.mp4)  
  Drivable areas, sidewalks, and zebra crossings are faithfully generated.  

- **Example 2**: Intersection, pedestrians, cones, turning.  
  ![Video](./data/Layout_controlability/ex3.mp4)  
  Objects are precisely rendered at correct locations, maintaining temporal consistency.

---

## 2. Qualitative Comparison

### 2.1 Comparison with Baseline
- **Panacea** result  
  ![Video](./data/Qualitative_comparison/temporal/Panacea/pred.mp4)

- **InstaDrive** result  
  ![Video](./data/Qualitative_comparison/temporal/InstaDrive/pred.mp4)

a. **Instance-Level Temporal Consistency**:  
In Panacea, the direction of the white car’s head changes over time.  
In InstaDrive, attributes are preserved, showing superior consistency.

- **Occlusion Comparison**  
  ![Video](./data/Qualitative_comparison/occlusion/Panacea/occ.mp4)


