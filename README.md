# InstaDrive: Instance-Aware Driving World Models for Realistic and Consistent Video Generation

## Project Links

| Category | Link | Badge |
|----------|------|-------|
| **Demo** | [Live Demo](https://shanpoyang654.github.io/InstaDrive/page.html) | ![Demo](https://img.shields.io/badge/Demo-Live-green) |
| **Paper** | [arXiv Paper](https://www.researchgate.net/publication/394053515_InstaDrive_Instance-Aware_Driving_World_Models_for_Realistic_and_Consistent_Video_Generation) | ![Paper](https://img.shields.io/badge/PDF-arXiv-blue) |

---

## Abstract

Autonomous driving relies on robust models trained on high-quality, large-scale multi-view driving videos for tasks such as perception, tracking, and planning. While world models offer a cost-effective solution for generating realistic driving videos, they struggle to maintain instance-level temporal consistency and spatial geometric fidelity. To address these challenges, we propose **InstaDrive**, a novel framework with two key advancements:

1. **Instance Flow Guider module** — extracts and propagates instance features across frames to ensure temporal consistency and preserve instance identity.  
2. **Spatial Geometric Aligner module** — improves spatial reasoning, ensures precise instance positioning, and explicitly models occlusion hierarchies.  

By incorporating these instance-aware mechanisms, InstaDrive achieves state-of-the-art video generation quality and improves downstream autonomous driving tasks on the nuScenes dataset. We also leverage CARLA’s autopilot to procedurally and stochastically simulate rare but safety-critical driving scenarios across different maps and regions.

![InstaDrive Overview](./data/teaser.png)  
![InstaDrive Method](./data/flow.png)

---

## 1. Multimodal Condition Controllability

### 1.1 Layout Controllability

* **Example A** — Oncoming bus, parked car on center divider, pedestrian, roundabout, parked trucks.


https://github.com/user-attachments/assets/8992eb1e-289a-445c-975a-7ab8ab28088e



Description: Drivable areas, sidewalks, and zebra crossings are faithfully generated according to the road map projections. Objects in the scene are accurately placed and sized.

* **Example B** — Waiting at intersection, pedestrians on sidewalk, turning right, cones, crossing intersection.  


https://github.com/user-attachments/assets/4119b0a6-8012-4946-a8a9-4ce844eb2efc


Description: Small and densely packed objects are rendered accurately at their correct positions following 3D bounding box coordinates. Objects maintain temporal consistency via instance flow guidance.

---

## 2. Qualitative Comparison

### 2.1 Comparison with Baseline

#### a. Instance-Level Temporal Consistency

* **MagicDrive-V2 (baseline)**  
[![MagicDrive-V2 Temporal](https://github.com/user-attachments/assets/abdabb8a-9c0d-466c-a552-5ff1ba2bf4ee)](https://github.com/user-attachments/assets/abdabb8a-9c0d-466c-a552-5ff1ba2bf4ee)

* **InstaDrive**  
[![InstaDrive Temporal](https://github.com/user-attachments/assets/09e0c4c7-f3dd-4fa6-819e-a024fab506c2)](https://github.com/user-attachments/assets/09e0c4c7-f3dd-4fa6-819e-a024fab506c2)

Explanation: In MagicDrive-V2, the front orientation of the white car (FrontLeft and BackLeft views) changes over time. InstaDrive preserves instance attributes, demonstrating superior temporal consistency.

#### b. Occlusion Hierarchy

* **Panacea (baseline occlusion)**  
[![Panacea Occlusion](https://github.com/user-attachments/assets/d1d25398-60c7-499f-b9e6-c59416eb5502)](https://github.com/user-attachments/assets/d1d25398-60c7-499f-b9e6-c59416eb5502)

* **InstaDrive (occlusion)**  
[![InstaDrive Occlusion](https://github.com/user-attachments/assets/d0f7df29-171f-418a-8a3d-444643e9f35b)](https://github.com/user-attachments/assets/d0f7df29-171f-418a-8a3d-444643e9f35b)

Explanation: A stationary box (FrontLeft view) is farther away, while a moving box is closer. In Panacea, the distant box incorrectly appears in front of the moving object. InstaDrive correctly renders the closer moving object in front.

#### c. Spatial Localization

* **Panacea (spatial example)**  
[![Panacea Spatial](https://github.com/user-attachments/assets/d29e875b-bccf-4d82-afc8-e8e70a67339b)](https://github.com/user-attachments/assets/d29e875b-bccf-4d82-afc8-e8e70a67339b)

* **InstaDrive (spatial example)**  
[![InstaDrive Spatial](https://github.com/user-attachments/assets/e65e07fa-3ed9-4e2a-ba20-9182e476c849)](https://github.com/user-attachments/assets/e65e07fa-3ed9-4e2a-ba20-9182e476c849)

Explanation: In some baselines like MagicDrive-V2, FrontRight view objects may deviate from the bounding box. InstaDrive preserves accurate spatial alignment.

---

### 2.2 Additional Results

* **a. Instance-Level Temporal Consistency**  
[![Temporal More](https://github.com/user-attachments/assets/fd350c61-46a2-4920-aaad-1b8e1c487e1c)](https://github.com/user-attachments/assets/fd350c61-46a2-4920-aaad-1b8e1c487e1c)  
Description: Demonstrates model consistency of instance attributes across complex scenarios.

* **b. Occlusion Hierarchy**  
[![Occlusion More](https://github.com/user-attachments/assets/5dca34c1-ab42-4988-befe-d9c376c909aa)](https://github.com/user-attachments/assets/5dca34c1-ab42-4988-befe-d9c376c909aa)  
Description: Further confirms correct handling of occlusion relationships.

* **c. Spatial Localization**  
[![Spatial More](https://github.com/user-attachments/assets/ec81aca4-6128-4370-93a7-5cd4f5080502)](https://github.com/user-attachments/assets/ec81aca4-6128-4370-93a7-5cd4f5080502)  
Description: Demonstrates model accuracy in spatial localization across different scenarios.

---

## 3. Scenario Simulation Using CARLA-Generated Layouts

### 3.1 Corner Cases in Autonomous Driving

* **Example A** — The vehicle ahead brakes, prompting the ego vehicle to decelerate and stop.  
Please refer to the [Project Page](https://shanpoyang654.github.io/InstaDrive/page.html) for the long-term demo.  
Description: Simulates sudden braking to visualize behavior and generation under emergency conditions.

* **Example B** — Vehicle cutting in from the right lane.  
[![Cut-in Example](https://github.com/user-attachments/assets/4b51e9dc-fb1c-49a1-bba2-f51455a01220)](https://github.com/user-attachments/assets/4b51e9dc-fb1c-49a1-bba2-f51455a01220)  
Description: Simulates lane cutting to test model stability and generation in complex traffic scenarios.

### 3.2 Long-term Generation (2x speed)

Please refer to the [Project Page](https://shanpoyang654.github.io/InstaDrive/page.html) for long-term video demos.  
Description: Demonstrates long-term generation consistency and coherent world modeling.

---


