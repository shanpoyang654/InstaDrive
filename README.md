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


https://github.com/user-attachments/assets/7acd3d1c-e0bf-4ee9-8454-4a143f5d978b


* **InstaDrive**  



https://github.com/user-attachments/assets/1b249f32-9b84-4d08-876c-6073aa24b14e


Explanation: In MagicDrive-V2, the front orientation of the white car (FrontLeft and BackLeft views) changes over time. InstaDrive preserves instance attributes, demonstrating superior temporal consistency.

#### b. Occlusion Hierarchy

* **Panacea (baseline occlusion)**  


https://github.com/user-attachments/assets/42c6bb6a-c20b-4ba0-b4bd-0dbe4a1dd00c


* **InstaDrive (occlusion)**  


https://github.com/user-attachments/assets/228d8eff-c1f8-4f10-8771-16d4602bcf17


Explanation: A stationary box (FrontLeft view) is farther away, while a moving box is closer. In Panacea, the distant box incorrectly appears in front of the moving object. InstaDrive correctly renders the closer moving object in front.

#### c. Spatial Localization

* **MagicDrive-V2 (spatial example)**  


https://github.com/user-attachments/assets/2a94f4d0-b06d-4f98-9887-c8627e3699db


* **InstaDrive (spatial example)**  


https://github.com/user-attachments/assets/5888df20-1af3-4d6b-8dc5-25725843cf62


Explanation: In some baselines like MagicDrive-V2, FrontRight view objects may deviate from the bounding box. InstaDrive preserves accurate spatial alignment.

---

### 2.2 Additional Results

* **a. Instance-Level Temporal Consistency**  


https://github.com/user-attachments/assets/21fcffe2-1419-4537-86f5-87e24725e370


Description: Demonstrates model consistency of instance attributes across complex scenarios.

* **b. Occlusion Hierarchy**  
  

https://github.com/user-attachments/assets/798ec166-58e2-47f5-ae37-90ff3c238a1f


Description: Further confirms correct handling of occlusion relationships.

* **c. Spatial Localization**  


https://github.com/user-attachments/assets/071bdc1c-d726-4e23-90ef-d54b0db9cd85


Description: Demonstrates model accuracy in spatial localization across different scenarios.

---

## 3. Scenario Simulation Using CARLA-Generated Layouts

### 3.1 Corner Cases in Autonomous Driving

* **Example A** — The vehicle ahead brakes, prompting the ego vehicle to decelerate and stop.  
Please refer to the [Project Page](https://shanpoyang654.github.io/InstaDrive/page.html) for the long-term demo.  
Description: Simulates sudden braking to visualize behavior and generation under emergency conditions.

* **Example B** — Vehicle cutting in from the right lane.  


https://github.com/user-attachments/assets/0523382a-e55a-4c73-b7df-96e1bf0de960


Description: Simulates lane cutting to test model stability and generation in complex traffic scenarios.

### 3.2 Long-term Generation (2x speed)

Please refer to the [Project Page](https://shanpoyang654.github.io/InstaDrive/page.html) for long-term video demos.  
Description: Demonstrates long-term generation consistency and coherent world modeling.

---




