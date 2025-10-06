# InstaDrive: Instance-Aware Driving World Models for Realistic and Consistent Video Generation

## Project Links

| Category              | Link                                                         | Badge                                                        |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Demo**  | [Live Demo](https://shanpoyang654.github.io/InstaDrive/page.html) | ![Demo](https://img.shields.io/badge/Demo-Live-green)        |
| **Paper**    | [arXiv Paper](https://www.researchgate.net/publication/394053515_InstaDrive_Instance-Aware_Driving_World_Models_for_Realistic_and_Consistent_Video_Generation)              | ![Paper](https://img.shields.io/badge/PDF-arXiv-blue)        |
 


---

## Abstract

Autonomous driving relies on robust models trained on high-quality, large-scale multi-view driving videos for tasks like perception, tracking, and planning. While world models offer a cost-effective solution for generating realistic driving videos, they struggle to maintain instance-level temporal consistency and spatial geometric fidelity. To address these challenges, we propose **InstaDrive**, a novel framework that enhances driving video realism through two key advancements:

1. **Instance Flow Guider module** — extracts and propagates instance features across frames to enforce temporal consistency, preserving instance identity over time.
2. **Spatial Geometric Aligner module** — improves spatial reasoning, ensures precise instance positioning, and explicitly models occlusion hierarchies.

By incorporating these instance-aware mechanisms, InstaDrive achieves state-of-the-art video generation quality and enhances downstream autonomous driving tasks on the nuScenes dataset. Additionally, we utilize CARLA's autopilot to procedurally and stochastically simulate rare yet safety-critical driving scenarios across diverse maps and regions, enabling rigorous safety evaluation for autonomous systems.

![InstaDrive Overview](./data/teaser.png)
![InstaDrive method](./data/flow.png)

---

## 1. Multimodal Condition Controllability

### 1.1 Layout Controllability

* **示例 A** — Oncoming bus, parked car on center divider, pedestrian, roundabout, parked trucks.

https://github.com/user-attachments/assets/3f3d0b02-1dcc-4231-bd5c-89a684134bfa


  描述：Drivable areas, sidewalks, and zebra crossings are faithfully generated following the road map projections. Objects in the scene are accurately placed and sized to align with their projected bounding boxes.

* **示例 B** — Wait at intersection, pedestrians on sidewalk, turn right, cones, cross intersection.



https://github.com/user-attachments/assets/3df501f5-76a8-4a2c-98e7-70904261647e


  描述：Small and densely packed objects are precisely rendered at their correct locations, following 3D bounding box coordinates. Objects track their previous attributes as guided by the instance flow, ensuring temporal consistency across frames.

---

## 2. Qualitative Comparison

### 2.1 Comparison with Baseline

#### a. Instance-Level Temporal Consistency

* **MagicDrive-V2 (baseline)** 

https://github.com/user-attachments/assets/abdabb8a-9c0d-466c-a552-5ff1ba2bf4ee


* **InstaDrive** 


https://github.com/user-attachments/assets/09e0c4c7-f3dd-4fa6-819e-a024fab506c2


说明：In Panacea, the direction of the white car's front head (in FrontLeft and BackLeft view) changes over time. In InstaDrive, our model preserves the white car's attributes, demonstrating superior temporal consistency.

#### b. Occlusion Hierarchy

* **Panacea (baseline occlusion example)** 


https://github.com/user-attachments/assets/d1d25398-60c7-499f-b9e6-c59416eb5502



* **InstaDrive (occlusion example)** 


https://github.com/user-attachments/assets/d0f7df29-171f-418a-8a3d-444643e9f35b



说明：例子中有一个静止的箱子（FrontLeft 视角停车位）位于更远处，移动箱子更近。在 Panacea 中，远处的静止箱子错误地放在前面，遮挡了更近的移动车辆，违反预期的遮挡层次。在 InstaDrive 中，我们正确渲染了更近的移动箱子在前，远处静止箱子被适当遮挡。

#### c. Spatial Localization（空间定位）

* **Panacea (空间示例)** 


https://github.com/user-attachments/assets/d29e875b-bccf-4d82-afc8-e8e70a67339b



* **InstaDrive (空间示例)** 



https://github.com/user-attachments/assets/e65e07fa-3ed9-4e2a-ba20-9182e476c849





说明：在 MagicDrive-V2 或其他 baseline 中，FrontRight 视角的车辆可能偏离 bounding box 控制信号。而在 InstaDrive 中，我们保持准确的空间定位，确保对象位置和尺寸符合控制信号。

---

### 2.2 More Results

* **a. Instance-Level Temporal Consistency (更多示例)** 


https://github.com/user-attachments/assets/fd350c61-46a2-4920-aaad-1b8e1c487e1c


  说明：展示模型在复杂场景中保持实例属性随时间一致性。

* **b. Occlusion Hierarchy (更多示例)** 


https://github.com/user-attachments/assets/5dca34c1-ab42-4988-befe-d9c376c909aa


  说明：进一步确认模型在遮挡关系处理上的正确性。

* **c. Spatial Localization (更多示例)** 


https://github.com/user-attachments/assets/ec81aca4-6128-4370-93a7-5cd4f5080502


  说明：展示模型在不同场景下维持对象空间定位的能力。

---

## 3. Scenario Simulation Using CARLA-Generated Layouts

### 3.1 Corner Case in Autonomous Driving

* **示例 a** — The vehicle ahead brakes, prompting the ego vehicle to decelerate and stop.

  描述：模拟前车刹车场景，评估在突发制动情况下的生成与行驶行为可视化。

* **示例 b** — Vehicle cutting in from the right lane. <video src="data/Carla/cut-in.mp4" width="640" controls></video>
  描述：模拟车辆并线切入，测试模型在复杂交通交互下的生成稳定性与合理性。

### 3.2 Long-term Generation (2x speed)

<video src="data/Carla/long3-final.mp4" width="640" controls></video>
描述：展示模型在长时序生成下保持世界一致性和内容连贯性的能力。

---


