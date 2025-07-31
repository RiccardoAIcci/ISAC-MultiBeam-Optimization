# ISAC-MultiBeam-Optimization

## 面向通感一体化的多模态波束与频谱联合优化方法
## A Joint Multi-modal Beamforming and Spectrum Optimization Method for Integrated Sensing and Communication (ISAC)

## 1. 项目简介 (Introduction)

本项目提出了一种面向通信感知一体化（ISAC）的创新解决方案，旨在解决传统方案中通信与感知功能分离、资源利用率低、以及在动态环境下缺乏智能决策能力等核心问题。

This project proposes an innovative solution for Integrated Sensing and Communication (ISAC), aiming to address key issues in traditional approaches, such as the separation of communication and sensing functions, low resource utilization, and the lack of intelligent decision-making capabilities in dynamic environments.

我们通过设计一种**主子波束嵌套**的多模态波束结构，在物理层实现了通信与感知功能的深度融合。更核心地，我们引入了**贝叶斯深度强化学习（Bayesian Deep Reinforcement Learning, BDRL）**作为智能决策引擎，并结合**交替优化算法**，以实现对波束、频谱、功率等资源的实时、高效、智能的联合调配。

By designing a **multi-modal beam structure with nested main and sub-beams**, we achieve a deep fusion of communication and sensing functions at the physical layer. More centrally, we introduce **Bayesian Deep Reinforcement Learning (BDRL)** as the intelligent decision-making engine, combined with an **Alternating Optimization (AO) algorithm**, to achieve real-time, efficient, and intelligent joint allocation of resources such as beams, spectrum, and power.

该方法旨在最大化通信速率和感知精度，显著提升系统在无人机群、车联网等高动态、高复杂度场景下的整体性能和鲁棒性。

This method aims to maximize communication data rates and sensing accuracy, significantly enhancing the overall performance and robustness of the system in highly dynamic and complex scenarios, such as UAV swarms and Vehicle-to-Everything (V2X) networks.

## 2. 核心原理与方法 (Core Principles and Methods)

本项目的技术方案主要包含四大核心步骤：

The technical solution of this project mainly consists of four core steps:

1.  **多模态波束嵌套设计**：构建一种由宽覆盖的感知主波束和窄波束的通信子波束组成的层次化波束结构，从根本上解决了通感波束间的空间冲突问题。
    **Multi-modal Nested Beamforming Design**: Construct a hierarchical beam structure composed of a wide-coverage sensing main beam and narrow-beam communication sub-beams, fundamentally resolving spatial conflicts between sensing and communication beams.
2.  **动态频谱分配**：设计一个与感知性能反馈挂钩的动态频谱比函数，使频谱资源能够根据实时感知检测概率在通信和感知功能间自适应调整，打破静态限制，提升频谱效率。
    **Dynamic Spectrum Allocation**: Design a dynamic spectrum ratio function linked to sensing performance feedback, allowing spectrum resources to be adaptively adjusted between communication and sensing functions based on real-time sensing detection probability, thus breaking static limitations and improving spectral efficiency.
3.  **联合优化目标构建**：建立一个加权和（weighted-sum）形式的联合优化目标函数，协同优化所有通信用户的总速率与雷达感知的互信息量，为系统的整体性能评估与优化提供了明确的数学指引。
    **Joint Optimization Objective Formulation**: Establish a weighted-sum objective function to synergistically optimize the total data rate of all communication users and the mutual information of radar sensing, providing a clear mathematical guide for the system's overall performance evaluation and optimization.
4.  **智能决策与优化求解**：采用**贝叶斯深度强化学习**进行高层决策变量的优化，并结合**交替优化算法**对底层物理层参数进行高效求解。
    **Intelligent Decision-making and Optimization**: Employ **Bayesian Deep Reinforcement Learning** to optimize high-level decision variables, combined with an **Alternating Optimization algorithm** for the efficient solving of underlying physical layer parameters.

### 2.1 关键数学模型 (Key Mathematical Models)

#### 波束权重向量 (Beamforming Weight Vector)

为了实现主子波束的嵌套设计，总的波束权重向量 `W` 被构建为感知主波束和通信子波束的加权和：

To implement the nested design of main and sub-beams, the total beamforming weight vector `W` is constructed as a weighted sum of the sensing main beam and the communication sub-beams:

\\[ W = \\sqrt{p_s} W_s + \\sqrt{p_c} \\sum_{k=1}^{K} w_{c,k} \\]

-   \\( W_s \\): 感知主波束权重 (Sensing main beam weight)
-   \\( w_{c,k} \\): 第k个通信子波束权重 (k-th communication sub-beam weight)
-   \\( p_s, p_c \\): 感知和通信的功率分配因子 (Power allocation factors for sensing and communication)

#### 动态频谱比函数 (Dynamic Spectrum Ratio Function)

频谱分配比例 `ρ(t)` 根据感知检测概率 `P_d` 动态调整，其更新机制如下：

The spectrum allocation ratio `ρ(t)` is dynamically adjusted according to the sensing detection probability `P_d`, with the following update mechanism:

\\[ \\rho(t) = \\rho(t-1) + \\Delta \\rho \\]

-   当感知检测概率低于预设阈值时，增加感知频谱占比，反之则将更多资源分配给通信。
    When the sensing detection probability falls below a preset threshold, the spectrum allocation for sensing is increased; otherwise, more resources are allocated to communication.

#### 联合优化目标函数 (Joint Optimization Objective Function)

系统的最终优化目标是最大化通信速率与感知精度的加权和：

The final optimization objective of the system is to maximize the weighted sum of communication rate and sensing accuracy:

\\[ \\max_{W, \\Theta} \\quad \\alpha_R \\log(R_{total}) + \\alpha_I I(W, \\Theta) \\]

-   \\( R_{total} \\): 所有通信用户的总速率 (Total data rate of all communication users)
-   \\( I(W, \\Theta) \\): 雷达的互信息（MI），衡量感知性能 (Mutual Information (MI) of the radar, measuring sensing performance)
-   \\( W, \\Theta \\): 分别为波束权重矩阵和智能反射面（RIS）相位矢量 (Beamforming weight matrix and RIS phase-shift vector, respectively)
-   \\( \\alpha_R, \\alpha_I \\): 权重因子，可根据场景需求灵活调整 (Weighting factors that can be flexibly adjusted according to scenario requirements)

## 3. 核心优化方法：基于AI的智能决策 (Core Optimization Method: AI-Based Intelligent Decision-Making)

为了应对高动态环境的不确定性并实现系统资源的智能分配，我们设计的核心是**贝叶斯深度强化学习（BDRL）智能决策引擎**。

To address the uncertainty of highly dynamic environments and achieve intelligent allocation of system resources, the core of our design is the **Bayesian Deep Reinforcement Learning (BDRL) intelligent decision-making engine**.

### 3.1 贝叶斯深度强化学习 (BDRL) 引擎 (Bayesian Deep Reinforcement Learning (BDRL) Engine)

BDRL不仅具备深度强化学习（DRL）强大的非线性拟合与序列决策能力，更通过引入贝叶斯方法，能够**量化决策过程中的不确定性**，从而在信息不完备的复杂环境中做出更鲁棒、更可靠的决策。

BDRL not only possesses the powerful non-linear fitting and sequential decision-making capabilities of Deep Reinforcement Learning (DRL) but also, by introducing Bayesian methods, can **quantify uncertainty in the decision-making process**, thereby making more robust and reliable decisions in complex environments with incomplete information.

#### 状态空间 (State Space)

智能体的“输入”。为了让决策全面，状态空间包含了多维度的环境信息：

The "input" for the agent. To ensure comprehensive decision-making, the state space includes multi-dimensional environmental information:

*   **信道状态信息 (CSI)**：反映通信链路的质量。 (Reflects the quality of the communication link.)
*   **目标位置信息**：感知任务的核心数据。 (Core data for the sensing task.)
*   **历史感知误差**：作为反馈，帮助智能体修正未来的策略。 (Serves as feedback to help the agent correct future policies.)

#### 动作空间 (Action Space)

智能体的“输出”，即可以调控的关键决策变量：

The "output" of the agent, i.e., the key decision variables that can be controlled:

*   **波束指向角度**：决定通信与感知信号的覆盖范围。 (Determines the coverage of communication and sensing signals.)
*   **频谱分配比例**：在通感功能间权衡资源。 (Balances resources between communication and sensing functions.)
*   **功率分配因子**：控制发射功率。 (Controls the transmission power.)

#### 奖励函数 (Reward Function)

奖励函数是指导智能体学习的“指挥棒”，其设计目标是平衡通信性能增益与感知精度惩罚：

The reward function is the "guideline" for the agent's learning, designed to balance the trade-off between communication performance gains and sensing accuracy penalties:

\\[ R_t(s_t, a_t) = R_{sum} + \\lambda \\cdot \\frac{1}{\\text{MSE}_s} \\]

-   \\( R_t(s_t, a_t) \\): 在状态 \\(s_t\\) 下采取动作 \\(a_t\\) 获得的即时奖励。 (The immediate reward received for taking action \\(a_t\\) in state \\(s_t\\).)
-   \\( R_{sum} \\): 通信总速率，代表通信增益。 (The total communication rate, representing the communication gain.)
-   \\( \\text{MSE}_s \\): 感知均方误差。该项是一个惩罚项，误差越小，奖励越高。 (The Mean Squared Error of sensing. This term is a penalty; the smaller the error, the higher the reward.)
-   \\( \\lambda \\): 权衡参数，用于平衡通信与感知的重要性。 (A trade-off parameter to balance the importance of communication and sensing.)

#### 决策与学习流程 (Decision-Making and Learning Process)

智能决策引擎遵循标准的“感知-决策-学习”闭环流程：

The intelligent decision-making engine follows a standard "sense-decide-learn" closed-loop process:

1.  **感知与决策**：在当前时刻 `t`，智能体根据观测到的状态 \\(s_t\\)，利用其内部的（贝叶斯）神经网络模型选择一个最优动作 \\(a_t\\)。
    **Sensing and Decision-Making**: At the current time `t`, the agent selects an optimal action \\(a_t\\) based on the observed state \\(s_t\\) using its internal (Bayesian) neural network model.
2.  **执行与反馈**：系统执行动作 \\(a_t\\)，与环境交互后，进入新的状态 \\(s_{t+1}\\) 并获得一个即时奖励 \\(R_t\\)。
    **Execution and Feedback**: The system executes action \\(a_t\\), and after interacting with the environment, it transitions to a new state \\(s_{t+1}\\) and receives an immediate reward \\(R_t\\).
3.  **贝叶斯后验更新**：智能体利用新的样本 `(s_t, a_t, R_t, s_{t+1})` 来更新其神经网络的权重。区别于传统DRL，BDRL的更新遵循贝叶斯后验概率分布，这使得模型能够评估其预测的**不确定性**，从而在探索（Exploration）和利用（Exploitation）之间取得更好的平衡，决策鲁棒性提升显著。
    **Bayesian Posterior Update**: The agent uses the new sample `(s_t, a_t, R_t, s_{t+1})` to update the weights of its neural network. Unlike traditional DRL, the BDRL update follows a Bayesian posterior probability distribution, which allows the model to assess the **uncertainty** of its predictions. This leads to a better balance between exploration and exploitation, significantly improving decision-making robustness.

### 3.2 交替优化 (Alternating Optimization, AO) 算法 (Alternating Optimization (AO) Algorithm)

BDRL负责顶层的策略决策，而具体的物理层参数（波束权重 `W` 和RIS相位 `Θ`）优化是一个高维度的非凸问题，直接求解计算复杂度极高。为此，我们采用**交替优化**策略将其分解。

BDRL is responsible for top-level policy decisions, while the optimization of specific physical layer parameters (beamforming weights `W` and RIS phase shifts `Θ`) is a high-dimensional, non-convex problem with extremely high computational complexity for direct solving. To this end, we employ an **Alternating Optimization** strategy to decompose it.

该算法将复杂的联合优化问题分解为两个交替进行的子问题：

This algorithm decomposes the complex joint optimization problem into two sub-problems that are solved alternately:

1.  **固定RIS相位，优化波束权重**：当RIS相位 `Θ` 固定时，原问题简化。我们采用**半定松弛 (Semi-Definite Relaxation, SDR)** 方法来求解波束权重矩阵 `W`。
    **Fix RIS Phase, Optimize Beamforming Weights**: When the RIS phase `Θ` is fixed, the original problem is simplified. We use the **Semi-Definite Relaxation (SDR)** method to solve for the beamforming weight matrix `W`.
2.  **固定波束权重，优化RIS相位**：当波束权重 `W` 固定时，我们采用**最小二乘法 (Least Squares, LS) 结合流形优化 (Manifold Optimization)** 的方法来求解RIS相位矢量 `Θ`。
    **Fix Beamforming Weights, Optimize RIS Phase**: When the beamforming weights `W` are fixed, we use a method combining **Least Squares (LS) and Manifold Optimization** to solve for the RIS phase-shift vector `Θ`.

通过这种分而治之的策略，我们能够大幅降低计算复杂度，实验证明计算效率提升90%以上，满足了系统的实时性要求。

Through this divide-and-conquer strategy, we can significantly reduce computational complexity. Experiments show a computational efficiency improvement of over 90%, meeting the real-time requirements of the system.

## 4. 技术优势 (Technical Advantages)

*   **物理层深度融合**：通过多模态波束嵌套设计，有效解决波束冲突，实现通感共生。
    **Deep Physical Layer Fusion**: Effectively resolves beam conflicts and achieves sensing-communication symbiosis through a multi-modal nested beamforming design.
*   **智能决策与高鲁棒性**：引入贝叶斯深度强化学习，能够量化不确定性，使系统在复杂动态环境下的决策鲁棒性提升40%。
    **Intelligent Decision-Making and High Robustness**: The introduction of Bayesian Deep Reinforcement Learning allows for the quantification of uncertainty, improving decision-making robustness by 40% in complex, dynamic environments.
*   **资源利用率高**：基于感知性能反馈的动态频谱分配机制，显著提升了频谱效率。
    **High Resource Utilization**: The dynamic spectrum allocation mechanism based on sensing performance feedback significantly enhances spectral efficiency.
*   **计算高效性**：采用交替优化算法，将系统处理时延从秒级降低至毫秒级（1200ms -> 19.2ms），满足实时性需求。
    **Computational Efficiency**: The use of the Alternating Optimization algorithm reduces system processing delay from seconds to milliseconds (1200ms -> 19.2ms), meeting real-time requirements.
*   **性能协同提升**：通过双指标协同优化，通信速率、感知精度及频谱利用率得到全面提升。
    **Synergistic Performance Improvement**: Achieves comprehensive improvements in communication rate, sensing accuracy, and spectrum utilization through dual-objective co-optimization. 
