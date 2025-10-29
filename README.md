# 🦸♀️ Avengers RL Hackathon: Small Specialists Unite!

Link to the Repository: https://github.com/shaginhekvs/Synthetic_Data_Hackathon
Link to Medium article: https://medium.com/@shaginhekvs/avengers-rl-small-specialists-take-on-a-giant-3fa036402697


Welcome to the **Synthetic Data Hackathon** repository, where we revolutionize reinforcement learning by demonstrating that specialized, compact agents coordinating together can outshine massive, untrained models! Our project combines innovative environments, cutting-edge training techniques, and a thrilling "Avengers RL" concept that pits tiny trained experts against colossal but naive giants.

## Team Avengers RL 🤝

- **Ali Alami Idrissi**
- **Kazuma Choji**
- **Keshav Singh**


## Main Contributions

### 1. New Environments: Mastering Control in Diverse Arenas

We developed a suite of custom reinforcement learning environments using the flexible OpenEnv framework, designed to test agents across varied control challenges:

- **CartPole**: Classic balancing act requiring precise feedback control
- **Gym Environments**: Integration with Gymnasium's full ecosystem, including classic control tasks
- **MountainCar**: Momentum-based challenge demanding strategic energy management
- **LunarLander**: Complex 2D aerial navigation with physics-based landing mechanics

**The Super Complex Environment**: But why stop at single tasks? We created a groundbreaking **Sequential Environment** that seamlessly combines all four environments into one epic, multi-phase challenge. This composite environment tests true versatility, requiring agents to transition between completely different control paradigms within a single episode.

### 2. GRPO Training: Unleashing LLAMA 3.2 3B Power

We harnessed the power of **Generalized Reward-based Policy Optimization (GRPO)** to train a **3B-parameter LLAMA 3.2 model** on each individual environment. GRPO's efficiency in optimizing language models for RL tasks allowed us to achieve mastery-level performance across:

- **CartPole**: Zero to hero in stabilization techniques
- **LunarLander**: Precision maneuvers and soft touchdowns
- **MountainCar**: Momentum mastery through intelligent hill ascensions
- **BipedalWalker**: Efficient gait patterns for rough terrain navigation

Each model was trained until surpassing the standard success thresholds, resulting in specialized experts ready to tackle their domains.

### 3. Inference Battle Royale: Challenged by Goliath

The climax arrives in our mixed Sequential Environment, where our trained 3B-parameter specialists face off against massive **OSS 20B models**. These open-source behemoths bring raw computational might but lack domain-specific training. While we haven't achieved clear superiority yet, our experiments show promising results with comparable performance in specific domains:

- **Specialized Coordination**: Our ensemble of small experts shows comparable performance to the untrained giants in some tasks
- **Efficiency Gains**: Order-of-magnitude smaller models with potential for future advantage
- **Versatility**: Demonstrates capability for switching between radically different control tasks, setting foundation for future improvements

The results suggest that intelligence through specialization and coordination has strong potential to compete with brute computational force, though more work is needed to achieve clear superiority.

## 🦸♀️ Avengers RL — Small Specialists, United They Stand

**Avengers RL** explores whether specialized agents with simple coordination can challenge massive models.

### Core Concept

We train separate 3B-parameter LLAMA models on individual environments (CartPole, MountainCar, LunarLander, BipedalWalker). During inference, a deterministic router switches between specialists based on the current environment phase in our multi-task Sequential Environment. This "Avengers" approach pits the coordinated specialists against massive untrained 20B-parameter OSS models.

#### Phase 1: The Cast of Specialists 🦸‍♂️🦸‍♀️

Our heroes hail from standard Gymnasium environments, each developing unique superpowers:

| Hero | Home Environment | Core Skill |
|------|------------------|------------|
| **CartPole Man** | CartPole-v1 | Lightning-fast balance and stabilization |
| **Walker** | BipedalWalker-v3 | Efficient locomotion over treacherous terrain |
| **Jumper** | MountainCarContinuous-v0 | Momentum generation and gap-crossing mastery |
| **Lander Girl** | LunarLanderContinuous-v2 | Precision thrust control and feather-light landings |

Each agent graduates only after reliably clearing their environment's success thresholds. Architecturally flexible, they can be neural policies or 3B-parameter LLAMA adapters fine-tuned for their specialty.

#### Phase 2: The Endgame Environment 🚀

Step into the **Endgame-v0**—our masterfully crafted composite environment that stitches multiple Gym tasks into one continuous, heart-pounding episode. No smooth transitions here—this is pure controlled chaos, where success demands adapting to wildly different control paradigms!

The environment unfolds across dramatic phases:

1. **Balance & Approach** 🎯 - Maintain pole stability (CartPole) while advancing toward objectives
2. **Bridge Run** 🏃‍♂️ - Navigate uneven terrain (BipedalWalker) with adaptive gaits
3. **Gap Jump** 🦘 - Build momentum and vault valleys (MountainCar)
4. **Final Landing** 🚀 - Control precision descent and achieve perfect touchdowns (LunarLander)

Our smart wrapper handles seamless phase transitions, environment resets, and reward standardization. Observations cleverly encode active phases, timers, and normalized sensor data, while action spaces transparently forward to current sub-environments.

#### Phase 3: Team Coordination Strategy 🧠

At the heart of Avengers RL is a **simple deterministic router logic** that switches between specialists based on the current phase of the environment:

- **CartPole Phase** → CartPole specialist takes control
- **MountainCar Phase** → MountainCar specialist takes control
- **LunarLander Phase** → LunarLander specialist takes control
- **BipedalWalker Phase** → BipedalWalker specialist takes control

The router reads the observation vector to determine which environment phase is active (encoded in the observation) and routes control to the appropriate specialist. No learning is involved—pure rule-based switching!

#### Phase 4: Implementation Architecture ⚙️

Our implementation keeps things simple and focused:

1. **Specialist Training** 🎓 - Train each specialist independently on their environment
2. **Sequential Environment** 🔗 - Combine all environments into composite multi-phase challenges
3. **Deterministic Routing** 🎭 - Hard-coded switching logic based on environment phase
4. **Ensemble Execution** 🧪 - Run all specialists together through the router, with single base model using lightweight LoRA adapters for each specialist

#### Phase 5: Evaluation Showdown 📊

We measure victory across comprehensive metrics:

- **Success Rate**: Percentage of complete multi-phase runs
- **Phase Scores**: Domain-specific mean rewards for each hero
- **Switch Frequency**: How often the team changes leaders
- **Energy Efficiency**: Cumulative control effort optimization
- **Computational Efficiency**: Inference cost vs. baseline models


#### Phase 6: The Final Showdown — Beating Thanos ⚔️

Enter **Thanos**: Our baseline adversary is a colossal **20B-parameter untrained quantized model**—pure raw potential without skill or training. He represents scale without wisdom!

Our **Avengers ensemble**: A handful of 3B-parameter trained specialists plus a deteminstic router logic.

**The Results**: While achievements show promise in matching performance in some domains, clear superiority over OSS models remains elusive. Our experiments demonstrate the viability of specialization + coordination approach but highlight areas for future improvement. **Work in progress: Specialization + Coordination ≈ Current Goliath, with potential ≫ Brute Size.**

#### Takeaway: Compositional Intelligence Has Promise 🎉

**Avengers RL** is both a playful tribute and a serious prototype in hierarchical RL. It demonstrates that multiple narrow experts, each competent in isolation, can be orchestrated to rival monolithic models, though more research is needed to achieve decisive victories.

This work creates concrete testbeds for studying mixture-of-experts routing, model efficiency, and emergent collaborative intelligence.

**Tagline**: Avengers didn't win Endgame V1, Thanos too strong... but they'll be back stronger for V2!

## Environment Server Startup Scripts

Before running any training notebooks, ensure the corresponding environment servers are running:

**CartPole Environment (Port 8030):**
```bash
start_cartpole_server.sh  # or ./OpenEnv/scripts/start_cartpole_server.sh
# Expected port: 8030 - serves CartPole environment for training
```

**MountainCar Environment (Port 8050):**
```bash
start_mountaincar_server.sh  # or ./OpenEnv/scripts/start_mountaincar_server.sh
# Expected port: 8050 - serves MountainCarContinuous environment for training
```

**LunarLander Environment (Port 8090):**
```bash
start_lunarlander_server.sh  # or ./OpenEnv/scripts/start_lunarlander_server.sh
# Expected port: 8090 - serves LunarLanderContinuous environment for training
```

**Sequential Environment (Port 8060):**
```bash
start_sequential_server.sh  # or ./OpenEnv/scripts/start_sequential_server.sh
# Expected port: 8060 - serves combined sequential environment for multi-experiment testing
```

**Note:** Start the environment servers in a separate terminal session before launching training notebooks. Each server provides HTTP REST API endpoints for environment interaction.

## Training Notebooks

**Specialist Training (GRPO on individual environments):**
- [CartPole Training](cartpole_codeGenStrategy_env_llama.ipynb) - GRPO training of LLAMA 3.2 3B for CartPole mastery (requires CartPole server on port 8030)
- [LunarLander Training](lunarlander_codeGenStrategy_env_llama.ipynb) - GRPO training of LLAMA 3.2 3B for LunarLander mastery (requires LunarLander server on port 8090)
- [MountainCar Training](mountaincart_codeGenStrategy_env_llama-Copy1.ipynb) - GRPO training of LLAMA 3.2 3B for MountainCar mastery (requires MountainCar server on port 8050)
- [BipedalWalker Training](bipedal_walker_final.ipynb) - Final implementation for BipedalWalker mastery (requires OpenAI Gym/BipedalWalker)

**Ensemble & Evaluation:**
- [Avengers RL Endgame](avengers_rl_endgame.ipynb) - Sequential environment inference, comparison against OSS models, router logic (requires Sequential server on port 8060)

## Environment Implementations

**Custom OpenEnv Environments (OpenEnv/src/envs/):**
- [CartPole Environment](OpenEnv/src/envs/cartpole_environment/) - RL environment wrapper for CartPole task (server: port 8030)
- [MountainCar Environment](OpenEnv/src/envs/mountaincarcontinuous_environment/) - RL environment wrapper for MountainCar (server: port 8050)
- [LunarLander Environment](OpenEnv/src/envs/lunarlander_environment/) - RL environment wrapper for LunarLander (server: port 8090)
- [Gym Environment](OpenEnv/src/envs/gym_environment/) - Framework integration with Gymnasium
- [Sequential Environment](OpenEnv/src/envs/sequential_environment/) - Multi-phase composite environment combining all environments (server: port 8060)

## Environment Testing

**Comprehensive Test Suite** (Synthetic_Data_Hackathon/OpenEnv/tests/):
- [test_cartpole_env.py](Synthetic_Data_Hackathon/OpenEnv/tests/test_cartpole_env.py) - CartPole environment API tests and integration
- [test_mountaincar_env.py](Synthetic_Data_Hackathon/OpenEnv/tests/test_mountaincar_env.py) - MountainCar environment API tests and integration
- [test_lunarEnv.py](Synthetic_Data_Hackathon/OpenEnv/tests/test_lunarEnv.py) - LunarLander environment API tests and integration
- [test_bipedalwalker_env.py](Synthetic_Data_Hackathon/OpenEnv/tests/test_bipedalwalker_env.py) - BipedalWalker environment API tests and integration
- [test_gym_environment.py](Synthetic_Data_Hackathon/OpenEnv/tests/test_gym_environment.py) - Generic Gym environment API tests
- [test_sequential_environment.py](Synthetic_Data_Hackathon/OpenEnv/tests/test_sequential_environment.py) - Sequential multi-environment integration tests

## Acknowledgments

Many thanks to **Unsloth**, **AMD**, and **Meta** for sponsoring the colossal **AMD M300 GPUs** that powered our fine-tuning and experimentation needs. This work was made possible through their generous hardware support and commitment to advancing AI research.
