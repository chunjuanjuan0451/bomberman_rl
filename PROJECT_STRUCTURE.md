# Final-project workspace

The official Bomberman environment files remain untouched. Project work lives in:

- `agent_code/model_a_dqn/`: Dueling Double-DQN target agent.
- `agent_code/model_b_linear/`: interpretable Q-learning baseline.
- `experiments/`: logs, ablation configurations and plotting scripts.
- `tools/`: inference and Docker checks plus module-level tests.

## First implementation milestones

1. Implement Model B BFS features and potential-based shaping; verify it in `coin-heaven`.
2. Implement and test Model A features, symmetry mappings and prioritized sampling separately.
3. Assemble Model A training, then progress through the curriculum defined in `bomberman_rl_full_prd.md` in the parent directory.

Do not package this whole repository for final submission: the requested zip contains
only the chosen agent directory.
