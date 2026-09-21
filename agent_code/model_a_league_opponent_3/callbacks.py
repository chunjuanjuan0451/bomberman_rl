from agent_code.model_a_league_opponent_common.callbacks import act_agent, setup_agent

def setup(self): setup_agent(self, 2)
def act(self, game_state): return act_agent(self, game_state)
