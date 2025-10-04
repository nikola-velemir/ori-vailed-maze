import pomdp_py


class PlannerFactory:
    @staticmethod
    def get_planner( name: str, agent: pomdp_py.Agent):
        if name.lower() == 'pomcp':
            return pomdp_py.POMCP(max_depth=20, discount_factor=0.8,
                                  exploration_const=70, num_sims=10000,
                                  rollout_policy=agent.policy_model)
        if name.lower() == "pouct":
            return pomdp_py.POUCT(max_depth=20, discount_factor=0.8,
                                  exploration_const=70, num_sims=10000,
                                  rollout_policy=agent.policy_model)
