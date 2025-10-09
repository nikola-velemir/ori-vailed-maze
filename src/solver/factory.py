import pomdp_py


class PlannerFactory:
    DEFAULT_CONFIG = {
        "max_depth": 25,
        "discount_factor": 0.9,
        "exploration_const": 100,
        "num_sims": 10000
    }
    @staticmethod
    def get_planner( name: str, agent: pomdp_py.Agent, **kwargs):
        config = {**PlannerFactory.DEFAULT_CONFIG,**kwargs}
        config.setdefault("rollout_policy",agent.policy_model)
        _name = name.strip().lower()
        print(config)
        planner_classes = {
            "pomcp": pomdp_py.POMCP,
            "pouct": pomdp_py.POUCT
        }

        if _name not in planner_classes:
            raise ValueError(f"Unknown planner '{name}'. "
                             f"Available: {list(planner_classes.keys())}")
        planner_class = planner_classes[_name]
        return planner_class(**config)
