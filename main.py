from utils.env_loader import load_env
from agents.solicitation_agent import SolicitationAgent

def main():
    config = load_env()
    agent = SolicitationAgent(config)
    agent.run()

if __name__ == "__main__":
    main()
