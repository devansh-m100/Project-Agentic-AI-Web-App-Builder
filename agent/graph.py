from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
llm = ChatGroq(model="openai/gpt-oss-120b")

from agent.prompts import *
from agent.states import *
from langgraph.constants import END
from langgraph.graph import StateGraph
from agent.tools import *
from langchain_core.globals import set_verbose, set_debug
from langchain.agents import create_agent

set_debug(True)
set_verbose(True)

def planner_agent(state: dict) -> dict:
    user_prompt = state["user_prompt"]
    resp = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))
    return {"plan": resp}

def architect_agent(state: dict) -> dict:
    """Creates TaskPlan from Plan."""
    plan: Plan = state["plan"]
    resp = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(plan=plan.model_dump_json())
    )
    if resp is None:
        raise ValueError("Planner did not return a valid response.")

    resp.plan = plan
    print(resp.model_dump_json())
    return {"task_plan": resp}

def coder_agent(state: dict) -> dict:
    steps = state['task_plan'].implementation_steps
    current_step_idx = 0
    current_task = steps[current_step_idx]
    existing_content = read_file.run(current_task.filepath)
    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        f"Existing content:\n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )
    coder_tools = [read_file, write_file, list_files, get_current_directory]
    system_prompt = coder_system_prompt()
    react_agent = create_agent(llm, coder_tools)
    react_agent.invoke({"messages": [{"role": "system", "content": system_prompt},
                                     {"role": "user", "content": user_prompt}]})



    return {}

graph = StateGraph(dict)
graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)
graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")

graph.set_entry_point("planner")

agent = graph.compile()

if __name__ == "__main__":

    user_prompt = "create a simple calculator web application"

    result = agent.invoke({"user_prompt": user_prompt})

    print(result)
