"""Proxy-Pointer RAG — entry point.

Bootstraps the four-agent RAG team and runs an interactive chat loop.

**Team structure**
  @Human  →  @RagManager  →  @Retriever
                          →  @Reranker
                          →  @Synthesizer

**5-step retrieval/synthesis contract**
  1. vector_search(query, k=200)          — broad recall from Qdrant
  2. dedup_by_pointer(hits)               — deduplicate by (doc_id, node_id)
  3. rerank_by_hierarchical_path(top_k=5) — LLM-based relevance ranking
  4. load_section(doc_id, node_id)        — read full .md slice from disk
  5. synthesize(query, sections)          — grounded answer with citations

Usage:
    python src/proxy_pointer_rag/main.py

    Then type your question. The team will retrieve, rerank, and synthesise
    a grounded answer with [doc_id#node_id] inline citations.

    Commands:
      /exit   — exit the chat loop
      /team   — show team roster
      /usage  — show LLM usage and cost per agent
      /help   — show this help message
"""

import logging
import time
from collections import defaultdict

import logfire
from proxy_pointer_rag.agents import (
    agent_cards,
    rag_manager_card,
    reranker_card,
    retriever_card,
    synthesizer_card,
)
from akgentic.agent import AgentMessage, BaseAgent, HumanProxy
from akgentic.core import ActorSystem, BaseConfig, EventSubscriber, Orchestrator
from akgentic.core.messages import Message
from akgentic.core.messages.orchestrator import EventMessage, SentMessage
from akgentic.llm import LlmUsageEvent, ToolCallEvent, aggregate_usage

logging.basicConfig(level=logging.ERROR, format="%(name)s - %(levelname)s - %(message)s")


class MessagePrinter(EventSubscriber):
    """Prints messages as they flow through the orchestrator.

    Subscribes to orchestrator events and prints relevant message traffic
    to provide visibility into agent interactions. Collects LlmUsageEvent
    per agent for on-demand cost reporting via get_usage_report().
    """

    EXCLUDED_ROLES = {"Orchestrator", "Human", "human"}

    def __init__(self) -> None:
        self._usage_events: dict[str, list[LlmUsageEvent]] = defaultdict(list)

    def set_restoring(self, restoring: bool) -> None:  # noqa: FBT001
        pass

    def on_stop(self) -> None:
        pass

    def on_stop_request(self) -> None:
        pass

    def on_message(self, message: Message) -> None:
        assert message.sender is not None
        sender = message.sender.name

        if isinstance(message, SentMessage):
            self.handle_sent_message(message, sender)
        elif isinstance(message, EventMessage):
            if isinstance(message.event, ToolCallEvent):
                self.handle_tool_call_event(message, sender)
            elif isinstance(message.event, LlmUsageEvent):
                if message.sender.role not in self.EXCLUDED_ROLES:
                    self._usage_events[sender].append(message.event)

    @staticmethod
    def _fmt_tokens(n: int) -> str:
        return f"{n:,}".replace(",", ".")

    @staticmethod
    def _fmt_cost(c: float) -> str:
        if c == 0.0:
            return "       —"
        formatted = f"{c:,.4f}".replace(",", " ").replace(".", ",").replace(" ", ".")
        return f"${formatted}"

    def get_usage_report(self) -> str:
        """Build a usage report across all tracked agents using aggregate_usage()."""
        lines = [
            f"\n{'Agent':<20} {'In':>10} {'Out':>10} {'Cache R':>10} {'Cache W':>10} {'Cost':>12}",
            "-" * 74,
        ]
        total_events: list[LlmUsageEvent] = []
        for agent_name, events in sorted(self._usage_events.items()):
            agg = aggregate_usage(events)
            total_events.extend(events)
            lines.append(
                f"{agent_name:<20} "
                f"{self._fmt_tokens(agg.total_input_tokens):>10} "
                f"{self._fmt_tokens(agg.total_output_tokens):>10} "
                f"{self._fmt_tokens(agg.total_cache_read_tokens):>10} "
                f"{self._fmt_tokens(agg.total_cache_write_tokens):>10} "
                f"{self._fmt_cost(agg.total_cost_usd):>12}"
            )
        if total_events:
            total = aggregate_usage(total_events)
            lines.append("-" * 74)
            lines.append(
                f"{'TOTAL':<20} "
                f"{self._fmt_tokens(total.total_input_tokens):>10} "
                f"{self._fmt_tokens(total.total_output_tokens):>10} "
                f"{self._fmt_tokens(total.total_cache_read_tokens):>10} "
                f"{self._fmt_tokens(total.total_cache_write_tokens):>10} "
                f"{self._fmt_cost(total.total_cost_usd):>12}"
            )
        return "\n".join(lines)

    def handle_tool_call_event(self, message: EventMessage, sender: str) -> None:
        assert isinstance(message.event, ToolCallEvent)
        print(f"  [tool] {sender} → {message.event.tool_name}({message.event.arguments})")

    def handle_sent_message(self, message: SentMessage, sender: str) -> None:
        assert message.recipient is not None
        msg = message.message
        recipient = message.recipient.name

        if hasattr(msg, "content"):
            content = getattr(msg, "content", "")
            type_ = getattr(msg, "type", "")
            print("-" * 100)
            print(f"[{sender}] -> {msg.__class__.__name__}({type_}) [{recipient}]: \n{content}\n")


def main() -> None:
    """Bootstrap the Proxy-Pointer RAG team and run an interactive chat loop."""

    logfire.configure(console=False)

    # 1. Create ActorSystem
    actor_system = ActorSystem()

    # 2. Create Orchestrator
    orchestrator_addr = actor_system.createActor(
        Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
    )
    orchestrator_proxy = actor_system.proxy_ask(orchestrator_addr, Orchestrator)

    # 3. Subscribe to orchestrator events
    printer = MessagePrinter()
    orchestrator_proxy.subscribe(printer)

    # 4. Register agent cards with orchestrator
    orchestrator_proxy.register_agent_profiles(agent_cards)

    # 5. Create HumanProxy
    human_config = BaseConfig(name="@Human", role="Human")
    human_addr = orchestrator_proxy.createActor(HumanProxy, config=human_config)
    human_proxy = actor_system.proxy_tell(human_addr, HumanProxy)

    # 6. Create RagManager agent
    manager_addr = orchestrator_proxy.createActor(
        BaseAgent,
        config=rag_manager_card.get_config_copy(),
        orchestrator=orchestrator_addr,
    )
    manager_proxy = actor_system.proxy_ask(manager_addr, BaseAgent)

    # 7. Create specialist sub-agents under RagManager
    manager_proxy.createActor(BaseAgent, config=retriever_card.get_config_copy())
    manager_proxy.createActor(BaseAgent, config=reranker_card.get_config_copy())
    manager_proxy.createActor(BaseAgent, config=synthesizer_card.get_config_copy())

    # 8. Wait for actors to initialise
    time.sleep(0.3)

    print()
    print(manager_proxy.cmd_get_team_roster())

    # 9. Interactive chat loop
    def print_help() -> None:
        print("Available commands:")
        print("  /exit   - Exit the chat loop")
        print("  /team   - Show team roster")
        print("  /usage  - Show LLM usage and cost per agent")
        print("  /help   - Show this help message")
        print()

    print(
        "\nType your question (start with @{agent_name} to route to a specific agent, "
        "'exit' to quit or '/help' for help):"
    )
    print("-" * 100)

    while True:
        user_input = input("")
        print()

        if user_input.strip().lower() in ["exit", "/exit"]:
            print("Exiting chat loop.")
            break

        if user_input.strip() == "":
            continue

        if user_input.startswith("/"):
            parts = user_input.split(" ", 1)
            command = parts[0][1:]
            if command == "team":
                print(manager_proxy.cmd_get_team_roster())
                print()
            elif command == "usage":
                print(printer.get_usage_report())
                print()
            else:
                print_help()
            continue

        if user_input.startswith("@"):
            target_name = user_input.split(" ")[0]
            actor_addr = orchestrator_proxy.get_team_member(target_name)
            if actor_addr is None:
                print(f"Error: Agent {target_name} not found")
                continue
            human_proxy.send(actor_addr, AgentMessage(content=user_input))
            continue

        human_proxy.send(manager_addr, AgentMessage(content=user_input))

    # 10. Shutdown
    actor_system.shutdown()


if __name__ == "__main__":
    main()
