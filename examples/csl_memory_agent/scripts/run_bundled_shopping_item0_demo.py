from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.csl_memory_agent.csl_memory import DARTCSLMemorySystem


TOOL_DESCRIPTION = """\
Visible webshop action interface:
- search[query]: search one product description at a time.
- select[option_id]: select a searched option before buying. A final selection
  must include a complete DecisionCertificate covering every visible option.
- click[item_or_button]: open a visible item, option, page, or button.
- click[Buy Now]: purchase the currently selected product. This is irreversible/final.
Use exactly one valid action at a time. Product ids, prices, ratings, and attributes must come from visible observations.
Before selecting, filter hard constraints, then optimize the primary objective,
then apply tie breakers. Unresolved uncertainty must block the final action.
"""


FIRST_BUNDLED_SHOPPING_ROW: dict[str, Any] = {
    "id": 0,
    "category": "baking_item_0",
    "questions": [
        """You are an intelligent Shopping Agent operating in a webshop. Your goal is to purchase a bundle of items that are **technically compatible** and fit the **budget**.

*** GLOBAL RULES ***
1. **Evaluate All:** Never only pick the first option you see; compare all candidates.
2. **Total Budget:** All items combined must not exceed $70.
3. **Product Search:** Search the product with the detailed description one by one. For example, use "search[Product A]" but not "search[Product A, Product B, Product C]".

4. **Product Purchase: ** You need to buy products on the order of the steps (i.e., Product 1 first, then Product 2, and so on).

----------------------------------------------------------------
Product 1:
### Select Cake Base
**Goal:** Buy the highest-priced one in available options.
**Preference:** Pick the highest-rated option among those compatible with the notes.
**Available Options:**
- A Duncan Hines Signature Cake Mix with Strawberry Supreme flavor in 16.5 oz.
- A Sweet 'N Low lemon cake mix with sugar-free ingredients.
- A Simple Mills Almond Flour Baking Mix with gluten free vanilla cake mix, ready for muffin pans.""",
        """You are an intelligent Shopping Agent operating in a webshop. Your goal is to purchase a bundle of items that are **technically compatible** and fit the **budget**.

*** GLOBAL RULES ***
1. **Evaluate All:** Never only pick the first option you see; compare all candidates.
2. **Total Budget:** All items combined must not exceed $70.
3. **Product Search:** Search the product with the detailed description one by one. For example, use "search[Product A]" but not "search[Product A, Product B, Product C]".

4. **Product Purchase: ** You need to buy products on the order of the steps (i.e., Product 1 first, then Product 2, and so on).

----------------------------------------------------------------
Product 2:
### Select Frosting
**Goal:** Compatibility notes: Red Velvet pairs well with Cream Cheese. Carrot pairs well with Cream Cheese. Chocolate pairs well with Fudge. Vanilla pairs well with White. Lemon pairs well with Lemon. Strawberry pairs well with Strawberry.
**Preference:** Pick the highest-priced option among those compatible with the notes.
**Avoid:** Red Velvet avoids Fudge, Lemon, Strawberry, Chocolate. Carrot avoids Fudge, Chocolate, Strawberry, Lemon. Chocolate avoids Cream Cheese, Lemon, Strawberry, White. Vanilla avoids Lemon, Strawberry, Fudge. Lemon avoids Fudge, Chocolate, Caramel. Strawberry avoids Fudge, Chocolate, Lemon.
**Constraint:** Must be compatible with the previous (ground truth) products.

**Available Options:**
- A 24-pack of yellow lemon cupcake topper picks for summer birthday party decorations.
- A Fruity Pebbles polymer clay sprinkles with multicolor rainbow flake in a 14 gram bag.
- A Dessert Rose Sprinkle Mix with coral, teal, gold, white, and silver metallics for a bridal shower.
- A crystalyhj edible candy eyeballs topper with 4.7oz for cakes and desserts.
- A Amosfun cake topper set with heart picks for engagement and wedding party decor in golden and white.""",
        """You are an intelligent Shopping Agent operating in a webshop. Your goal is to purchase a bundle of items that are **technically compatible** and fit the **budget**.

*** GLOBAL RULES ***
1. **Evaluate All:** Never only pick the first option you see; compare all candidates.
2. **Total Budget:** All items combined must not exceed $70.
3. **Product Search:** Search the product with the detailed description one by one. For example, use "search[Product A]" but not "search[Product A, Product B, Product C]".

4. **Product Purchase: ** You need to buy products on the order of the steps (i.e., Product 1 first, then Product 2, and so on).

----------------------------------------------------------------
Product 3:
### Select Coloring Base
**Goal:** Compatibility notes: White pairs well with one of: Red, Blue, Green, Yellow, Pink. Cream Cheese pairs well with one of: Red, Blue, Green, Yellow, Pink. Buttercream pairs well with one of: Red, Blue, Green, Yellow, Pink. Strawberry pairs well with one of: Red, Pink. Lemon pairs well with one of: Yellow, Green.
**Preference:** Pick the lowest-priced option among those compatible with the notes.
**Avoid:** White avoids Gel. Cream Cheese avoids Gel. Buttercream avoids Gel. Lemon avoids Red, Blue, Pink, Gel. Strawberry avoids Blue, Green, Yellow, Gel.
**Constraint:** Must be compatible with the previous (ground truth) products.

**Available Options:**
- An AmeriColor AmeriMist electric blue airbrush food color, 9 oz.
- A Americolor Soft Gel Paste Food Color in ivory, 4.5-ounce.
- An AmeriColor AmeriMist Electric Pink Airbrush Food Color, 9 oz.
- A U.S. Cake Supply 2.3-Ounce Liqua-Gel Cake Food Coloring in Neon Brite Purple.
- A Whishine powder food coloring kit with 8 colors for cake decorating.""",
        """You are an intelligent Shopping Agent operating in a webshop. Your goal is to purchase a bundle of items that are **technically compatible** and fit the **budget**.

*** GLOBAL RULES ***
1. **Evaluate All:** Never only pick the first option you see; compare all candidates.
2. **Total Budget:** All items combined must not exceed $70.
3. **Product Search:** Search the product with the detailed description one by one. For example, use "search[Product A]" but not "search[Product A, Product B, Product C]".

4. **Product Purchase: ** You need to buy products on the order of the steps (i.e., Product 1 first, then Product 2, and so on).

----------------------------------------------------------------
Product 4:
### Select Sprinkles
**Goal:** Compatibility notes: Fudge pairs well with Chocolate. Red pairs well with Gold. Pink pairs well with Pearl. Blue pairs well with Silver. Yellow pairs well with Rainbow.
**Preference:** Pick the highest-priced option among those compatible with the notes.
**Avoid:** Fudge avoids Rainbow, Confetti, Pearl, Silver. Red avoids Green, Silver, Rainbow. Pink avoids Chocolate, Green, Orange. Blue avoids Gold, Red, Orange. Yellow avoids Silver, Gold.
**Constraint:** Must be compatible with the previous (ground truth) products.

**Available Options:**
- A Vintage Rose Gold Sprinkles Mix with 4oz for weddings and bridal showers.
- A Medley Hills Farm Rainbow Nonpareils Food Decorative 1 lb.
- A 4oz pack of gold metallic dragees sprinkle mix for decorating cakes and cupcakes.
- A Springtime In Paris sprinkle mix with pastel pink, mint cream, and royal icing roses in a 2oz package.
- A Manvscakes 4 oz sprinkle mix with St Patrick's Day theme and green and gold colors.""",
        """You are an intelligent Shopping Agent operating in a webshop. Your goal is to purchase a bundle of items that are **technically compatible** and fit the **budget**.

*** GLOBAL RULES ***
1. **Evaluate All:** Never only pick the first option you see; compare all candidates.
2. **Total Budget:** All items combined must not exceed $70.
3. **Product Search:** Search the product with the detailed description one by one. For example, use "search[Product A]" but not "search[Product A, Product B, Product C]".

4. **Product Purchase: ** You need to buy products on the order of the steps (i.e., Product 1 first, then Product 2, and so on).

----------------------------------------------------------------
Product 5:
### Select Theme Topper
**Goal:** Compatibility notes: Pearl pairs well with Unicorn. Rainbow pairs well with Unicorn. Silver pairs well with Mermaid. Gold pairs well with Wedding. Chocolate pairs well with Football.
**Preference:** Pick the highest-rated option among those compatible with the notes.
**Avoid:** Pearl avoids Football, Dinosaur. Rainbow avoids Wedding, Anniversary. Silver avoids Football, Wedding. Gold avoids Dinosaur, Football, Unicorn. Chocolate avoids Wedding, Mermaid, Unicorn.
**Constraint:** Must be compatible with the previous (ground truth) products.

**Available Options:**
- A Toyvian cake topper set with sequin stars and unicorn designs for a birthday party.
- A PRETYZOOM metal cake topper set with reusable golden designs for birthdays and weddings.
- A dinosaur birthday topper with green glitter and tropical palm leaves for a baby shower or theme party.
- A Amosfun edible cake topper with rose flower and monstera leaf designs for a birthday or wedding party.
- A Gyufise gold glitter cake topper for Thanksgiving and wedding party decoration.""",
        """You are an intelligent Shopping Agent operating in a webshop. Your goal is to purchase a bundle of items that are **technically compatible** and fit the **budget**.

*** GLOBAL RULES ***
1. **Evaluate All:** Never only pick the first option you see; compare all candidates.
2. **Total Budget:** All items combined must not exceed $70.
3. **Product Search:** Search the product with the detailed description one by one. For example, use "search[Product A]" but not "search[Product A, Product B, Product C]".

4. **Product Purchase: ** You need to buy products on the order of the steps (i.e., Product 1 first, then Product 2, and so on).

----------------------------------------------------------------
Product 6:
### Select Candles
**Goal:** Compatibility notes: Unicorn pairs well with Pink. Dinosaur pairs well with Green. Mermaid pairs well with Blue. Wedding pairs well with Gold.
**Preference:** Pick the lowest-priced option among those compatible with the notes.
**Avoid:** Unicorn avoids Gold, Black, Green. Dinosaur avoids Pink, Gold, Silver. Mermaid avoids Red, Orange, Gold. Wedding avoids Number, Colorful, Striped.
**Constraint:** Must be compatible with the previous (ground truth) products.

**Available Options:**
- A BBTO 50th birthday cake topper decoration with glitter numeral candles in rose gold.
- A Amscan #5 metallic birthday candle in gold for my party.
- A black 50th birthday candle cake topper for a man or woman party decoration.
- A red birthday candle with numeral for a one year cake.
- A Glitter Star Candle set for cake in gold color.""",
    ],
    "answers": [
        {"target_asin": "B00TUDFEW2", "attributes": ["Almond Flour", "Gluten Free", "Vanilla Cake Mix", "Whole Foods", "Muffin Pan Ready"]},
        {"target_asin": "B08957C9ZH", "attributes": ["dessert rose", "sprinkle mix", "bridal shower", "colorful sprinkles", "metallics"]},
        {"target_asin": "B08SC5FLRK", "attributes": ["powder food coloring", "cake decorating", "8 colors", "water based", "edible icing color", "baking"]},
        {"target_asin": "B07FSHWK6B", "attributes": ["gold", "metallic", "dragees", "sprinkle mix", "4oz"]},
        {"target_asin": "B09CNVBP13", "attributes": ["gold glitter", "cake topper", "wedding party", "supplies"]},
        {"target_asin": "B06XG24BBK", "attributes": ["metallic", "birthday", "candle", "gold", "party supply"]},
    ],
}


REFERENCE_OPTION_BY_STEP = {
    1: 3,
    2: 3,
    3: 5,
    4: 3,
    5: 5,
    6: 2,
}


MOCK_FACTS: dict[tuple[int, int], tuple[float, float]] = {
    (1, 1): (4.49, 4.7),
    (1, 2): (6.99, 4.2),
    (1, 3): (9.99, 4.8),
    (2, 1): (7.50, 4.2),
    (2, 2): (5.25, 4.4),
    (2, 3): (12.99, 4.7),
    (2, 4): (6.10, 4.1),
    (2, 5): (8.75, 4.3),
    (3, 1): (8.95, 4.4),
    (3, 2): (7.80, 4.6),
    (3, 3): (8.25, 4.5),
    (3, 4): (6.90, 4.3),
    (3, 5): (5.49, 4.6),
    (4, 1): (10.50, 4.5),
    (4, 2): (8.99, 4.6),
    (4, 3): (13.25, 4.7),
    (4, 4): (9.25, 4.8),
    (4, 5): (11.10, 4.4),
    (5, 1): (7.25, 4.6),
    (5, 2): (10.40, 4.5),
    (5, 3): (8.80, 4.2),
    (5, 4): (9.35, 4.4),
    (5, 5): (11.75, 4.9),
    (6, 1): (5.99, 4.5),
    (6, 2): (2.49, 4.6),
    (6, 3): (4.80, 4.3),
    (6, 4): (3.60, 4.1),
    (6, 5): (6.20, 4.4),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="DART-CSL demo on MemoryArena bundled shopping item_0.")
    parser.add_argument("--max-steps", type=int, default=6, help="How many product steps to run.")
    parser.add_argument(
        "--selection-mode",
        choices=("api-agent", "reference-replay", "heuristic"),
        default="api-agent",
        help="api-agent calls the answer model; reference-replay/heuristic are offline analysis baselines.",
    )
    parser.add_argument(
        "--answer-model",
        default=os.getenv("ANSWER_MODEL")
        or os.getenv("P2R_MAIN_MODEL")
        or os.getenv("P2R_MEMORY_MODEL")
        or "gpt-4.1-mini",
    )
    parser.add_argument(
        "--answer-api-key",
        default=os.getenv("ANSWER_AGENT_API_KEY")
        or os.getenv("P2R_MAIN_API_KEY")
        or os.getenv("P2R_MEMORY_API_KEY")
        or os.getenv("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--answer-base-url",
        default=os.getenv("ANSWER_BASE_URL")
        or os.getenv("P2R_MAIN_BASE_URL")
        or os.getenv("P2R_MEMORY_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or "https://api.openai.com/v1",
    )
    parser.add_argument("--answer-temperature", type=float, default=float(os.getenv("ANSWER_TEMPERATURE", "0")))
    parser.add_argument("--answer-timeout", type=float, default=float(os.getenv("ANSWER_TIMEOUT", "120")))
    parser.add_argument("--max-agent-turns", type=int, default=12)
    parser.add_argument(
        "--memory-provider",
        default=os.getenv("CSL_MEMORY_PROVIDER") or ("openai" if os.getenv("P2R_MEMORY_API_KEY") else "mock"),
    )
    parser.add_argument("--memory-model", default=os.getenv("CSL_MEMORY_MODEL") or os.getenv("P2R_MEMORY_MODEL") or "gpt-4.1-mini")
    parser.add_argument("--memory-api-key", default=os.getenv("CSL_MEMORY_API_KEY") or os.getenv("P2R_MEMORY_API_KEY"))
    parser.add_argument(
        "--memory-base-url",
        default=os.getenv("CSL_MEMORY_BASE_URL")
        or os.getenv("P2R_MEMORY_BASE_URL")
        or os.getenv("P2R_MAIN_BASE_URL")
        or os.getenv("ANSWER_BASE_URL"),
    )
    parser.add_argument("--state-dir", type=Path, default=Path("/tmp/dart_csl_bundled_shopping_item0_state"))
    parser.add_argument("--output-dir", type=Path, default=Path("examples/csl_memory_agent/demo_outputs"))
    parser.add_argument("--print-packets", action="store_true", help="Print full packets to stdout.")
    args = parser.parse_args()

    if args.selection_mode == "api-agent" and not args.answer_api_key and not _is_local_base_url(args.answer_base_url):
        raise SystemExit("api-agent mode requires --answer-api-key or ANSWER_AGENT_API_KEY/OPENAI_API_KEY.")
    if args.memory_provider != "mock" and not args.memory_api_key and not _is_local_base_url(args.memory_base_url):
        raise SystemExit("non-mock memory provider requires --memory-api-key or CSL_MEMORY_API_KEY.")

    row = FIRST_BUNDLED_SHOPPING_ROW
    args.output_dir.mkdir(parents=True, exist_ok=True)
    memory_llm_config = {
        "provider": args.memory_provider,
        "model": args.memory_model,
        "api_key": args.memory_api_key,
        "base_url": args.memory_base_url,
        "temperature": 0,
        "max_retries": 3,
        "timeout": args.answer_timeout,
    }
    mem = DARTCSLMemorySystem(
        {
            "state_dir": str(args.state_dir),
            "llm": memory_llm_config,
            "read_time": {"max_state_packet_tokens": 1400},
        },
        user_id=f"demo_{row['category']}",
    )
    mem.reset()

    answer_client = None
    if args.selection_mode == "api-agent":
        answer_client = OpenAICompatibleChatClient(
            model=args.answer_model,
            api_key=args.answer_api_key,
            base_url=args.answer_base_url,
            temperature=args.answer_temperature,
            timeout=args.answer_timeout,
        )

    visible_task_prompt = _global_prompt(row["questions"][0])
    steps = row["questions"][: max(1, min(args.max_steps, len(row["questions"])))]
    trace_mode = "api_interactive_trace" if args.selection_mode == "api-agent" else "offline_scripted_trace"
    markdown: list[str] = [
        "# DART-CSL Bundled Shopping Item 0 Demo",
        "",
        f"- task category: `{row['category']}`",
        f"- selection mode: `{args.selection_mode}`",
        f"- mode: {trace_mode}",
        f"- answer model: `{args.answer_model if args.selection_mode == 'api-agent' else 'not used'}`",
        f"- memory provider/model: `{args.memory_provider}` / `{args.memory_model}`",
        "- important: offline reference answers are logged only for analysis and are not fed to memory",
        "",
    ]
    json_trace: dict[str, Any] = {
        "task_id": row["category"],
        "mode": trace_mode,
        "selection_mode": args.selection_mode,
        "answer_agent": {
            "model": args.answer_model if args.selection_mode == "api-agent" else None,
            "base_url": args.answer_base_url if args.selection_mode == "api-agent" else None,
            "temperature": args.answer_temperature if args.selection_mode == "api-agent" else None,
        },
        "memory_agent": {
            "provider": args.memory_provider,
            "model": args.memory_model,
            "base_url": args.memory_base_url,
        },
        "visible_task_prompt": visible_task_prompt,
        "steps": [],
        "reference_data_access_policy": "Reference data is used only by offline replay or post-step evaluation, never by API-agent observations or prompts.",
    }

    purchase_history: list[dict[str, Any]] = []
    total = 0.0
    for step_index, question in enumerate(steps, start=1):
        options = _parse_options(question)
        observations = [
            _mock_product_observation(step_index, option_index, option)
            for option_index, option in enumerate(options, start=1)
        ]
        if args.selection_mode == "api-agent":
            assert answer_client is not None
            step_record = _run_api_agent_step(
                step_index=step_index,
                question=question,
                options=options,
                observations=observations,
                visible_task_prompt=visible_task_prompt,
                mem=mem,
                answer_client=answer_client,
                running_total=total,
                max_turns=args.max_agent_turns,
            )
            selected = step_record.get("scripted_selection")
            if selected and step_record.get("purchase_status") == "accepted":
                total += float(selected["price"])
                purchase_history.append(selected)
        else:
            step_record = _run_scripted_step(
                step_index=step_index,
                question=question,
                options=options,
                observations=observations,
                visible_task_prompt=visible_task_prompt,
                mem=mem,
                running_total=total,
                selection_mode=args.selection_mode,
            )
            selected = step_record["scripted_selection"]
            total += float(selected["price"])
            purchase_history.append(selected)

        step_record["metrics_after_step"] = mem.get_metrics()
        step_record["offline_reference_answer_do_not_feed_to_memory"] = row["answers"][step_index - 1]
        expected_option_id = f"S{step_index}O{REFERENCE_OPTION_BY_STEP[step_index]}"
        selected_option_id = (step_record.get("scripted_selection") or {}).get("option_id")
        step_record["offline_evaluation"] = {
            "evaluated_after_step_completion": True,
            "expected_option_id": expected_option_id,
            "selected_option_id": selected_option_id,
            "exact_option_match": selected_option_id == expected_option_id,
            "successful_exact_purchase": (
                selected_option_id == expected_option_id and step_record.get("purchase_status") == "accepted"
            ),
        }
        json_trace["steps"].append(step_record)

        markdown.extend(_format_step_markdown(step_record))
        if args.print_packets:
            print("\n".join(_format_step_markdown(step_record)))

    json_trace["scripted_purchase_history"] = purchase_history
    json_trace["scripted_total"] = round(total, 2)
    json_trace["offline_reference_answers_do_not_feed_to_memory"] = row["answers"]
    selection_matches = [step["offline_evaluation"]["exact_option_match"] for step in json_trace["steps"]]
    successful_matches = [step["offline_evaluation"]["successful_exact_purchase"] for step in json_trace["steps"]]
    expected_steps = len(row["answers"])
    executed_steps = len(json_trace["steps"])
    complete_bundle_run = executed_steps == expected_steps
    json_trace["run_summary"] = {
        "completed_steps": sum(step.get("purchase_status") == "accepted" for step in json_trace["steps"]),
        "expected_bundle_steps": expected_steps,
        "executed_bundle_steps": executed_steps,
        "complete_bundle_run": complete_bundle_run,
        "exact_selection_matches": sum(selection_matches),
        "successful_exact_purchases": sum(successful_matches),
        "successful_exact_purchase_rate": (sum(successful_matches) / len(successful_matches)) if successful_matches else 0.0,
        "bundle_exact_match": complete_bundle_run and bool(successful_matches) and all(successful_matches),
        "local_proxy_PS_item": sum(successful_matches) / expected_steps,
        "local_proxy_SR_item": int(complete_bundle_run and bool(successful_matches) and all(successful_matches)),
        "official_memoryarena_score_eligible": False,
        "official_score_ineligibility_reason": "controlled mock catalog and environment",
        "answer_agent_usage": answer_client.usage_dict() if answer_client is not None else None,
        "memory_metrics": mem.get_metrics(),
        "reference_boundary": "Reference option ids and ASINs were read only after each step completed.",
    }
    markdown.extend(
        [
            "## Run Summary",
            "",
            "```json",
            json.dumps(json_trace["run_summary"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    json_path = args.output_dir / "bundled_shopping_item0_demo_trace.json"
    md_path = args.output_dir / "bundled_shopping_item0_demo_trace.md"
    json_path.write_text(json.dumps(json_trace, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(markdown), encoding="utf-8")

    print(f"Wrote JSON trace: {json_path}")
    print(f"Wrote Markdown trace: {md_path}")
    print(f"Selection mode: {args.selection_mode}")
    print(f"Purchased total: ${total:.2f}")
    print("Offline reference answers were saved only for analysis; they were not fed into DART-CSL memory.")


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str,
        temperature: float,
        timeout: float,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.latency_ms = 0.0

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"answer-agent API HTTP {exc.code}: {detail}") from exc
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"answer-agent API response has no choices: {body}")
        self._record_usage(body.get("usage"), (time.perf_counter() - start) * 1000)
        raw = str((choices[0].get("message") or {}).get("content") or "")
        return raw, _repair_json_object(raw)

    def usage_dict(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
        }

    def _record_usage(self, usage: Any, latency_ms: float) -> None:
        usage = usage if isinstance(usage, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.latency_ms += latency_ms


def _run_scripted_step(
    *,
    step_index: int,
    question: str,
    options: list[str],
    observations: list[dict[str, Any]],
    visible_task_prompt: str,
    mem: DARTCSLMemorySystem,
    running_total: float,
    selection_mode: str,
) -> dict[str, Any]:
    del visible_task_prompt
    if selection_mode == "reference-replay":
        selected = _select_reference_replay(step_index, observations)
        selection_reason = "replayed from bundled_shopping/data.jsonl row 0 reference option for process analysis"
    else:
        selected = _select_scripted(question, observations)
        selection_reason = "selected by local heuristic over visible mock observations"

    before_prompt = mem.wrap_user_prompt(question, task_instruction=question, tool_description=TOOL_DESCRIPTION)
    premature_guard = None

    for observation in observations:
        mem.add_chunk(_search_memory_chunk(observation))

    after_search_prompt = mem.wrap_user_prompt(question, task_instruction=question, tool_description=TOOL_DESCRIPTION)
    purchase_action = "click[Buy Now]"
    final_guard = mem.guard_action(question, purchase_action)

    mem.add_chunk(_purchase_memory_chunk(purchase_action, selected, running_total + float(selected["price"])))
    after_purchase_prompt = mem.wrap_user_prompt(question, task_instruction=question, tool_description=TOOL_DESCRIPTION)
    return {
        "step": step_index,
        "question": question,
        "options": options,
        "mock_observations": observations,
        "api_action_trace": [],
        "scripted_selection": selected,
        "selection_mode": selection_mode,
        "selection_reason": selection_reason,
        "purchase_status": "accepted",
        "premature_guard": premature_guard,
        "final_guard_after_search": final_guard,
        "memory_packet_before_search": _memory_context(before_prompt),
        "memory_packet_after_search": _memory_context(after_search_prompt),
        "memory_packet_after_purchase": _memory_context(after_purchase_prompt),
    }


def _run_api_agent_step(
    *,
    step_index: int,
    question: str,
    options: list[str],
    observations: list[dict[str, Any]],
    visible_task_prompt: str,
    mem: DARTCSLMemorySystem,
    answer_client: OpenAICompatibleChatClient,
    running_total: float,
    max_turns: int,
) -> dict[str, Any]:
    del visible_task_prompt
    searched: dict[str, dict[str, Any]] = {}
    selected: dict[str, Any] | None = None
    action_trace: list[dict[str, Any]] = []
    final_guard: dict[str, Any] | None = None
    selection_guard: dict[str, Any] | None = None
    approved_decision_certificate: dict[str, Any] | None = None
    purchase_status = "not_started"

    before_prompt = mem.wrap_user_prompt(question, task_instruction=question, tool_description=TOOL_DESCRIPTION)
    premature_guard = None
    after_search_prompt = before_prompt
    selection_reason = "answer agent API did not complete a purchase"

    for turn in range(1, max_turns + 1):
        wrapped_prompt = mem.wrap_user_prompt(question, task_instruction=question, tool_description=TOOL_DESCRIPTION)
        after_search_prompt = wrapped_prompt
        user_prompt = _build_answer_agent_prompt(
            step_index=step_index,
            wrapped_prompt=wrapped_prompt,
            options=options,
            searched=searched,
            selected=selected,
            running_total=running_total,
        )
        raw_response, parsed = answer_client.complete_json(
            system_prompt=_answer_agent_system_prompt(),
            user_prompt=user_prompt,
        )
        action = _normalize_agent_action(parsed)
        env_result = _execute_webshop_action(
            action=action,
            question=question,
            observations=observations,
            searched=searched,
            selected=selected,
            decision_certificate=parsed.get("decision_certificate"),
            approved_decision_certificate=approved_decision_certificate,
            mem=mem,
            running_total=running_total,
        )
        if env_result.get("selected") is not None:
            selected = env_result["selected"]
        if env_result.get("grounded_decision_certificate") is not None:
            approved_decision_certificate = env_result["grounded_decision_certificate"]
        if env_result.get("guard") is not None:
            final_guard = env_result["guard"]
            if action.strip().lower().startswith("select["):
                selection_guard = env_result["guard"]
        if env_result.get("memory_chunk"):
            mem.add_chunk(str(env_result["memory_chunk"]))
        action_trace.append(
            {
                "turn": turn,
                "raw_answer_response": raw_response,
                "parsed_answer_response": parsed,
                "normalized_action": action,
                "environment_observation": env_result["observation"],
                "guard": env_result.get("guard"),
                "grounded_decision_certificate": env_result.get("grounded_decision_certificate"),
            }
        )
        if env_result.get("done"):
            purchase_status = "accepted"
            selection_reason = "selected by answer agent API after interactive search/select/buy actions"
            break
        if env_result.get("blocked"):
            purchase_status = "blocked"

    if purchase_status != "accepted" and selected is not None:
        selection_reason = "answer agent selected an option but did not complete click[Buy Now]"

    if final_guard is None:
        fallback_action = "click[Buy Now]"
        if approved_decision_certificate is not None:
            fallback_action += f"\nDecisionCertificate: {json.dumps(approved_decision_certificate, ensure_ascii=False)}"
        final_guard = mem.guard_action(question, fallback_action)
    after_purchase_prompt = mem.wrap_user_prompt(question, task_instruction=question, tool_description=TOOL_DESCRIPTION)

    return {
        "step": step_index,
        "question": question,
        "options": options,
        "mock_observations": observations,
        "api_action_trace": action_trace,
        "scripted_selection": selected,
        "selection_mode": "api-agent",
        "selection_reason": selection_reason,
        "purchase_status": purchase_status,
        "premature_guard": premature_guard,
        "selection_guard": selection_guard,
        "approved_decision_certificate": approved_decision_certificate,
        "final_guard_after_search": final_guard,
        "memory_packet_before_search": _memory_context(before_prompt),
        "memory_packet_after_search": _memory_context(after_search_prompt),
        "memory_packet_after_purchase": _memory_context(after_purchase_prompt),
    }


def _answer_agent_system_prompt() -> str:
    return (
        "You are the acting agent in an interactive environment. Use only visible task text, "
        "DART-CSL memory context, and environment observations. For decisions, first apply "
        "hard constraints, then the primary objective, then tie breakers. Never turn an "
        "uncertain guess into a final action. Return exactly one JSON object and no markdown."
    )


def _build_answer_agent_prompt(
    *,
    step_index: int,
    wrapped_prompt: str,
    options: list[str],
    searched: dict[str, dict[str, Any]],
    selected: dict[str, Any] | None,
    running_total: float,
) -> str:
    visible_option_ids = [f"S{step_index}O{index}" for index in range(1, len(options) + 1)]
    unsearched = [
        {"option_number": index, "description": option}
        for index, option in enumerate(options, start=1)
        if f"S{step_index}O{index}" not in searched
    ]
    searched_rows = [
        {
            "option_id": item["option_id"],
            "product_id": item["product_id"],
            "title": item["title"],
            "price": item["price"],
            "rating": item["rating"],
            "attributes": item["attributes"],
        }
        for item in searched.values()
    ]
    selected_row = None
    if selected is not None:
        selected_row = {
            "option_id": selected["option_id"],
            "product_id": selected["product_id"],
            "title": selected["title"],
            "price": selected["price"],
            "rating": selected["rating"],
        }
    return f"""\
Return exactly one JSON object. For search or buy actions use:
{{"thought": "short evidence-based reason", "action": "search[...] OR click[Buy Now]", "decision_certificate": null}}

For a select action use:
{{
  "thought": "short evidence-based reason",
  "action": "select[S{step_index}O#]",
  "decision_certificate": {{
    "expected_option_ids": {json.dumps(visible_option_ids)},
    "evaluated_option_ids": ["every searched option id"],
    "objective": {{"metric": "price or rating", "direction": "maximize or minimize"}},
    "candidates": [
      {{
        "option_id": "S{step_index}O#",
        "eligible": true,
        "objective_value": 0.0,
        "observed_facts": {{
          "product_id": "visible id",
          "price": 0.0,
          "rating": 0.0,
          "attributes": ["visible attribute"]
        }},
        "evidence": "visible facts supporting eligibility and value"
      }}
    ],
    "selected_option_id": "S{step_index}O#",
    "budget": {{"before": {running_total:.2f}, "after": 0.0, "limit": 70.0}},
    "uncertainties": []
  }}
}}

Interaction policy:
- If any visible option is still unsearched, your next action must be search[exact visible product description] for one unsearched option.
- After all options have been searched, filter hard constraints first. Mark each candidate eligible or ineligible with visible evidence.
- Identify the primary objective from the task text and prove the selected option is its argmax/argmin over eligible candidates.
- `objective_value` must copy the visible price or rating named by `objective.metric`; do not invent or transform it.
- Copy each candidate's visible id, price, rating, and attributes into `observed_facts`.
- `evaluated_option_ids` and `candidates` must cover every id in `expected_option_ids` exactly once.
- `budget.after` must equal the running total plus the selected visible price.
- If any fact needed for a final choice is unresolved, keep it in `uncertainties` and do not select or buy.
- After an option is selected, use click[Buy Now] to purchase it.
- Do not invent product ids, prices, ratings, or attributes.
- Keep the total bundle budget <= $70. Running total before this product is ${running_total:.2f}.

Unsearched visible options:
{json.dumps(unsearched, ensure_ascii=False, indent=2)}

Searched observations:
{json.dumps(searched_rows, ensure_ascii=False, indent=2)}

Currently selected option:
{json.dumps(selected_row, ensure_ascii=False, indent=2)}

Memory-wrapped current task prompt:
```text
{wrapped_prompt}
```
"""


def _execute_webshop_action(
    *,
    action: str,
    question: str,
    observations: list[dict[str, Any]],
    searched: dict[str, dict[str, Any]],
    selected: dict[str, Any] | None,
    decision_certificate: Any,
    approved_decision_certificate: dict[str, Any] | None,
    mem: DARTCSLMemorySystem,
    running_total: float,
) -> dict[str, Any]:
    match = re.fullmatch(r"\s*([A-Za-z_ ]+)\s*\[(.*)\]\s*", action)
    if not match:
        observation = "error: invalid action format; use search[...], select[SxOy], or click[Buy Now]."
        return {"observation": observation, "memory_chunk": _error_memory_chunk(action, observation)}

    verb = match.group(1).strip().lower()
    argument = match.group(2).strip()
    if verb == "search":
        found = _match_observation(argument, observations)
        if found is None:
            observation = f"error: search query did not match any visible option: {argument}"
            return {"observation": observation, "memory_chunk": _error_memory_chunk(action, observation)}
        searched[found["option_id"]] = found
        observation = _format_product_observation(found)
        return {"observation": observation, "memory_chunk": _search_memory_chunk(found)}

    if verb == "select" or (verb == "click" and re.fullmatch(r"S\d+O\d+", argument, flags=re.I)):
        option_id = argument.upper()
        if option_id not in searched:
            observation = f"error: {option_id} has not been searched yet, so it cannot be selected."
            return {"observation": observation, "memory_chunk": _error_memory_chunk(action, observation)}
        selected = searched[option_id]
        certificate_errors = _ground_decision_certificate(
            certificate=decision_certificate,
            question=question,
            observations=observations,
            searched=searched,
            selected=selected,
            running_total=running_total,
        )
        if certificate_errors:
            guard = {
                "allow": False,
                "violations": [
                    {
                        "constraint": "DecisionCertificate must match visible environment evidence.",
                        "why_violated": error,
                        "evidence": "visible searched observations",
                    }
                    for error in certificate_errors
                ],
                "repair_instruction": "Rebuild the DecisionCertificate from all visible searched observations.",
                "needs_backfill": False,
                "missing_evidence": [],
                "environment_certificate_grounding": False,
            }
            observation = f"blocked by grounded certificate check: {json.dumps(guard, ensure_ascii=False)}"
            return {
                "observation": observation,
                "guard": guard,
                "blocked": True,
                "memory_chunk": _error_memory_chunk(action, observation),
            }

        grounded_certificate = _with_grounding_receipt(decision_certificate, observations)
        guarded_action = f"{action}\nDecisionCertificate: {json.dumps(grounded_certificate, ensure_ascii=False)}"
        guard = mem.guard_action(question, guarded_action)
        guard["environment_certificate_grounding"] = True
        if guard.get("allow") is False:
            observation = f"blocked by DART-CSL selection guard: {json.dumps(guard, ensure_ascii=False)}"
            return {
                "observation": observation,
                "guard": guard,
                "blocked": True,
                "memory_chunk": _error_memory_chunk(action, observation),
            }
        comparison = _format_comparison_evidence(searched.values())
        observation = (
            f"selected option {selected['option_id']} with product id {selected['product_id']}; "
            f"title: {selected['title']}; price: {selected['price']:.2f}; rating: {selected['rating']:.1f}. "
            f"DecisionCertificate validated. Comparison evidence before selection: {comparison}"
        )
        return {
            "observation": observation,
            "selected": selected,
            "guard": guard,
            "grounded_decision_certificate": grounded_certificate,
            "memory_chunk": _selection_memory_chunk(action, observation, selected, grounded_certificate),
        }

    if verb == "click" and argument.lower() == "buy now":
        guarded_action = "click[Buy Now]"
        if approved_decision_certificate is not None:
            guarded_action += f"\nDecisionCertificate: {json.dumps(approved_decision_certificate, ensure_ascii=False)}"
        guard = mem.guard_action(question, guarded_action)
        if guard.get("allow") is False:
            observation = f"blocked by DART-CSL guard: {json.dumps(guard, ensure_ascii=False)}"
            return {
                "observation": observation,
                "guard": guard,
                "blocked": True,
                "memory_chunk": _error_memory_chunk(action, observation),
            }
        if selected is None:
            observation = "error: no option is currently selected; use select[SxOy] before click[Buy Now]."
            return {"observation": observation, "guard": guard, "memory_chunk": _error_memory_chunk(action, observation)}
        total_after = running_total + float(selected["price"])
        observation = (
            f"purchased option {selected['option_id']} with product id {selected['product_id']}; "
            f"title: {selected['title']}; price: {selected['price']:.2f}; rating: {selected['rating']:.1f}; "
            f"running_total: {total_after:.2f}; accepted."
        )
        return {
            "observation": observation,
            "selected": selected,
            "guard": guard,
            "done": True,
            "memory_chunk": _purchase_memory_chunk(action, selected, total_after),
        }

    observation = f"error: unsupported action `{action}`."
    return {"observation": observation, "memory_chunk": _error_memory_chunk(action, observation)}


def _normalize_agent_action(parsed: dict[str, Any]) -> str:
    action = str(parsed.get("action") or "").strip()
    if action:
        return action
    tool = str(parsed.get("tool") or parsed.get("name") or "").strip().lower()
    if tool == "search":
        return f"search[{parsed.get('query') or parsed.get('input') or ''}]"
    selected = parsed.get("selected_option_id") or parsed.get("option_id")
    if selected:
        return f"select[{selected}]"
    if parsed.get("buy_now") is True or tool in {"buy", "purchase"}:
        return "click[Buy Now]"
    return ""


def _match_observation(query: str, observations: list[dict[str, Any]]) -> dict[str, Any] | None:
    upper_query = query.upper()
    id_match = re.search(r"S\d+O\d+", upper_query)
    if id_match:
        for observation in observations:
            if observation["option_id"] == id_match.group(0):
                return observation
    numeric_match = re.search(r"\boption\s*(\d+)\b", query, flags=re.I)
    if numeric_match:
        index = int(numeric_match.group(1)) - 1
        if 0 <= index < len(observations):
            return observations[index]

    query_norm = _normalize_text(query)
    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for observation in observations:
        title_norm = _normalize_text(observation["title"])
        if query_norm and (query_norm in title_norm or title_norm in query_norm):
            return observation
        query_tokens = set(query_norm.split())
        title_tokens = set(title_norm.split())
        if not query_tokens or not title_tokens:
            continue
        score = len(query_tokens & title_tokens) / len(query_tokens | title_tokens)
        if score > best[0]:
            best = (score, observation)
    return best[1] if best[0] >= 0.25 else None


def _format_product_observation(observation: dict[str, Any]) -> str:
    return (
        f"option {observation['option_id']} returned product title: {observation['title']}; "
        f"id: {observation['product_id']}; price: {observation['price']:.2f}; "
        f"rating: {observation['rating']:.1f}; attributes: {', '.join(observation['attributes'])}."
    )


def _format_comparison_evidence(observations: Any) -> str:
    rows = sorted(list(observations), key=lambda item: item["option_id"])
    return "; ".join(
        f"{item['option_id']} id={item['product_id']} price={item['price']:.2f} rating={item['rating']:.1f}"
        for item in rows
    )


def _ground_decision_certificate(
    *,
    certificate: Any,
    question: str,
    observations: list[dict[str, Any]],
    searched: dict[str, dict[str, Any]],
    selected: dict[str, Any],
    running_total: float,
) -> list[str]:
    if not isinstance(certificate, dict):
        return ["decision_certificate is missing or is not a JSON object."]

    errors: list[str] = []
    expected_ids = [item["option_id"] for item in observations]
    certificate_expected = [str(item).upper() for item in certificate.get("expected_option_ids", [])]
    evaluated_ids = [str(item).upper() for item in certificate.get("evaluated_option_ids", [])]
    searched_ids = sorted(searched)
    if certificate_expected != expected_ids:
        errors.append(f"expected_option_ids must be {expected_ids}, got {certificate_expected}.")
    if sorted(evaluated_ids) != sorted(expected_ids) or len(evaluated_ids) != len(expected_ids):
        errors.append("evaluated_option_ids must cover every visible option exactly once.")
    if searched_ids != sorted(expected_ids):
        errors.append(f"Only {searched_ids} were searched; all {sorted(expected_ids)} are required.")

    selected_option_id = str(certificate.get("selected_option_id") or "").upper()
    if selected_option_id != selected["option_id"]:
        errors.append("selected_option_id does not match the proposed select action.")

    expected_objective = _visible_objective(question)
    objective = certificate.get("objective") if isinstance(certificate.get("objective"), dict) else {}
    metric = str(objective.get("metric") or "").lower()
    direction = str(objective.get("direction") or "").lower()
    if expected_objective is None:
        errors.append("The visible primary objective could not be compiled.")
    elif {"metric": metric, "direction": direction} != expected_objective:
        errors.append(f"objective must be {expected_objective}, got {{'metric': '{metric}', 'direction': '{direction}'}}.")

    candidates = certificate.get("candidates")
    candidate_rows = candidates if isinstance(candidates, list) else []
    row_by_id = {
        str(row.get("option_id") or "").upper(): row
        for row in candidate_rows
        if isinstance(row, dict) and row.get("option_id")
    }
    if set(row_by_id) != set(expected_ids) or len(candidate_rows) != len(expected_ids):
        errors.append("candidates must contain exactly one grounded row for each visible option.")
    for observation in observations:
        option_id = observation["option_id"]
        row = row_by_id.get(option_id)
        if row is None:
            continue
        if not isinstance(row.get("eligible"), bool):
            errors.append(f"{option_id}.eligible must be boolean.")
        expected_value = observation.get(metric) if metric in {"price", "rating"} else None
        value = row.get("objective_value")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or expected_value is None:
            errors.append(f"{option_id}.objective_value is missing or uses an unsupported metric.")
        elif abs(float(value) - float(expected_value)) > 1e-9:
            errors.append(f"{option_id}.objective_value={value} does not match visible {metric}={expected_value}.")
        if not str(row.get("evidence") or "").strip():
            errors.append(f"{option_id} has no visible evidence summary.")

    has_explicit_compatibility_rules = "compatibility notes" in question.lower() or "**avoid:**" in question.lower()
    if not has_explicit_compatibility_rules:
        incorrectly_excluded = [option_id for option_id, row in row_by_id.items() if row.get("eligible") is not True]
        if incorrectly_excluded:
            errors.append(f"No explicit compatibility rule excludes candidates, but these were marked ineligible: {incorrectly_excluded}.")

    budget = certificate.get("budget") if isinstance(certificate.get("budget"), dict) else {}
    budget_limit = _visible_budget_limit(question)
    expected_after = round(running_total + float(selected["price"]), 2)
    numeric_budget = all(
        isinstance(budget.get(key), (int, float)) and not isinstance(budget.get(key), bool)
        for key in ["before", "after", "limit"]
    )
    if not numeric_budget:
        errors.append("budget.before, budget.after, and budget.limit must be numeric.")
    else:
        if abs(float(budget["before"]) - running_total) > 1e-9:
            errors.append("budget.before does not match the visible running total.")
        if abs(float(budget["after"]) - expected_after) > 1e-9:
            errors.append(f"budget.after must be {expected_after:.2f}.")
        if budget_limit is not None and abs(float(budget["limit"]) - budget_limit) > 1e-9:
            errors.append(f"budget.limit must be the visible limit {budget_limit:.2f}.")

    uncertainties = certificate.get("uncertainties")
    if not isinstance(uncertainties, list):
        errors.append("uncertainties must be a list.")
    elif any(str(item).strip() for item in uncertainties):
        errors.append("unresolved uncertainties must be repaired before selection.")
    return errors


def _with_grounding_receipt(certificate: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    grounded = json.loads(json.dumps(certificate, ensure_ascii=False))
    facts_by_id = {
        item["option_id"]: {
            "product_id": item["product_id"],
            "price": item["price"],
            "rating": item["rating"],
            "attributes": item["attributes"],
        }
        for item in observations
    }
    for row in grounded.get("candidates", []):
        if not isinstance(row, dict):
            continue
        option_id = str(row.get("option_id") or "").upper()
        if option_id in facts_by_id:
            row["observed_facts"] = facts_by_id[option_id]
    grounded["grounding_receipt"] = {
        "validated_against_environment": True,
        "validated_option_ids": [item["option_id"] for item in observations],
        "validated_fields": ["product_id", "price", "rating", "attributes"],
        "binding": "one candidate row to one searched observation",
    }
    return grounded


def _visible_objective(question: str) -> dict[str, str] | None:
    sections = []
    for label in ["Goal", "Preference"]:
        match = re.search(rf"\*\*{label}:\*\*\s*([^\n]+)", question, flags=re.I)
        if match:
            sections.append(match.group(1).lower())
    for text in sections:
        if "highest-priced" in text or "highest priced" in text or "highest price" in text:
            return {"metric": "price", "direction": "maximize"}
        if "lowest-priced" in text or "lowest priced" in text or "lowest price" in text:
            return {"metric": "price", "direction": "minimize"}
        if "highest-rated" in text or "highest rated" in text or "highest rating" in text:
            return {"metric": "rating", "direction": "maximize"}
        if "lowest-rated" in text or "lowest rated" in text or "lowest rating" in text:
            return {"metric": "rating", "direction": "minimize"}
    return None


def _visible_budget_limit(question: str) -> float | None:
    match = re.search(r"(?:not exceed|budget(?:\s+is|\s+of|:)?)\s*\$\s*(\d+(?:\.\d+)?)", question, flags=re.I)
    return float(match.group(1)) if match else None


def _search_memory_chunk(observation: dict[str, Any]) -> str:
    action = f"search[{observation['query']}]"
    return _structured_memory_event(
        {
            "event_type": "search_result",
            "action": action,
            "observation": _format_product_observation(observation),
            "entities": [
                {
                    "subject": f"option {observation['option_id']}",
                    "status": "candidate",
                    "confidence": 0.99,
                    "properties": {
                        "id": observation["product_id"],
                        "title": observation["title"],
                        "price": observation["price"],
                        "rating": observation["rating"],
                        "attributes": observation["attributes"],
                        "searched": True,
                    },
                }
            ],
        }
    )


def _selection_memory_chunk(
    action: str,
    observation: str,
    selected: dict[str, Any],
    certificate: dict[str, Any],
) -> str:
    return _structured_memory_event(
        {
            "event_type": "selection_committed",
            "action": action,
            "observation": observation,
            "is_final_action": True,
            "boundary_signal": True,
            "entities": [
                {
                    "subject": f"option {selected['option_id']}",
                    "status": "final",
                    "confidence": 0.99,
                    "properties": {
                        "selected": True,
                        "decision_certificate_status": "validated",
                        "decision_certificate": certificate,
                    },
                }
            ],
            "decision": {
                "subject": f"option {selected['option_id']}",
                "variable": "decision",
                "value": {
                    "action": action,
                    "option_id": selected["option_id"],
                    "product_id": selected["product_id"],
                    "certificate_status": "validated",
                },
            },
        }
    )


def _purchase_memory_chunk(action: str, selected: dict[str, Any], total_after: float) -> str:
    observation = (
        f"purchased option {selected['option_id']} with product id {selected['product_id']}; "
        f"title: {selected['title']}; price: {selected['price']:.2f}; rating: {selected['rating']:.1f}; "
        f"running_total: {total_after:.2f}; accepted."
    )
    return _structured_memory_event(
        {
            "event_type": "purchase_accepted",
            "action": action,
            "observation": observation,
            "reward": "controlled purchase accepted",
            "is_final_action": True,
            "is_success": True,
            "boundary_signal": True,
            "entities": [
                {
                    "subject": f"option {selected['option_id']}",
                    "status": "final",
                    "confidence": 0.99,
                    "properties": {
                        "purchase_status": "accepted",
                        "running_total": round(total_after, 2),
                    },
                }
            ],
            "decision": {
                "subject": f"option {selected['option_id']}",
                "variable": "decision",
                "value": {
                    "action": action,
                    "option_id": selected["option_id"],
                    "product_id": selected["product_id"],
                    "running_total": round(total_after, 2),
                    "status": "accepted",
                },
            },
        }
    )


def _error_memory_chunk(action: str, observation: str) -> str:
    return _structured_memory_event(
        {
            "event_type": "error",
            "action": action,
            "observation": observation,
            "feedback": "invalid or blocked action; repair and continue",
            "is_failure": True,
            "boundary_signal": True,
        }
    )


def _structured_memory_event(payload: dict[str, Any]) -> str:
    return "DART_EVENT_JSON:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _repair_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if match is None:
            raise RuntimeError(f"answer-agent response is not JSON: {text}")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"answer-agent JSON response is not an object: {text}")
    return parsed


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _is_local_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    return bool(re.search(r"//(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?", base_url))


def _global_prompt(question: str) -> str:
    match = re.search(r"\n-{10,}\nProduct\s+1:", question, flags=re.I)
    if match:
        return question[: match.start()].strip()
    product_match = re.search(r"Product\s+1:", question, flags=re.I)
    return question[: product_match.start()].strip() if product_match else question


def _parse_options(question: str) -> list[str]:
    match = re.search(r"\*\*Available Options:\*\*(.*)$", question, flags=re.S)
    if not match:
        return []
    options = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            options.append(stripped[2:].strip())
    return options


def _mock_product_observation(step: int, option_index: int, option: str) -> dict[str, Any]:
    product_id = f"DEMO-S{step}O{option_index}"
    price, rating = MOCK_FACTS.get((step, option_index), (7.99, 4.3))
    return {
        "option_id": f"S{step}O{option_index}",
        "product_id": product_id,
        "query": option,
        "title": option,
        "price": round(price, 2),
        "rating": round(min(rating, 5.0), 1),
        "attributes": _option_keywords(option),
    }


def _select_reference_replay(step: int, observations: list[dict[str, Any]]) -> dict[str, Any]:
    target_option_id = f"S{step}O{REFERENCE_OPTION_BY_STEP[step]}"
    for observation in observations:
        if observation["option_id"] == target_option_id:
            return observation
    return observations[0]


def _select_scripted(question: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
    preference_match = re.search(r"\*\*Preference:\*\*\s*(.+)", question)
    preference = preference_match.group(1).lower() if preference_match else ""
    goal_match = re.search(r"\*\*Goal:\*\*\s*(.+)", question)
    goal = goal_match.group(1).lower() if goal_match else ""
    rule = preference + " " + goal
    if "lowest-priced" in rule or "lowest price" in rule:
        return min(observations, key=lambda item: item["price"])
    if "highest-rated" in rule or "highest rated" in rule:
        return max(observations, key=lambda item: item["rating"])
    if "highest-priced" in rule or "highest price" in rule:
        return max(observations, key=lambda item: item["price"])
    return observations[0]


def _option_keywords(option: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z0-9#'.-]+", option):
        cleaned = token.strip(".,").lower()
        if len(cleaned) >= 4 and cleaned not in {"with", "from", "that", "this", "pack", "cake"}:
            tokens.append(cleaned)
    seen = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
    return seen[:8]


def _memory_context(prompt: str) -> str:
    start = prompt.find("<memory_context>")
    end = prompt.find("</memory_context>")
    if start == -1 or end == -1:
        return prompt
    return prompt[start + len("<memory_context>") : end].strip()


def _format_step_markdown(step_record: dict[str, Any]) -> list[str]:
    selected = step_record.get("scripted_selection")
    lines = [
        f"## Step {step_record['step']}",
        "",
        "### Visible Current Instruction",
        "",
        "```text",
        step_record["question"],
        "```",
        "",
        "### Optional Counterfactual Guard Probe",
        "",
        "```json",
        json.dumps(step_record["premature_guard"], ensure_ascii=False, indent=2),
        "```",
        "",
        "### Environment Candidate Catalog",
        "",
    ]
    lines.extend(
        f"- `{obs['option_id']}` `{obs['product_id']}` price=${obs['price']:.2f} rating={obs['rating']:.1f}: {obs['title']}"
        for obs in step_record["mock_observations"]
    )
    lines.extend(
        [
            "",
        ]
    )
    if step_record.get("api_action_trace"):
        lines.extend(["### Answer Agent API Trace", ""])
        for turn in step_record["api_action_trace"]:
            lines.extend(
                [
                    f"- turn {turn['turn']}: `{turn['normalized_action']}`",
                    f"  observation: {turn['environment_observation']}",
                ]
            )
            if turn.get("guard") is not None:
                lines.extend(
                    [
                        "  guard:",
                        "  ```json",
                        json.dumps(turn["guard"], ensure_ascii=False, indent=2),
                        "  ```",
                    ]
                )
        lines.append("")
    if step_record.get("selection_guard") is not None:
        lines.extend(
            [
                "### Selection Guard",
                "",
                "```json",
                json.dumps(step_record["selection_guard"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "### DART-CSL Packet After Search",
            "",
            "```text",
            step_record["memory_packet_after_search"],
            "```",
            "",
            "### Guard After Search",
            "",
            "```json",
            json.dumps(step_record["final_guard_after_search"], ensure_ascii=False, indent=2),
            "```",
            "",
            "### DART-CSL Packet After Purchase",
            "",
            "```text",
            step_record.get("memory_packet_after_purchase", ""),
            "```",
            "",
            "### Purchase",
            "",
            f"- selection mode: `{step_record['selection_mode']}`",
            f"- purchase status: `{step_record.get('purchase_status', 'unknown')}`",
            f"- reason: {step_record['selection_reason']}",
        ]
    )
    if selected:
        lines.append(
            f"- selected `{selected['option_id']}` / `{selected['product_id']}` price=${selected['price']:.2f} rating={selected['rating']:.1f}"
        )
    else:
        lines.append("- selected: none")
    lines.extend(
        [
            "",
            "### Offline Reference Answer",
            "",
            "This is for analysis only and was not fed to memory:",
            "",
            "```json",
            json.dumps(step_record["offline_reference_answer_do_not_feed_to_memory"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return lines


if __name__ == "__main__":
    main()
