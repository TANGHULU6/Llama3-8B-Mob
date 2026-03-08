import argparse
import json
import os
import re
from typing import List, Optional, Tuple

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Next POI prediction with Acc@k / MRR using beam search."
    )
    parser.add_argument(
        "--city",
        type=str,
        default="nyc",
        help="City tag used to select dataset and adapter, e.g. nyc/ca/tky.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help=(
            "Base model path or HF id. If omitted, will auto-detect from the LoRA adapter's "
            "adapter_config.json when available."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional directory to save eval summary and details.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=10,
        help="Beam size for beam search.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for debugging).",
    )
    return parser.parse_args()


def resolve_base_model(base_model: Optional[str], adapter_path: str) -> str:
    if isinstance(base_model, str) and base_model.strip():
        return base_model.strip()

    adapter_cfg = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        try:
            with open(adapter_cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            inferred = cfg.get("base_model_name_or_path")
            if isinstance(inferred, str) and inferred.strip():
                return inferred.strip()
        except Exception:
            pass

    return "unsloth/llama-3-8b-Instruct-bnb-4bit"


def extract_poi_id(text: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None
    match = re.search(r"\bpoi\d+\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(0).lower()
    token = text.split()[0].strip().strip(",.。;；:：'\"()[]{}<>")
    return token or None


def tokenizer_expects_from_value_format(tokenizer) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        return False
    return ("message['from']" in template or 'message["from"]' in template) and (
        "message['value']" in template or 'message["value"]' in template
    )


def to_from_value_messages(messages: List[dict]) -> List[dict]:
    out = []
    for m in messages:
        role = (m.get("role") or m.get("from") or "").strip()
        content = m.get("content") if "content" in m else m.get("value")
        if content is None:
            continue
        from_role = role
        if role == "user":
            from_role = "human"
        elif role == "assistant":
            from_role = "gpt"
        out.append({"from": from_role, "value": content})
    return out


def to_role_content_messages(messages: List[dict]) -> List[dict]:
    out = []
    for m in messages:
        role = (m.get("role") or m.get("from") or "").strip()
        content = m.get("content") if "content" in m else m.get("value")
        if not role or content is None:
            continue
        normalized_role = role
        if role == "human":
            normalized_role = "user"
        elif role == "gpt":
            normalized_role = "assistant"
        out.append({"role": normalized_role, "content": content})
    return out


def load_test_data(data_path: str) -> List[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for item in data:
        messages = item.get("messages", [])
        if len(messages) < 2:
            continue
        user_msg = next(
            (m for m in messages if (m.get("role") or m.get("from")) in {"user", "human"}),
            None,
        )
        assistant_msg = next(
            (
                m
                for m in messages
                if (m.get("role") or m.get("from")) in {"assistant", "gpt"}
            ),
            None,
        )
        if not user_msg:
            continue
        target_poi_id = None
        if assistant_msg:
            content = (assistant_msg.get("content") or assistant_msg.get("value") or "").strip()
            target_poi_id = extract_poi_id(content)
        out.append(
            {
                "messages": messages,
                "ground_truth": target_poi_id,
            }
        )
    return out


def ranked_predictions_by_confidence(
    pred_poi_with_score: List[Tuple[Optional[str], float]]
) -> List[str]:
    best_score = {}
    for poi, score in pred_poi_with_score:
        if poi is None:
            continue
        if poi not in best_score or score > best_score[poi]:
            best_score[poi] = score
    return sorted(best_score.keys(), key=lambda p: -best_score[p])


def main():
    args = parse_args()

    city_tag = (args.city or "nyc").lower()
    if city_tag not in {"nyc", "ca", "tky"}:
        raise ValueError(f"Unsupported city '{city_tag}', expected one of nyc/ca/tky.")

    adapter_path = os.path.join(f"outputs-poi-{city_tag}", "lora_model")
    test_data_path = f"datasets/nextpoi/{city_tag}_mobility_test.json"
    base_model = resolve_base_model(args.base_model, adapter_path)

    print(f"\n{'='*60}")
    print("Evaluate Next POI prediction (Acc@1/5/10, MRR) with beam search")
    print(f"Base model: {base_model}")
    print(f"Adapter:    {adapter_path}")
    print(f"Test data:  {test_data_path}")
    print(f"num_beams:  {args.num_beams}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print(f"{'='*60}\n")

    tokenizer_source = adapter_path if os.path.isdir(adapter_path) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_cuda = torch.cuda.is_available()
    torch_dtype = (
        (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        if use_cuda
        else torch.float32
    )
    device_map = {"": 0} if use_cuda else None
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        try:
            model = model.merge_and_unload()
        except Exception as e:
            print(
                f"Warning: merge_and_unload failed ({type(e).__name__}: {e}); "
                "continuing with PEFT model without merging.",
                flush=True,
            )
    model.eval()

    samples = load_test_data(test_data_path)
    if args.max_samples:
        samples = samples[: args.max_samples]
    n = len(samples)

    acc1_sum = acc5_sum = acc10_sum = mrr_sum = 0.0
    valid = 0
    results_detail = [] if args.output_dir else None

    pbar = tqdm(samples, total=n, desc="Evaluating", unit="sample")
    for i, sample in enumerate(pbar, start=1):
        messages = sample["messages"]
        input_messages = [
            m
            for m in messages
            if (m.get("role") or m.get("from")) not in {"assistant", "gpt"}
        ]
        gt = sample["ground_truth"]
        if gt is None:
            if results_detail is not None:
                results_detail.append(
                    {
                        "ground_truth": None,
                        "rank": None,
                        "acc1": 0,
                        "acc5": 0,
                        "acc10": 0,
                        "mrr": 0.0,
                    }
                )
            continue
        try:
            chat_messages = (
                to_from_value_messages(input_messages)
                if tokenizer_expects_from_value_format(tokenizer)
                else to_role_content_messages(input_messages)
            )
            input_ids = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=False,
            )
            if input_ids.shape[1] > 8192:
                if results_detail is not None:
                    results_detail.append(
                        {
                            "ground_truth": gt,
                            "rank": None,
                            "acc1": 0,
                            "acc5": 0,
                            "acc10": 0,
                            "mrr": 0.0,
                        }
                    )
                continue
            input_device = model.device
            input_ids = input_ids.to(input_device)
            attention_mask = torch.ones_like(
                input_ids, dtype=torch.long, device=input_device
            )
            input_len = input_ids.shape[1]
            num_return = min(args.num_beams, 10)
            pred_poi_with_score: List[Tuple[Optional[str], float]] = []
            with torch.inference_mode():
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=args.num_beams,
                    num_return_sequences=num_return,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
                seqs = out.sequences
                scores = out.sequences_scores
                for j in range(seqs.shape[0]):
                    new_tokens = seqs[j][input_len:]
                    text = tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    ).strip()
                    poi = extract_poi_id(text)
                    sc = float(scores[j].item()) if scores is not None else 0.0
                    pred_poi_with_score.append((poi, sc))
            ranked = ranked_predictions_by_confidence(pred_poi_with_score)
            try:
                rank = ranked.index(gt) + 1
            except ValueError:
                rank = 0
            acc1 = 1.0 if rank == 1 else 0.0
            acc5 = 1.0 if rank and rank <= 5 else 0.0
            acc10 = 1.0 if rank and rank <= 10 else 0.0
            mrr = 1.0 / rank if rank else 0.0
            acc1_sum += acc1
            acc5_sum += acc5
            acc10_sum += acc10
            mrr_sum += mrr
            valid += 1
            if results_detail is not None:
                results_detail.append(
                    {
                        "ground_truth": gt,
                        "rank": rank,
                        "acc1": acc1,
                        "acc5": acc5,
                        "acc10": acc10,
                        "mrr": mrr,
                    }
                )
        except Exception as e:
            if results_detail is not None:
                results_detail.append(
                    {
                        "ground_truth": gt,
                        "error": str(e),
                        "rank": None,
                        "acc1": 0,
                        "acc5": 0,
                        "acc10": 0,
                        "mrr": 0.0,
                    }
                )
        if valid and i % 50 == 0:
            pbar.set_postfix(
                acc1=f"{acc1_sum/valid:.4f}",
                mrr=f"{mrr_sum/valid:.4f}",
                valid=valid,
            )

    if valid == 0:
        print("No valid samples.")
        return

    acc1 = acc1_sum / valid
    acc5 = acc5_sum / valid
    acc10 = acc10_sum / valid
    mrr = mrr_sum / valid

    print(f"\nValid samples: {valid}")
    print(f"Acc@1  = {acc1:.4f}")
    print(f"Acc@5  = {acc5:.4f}")
    print(f"Acc@10 = {acc10:.4f}")
    print(f"MRR    = {mrr:.4f}")

    summary = {
        "adapter_path": adapter_path,
        "test_data": test_data_path,
        "num_beams": args.num_beams,
        "valid_samples": valid,
        "Acc@1": round(acc1, 4),
        "Acc@5": round(acc5, 4),
        "Acc@10": round(acc10, 4),
        "MRR": round(mrr, 4),
    }
    if args.output_dir:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        with open(
            os.path.join(out_dir, "eval_topk_mrr_summary.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(
            os.path.join(out_dir, "eval_topk_mrr_details.jsonl"),
            "w",
            encoding="utf-8",
        ) as f:
            for r in results_detail:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nResults written to: {out_dir}")

    print("\n--- Single line (for table) ---")
    print(
        f"Acc@1={acc1:.4f}\tAcc@5={acc5:.4f}\tAcc@10={acc10:.4f}\tMRR={mrr:.4f}"
    )


if __name__ == "__main__":
    main()
