import argparse
import logging
import os
from datetime import datetime
from models import load_model
import json
from tqdm import tqdm
import torch
from typing import Optional


class Certainty_classifier:
    def __init__(self, llm, problem, threshold):
        self.llm = llm
        self.problem = problem
        self.threshold = threshold

        self.spp_nll: Optional[float] = None
        self.spp_ppl: Optional[float] = None


    # ----- Self-Perplexity 계산 ----- #
    @torch.no_grad()
    def _compute_spp(self):
        model = self.llm.model
        tokenizer = self.llm.tokenizer
        device = next(model.parameters()).device

        enc = tokenizer(self.problem, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

        if input_ids.shape[1] < 2:
            self.spp_nll, self.spp_ppl = 0.0, 1.0
            return

        out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        mask = attn_mask[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        nll = -(token_lp * mask).sum() / (mask.sum() + 1e-9)

        ppl = float(torch.exp(nll))
        self.spp_nll = float(nll)
        self.spp_ppl = ppl


    def __call__(self):
        self._compute_spp()

        # Self-Perplexity를 기반으로 한 Certain / Uncertain 여부 판단
        if self.spp_ppl >= self.threshold:
            return "Uncertain"
        else:
            return "Certain"


        
def setup_logging(log_file, log_level):
    logger = logging.getLogger("__name__")
    logger.setLevel(getattr(logging, log_level))
    handler = logging.FileHandler(log_file)
    handler.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


model_version = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "deepseek-v2-16b": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3"
}


def main():
    parser = argparse.ArgumentParser(description="Certain/Uncertain Classifier")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="qwen3-14b")
    parser.add_argument("--dataset_name", type=str, default="HotpotQA", help="HotpotQA, StrategyQA, Musique, MATH500, T4D")
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--force_eager_attn", action="store_true")
    parser.add_argument("--disable_sdp_flash", action="store_true")
    parser.add_argument("--threshold", type=float, default=50.0)

    args = parser.parse_args()
    
    log_dir = os.path.join(args.log_dir, args.dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{args.model_name}_{timestamp}.log")

    logger = setup_logging(log_file, args.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Classifying using {args.model_name} for {args.dataset_name}")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Logs saved to {os.path.abspath(log_file)}")


    output_dir = os.path.join(args.output_dir, args.dataset_name)
    output_path = os.path.join(output_dir, f"{args.model_name}.json")
    os.makedirs(output_dir, exist_ok=True)

    args.model_id = model_version[args.model_name]

    model = load_model(args)


    data_path = os.path.join(args.data_dir, args.dataset_name, f"{args.dataset_name}.json")
    with open(data_path, 'r') as f:
        dataset = json.load(f)

    res = []
    for idx, data in tqdm(enumerate(dataset)):
        runner = Certainty_classifier(model, data['question'], threshold=args.threshold)
        result = runner()
        tmp = {
            "id": data['id'],
            "question": data['question'],
            "answer": data.get('answer'),
            "spp_nll": runner.spp_nll,
            "spp_ppl": runner.spp_ppl,
            "cls_result": result
        }
        res.append(tmp)

        if idx % args.save_interval == 0:
            with open(output_path, 'w') as f:
                json.dump(res, f, indent=4)
            logger.info(f"Results saved to {os.path.abspath(output_path)}")

    with open(output_path, 'w') as f:
        json.dump(res, f, indent=4)
    logger.info(f"All results saved to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()


