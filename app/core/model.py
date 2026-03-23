from __future__ import annotations

import ast
import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class MockRecommender:
    """A cheap backend for smoke testing without downloading any model."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _clean_list_field(x: Any) -> List[str]:
        if pd.isna(x):
            return []
        x = str(x).strip()
        if not x:
            return []
        return [t.strip() for t in x.split("||") if t.strip()]

    @staticmethod
    def _parse_genres(genres_text: str) -> List[str]:
        if not genres_text or pd.isna(genres_text):
            return []
        return [g.strip() for g in str(genres_text).split("|") if g.strip()]

    @staticmethod
    def _parse_genre_pref(raw: Any) -> Dict[str, int]:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return {}
        text = str(raw).strip()
        if not text:
            return {}
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(text)
                if isinstance(obj, dict):
                    return {str(k): int(v) for k, v in obj.items()}
            except Exception:
                pass
        return {}

    def score_row(self, row: pd.Series) -> Tuple[float, int, str]:
        genres = self._parse_genres(row.get("movie_genres", ""))
        genre_pref = self._parse_genre_pref(row.get("user_top_positive_genres", ""))
        genre_score = sum(genre_pref.get(g, 0) for g in genres)
        popularity = float(row.get("movie_train_positive_count", 0)) / max(float(row.get("movie_train_count", 1)), 1.0)
        rating_signal = float(row.get("movie_train_avg_rating", 0.0)) / 5.0
        liked = self._clean_list_field(row.get("user_liked_movie_titles", ""))
        disliked = self._clean_list_field(row.get("user_disliked_movie_titles", ""))
        history_balance = 0.5 if len(liked) >= len(disliked) else 0.0

        # bounded score in [0,1]
        score = 0.45 * min(genre_score / 10.0, 1.0) + 0.35 * popularity + 0.15 * rating_signal + 0.05 * history_balance
        score = max(0.0, min(1.0, score))
        pred = int(score >= 0.5)
        raw = f"mock_score={score:.4f}; pred={pred}"
        return score, pred, raw


class HFRecommender:
    def __init__(self, model_name: str, use_4bit_if_available: bool = True) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_4bit = bool(torch.cuda.is_available() and use_4bit_if_available)
        if use_4bit:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

    @staticmethod
    def _clean_list_field(x: Any) -> List[str]:
        if pd.isna(x):
            return []
        x = str(x).strip()
        if not x:
            return []
        return [t.strip() for t in x.split("||") if t.strip()]

    def build_prompt_from_row(self, row: pd.Series) -> str:
        liked = self._clean_list_field(row.get("user_liked_movie_titles", ""))
        disliked = self._clean_list_field(row.get("user_disliked_movie_titles", ""))
        genres = str(row.get("movie_genres", "")).strip()
        liked_text = ", ".join(liked[:8]) if liked else "None"
        disliked_text = ", ".join(disliked[:8]) if disliked else "None"

        return f"""
User profile:
- age: {row.get('user_age', 'unknown')}
- gender: {row.get('user_gender', 'unknown')}
- occupation: {row.get('user_occupation', 'unknown')}

User history from training data:
- liked movies: {liked_text}
- disliked movies: {disliked_text}

Candidate movie:
- title: {row.get('movie_title', 'unknown')}
- genres: {genres if genres else 'unknown'}
- train positivity ratio: {row.get('movie_train_positive_count', 0)}/{max(row.get('movie_train_count', 0), 1)}
- train average rating: {row.get('movie_train_avg_rating', 0.0)}

Task:
Estimate recommendation confidence for this user and this movie.

Rules:
- Return JSON only.
- Format exactly as {{"score": <number between 0 and 1>, "pred": <0 or 1>}}.
- Do not add explanations.
""".strip()

    @staticmethod
    def _extract_json(raw_text: str) -> Tuple[float, int]:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match:
            snippet = match.group(0)
            try:
                obj = json.loads(snippet)
                score = float(obj.get("score", obj.get("prob", 0.0)))
                pred = int(obj.get("pred", int(score >= 0.5)))
                return max(0.0, min(1.0, score)), int(pred)
            except Exception:
                pass
        matches = re.findall(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", raw_text)
        score = float(matches[0]) if matches else 0.0
        pred_matches = re.findall(r"\b[01]\b", raw_text)
        pred = int(pred_matches[-1]) if pred_matches else int(score >= 0.5)
        return max(0.0, min(1.0, score)), pred

    def score_row(self, row: pd.Series) -> Tuple[float, int, str]:
        prompt = self.build_prompt_from_row(row)
        messages = [
            {"role": "system", "content": "You are a precise recommendation scoring model. Return JSON only."},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"System: You are a precise recommendation scoring model. Return JSON only.\nUser: {prompt}\nAssistant:"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        score, pred = self._extract_json(decoded)
        return score, pred, decoded


@lru_cache(maxsize=4)
def get_recommender(backend: str, model_name: str, use_4bit_if_available: bool = True):
    backend = backend.lower()
    if backend == "mock":
        return MockRecommender()
    if backend == "hf":
        return HFRecommender(model_name=model_name, use_4bit_if_available=use_4bit_if_available)
    raise ValueError(f"Unsupported backend: {backend}")
