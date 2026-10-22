from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.metrics import auc
import numpy as np
from scipy.optimize import linear_sum_assignment
from difflib import SequenceMatcher

def ensure_nltk_resources():
    """Resolve preprocessing resources on demand, never during module import."""
    for package, resource in (
        ('punkt', 'tokenizers/punkt'), ('punkt_tab', 'tokenizers/punkt_tab'),
        ('wordnet', 'corpora/wordnet'), ('omw-1.4', 'corpora/omw-1.4'),
        ('stopwords', 'corpora/stopwords'),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.data.find(resource + '.zip')
            except LookupError:
                if not nltk.download(package, quiet=True):
                    raise RuntimeError(f"Cannot install required NLTK resource: {package}")


def aggregate_soft_scores(results):
    """Original reference-count-weighted P/R, then harmonic F1 (not macro F1)."""
    results = list(results)
    weight = sum(row['reference_goal_count'] for row in results)
    precision = sum(row['precision'] * row['reference_goal_count'] for row in results) / weight if weight else 0.0
    recall = sum(row['recall'] * row['reference_goal_count'] for row in results) / weight if weight else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'reference_goal_count': weight, 'dataset_count': len(results)}



def weighted_metric_table(frame, group_columns):
    """Reference-weighted P/R and harmonic F1 per cohort, distinct from macros.

    Repeated observations of one dataset are averaged before weighting so a
    dataset does not gain weight from extra repetitions. Reference counts must
    remain identical within that dataset/cohort. Missing results are excluded,
    not replaced by zero; the dataset count describes the actual cohort.
    """
    import pandas as pd
    columns = [*group_columns, 'precision', 'recall', 'f1', 'reference_goal_count',
               'dataset_count', 'aggregation']
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for keys, group in frame.groupby(list(group_columns), dropna=False):
        identity = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        if (group.groupby('dataset')['reference_goal_count'].nunique() > 1).any():
            raise ValueError('Reference counts changed within an evaluation cohort')
        datasets = group.groupby('dataset', as_index=False).agg(
            precision=('precision', 'mean'), recall=('recall', 'mean'),
            reference_goal_count=('reference_goal_count', 'first'))
        result = aggregate_soft_scores(datasets.to_dict('records'))
        rows.append({**identity, **result, 'aggregation': 'reference_weighted_precision_recall_harmonic_f1'})
    return pd.DataFrame(rows, columns=columns)


def goal_descriptions(container, field):
    """Use generated descriptions, as exported by the original notebook."""
    items = container.get(field) if isinstance(container, dict) else container
    if not isinstance(items, list):
        raise ValueError(f"Missing goal collection: {field}")
    texts = []
    for item in items:
        text = item if isinstance(item, str) else item.get('description') if isinstance(item, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Evaluation requires a nonblank goal description")
        texts.append(text)
    return texts


class GoalEvaluator:
    def __init__(self, model_name="bert-base-uncased", preprocess=True):
        self.summary = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.preprocess = preprocess
        if preprocess:
            ensure_nltk_resources()
            self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        tokens = word_tokenize(text)
        filtered_tokens = [t for t in tokens if t not in self.stop_words and t.isalnum()]
        stemmed_tokens = [self.stemmer.stem(token) for token in filtered_tokens]
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in stemmed_tokens]
        return " ".join(lemmatized_tokens)

    def encode_texts(self, texts):
        if not texts:
            return torch.empty((0, self.model.config.hidden_size), device=self.device)
        if self.preprocess:
            texts = [self.preprocess_text(text) for text in texts]

        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Mean Pooling
        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def compute_similarity(self, generated_goals, reference_goals):
        if not generated_goals or not reference_goals:
            return np.zeros((len(generated_goals), len(reference_goals)), dtype=float)
        gen_embeddings = self.encode_texts(generated_goals)
        ref_embeddings = self.encode_texts(reference_goals)

        # Cosine similarity matrix
        gen_norm = torch.nn.functional.normalize(gen_embeddings, p=2, dim=1)
        ref_norm = torch.nn.functional.normalize(ref_embeddings, p=2, dim=1)
        sim_matrix = torch.mm(gen_norm, ref_norm.transpose(0, 1))

        return sim_matrix.cpu().numpy()

    @staticmethod
    def string_similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    def calculate_soft_f1(self, preds, refs):
        sim_matrix = self.compute_similarity(preds, refs)

        if preds and refs:
            row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        else:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)
        total_similarity = sim_matrix[row_ind, col_ind].sum()

        precision = total_similarity / len(preds) if preds else 0.0
        recall = total_similarity / len(refs) if refs else 0.0

        f1 = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

        rate_table = []
        optimized_similarities = sim_matrix[row_ind, col_ind]
        for tau in np.linspace(0, 1, 101):
            tp = np.sum(optimized_similarities >= tau)
            p = tp / len(preds) if len(preds) > 0 else 0
            r = tp / len(refs) if len(refs) > 0 else 0
            rate_table.append({"threshold": tau, "precision": p, "recall": r})

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "f1": f1,
            "total_similarity": float(total_similarity),
            "matched_goal_count": len(row_ind),
            "matches": [{"generated_index": int(i), "reference_index": int(j),
                         "generated_text": preds[i], "reference_text": refs[j],
                         "cosine": float(sim_matrix[i, j]), "selected": True}
                        for i, j in zip(row_ind, col_ind)],
            "similarities": sim_matrix,
            "rate_table": rate_table
        }

    def score_goals(self, preds, refs):
        """Tabular notebook output; every assigned similarity contributes, without a threshold."""
        result = self.calculate_soft_f1(preds, refs)
        return {key: result[key] for key in (
            'precision', 'recall', 'f1', 'total_similarity', 'matched_goal_count', 'matches')}

    def evaluate_datasets(self, generated, references):
        """Score a matched dataset cohort and aggregate using reference counts."""
        if set(generated) != set(references):
            raise ValueError("Generated and reference datasets must have identical keys")
        rows = [{"dataset": name, "reference_goal_count": len(references[name]),
                 **self.score_goals(preds, references[name])} for name, preds in generated.items()]
        return rows, aggregate_soft_scores(rows)

    def print_prec_rec_curve(self, generated_goals_dict: dict[str, list[str]],
                             reference_goals_dict: dict[str, list[str]], hide_prec_rec=False,
                             save_to_file=False, output_file="output.txt", threshold=0.9, rate_table=False):

        with open(output_file, "w") as f:
            f.write("dataset name; recall; precision; f1_score;\n")

        weighted_recalls = np.array([0.0] * 101)
        weighted_precisions = np.array([0.0] * 101)

        weighted_r = 0
        weighted_p = 0

        for dataset_name, generated_goals in generated_goals_dict.items():
            weight = len(reference_goals_dict[dataset_name])
            reference_goals = reference_goals_dict[dataset_name]
            results = self.calculate_soft_f1(generated_goals, reference_goals)

            weighted_p += (results["precision"] * weight)
            weighted_r += (results["recall"] * weight)

            summary_str = f"{dataset_name}; {results['recall']}; {results['precision']}; {results['f1_score']};"
            self.summary.append(summary_str)

            if save_to_file:
                with open(output_file, "a") as f:
                    f.write(summary_str + "\n")

            if rate_table:
                recalls = np.array([point['recall'] for point in results["rate_table"]])
                precisions = np.array([point['precision'] for point in results["rate_table"]])
                weighted_recalls += (recalls * weight)
                weighted_precisions += (precisions * weight)

        reference_count = sum(len(l) for l in reference_goals_dict.values())
        normalization_term = 1 / reference_count if reference_count else 0.0

        if rate_table:
            weighted_recalls *= normalization_term
            weighted_precisions *= normalization_term

        weighted_r *= normalization_term
        weighted_p *= normalization_term
        weighted_f1 = 2 * weighted_p * weighted_r / (weighted_p + weighted_r) if weighted_r + weighted_p else 0
        summary_str = f"Precision: {weighted_p}; Recall: {weighted_r}; F1-score: {weighted_f1};"

        if save_to_file:
            with open(output_file, "a") as f:
                f.write("Weighted averages; "+ summary_str + "\n")

        auc_prec_rec = auc(weighted_recalls[::-1], weighted_precisions[::-1])

        if not hide_prec_rec:
            plt.figure(figsize=(8, 6))
            plt.plot(weighted_recalls, weighted_precisions, marker='o', linestyle='-', color='b',
                     label="Precision-Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend()
            plt.grid()
            plt.show()

        return auc_prec_rec, summary_str

    def print_best_f1(self, generated_goals_dict: dict[str, list[str]], reference_goals_dict: dict[str, list[str]],
                      save_to_file=False, output_file="output.txt", ):

        best_f1 = 0
        best_summary = ""

        for th in range(1, 101):
            _, summary_str, p, r, f1 = self.print_prec_rec_curve(generated_goals_dict, reference_goals_dict,
                                                                 threshold=th, hide_prec_rec=True)

            if f1 > best_f1:
                best_f1 = f1
                best_summary = summary_str

        print(best_summary)
        return best_summary
