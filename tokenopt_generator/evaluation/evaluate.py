import os
from typing import List, Dict
from output_dir_reader import create_eval_row_from_configuration
from metrics_eval import EvalRow,MetricsEvaluator,PyiqaMetric,PyiqaFIDMetric,AggregatedScores
from pathlib import Path
import json
from plots_from_scores import generate_plots_from_scores



def evaluate(rows:List[EvalRow],N:int,fid_T:int=200):
    qualiCLIP = PyiqaMetric("qualiclip", "cuda")
    nima = PyiqaMetric("nima", "cuda")
    fid = PyiqaFIDMetric("cuda")
    evaluator = MetricsEvaluator(qualiCLIP, nima, fid)
    return evaluator.evaluate_all(rows,N,fid_T)

def _annidate_dict(dictionary:Dict) -> Dict:
    nested: Dict[str, Dict[str, float]] = {}
    for (k1,k2),v in dictionary.items():
        nested.setdefault(k1,{})[k2]=v
    return nested

def append_score(metric_name:str,value,path="scores.json"):
    with open(path,"r", encoding="utf8") as f:
        data=json.load(f)
    data[metric_name].append(value)
    with open(path,"w", encoding="utf8") as f:
        json.dump(data,f,indent=2)

def initialize_json(path:str,configs:list[str]):
    status=[False for _ in range(len(configs))]
    dict_json = {
        "QualiCLIP": [],
        "Nima": [],
        "FID": [],
    }
    with open(f"scores.json", 'w') as file:
        json.dump(dict_json, file, indent=2)


def merge_dict(dict1,dict2):
    d=dict1.copy()
    d.update(dict2)
    return d


def main():
    base_dir="C:/Users/nicco/OneDrive/UNIFI/Tesi/"
    configuration_dir="outputs/inpainting/"
    configs=os.listdir(base_dir+configuration_dir)
    json_file = "scores.json"
    if not os.path.exists("scores.json"):
        initialize_json(json_file,configs)

    scores=AggregatedScores(
        qualiclip={},
        nima={},
        fid_mean={},
        fid_std={},
        qualiclip_per_input={},
        nima_per_input={},
        fid_per_iteration={},
        qualiclip_per_output={},
        nima_per_output={},
    )

    for config in configs:
        qualiCLIP = PyiqaMetric("qualiclip", "cuda")
        nima = PyiqaMetric("nima", "cuda")
        fid = PyiqaFIDMetric("cuda")
        evaluator = MetricsEvaluator(qualiCLIP, nima, fid)

        rows=create_eval_row_from_configuration(Path(base_dir),Path(configuration_dir+"/"+config))

        qauliclip_scores=evaluator.compute_qualiclip(rows)
        qualiclip={
            "config_evaluation":qauliclip_scores[0],
            "input_averaged_evaluation":_annidate_dict(qauliclip_scores[1]),
            "input_normal_evaluation":_annidate_dict(qauliclip_scores[2]),
        }
        append_score("QualiCLIP",qualiclip,json_file)

        nima_scores=evaluator.compute_nima(rows)
        nima={
            "config_evaluation":nima_scores[0],
            "input_averaged_evaluation":_annidate_dict(nima_scores[1]),
            "input_normal_evaluation":_annidate_dict(nima_scores[2]),
        }
        append_score("Nima",nima,json_file)

        fid_scores=evaluator.compute_fid(rows,14,70)
        fid={
            "config_fid_mean":fid_scores[0],
            "config_fid_std":fid_scores[1],
            "fid_per_iteration":fid_scores[2],
        }
        append_score("FID",fid,json_file)

        scores.qualiclip=merge_dict(scores.qualiclip,qauliclip_scores[0])
        scores.qualiclip_per_input=merge_dict(scores.qualiclip_per_input,qauliclip_scores[1])
        scores.qualiclip_per_output=merge_dict(scores.qualiclip_per_output,qauliclip_scores[2])

        scores.nima=merge_dict(scores.nima,nima_scores[0])
        scores.nima_per_input=merge_dict(scores.nima_per_input,nima_scores[1])
        scores.nima_per_output=merge_dict(scores.nima_per_output,nima_scores[2])

        scores.fid_mean=merge_dict(scores.fid_mean,fid_scores[0])
        scores.fid_std=merge_dict(scores.fid_std,fid_scores[1])
        scores.fid_per_iteration=merge_dict(scores.fid_per_iteration,fid_scores[2])

    generate_plots_from_scores(scores, "plots_images")


if __name__ == "__main__":
    main()