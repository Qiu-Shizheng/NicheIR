"""Run after installing nicheir: python examples/inference.py"""
from nicheir import HalfPlane,load_pretrained,prepare_cells
from nicheir.example import synthetic_cells

cells,query_ids=synthetic_cells(seed=42)
batch=prepare_cells(cells,query_ids,HalfPlane(cut=0.0))
predictor=load_pretrained()
result=predictor.predict(batch.inputs)
print(predictor.class_names)
for identifier,composition in zip(query_ids,result.composition[result.valid]):
    print(identifier,composition)
