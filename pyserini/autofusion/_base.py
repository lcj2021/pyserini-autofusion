#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from enum import Enum
from pyserini.trectools import AggregationMethod, RescoreMethod, TrecRun
from typing import List
import pandas as pd
from tqdm import tqdm


class FusionMethod(Enum):
    RRF = 'rrf'
    INTERPOLATION = 'interpolation'
    AVERAGE = 'average'
    CUSTOM = 'custom'
    HALFTOPS = 'halftops'


def reciprocal_rank_fusion(runs: List[TrecRun], rrf_k: int = 60, depth: int = None, nq: float=-1, k: int = None):
    """Perform reciprocal rank fusion on a list of ``TrecRun`` objects. Implementation follows Cormack et al.
    (SIGIR 2009) paper titled "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods."

    Parameters
    ----------
    runs : List[TrecRun]
        List of ``TrecRun`` objects.
    rrf_k : int
        Parameter to avoid vanishing importance of lower-ranked documents. Note that this is different from the *k* in
        top *k* retrieval; set to 60 by default, per Cormack et al.
    depth : int
        Maximum number of results from each input run to consider. Set to ``None`` by default, which indicates that
        the complete list of results is considered.
    k : int
        Length of final results list.  Set to ``None`` by default, which indicates that the union of all input documents
        are ranked.

    Returns
    -------
    TrecRun
        Output ``TrecRun`` that combines input runs via reciprocal rank fusion.
    """

    num_queries = round(nq)
    if nq < 0:
      num_queries = round(len(runs[0].run_data) / 1000)
    elif nq < 1:
      num_queries = round(len(runs[0].run_data) / 1000 * nq)
    elif nq == 1:
      num_queries = 1

    num_queries *= 1000
    for run in runs:
        run.run_data = run.run_data[:num_queries]

    # TODO: Add option to *not* clone runs, thus making the method destructive, but also more efficient.
    rrf_runs = [run.clone().rescore(method=RescoreMethod.RRF, rrf_k=rrf_k) for run in runs]
    return TrecRun.merge(rrf_runs, AggregationMethod.SUM, depth=depth, k=k)


def reciprocal_rank_fusion_custom(runs: List[TrecRun], path: str, rrf_k: int = 60, depth: int = None, nq: float=-1, k: int = None):
    print('reciprocal rank custom')
    
    num_queries = round(nq)
    if nq < 0:
      num_queries = round(len(runs[0].run_data) / 1000)
    elif nq < 1:
      num_queries = round(len(runs[0].run_data) / 1000 * nq)
    elif nq == 1:
      num_queries = 1

    num_queries *= 1000
    for run in runs:
        run.run_data = run.run_data[:num_queries]

    # print(runs[1].run_data.loc[0 * 1000: 0 * 1000 + 100 - 1])
    # print(runs[0].run_data.loc[1 * 1000: 1 * 1000 + 100 - 1])

    with open(path, 'r') as file:
        alphas = file.readlines()
        alphas = [float(alpha) for alpha in alphas]
    
    print(f'{len(set(alphas))} alpha: {set(alphas)}')

    # TODO: Add option to *not* clone runs, thus making the method destructive, but also more efficient.
    rrf_runs = [run.clone().rescore(method=RescoreMethod.RRF, rrf_k=rrf_k) for run in runs]
    rrf_runs_merged = TrecRun.merge(rrf_runs, AggregationMethod.SUM, depth=depth, k=k)

    print('Merge fininshed')

    num_replaced = 0
    for idx, alpha in enumerate(alphas):
        if alpha == 0.5:  continue
        rrf_runs_merged.run_data.iloc[idx * 100: idx * 100 + 100] = runs[1].run_data.iloc[idx * 1000: idx * 1000 + 100].values
        num_replaced += 1
    print(f'{num_replaced} replaced to BM25')
    # print(f'idx: {idx}')

    print(rrf_runs_merged.run_data.iloc[100 - 1: 102])
    print(rrf_runs_merged.run_data.iloc[200 - 1: 202])
    
    return rrf_runs_merged

def half_tops_custom(runs: List[TrecRun], path: str = None, rrf_k: int = 60, depth: int = None, nq: float=-1, k: int = None):
    print('Half tops custom')

    num_queries = round(nq)
    if nq < 0:
      num_queries = round(len(runs[0].run_data) / 1000)
    elif nq < 1:
      num_queries = round(len(runs[0].run_data) / 1000 * nq)
    elif nq == 1:
      num_queries = 1

    num_queries *= 1000
    for run in runs:
        run.run_data = run.run_data[:num_queries]

    half_tops = TrecRun()
    for idx in tqdm(range(int(num_queries / 1000))):
        half_tops.run_data = pd.concat([half_tops.run_data, runs[0].run_data.iloc[idx * 1000: idx * 1000 + 50]])
        half_tops.run_data = pd.concat([half_tops.run_data, runs[1].run_data.iloc[idx * 1000: idx * 1000 + 50]])


    print('Merge fininshed')

    print(half_tops.run_data.iloc[0: 102])
    
    return half_tops



def interpolation(runs: List[TrecRun], alpha: float = 0.5, depth: int = None, nq: float=-1, k: int = None):
    """Perform fusion by interpolation on a list of exactly two ``TrecRun`` objects.
    new_score = first_run_score * alpha + (1 - alpha) * second_run_score.

    Parameters
    ----------
    runs : List[TrecRun]
        List of ``TrecRun`` objects. Exactly two runs.
    alpha : int
        Parameter alpha will be applied on the first run and (1 - alpha) will be applied on the second run.
    depth : int
        Maximum number of results from each input run to consider. Set to ``None`` by default, which indicates that
        the complete list of results is considered.
    k : int
        Length of final results list.  Set to ``None`` by default, which indicates that the union of all input documents
        are ranked.

    Returns
    -------
    TrecRun
        Output ``TrecRun`` that combines input runs via interpolation.
    """

    if len(runs) != 2:
        raise Exception('Interpolation must be performed on exactly two runs.')

    scaled_runs = []

    num_queries = round(nq)
    if nq < 0:
      num_queries = round(len(runs[0].run_data) / 1000)
    elif nq < 1:
      num_queries = round(len(runs[0].run_data) / 1000 * nq)
    elif nq == 1:
      num_queries = 1

    num_queries *= 1000

    print(f'num_queries: {num_queries}')
    for run in runs:
        run.run_data = run.run_data[:num_queries]

    scaled_runs.append(runs[0].clone().rescore(method=RescoreMethod.SCALE, scale=alpha))
    scaled_runs.append(runs[1].clone().rescore(method=RescoreMethod.SCALE, scale=(1-alpha)))

    return TrecRun.merge(scaled_runs, AggregationMethod.SUM, depth=depth, k=k)

def interpolation_custom(runs: List[TrecRun], path: str, depth: int = None, nq: float=12, k: int = None):
    """Perform fusion by interpolation on a list of exactly two ``TrecRun`` objects.
    new_score = first_run_score * alpha + (1 - alpha) * second_run_score.

    Parameters
    ----------
    runs : List[TrecRun]
        List of ``TrecRun`` objects. Exactly two runs.
    alpha : int
        Parameter alpha will be applied on the first run and (1 - alpha) will be applied on the second run.
    depth : int
        Maximum number of results from each input run to consider. Set to ``None`` by default, which indicates that
        the complete list of results is considered.
    k : int
        Length of final results list.  Set to ``None`` by default, which indicates that the union of all input documents
        are ranked.

    Returns
    -------
    TrecRun
        Output ``TrecRun`` that combines input runs via interpolation.
    """

    if len(runs) != 2:
        raise Exception('Interpolation must be performed on exactly two runs.')

    print('Interpolation custom')

    scaled_runs = []

    num_queries = round(nq)
    if nq < 0:
      num_queries = round(len(runs[0].run_data) / 1000)
    elif nq < 1:
      num_queries = round(len(runs[0].run_data) / 1000 * nq)
    elif nq == 1:
      num_queries = 1

    num_queries *= 1000

    print(f'num_queries: {num_queries}')
    for run in runs:
        run.run_data = run.run_data[:num_queries]

    with open(path, 'r') as file:
        alphas = file.readlines()
        alphas = [float(alpha) for alpha in alphas]
    
    print(f'{len(set(alphas))} alpha: {set(alphas)}')

    scaled_run0 = runs[0].clone()
    scaled_run1 = runs[1].clone()
    for idx, alpha in enumerate(alphas):
        # if idx < 100:
        #     print(f"alpha: {alpha} | run0: {scaled_run0.run_data.loc[idx * 1000: (idx + 1) * 1000 - 1, 'score'].values} | run1: {scaled_run1.run_data.loc[idx * 1000: (idx + 1) * 1000 - 1, 'score'].values}")
        scaled_run0.run_data.loc[idx * 1000: (idx + 1) * 1000 - 1, 'score'] *= alpha
        scaled_run1.run_data.loc[idx * 1000: (idx + 1) * 1000 - 1, 'score'] *= 1 - alpha
    #    scaled_run0.run_data[idx]['score'] = scaled_run0.run_data[idx]['score'].values * alpha
    #    scaled_run1.run_data[idx]['score'] = scaled_run1.run_data[idx]['score'].values * (1-alpha)

    scaled_runs.append(scaled_run0)
    scaled_runs.append(scaled_run1)

    return TrecRun.merge(scaled_runs, AggregationMethod.SUM, depth=depth, k=k)


def average(runs: List[TrecRun], depth: int = None, k: int = None):
    """Perform fusion by averaging on a list of ``TrecRun`` objects.

    Parameters
    ----------
    runs : List[TrecRun]
        List of ``TrecRun`` objects.
    depth : int
        Maximum number of results from each input run to consider. Set to ``None`` by default, which indicates that
        the complete list of results is considered.
    k : int
        Length of final results list.  Set to ``None`` by default, which indicates that the union of all input documents
        are ranked.

    Returns
    -------
    TrecRun
        Output ``TrecRun`` that combines input runs via averaging.
    """

    scaled_runs = [run.clone().rescore(method=RescoreMethod.SCALE, scale=(1/len(runs))) for run in runs]
    return TrecRun.merge(scaled_runs, AggregationMethod.SUM, depth=depth, k=k)
