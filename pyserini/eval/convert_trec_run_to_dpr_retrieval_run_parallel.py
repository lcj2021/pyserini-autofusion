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

import argparse
import json
import os
from tqdm import tqdm

from pyserini.search import get_topics, get_topics_with_reader
from pyserini.search.lucene import LuceneSearcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert an TREC run to DPR retrieval result json.')
    parser.add_argument('--topics', help='topic name')
    parser.add_argument('--topics-file', help='path to a topics file')
    parser.add_argument('--topics-reader', help='anserini TopicReader class')
    parser.add_argument('--index', required=True, help='Anserini Index that contains raw')
    parser.add_argument('--input', required=True, help='Input TREC run file.')
    parser.add_argument('--store-raw', action='store_true', help='Store raw text of passage')
    parser.add_argument('--regex', action='store_true', default=False, help="regex match")
    parser.add_argument('--combine-title-text', action='store_true', help="Make context the concatenation of title and text.")
    parser.add_argument('--output', required=True, help='Output DPR Retrieval json file.')
    args = parser.parse_args()

    if args.topics_file:
        qas = get_topics_with_reader(args.topics_reader, args.topics_file)
    elif args.topics:
        qas = get_topics(args.topics)
    else:
        print("No topics file or topics name was provided")

    if os.path.exists(args.index):
        searcher = LuceneSearcher(args.index)
    else:
        searcher = LuceneSearcher.from_prebuilt_index(args.index)
    if not searcher:
        exit()

    tokenizer = SimpleTokenizer()

    import concurrent.futures

    from tqdm import tqdm

    import itertools

    def read_lines(start_line, end_line):
        with open(args.input, 'r') as f:
            selected_lines = list(itertools.islice(f, start_line, end_line))
            return selected_lines
        
    from multiprocessing import Manager, Process, Lock

    manager = Manager()
    retrieval = manager.dict()
    doc_contents = manager.dict()
    locks = manager.dict()

    def read_doc_content(doc_id):
        if doc_id not in doc_contents:
            doc_contents[doc_id] = json.loads(searcher.doc(doc_id).raw())

    def process_line(line, retrieval):
        question_id, _, doc_id, _, score, _ = line.strip().split()
        question_id = int(question_id)
        question = qas[question_id]['title']
        answers = qas[question_id]['answers']
        if answers[0] == '"':
            answers = answers[1:-1].replace('""', '"')
        answers = eval(answers)
        # print(f'{question_id} {question} {doc_id} {score}')
        
        if args.combine_title_text:
            # read_doc_content(doc_id)
            ctx = doc_contents[doc_id]['title'] + "\n" + doc_contents[doc_id]['text']
        else:
            # read_doc_content(doc_id)
            ctx = doc_contents[doc_id]['contents']
        # print(f'{ctx}')
            

        lock = Lock()
        if question_id not in retrieval:
            # lock.acquire()
            retrieval[question_id] = {'question': question, 'answers': answers, 'contexts': manager.list()}
            # lock.release()
            

        title, text = ctx.split('\n')
        answer_exist = has_answers(text, answers, tokenizer, args.regex)
        
        if args.store_raw:
            retrieval[question_id]['contexts'].append(
                {'docid': doc_id,
                'score': score,
                'text': ctx,
                'has_answer': answer_exist}
            )
        else:
            # lock.acquire()
            retrieval[question_id]['contexts'].append(
                {'docid': doc_id, 'score': score, 'has_answer': answer_exist}
            )
            # lock.release()
        # print(retrieval[question_id]['contexts'])
        # print(f'{question_id} {question} {doc_id} {score} {answer_exist}')

    def process_lines(start_line, end_line):
        lines = read_lines(start_line, end_line)
        for line in lines:
            process_line(line, retrieval)
        
    num_threads = 8
    num_processes = num_threads
    lines_per_worker = 0  # 每个线程处理的行数
    k = 0

    with open(args.input, 'r') as file:
        while True:
            line = file.readline()
            question_id, _, doc_id, _, score, _ = line.split()
            if int(question_id) == 0:
                k += 1
            else:
                break

    score_dict = {}

    with open(args.input) as file:
        total_lines = sum(1 for _ in file)
    total_lines = total_lines // 4
    print(f'total_lines: {total_lines}')
    lines_per_worker = int((int((total_lines + num_processes - 1) / num_processes) + k - 1) / k) * k
    print(f'lines_per_worker: {lines_per_worker}')

    pre = read_lines(0, total_lines)
    for line in pre:
        question_id, _, doc_id, _, score, _ = line.strip().split()
        question_id = int(question_id)
        read_doc_content(doc_id)

    print('Starting to process lines...')
    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    #     futures = []
    #     progress_bar = tqdm(total=total_lines)

    #     for i in range(0, total_lines, lines_per_worker):
    #         start_line = i
    #         end_line = min(i + lines_per_worker, total_lines)
    #         futures.append(executor.submit(process_lines, start_line, end_line, score_dict))
    #         progress_bar.update(lines_per_worker)

    #     concurrent.futures.wait(futures)

    #     progress_bar.close()

    # score_dict = sorted(score_dict.items(), key=lambda x: int(x[0]))
    
    processes = []
    progress_bar = tqdm(total=total_lines)
    
    for i in range(0, total_lines, lines_per_worker):
        start_line = i
        end_line = min(i + lines_per_worker, total_lines)
        p = Process(target=process_lines, args=(start_line, end_line))
        p.start()
        processes.append(p)
        progress_bar.update(lines_per_worker)
    
    for p in processes:
        p.join()
    
    progress_bar.close()

    # retrieval = dict(sorted(retrieval.items(), key=lambda x: int(x[0])))
    retrieval_dict = dict(retrieval)
    for question_id in retrieval_dict:
        retrieval_dict[question_id]['contexts'] = list(retrieval_dict[question_id]['contexts'])

    json.dump(retrieval_dict, open(args.output, 'w'), indent=4, ensure_ascii=False)