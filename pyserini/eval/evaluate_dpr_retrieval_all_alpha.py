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

"""
Most of the tokenization code here is copied from Facebook/DPR & DrQA codebase to avoid adding an extra dependency
"""

import argparse
import copy
import json
import logging
import re
import unicodedata
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

import regex

logger = logging.getLogger(__name__)


class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def has_answers(text, answers, tokenizer, regex=False):
    text = _normalize(text)
    if regex:
        for ans in answers:
            ans = _normalize(ans)
            if regex_match(text, ans):
                return True
    else:
        text = tokenizer.tokenize(text).words(uncased=True)
        for ans in answers:
            ans = _normalize(ans)
            ans = tokenizer.tokenize(ans).words(uncased=True)
            for i in range(0, len(text) - len(ans) + 1):
                if ans == text[i: i + len(ans)]:
                    return True
    return False


def evaluate_retrieval(retrieval_file, topk, alpha, qid_alpha_numhit, regex=False):
    tokenizer = SimpleTokenizer()
    retrieval_file_path = retrieval_file.replace('?', f'{round(alpha, 2)}')
    retrieval = json.load(open(retrieval_file_path))
    max_k = max(topk)

    for idx, qid in tqdm(enumerate(list(retrieval.keys()))):

        answers = retrieval[qid]['answers']
        contexts = retrieval[qid]['contexts']

        num_hit = 0
        for idx, ctx in enumerate(contexts):
            if idx >= max_k:
                break
            if 'has_answer' in ctx:
                if ctx['has_answer']:
                    has_ans_idx = idx
                    num_hit += 1
            else:
                text = ctx['text'].split('\n')[1]  # [0] is title, [1] is text
                if has_answers(text, answers, tokenizer, regex):
                    has_ans_idx = idx
                    num_hit += 1
        qid = int(qid)
        qid_alpha_numhit[qid][alpha] = num_hit

def evaluate_retrieval_rrf(retrieval_file, topk, alpha, qid_alpha_numhit, regex=False):
    tokenizer = SimpleTokenizer()
    retrieval_file_path = retrieval_file.replace('?', 'rrf')
    retrieval = json.load(open(retrieval_file_path))
    max_k = max(topk)

    for idx, qid in tqdm(enumerate(list(retrieval.keys()))):

        answers = retrieval[qid]['answers']
        contexts = retrieval[qid]['contexts']

        num_hit = 0
        for idx, ctx in enumerate(contexts):
            if idx >= max_k:
                break
            if 'has_answer' in ctx:
                if ctx['has_answer']:
                    has_ans_idx = idx
                    num_hit += 1
            else:
                text = ctx['text'].split('\n')[1]  # [0] is title, [1] is text
                if has_answers(text, answers, tokenizer, regex):
                    has_ans_idx = idx
                    num_hit += 1
        qid = int(qid)
        qid_alpha_numhit[qid][alpha] = num_hit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval', type=str, help="Path to retrieval output file.")
    parser.add_argument('--topk', type=int, nargs='+', help="topk to evaluate")
    parser.add_argument('--regex', action='store_true', default=False, help="regex match")
    parser.add_argument('--traindata', default=True, help="regex match")
    args = parser.parse_args()

    # write code to load retrieval file at alpha 0
    retrieval_file_0_path = args.retrieval.replace('?', '0.0')
    # print(retrieval_file_0_path)


    retrieval_file_0 = json.load(open(retrieval_file_0_path))
    nq = len(retrieval_file_0)

    # print(f'Evaluating {nq} queries')

    """
    alphas estimate from np.arange(0, 1.05, 0.05), using softmax
    """

    # # key: qid, value: {alpha: answer_hit}
    # qid_alpha_numhit = [{k: {} for k in np.arange(0, 1.05, 0.05)} for _ in range(nq)]

    # for alpha in tqdm(np.arange(0, 1.05, 0.05)):
    #     evaluate_retrieval(args.retrieval, args.topk, alpha, qid_alpha_numhit, args.regex)

    # final_qid_alpha_numhit = []
    # avg_alpha = 0.
    
    # num_c = 7
    # num_critical = [0] * num_c
    # critical_hits = []
    # coverage = 0.
    # hard_fail = 0
    # for qid, alpha_numhit in enumerate(qid_alpha_numhit):
    #     total_hit = 0
    #     weighted_alpha = 0.
    #     max_answer_hit = max(alpha_numhit.values())
    #     total_hit = sum(alpha_numhit.values())

    #     hits = [hit for _, hit in alpha_numhit.items()]
    #     alphas = [alpha for alpha, _ in alpha_numhit.items()]

    #     hits_np = np.array(hits).astype(float)
    #     alphas_np = np.array(alphas).astype(float)
        
    #     is_critical = False
    #     for c in range(0, num_c):
    #         if np.sum(hits_np != 0) == c + 1:
    #             is_critical = True
    #             num_critical[c] += 1
    #             critical_hits.append(alpha_numhit.values())
    #     # if not is_critical:
    #     coverage += alpha_numhit[0.5] > 0

    #     if total_hit == 0:
    #         weighted_alpha = 0.5
    #         final_qid_alpha_numhit.append((qid, weighted_alpha, total_hit))

    #         # print(f'qid: {qid}\thits: {list(hits)}\tsoftmax_hits: {list(softmax_hits)}\tweighted_alpha: {weighted_alpha}')
    #         avg_alpha += weighted_alpha
    #         hard_fail += 1
    #         continue

    #     hits_np /= np.max(hits_np)
    #     hits_np[hits_np == 0] -= 1e6

    #     # softmax hits_np:
    #     softmax_hits = np.exp(hits_np) / np.sum(np.exp(hits_np))
    #     softmax_hits = np.round(softmax_hits, decimals=2)

    #     weighted_alpha = np.dot(softmax_hits, alphas_np)

    #     weighted_alpha = round(weighted_alpha, 2)

    #     # max_index = np.argmax(hits)
    #     # weighted_alpha = round(alphas[max_index], 2)
    #     # assert hits[max_index] > 0

    #     final_qid_alpha_numhit.append((qid, weighted_alpha, total_hit))

    #     # print(f'qid: {qid}\thits: {list(hits)}\tsoftmax_hits: {list(softmax_hits)}\tweighted_alpha: {weighted_alpha}')
    #     avg_alpha += weighted_alpha
    
    # # avg_alpha /= len(final_qid_alpha_numhit)
    # # print(f'Average alpha: {avg_alpha:.4f}')

    # # coverage /= (len(qid_alpha_numhit))
    # # print(f'Coverage: {coverage:.4f}')

    # # for c in range(0, num_c):
    # #     print(f'Num critical[{c + 1}]: {num_critical[c]}')
    # # for critical_alpha in critical_hits:
    # #     print(f'Critical alpha: {critical_alpha}')
    
    # for qid, best_alpha, answer_hit in final_qid_alpha_numhit:
    #     print(f'{qid}\t{best_alpha}\t{answer_hit}')
    # # print(f'Hard fail: {hard_fail}')
    # # print(f'total: {len(final_qid_alpha_numhit)}')


    """
    alphas only in [0, 0.5, 1]
    """


    # key: qid, value: {alpha: answer_hit}
    qid_alpha_numhit = [{k: {} for k in np.arange(0, 1.05, 0.5)} for _ in range(nq)]

    for alpha in tqdm(np.arange(0, 1.05, 0.5)):
        if alpha != 0.5:
            evaluate_retrieval(args.retrieval, args.topk, alpha, qid_alpha_numhit, args.regex)
        else:
            evaluate_retrieval_rrf(args.retrieval, args.topk, alpha, qid_alpha_numhit, args.regex)

    num_classes = [0] * 3
    final_qid_alpha_numhit = []
    for qid, alpha_numhit in enumerate(qid_alpha_numhit):
        total_hit = 0
        label_class = -1

        hits = [1 if hit else 0 for _, hit in alpha_numhit.items()]
        total_hit = sum(hits)
        alphas = [alpha for alpha, _ in alpha_numhit.items()]

        if total_hit == 0:
            label_class = 1
            num_classes[label_class] += 1
            final_qid_alpha_numhit.append((qid, label_class, total_hit))

            print(f'qid: {qid}\thits: {list(hits)}')
            continue
        if total_hit == 1:
            label_class = hits.index(1)
        elif total_hit == 2:
            zero_index = hits.index(0)
            label_class = 2 - zero_index
        elif total_hit == 3:
            label_class = 1
        else:
            raise Exception('total_hit should be 0, 1, 2, or 3')

        num_classes[label_class] += 1
        final_qid_alpha_numhit.append((qid, label_class, total_hit))

        print(f'qid: {qid}\thits: {list(hits)}')
    
    # for qid, best_alpha, answer_hit in final_qid_alpha_numhit:
    #     print(f'{qid}\t{best_alpha}\t{answer_hit}')
    # print(f'Num classes: {num_classes}')