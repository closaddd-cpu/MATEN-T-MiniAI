# BY MATEN/T | MATEN/T MiniAI v1.1 | M/ License v1.0
"""Train SentencePiece BPE using the training split of TXT/JSONL documents."""
from __future__ import annotations
from brand_guard import enforce_attribution
enforce_attribution()
import argparse
import os
from pathlib import Path
from pretrain import source_files,iter_documents

ROOT=Path(__file__).resolve().parent


def prepare(corpus,output,vocab_size=16384,max_documents=200000,force=False):
    import sentencepiece as spm
    output=Path(output)
    output.parent.mkdir(exist_ok=True,parents=True)
    if output.is_file() and not force:
        print(f'BY MATEN/T | reuse tokenizer {output}')
        return output
    files=source_files(corpus)
    corpus_txt=output.parent/'tokenizer_train_corpus.tmp.txt'
    count=0
    try:
        with corpus_txt.open('w',encoding='utf-8') as handle:
            for text in iter_documents(files,'train',limit=max_documents):
                line=' '.join(text.strip().split())
                if not line:continue
                handle.write(line[:16384]+'\n')
                count+=1
        if count<2:raise ValueError('At least two training documents required for a tokenizer')
        spm.SentencePieceTrainer.train(input=str(corpus_txt),model_prefix=str(output.with_suffix('')),
            vocab_size=vocab_size,model_type='bpe',character_coverage=1.0,hard_vocab_limit=False,
            bos_id=1,eos_id=2,pad_id=3,unk_id=0,
            user_defined_symbols=['<|user|>','<|assistant|>','<|end|>'],
            num_threads=min(8,os.cpu_count() or 2),max_sentence_length=16384)
    finally:corpus_txt.unlink(missing_ok=True)
    print(f'BY MATEN/T | tokenizer trained on {count:,} training documents -> {output}')
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',required=True)
    p.add_argument('--output',default=str(ROOT/'tokenizer'/'tokenizer.model'))
    p.add_argument('--vocab-size',type=int,default=16384)
    p.add_argument('--max-documents',type=int,default=200000)
    p.add_argument('--force',action='store_true')
    a=p.parse_args();prepare(a.corpus,a.output,a.vocab_size,a.max_documents,a.force)

if __name__=='__main__':main()
